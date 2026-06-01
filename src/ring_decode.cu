/// @file
/// Multi-rank ring decode loop.
///
/// One step processes `cp_size` ring rotations. Per rank:
///   - Originating rank at step s is `o = (rank - s + cp_size) % cp_size`.
///   - Per-step seq_k = current_len_per_rank[o] — variable across ranks/steps.
///   - K/V are packed before the first kernel call so the kernel sees a
///     contiguous (B, kv_H, Sk, D) layout (cache stride is S_max).
///   - Double-buffered transit (K_cur / K_recv, V_cur / V_recv) sized to the
///     maximum chunk across all ranks.
///   - Causal=false in the kernel: the cache only holds past tokens, so all
///     cached positions are admissible (online-softmax accumulates them
///     unmasked).

#include <cuda_fp16.h>
#include <math_constants.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "device_tensor.hpp"
#include "kv_cache.hpp"
#include "ring_decode.hpp"
#include "ring_loop.hpp"

#ifdef RING_USE_NCCL
#include "nccl_utils.hpp"
#endif

namespace ring_attention {

namespace {

/// Merge `cp_size` per-rank online-softmax partials into the global
/// (O, m, l) using the FlashAttention partial-merge recurrence.
///
/// `gathered` is the all-gathered partial buffer, laid out as `cp_size`
/// contiguous blocks of `B*H*(D+2)` floats: `[O (B*H*D) | m (B*H) | l (B*H)]`.
/// O_r is the *unnormalized* weighted sum on rank r; m_r/l_r its running max
/// and denominator. Output: `out` ← global unnormalized O, `m_out` ← global
/// max, `l_out` ← global denominator (the caller then divides via
/// `launch_attention_finalize`). One block per (b,h) row, `D` threads.
__global__ void decode_partial_merge_kernel(const float* __restrict__ gathered,
                                            float* __restrict__ out, float* __restrict__ m_out,
                                            float* __restrict__ l_out, int cp_size, int BH, int D) {
  const int bh = blockIdx.x;
  const int d = threadIdx.x;
  if (bh >= BH || d >= D) return;

  const long block = (long)BH * D + 2 * BH;  // floats per rank
  const long o_base = (long)bh * D + d;
  const int m_off = BH * D + bh;
  const int l_off = BH * D + BH + bh;

  // Global max over ranks (every thread recomputes — cheap, cp_size is small).
  float m_max = -CUDART_INF_F;
  for (int r = 0; r < cp_size; ++r) {
    const float m_r = gathered[(long)r * block + m_off];
    m_max = fmaxf(m_max, m_r);
  }

  float o_acc = 0.f, l_acc = 0.f;
  for (int r = 0; r < cp_size; ++r) {
    const float m_r = gathered[(long)r * block + m_off];
    const float scale = (m_r == -CUDART_INF_F) ? 0.f : __expf(m_r - m_max);
    o_acc += scale * gathered[(long)r * block + o_base];
    if (d == 0) l_acc += scale * gathered[(long)r * block + l_off];
  }
  out[(long)bh * D + d] = o_acc;
  if (d == 0) {
    m_out[bh] = m_max;
    l_out[bh] = l_acc;
  }
}

}  // namespace

RingResult run_ring_decode_step(const RingDecodeConfig& cfg, const float* q, const float* k_new,
                                const float* v_new, int owner_rank, DeviceKVCache<float>& cache,
                                const std::vector<int>& current_len_per_rank, float* out) {
#ifndef RING_USE_NCCL
  (void)cfg;
  (void)q;
  (void)k_new;
  (void)v_new;
  (void)owner_rank;
  (void)cache;
  (void)current_len_per_rank;
  (void)out;
  fprintf(stderr, "run_ring_decode_step requires NCCL (build with USE_NCCL=ON)\n");
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  return RingResult{};  // unreachable; satisfy compiler
#else
  const int rank = cfg.rank;
  const int cp_size = cfg.cp_size;
  const int B = cfg.batch;
  const int H = cfg.heads;
  const int kv_H = cache.kv_heads();
  const int D = cache.head_dim();

  if (static_cast<int>(current_len_per_rank.size()) != cp_size) {
    fprintf(stderr,
            "run_ring_decode_step: current_len_per_rank.size()=%zu does not match cp_size=%d\n",
            current_len_per_rank.size(), cp_size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // 1. Owner appends new K/V into its local cache.
  if (rank == owner_rank) {
    cache.append(k_new, v_new);
  }
  if (cache.current_len() != current_len_per_rank[rank]) {
    fprintf(stderr,
            "run_ring_decode_step: cache.current_len()=%d on rank %d disagrees with "
            "current_len_per_rank[rank]=%d\n",
            cache.current_len(), rank, current_len_per_rank[rank]);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // 2. Compute this rank's local partial, then merge partials across ranks.
  //    Q is a single token replicated on every rank and the KV is sharded, so
  //    instead of rotating the (large) KV around the ring we keep KV resident
  //    and compute attention(q, local_shard) -> partial (O, m, l) locally. The
  //    partials are then all-gathered (a few KB) and merged with the
  //    FlashAttention recurrence. Online softmax is associative, so this is
  //    bit-for-bit the sequential ring's accumulation but moves ~B*H*(D+2)
  //    floats per rank instead of the whole KV shard each step — at cp_size=8
  //    that is ~33 KB total vs. tens of MB.  See KERNEL_OPTIMIZATIONS.md
  //    Round 13.
  const int local_len = current_len_per_rank[rank];

  // Online-softmax state for one query row.
  AttentionShape init_sh{B, H, 1, local_len, D};
  init_sh.kv_heads = kv_H;
  const std::size_t m_count = static_cast<std::size_t>(B) * H;
  DeviceTensor<float> m_d(m_count), l_d(m_count);
  launch_attention_init(out, m_d.data(), l_d.data(), init_sh, m_count, 0);

  // Partial layout for the all-gather: [O (B*H*D) | m (B*H) | l (B*H)].
  const int BH = B * H;
  const std::size_t partial_elem = static_cast<std::size_t>(BH) * D + 2 * BH;
  DeviceTensor<float> send_partial(partial_elem);
  DeviceTensor<float> recv_partials(partial_elem * cp_size);

  cudaEvent_t ev_comm_start, ev_comm_end, ev_comp_start, ev_comp_end;
  cudaEventCreate(&ev_comm_start);
  cudaEventCreate(&ev_comm_end);
  cudaEventCreate(&ev_comp_start);
  cudaEventCreate(&ev_comp_end);

  // Leading barrier aligns the ranks before the collective all-gather. It is
  // not pure overhead: without it, inter-token rank skew falls into the timed
  // all-gather and *raises* the measured per-token latency (Round 15 tried
  // removing it and regressed cp4 0.232 -> 0.388 ms — see
  // KERNEL_OPTIMIZATIONS.md).
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();

  // 3. Local partial: attention(q, local KV shard). The decode kernel reads the
  //    KV cache *in place* (per-head row stride = cache.s_max()), so there is
  //    no per-token de-stride pack copy (Round 17). Leaves (out, m_d, l_d) as
  //    the *unnormalized* online-softmax partial (no finalize yet).
  AttentionShape step_sh{B, H, 1, local_len, D};
  step_sh.kv_heads = kv_H;
  cudaEventRecord(ev_comp_start, 0);
  launch_attention_decode_step(q, cache.k_data(), cache.v_data(), out, m_d.data(), l_d.data(),
                               step_sh, /*q_offset=*/0, /*k_offset=*/0, /*causal=*/false,
                               /*stream=*/0, /*kv_row_stride=*/cache.s_max());
  cudaEventRecord(ev_comp_end, 0);

  // 4. Pack the partial contiguously and all-gather it across the ring.
  cudaCheck(cudaMemcpyAsync(send_partial.data(), out, (std::size_t)BH * D * sizeof(float),
                            cudaMemcpyDeviceToDevice, 0));
  cudaCheck(cudaMemcpyAsync(send_partial.data() + (std::size_t)BH * D, m_d.data(),
                            (std::size_t)BH * sizeof(float), cudaMemcpyDeviceToDevice, 0));
  cudaCheck(cudaMemcpyAsync(send_partial.data() + (std::size_t)BH * D + BH, l_d.data(),
                            (std::size_t)BH * sizeof(float), cudaMemcpyDeviceToDevice, 0));

  cudaEventRecord(ev_comm_start, 0);
  NCCL_CHECK(ncclAllGather(send_partial.data(), recv_partials.data(), partial_elem, ncclFloat,
                           cfg.nccl_comm, 0));
  cudaEventRecord(ev_comm_end, 0);

  // 5. Merge the cp_size partials -> global (O, m, l), then finalize (O / l).
  {
    const dim3 grid(BH);
    const dim3 block(D);
    decode_partial_merge_kernel<<<grid, block, 0, 0> > >(recv_partials.data(), out, m_d.data(),
                                                         l_d.data(), cp_size, BH, D);
    cudaCheck(cudaGetLastError());
  }
  AttentionShape final_sh{B, H, 1, 1, D};
  final_sh.kv_heads = kv_H;
  launch_attention_finalize(out, l_d.data(), final_sh, 0);
  cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  float comp_ms = 0.f, comm_ms = 0.f;
  cudaEventElapsedTime(&comp_ms, ev_comp_start, ev_comp_end);
  cudaEventElapsedTime(&comm_ms, ev_comm_start, ev_comm_end);

  cudaEventDestroy(ev_comm_start);
  cudaEventDestroy(ev_comm_end);
  cudaEventDestroy(ev_comp_start);
  cudaEventDestroy(ev_comp_end);

  RingResult res;
  res.comm_ms = comm_ms;
  res.comp_ms = comp_ms;
  res.wait_ms = 0.0;
  res.total_ms = (t_end - t_start) * 1e3;
  res.max_err = -1.f;
  return res;
#endif
}

}  // namespace ring_attention
