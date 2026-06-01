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

/// Pack the populated `Sk` rows of the cache into a contiguous
/// `(B, kv_H, Sk, D)` tile on the same stream.
void pack_local_cache(const DeviceKVCache<float>& cache, float* k_dst, float* v_dst, int Sk,
                      cudaStream_t stream) {
  const int B = cache.batch();
  const int kv_H = cache.kv_heads();
  const int D = cache.head_dim();
  const int S_max = cache.s_max();
  const std::size_t width = static_cast<std::size_t>(Sk) * D * sizeof(float);
  const std::size_t spitch = static_cast<std::size_t>(S_max) * D * sizeof(float);
  const std::size_t height = static_cast<std::size_t>(B) * kv_H;
  cudaCheck(cudaMemcpy2DAsync(k_dst, width, cache.k_data(), spitch, width, height,
                              cudaMemcpyDeviceToDevice, stream));
  cudaCheck(cudaMemcpy2DAsync(v_dst, width, cache.v_data(), spitch, width, height,
                              cudaMemcpyDeviceToDevice, stream));
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

  // 2. Allocate transit buffers sized for the largest chunk across ranks.
  const int max_chunk = *std::max_element(current_len_per_rank.begin(), current_len_per_rank.end());
  const std::size_t transit_elem = static_cast<std::size_t>(B) * kv_H * max_chunk * D;
  DeviceTensor<float> K_a(transit_elem), K_b(transit_elem);
  DeviceTensor<float> V_a(transit_elem), V_b(transit_elem);
  float* K_cur = K_a.data();
  float* V_cur = V_a.data();
  float* K_recv = K_b.data();
  float* V_recv = V_b.data();

  // 3. Pack local cache into K_cur/V_cur.
  pack_local_cache(cache, K_cur, V_cur, current_len_per_rank[rank], 0);

  // 4. Init online-softmax state for one query row.
  AttentionShape init_sh{B, H, 1, current_len_per_rank[rank], D};
  init_sh.kv_heads = kv_H;
  const std::size_t m_count = static_cast<std::size_t>(B) * H;
  DeviceTensor<float> m_d(m_count), l_d(m_count);
  launch_attention_init(out, m_d.data(), l_d.data(), init_sh, m_count, 0);

  const int next_rank = (rank + 1) % cp_size;
  const int prev_rank = (rank + cp_size - 1) % cp_size;

  cudaEvent_t ev_comm_start, ev_comm_end, ev_comp_start, ev_comp_end;
  cudaEventCreate(&ev_comm_start);
  cudaEventCreate(&ev_comm_end);
  cudaEventCreate(&ev_comp_start);
  cudaEventCreate(&ev_comp_end);

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  double comm_acc = 0.0, comp_acc = 0.0;

  // 5. Ring loop: cp_size steps. At step s, K_cur holds the chunk originating
  //    from rank `o = (rank - s + cp_size) % cp_size`. We send K_cur to next
  //    (which will use it at step s+1) and receive K_recv from prev (the chunk
  //    we'll use at step s+1).
  for (int step = 0; step < cp_size; ++step) {
    const int origin = (rank - step + cp_size) % cp_size;
    const int chunk_len = current_len_per_rank[origin];

    // 5a. Comm: send K_cur, recv into K_recv (skip on last step).
    if (step < cp_size - 1) {
      const int next_origin = (rank - step - 1 + cp_size) % cp_size;
      const int recv_chunk_len = current_len_per_rank[next_origin];
      const std::size_t send_count = static_cast<std::size_t>(B) * kv_H * chunk_len * D;
      const std::size_t recv_count = static_cast<std::size_t>(B) * kv_H * recv_chunk_len * D;

      cudaEventRecord(ev_comm_start, 0);
      ncclGroupStart();
      NCCL_CHECK(ncclSend(K_cur, send_count, ncclFloat, next_rank, cfg.nccl_comm, 0));
      NCCL_CHECK(ncclRecv(K_recv, recv_count, ncclFloat, prev_rank, cfg.nccl_comm, 0));
      NCCL_CHECK(ncclSend(V_cur, send_count, ncclFloat, next_rank, cfg.nccl_comm, 0));
      NCCL_CHECK(ncclRecv(V_recv, recv_count, ncclFloat, prev_rank, cfg.nccl_comm, 0));
      NCCL_CHECK(ncclGroupEnd());
      cudaEventRecord(ev_comm_end, 0);
    }

    // 5b. Compute: launch_attention_step over the current chunk.
    AttentionShape step_sh{B, H, 1, chunk_len, D};
    step_sh.kv_heads = kv_H;
    cudaEventRecord(ev_comp_start, 0);
    launch_attention_step(q, K_cur, V_cur, out, m_d.data(), l_d.data(), step_sh,
                          /*q_offset=*/0, /*k_offset=*/0, /*causal=*/false, 0);
    cudaEventRecord(ev_comp_end, 0);
    cudaEventSynchronize(ev_comp_end);

    float comp_ms = 0.f;
    cudaEventElapsedTime(&comp_ms, ev_comp_start, ev_comp_end);
    comp_acc += comp_ms;
    if (step < cp_size - 1) {
      float comm_ms = 0.f;
      cudaEventElapsedTime(&comm_ms, ev_comm_start, ev_comm_end);
      comm_acc += comm_ms;
    }

    // 5c. Promote received → current for next step.
    std::swap(K_cur, K_recv);
    std::swap(V_cur, V_recv);
  }

  // 6. Finalize: divide out by accumulated softmax denominator.
  AttentionShape final_sh{B, H, 1, 1, D};
  final_sh.kv_heads = kv_H;
  launch_attention_finalize(out, l_d.data(), final_sh, 0);
  cudaDeviceSynchronize();

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  cudaEventDestroy(ev_comm_start);
  cudaEventDestroy(ev_comm_end);
  cudaEventDestroy(ev_comp_start);
  cudaEventDestroy(ev_comp_end);

  RingResult res;
  res.comm_ms = comm_acc;
  res.comp_ms = comp_acc;
  res.wait_ms = 0.0;
  res.total_ms = (t_end - t_start) * 1e3;
  res.max_err = -1.f;
  return res;
#endif
}

}  // namespace ring_attention
