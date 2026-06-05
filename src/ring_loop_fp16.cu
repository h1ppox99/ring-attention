/// @file
/// FP16 / Tensor-Core ring-attention orchestrator. Sibling of `ring_loop.cu`.
/// Same three modes (allgather, ring-blocking, ring-overlap) and same timing
/// instrumentation; differences vs the FP32 path:
///   - K/V live in device `__half` buffers (half the bytes per element).
///   - MPI transports the raw bytes (`MPI_BYTE`) — halves wire traffic.
///   - Q is uploaded once as fp32 then cast to `__half` on the device.
///   - Persistent state (O, m, l) stays fp32 — the online-softmax accumulators
///     need the range; `launch_attention_init` / `launch_attention_finalize`
///     are reused unchanged.
///   - Kernel calls dispatch to `launch_attention_step_fp16`.
///
/// The FP32 path is bit-for-bit untouched; both can be selected at runtime
/// from `run_ring_attention` via `RingConfig::dtype`.

#include <cuda_fp16.h>
#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "attention.hpp"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"
#include "nccl_utils.hpp"
#include "ring_gen.hpp"
#include "ring_loop.hpp"
#include "ring_loop_common.hpp"
#include "ring_partition.hpp"

namespace {

// The MPI guards, gen_elem fills, timed-loop driver, and CPU-reference
// verification are shared with the FP32 path; see ring_loop_common.hpp.
using ring_attention::detail::fill_host_tensor;
using ring_attention::detail::fill_region;
using ring_attention::detail::mpi_die;
using ring_attention::detail::mpi_int_count;

/// Upload a host fp32 buffer and cast it on the device into a __half buffer.
/// Scratch is reused across calls to amortise allocation; `scratch_count` must
/// be ≥ `host.size()`.
void upload_as_half(const std::vector<float>& host, ring_attention::DeviceTensor<float>& scratch,
                    __half* dst, std::size_t n) {
  scratch.copy_from_host(host);
  ring_attention::launch_float_to_half(scratch.data(), dst, n);
}

// ---------------------------------------------------------------------------
// AllGather baseline (fp16)
// ---------------------------------------------------------------------------

ring_attention::RingResult run_allgather_fp16(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;
  const int Sl = S / P;
  const int q_off = R * Sl;

  const std::size_t q_local_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t kv_local_elem = static_cast<std::size_t>(B) * kv_H * Sl * D;
  const std::size_t kv_full_elem = static_cast<std::size_t>(B) * kv_H * S * D;
  const std::size_t m_count = static_cast<std::size_t>(B) * H * Sl;
  const std::size_t kv_local_bytes = kv_local_elem * sizeof(__half);  // MPI byte count

  // Host-side fill (fp32) — constant across iterations.
  std::vector<float> q_h, k_h, v_h;
  fill_host_tensor(q_h, cfg.seed, 0, B, H, Sl, D, q_off);
  fill_host_tensor(k_h, cfg.seed, 1, B, kv_H, Sl, D, q_off);
  fill_host_tensor(v_h, cfg.seed, 2, B, kv_H, Sl, D, q_off);

  // Convert K/V to fp16 once (payload that rides MPI).
  std::vector<__half> k_h16(kv_local_elem), v_h16(kv_local_elem);
  for (std::size_t i = 0; i < kv_local_elem; ++i) {
    k_h16[i] = __float2half(k_h[i]);
    v_h16[i] = __float2half(v_h[i]);
  }

  // Q stays put as __half on the device.
  DeviceTensor<float> q_scratch_f(q_local_elem);
  DeviceTensor<__half> q_d(q_local_elem);
  upload_as_half(q_h, q_scratch_f, q_d.data(), q_local_elem);

  std::vector<__half> k_gathered(kv_full_elem), v_gathered(kv_full_elem);
  std::vector<__half> full_k_h(kv_full_elem), full_v_h(kv_full_elem);
  DeviceTensor<__half> k_d(kv_full_elem), v_d(kv_full_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);
  AttentionShape shape{B, H, Sl, S, D};
  shape.kv_heads = kv_H;

  const int kv_local_bytes_int = mpi_int_count(kv_local_bytes, "run_allgather_fp16/MPI_Allgather");

  auto one_pass = [&]() -> std::tuple<double, double, double> {
    const double tc0 = MPI_Wtime();
    MPI_Allgather(k_h16.data(), kv_local_bytes_int, MPI_BYTE, k_gathered.data(), kv_local_bytes_int,
                  MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(v_h16.data(), kv_local_bytes_int, MPI_BYTE, v_gathered.data(), kv_local_bytes_int,
                  MPI_BYTE, MPI_COMM_WORLD);

    // Rearrange (P,B,kv_H,Sl,D) → (B,kv_H,S,D).
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < kv_H; ++h)
        for (int p = 0; p < P; ++p) {
          const std::size_t src = static_cast<std::size_t>(p) * kv_local_elem +
                                  static_cast<std::size_t>(b * kv_H + h) * Sl * D;
          const std::size_t dst = (static_cast<std::size_t>(b * kv_H + h) * S + p * Sl) * D;
          std::copy(k_gathered.data() + src, k_gathered.data() + src + Sl * D,
                    full_k_h.data() + dst);
          std::copy(v_gathered.data() + src, v_gathered.data() + src + Sl * D,
                    full_v_h.data() + dst);
        }

    cudaMemcpy(k_d.data(), full_k_h.data(), kv_full_elem * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d.data(), full_v_h.data(), kv_full_elem * sizeof(__half), cudaMemcpyHostToDevice);
    const double tc1 = MPI_Wtime();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), shape, m_count);
    cudaEventRecord(ev0);
    launch_attention_step_fp16(q_d.data(), k_d.data(), v_d.data(), out_d.data(), m_d.data(),
                               l_d.data(), shape, q_off, /*k_offset=*/0, cfg.causal);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    launch_attention_finalize(out_d.data(), l_d.data(), shape);
    cudaDeviceSynchronize();
    float comp_float_ms = 0.f;
    cudaEventElapsedTime(&comp_float_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return {(tc1 - tc0) * 1e3, static_cast<double>(comp_float_ms), 0.0};
  };

  RingResult res;
  detail::run_timed_passes(cfg, one_pass, res);

  if (cfg.verify) {
    one_pass();
    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);
    res.max_err =
        detail::verify_local_output(cfg, B, H, S, D, kv_H, Sl, q_local_elem, {q_off}, dev_out_h);
  }
  return res;
}

// ---------------------------------------------------------------------------
// Ring rotation, blocking style (fp16)
// ---------------------------------------------------------------------------

ring_attention::RingResult run_ring_blocking_fp16(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;

  const RingPartition::Mode mode =
      cfg.zigzag ? RingPartition::Mode::Zigzag : RingPartition::Mode::Contiguous;
  RingPartition part(P, R, S, mode);
  const int nsg = part.num_sub_groups();
  const int Sl = part.local_chunk_len();
  const int Sl_local = Sl * nsg;
  const std::size_t q_sg_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t q_local_elem = static_cast<std::size_t>(nsg) * q_sg_elem;
  const std::size_t kv_sg_elem = static_cast<std::size_t>(B) * kv_H * Sl * D;
  const std::size_t kv_local_elem = static_cast<std::size_t>(nsg) * kv_sg_elem;
  const std::size_t m_sg = static_cast<std::size_t>(B) * H * Sl;
  const std::size_t m_count = static_cast<std::size_t>(nsg) * m_sg;
  // Used only on the MPI path; the NCCL build never references it.
  [[maybe_unused]] const std::size_t kv_bytes = kv_local_elem * sizeof(__half);

  const int next_rank = part.next_rank();
  const int prev_rank = part.prev_rank();

  std::vector<float> q_h(q_local_elem), k_h_init(kv_local_elem), v_h_init(kv_local_elem);
  for (int sg = 0; sg < nsg; ++sg) {
    fill_region(q_h, sg * q_sg_elem, cfg.seed, 0, B, H, Sl, D, part.q_offset(sg));
    fill_region(k_h_init, sg * kv_sg_elem, cfg.seed, 1, B, kv_H, Sl, D,
                part.k_offset_for_step(0, sg));
    fill_region(v_h_init, sg * kv_sg_elem, cfg.seed, 2, B, kv_H, Sl, D,
                part.k_offset_for_step(0, sg));
  }

  // Q (fp16 on device) — uploaded once. Scratch is reused below to stage K/V,
  // so size it to fit the largest payload (kv_local_elem can exceed q_local_elem
  // if kv_heads > heads).
  DeviceTensor<float> scratch_f(std::max(q_local_elem, kv_local_elem));
  DeviceTensor<__half> q_d(q_local_elem);
  upload_as_half(q_h, scratch_f, q_d.data(), q_local_elem);

  // Double-buffered K/V (fp16 on device), sized by kv_H.
  DeviceTensor<__half> K_a(kv_local_elem), K_b(kv_local_elem);
  DeviceTensor<__half> V_a(kv_local_elem), V_b(kv_local_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);

  const AttentionShape init_shape{B, H, Sl_local, S, D};
  AttentionShape sg_shape{B, H, Sl, Sl, D};
  sg_shape.kv_heads = kv_H;

#ifdef RING_USE_NCCL
  // NCCL communicator: GPU-to-GPU direct, no host staging needed.
  ncclComm_t nccl_comm = ring_attention::nccl_init(R, P);
#else
  // Pinned host stagers, sized in fp16 bytes for K/V.
  __half *K_send_h = nullptr, *V_send_h = nullptr, *K_recv_h = nullptr, *V_recv_h = nullptr;
  cudaHostAlloc(&K_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&K_recv_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_recv_h, kv_bytes, cudaHostAllocDefault);
#endif

  auto one_pass = [&]() -> std::tuple<double, double, double> {
    // Reset K/V to local chunk. scratch_f was sized to max(q_local_elem, kv_local_elem)
    // so it is large enough to serve as the fp32 staging buffer for any valid kv_H.
    upload_as_half(k_h_init, scratch_f, K_a.data(), kv_local_elem);
    upload_as_half(v_h_init, scratch_f, V_a.data(), kv_local_elem);
    __half* K_cur = K_a.data();
    __half* V_cur = V_a.data();
    __half* K_recv = K_b.data();
    __half* V_recv = V_b.data();

    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), init_shape, m_count);

    double comm_acc = 0.0, comp_acc = 0.0, wait_acc = 0.0;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
#ifdef RING_USE_NCCL
    // Bracket NCCL with its own events; otherwise the on-device transfer time
    // is invisible in every sub-metric (see run_ring_blocking for details).
    cudaEvent_t ev_nccl0, ev_nccl1;
    cudaEventCreate(&ev_nccl0);
    cudaEventCreate(&ev_nccl1);
#endif

    for (int step = 0; step < P; ++step) {
#ifndef RING_USE_NCCL
      MPI_Request reqs[4];
      int n_req = 0;
#endif

      const double t_post0 = MPI_Wtime();
      if (step < P - 1) {
#ifdef RING_USE_NCCL
        cudaEventRecord(ev_nccl0, 0);
        ncclGroupStart();
        NCCL_CHECK(ncclSend(K_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, 0));
        NCCL_CHECK(ncclRecv(K_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, 0));
        NCCL_CHECK(ncclSend(V_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, 0));
        NCCL_CHECK(ncclRecv(V_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, 0));
        NCCL_CHECK(ncclGroupEnd());
        cudaEventRecord(ev_nccl1, 0);
#else
        cudaMemcpy(K_send_h, K_cur, kv_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(V_send_h, V_cur, kv_bytes, cudaMemcpyDeviceToHost);
        const int n = mpi_int_count(kv_bytes, "run_ring_blocking_fp16/MPI_Isend|Irecv");
        MPI_Isend(K_send_h, n, MPI_BYTE, next_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Irecv(K_recv_h, n, MPI_BYTE, prev_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Isend(V_send_h, n, MPI_BYTE, next_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Irecv(V_recv_h, n, MPI_BYTE, prev_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[n_req++]);
#endif
      }
      const double t_post1 = MPI_Wtime();
      comm_acc += (t_post1 - t_post0) * 1e3;

      cudaEventRecord(ev0);
      for (int sg_q = 0; sg_q < nsg; ++sg_q) {
        for (int sg_k = 0; sg_k < nsg; ++sg_k) {
          const int q_off_sg = part.q_offset(sg_q);
          const int k_off_sg = part.k_offset_for_step(step, sg_k);
          if (cfg.causal && k_off_sg > q_off_sg + Sl - 1) continue;
          launch_attention_step_fp16(q_d.data() + sg_q * q_sg_elem, K_cur + sg_k * kv_sg_elem,
                                     V_cur + sg_k * kv_sg_elem, out_d.data() + sg_q * q_sg_elem,
                                     m_d.data() + sg_q * m_sg, l_d.data() + sg_q * m_sg, sg_shape,
                                     q_off_sg, k_off_sg, cfg.causal);
        }
      }
      cudaEventRecord(ev1);
      cudaEventSynchronize(ev1);
      float comp_float_ms = 0.f;
      cudaEventElapsedTime(&comp_float_ms, ev0, ev1);
      comp_acc += comp_float_ms;

      if (step < P - 1) {
#ifdef RING_USE_NCCL
        // Stream 0 already drained by cudaEventSynchronize(ev1); wait ≈ 0 by
        // construction. The NCCL transfer cost is attributed to comm_ms below.
        const double t_wait0 = MPI_Wtime();
        cudaStreamSynchronize(0);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;
        float nccl_ms = 0.f;
        cudaEventElapsedTime(&nccl_ms, ev_nccl0, ev_nccl1);
        comm_acc += nccl_ms;
#else
        const double t_wait0 = MPI_Wtime();
        MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;

        const double t_h2d0 = MPI_Wtime();
        cudaMemcpy(K_recv, K_recv_h, kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(V_recv, V_recv_h, kv_bytes, cudaMemcpyHostToDevice);
        const double t_h2d1 = MPI_Wtime();
        comm_acc += (t_h2d1 - t_h2d0) * 1e3;
#endif
        std::swap(K_cur, K_recv);
        std::swap(V_cur, V_recv);
      }
    }

    for (int sg = 0; sg < nsg; ++sg)
      launch_attention_finalize(out_d.data() + sg * q_sg_elem, l_d.data() + sg * m_sg, sg_shape);
    cudaDeviceSynchronize();
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
#ifdef RING_USE_NCCL
    cudaEventDestroy(ev_nccl0);
    cudaEventDestroy(ev_nccl1);
#endif
    return {comm_acc, comp_acc, wait_acc};
  };

  RingResult res;
  detail::run_timed_passes(cfg, one_pass, res);

  if (cfg.verify) {
    one_pass();
    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);
    std::vector<int> q_offsets(nsg);
    for (int sg = 0; sg < nsg; ++sg) q_offsets[sg] = part.q_offset(sg);
    res.max_err =
        detail::verify_local_output(cfg, B, H, S, D, kv_H, Sl, q_sg_elem, q_offsets, dev_out_h);
  }

#ifdef RING_USE_NCCL
  ncclCommDestroy(nccl_comm);
#else
  cudaFreeHost(K_send_h);
  cudaFreeHost(V_send_h);
  cudaFreeHost(K_recv_h);
  cudaFreeHost(V_recv_h);
#endif
  return res;
}

// ---------------------------------------------------------------------------
// Ring rotation with compute/comm overlap (fp16)
// ---------------------------------------------------------------------------

ring_attention::RingResult run_ring_overlap_fp16(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;

  const RingPartition::Mode mode =
      cfg.zigzag ? RingPartition::Mode::Zigzag : RingPartition::Mode::Contiguous;
  RingPartition part(P, R, S, mode);
  const int nsg = part.num_sub_groups();
  const int Sl = part.local_chunk_len();
  const int Sl_local = Sl * nsg;
  const std::size_t q_sg_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t q_local_elem = static_cast<std::size_t>(nsg) * q_sg_elem;
  const std::size_t kv_sg_elem = static_cast<std::size_t>(B) * kv_H * Sl * D;
  const std::size_t kv_local_elem = static_cast<std::size_t>(nsg) * kv_sg_elem;
  const std::size_t m_sg = static_cast<std::size_t>(B) * H * Sl;
  const std::size_t m_count = static_cast<std::size_t>(nsg) * m_sg;
  const std::size_t kv_bytes = kv_local_elem * sizeof(__half);

  const int next_rank = part.next_rank();
  const int prev_rank = part.prev_rank();

  // In mem_probe mode we only test whether the device buffers fit, so skip both
  // the host allocations and the fills (pure CPU work over S*H*D elements that
  // would also balloon host RAM at large S).
  std::vector<float> q_h, k_h_init, v_h_init;
  if (!cfg.mem_probe) {
    q_h.resize(q_local_elem);
    k_h_init.resize(kv_local_elem);
    v_h_init.resize(kv_local_elem);
    for (int sg = 0; sg < nsg; ++sg) {
      fill_region(q_h, sg * q_sg_elem, cfg.seed, 0, B, H, Sl, D, part.q_offset(sg));
      fill_region(k_h_init, sg * kv_sg_elem, cfg.seed, 1, B, kv_H, Sl, D,
                  part.k_offset_for_step(0, sg));
      fill_region(v_h_init, sg * kv_sg_elem, cfg.seed, 2, B, kv_H, Sl, D,
                  part.k_offset_for_step(0, sg));
    }
  }

  DeviceTensor<float> scratch_f(q_local_elem);
  DeviceTensor<__half> q_d(q_local_elem);
  if (!cfg.mem_probe) upload_as_half(q_h, scratch_f, q_d.data(), q_local_elem);

  DeviceTensor<__half> K_a(kv_local_elem), K_b(kv_local_elem);
  DeviceTensor<__half> V_a(kv_local_elem), V_b(kv_local_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);

  const AttentionShape init_shape{B, H, Sl_local, S, D};
  AttentionShape sg_shape{B, H, Sl, Sl, D};
  sg_shape.kv_heads = kv_H;

#ifdef RING_USE_NCCL
  ncclComm_t nccl_comm = ring_attention::nccl_init(R, P);
#else
  __half *K_send_h = nullptr, *V_send_h = nullptr, *K_recv_h = nullptr, *V_recv_h = nullptr;
  cudaHostAlloc(&K_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&K_recv_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_recv_h, kv_bytes, cudaHostAllocDefault);
#endif

  cudaStream_t stream_compute = nullptr, stream_copy = nullptr;
  cudaStreamCreate(&stream_compute);
  cudaStreamCreate(&stream_copy);
  cudaEvent_t comm_done;
  cudaEventCreate(&comm_done);

  std::vector<cudaEvent_t> ev_starts(P), ev_ends(P);
  std::vector<cudaEvent_t> ev_h2d_starts(P), ev_h2d_ends(P);
  for (int s = 0; s < P; ++s) {
    cudaEventCreate(&ev_starts[s]);
    cudaEventCreate(&ev_ends[s]);
    cudaEventCreate(&ev_h2d_starts[s]);
    cudaEventCreate(&ev_h2d_ends[s]);
  }

  // Memory-capacity probe: every device buffer (q/K_a/K_b/V_a/V_b/out/m/l) and
  // the NCCL transport are now allocated — an OOM would already have aborted via
  // cudaCheck. A clean return here means this config fits, without paying the
  // O(S^2) forward pass. NCCL's lazy per-channel buffers (a few MB) are the only
  // footprint not yet counted; negligible beside the multi-GB tensors above.
  if (cfg.mem_probe) {
    cudaDeviceSynchronize();
    // Tear down everything allocated up to this point (streams, events, comm /
    // pinned staging). The init buffers k_init_h16/v_init_h16 are allocated only
    // below, so they are not held here.
    for (int s = 0; s < P; ++s) {
      cudaEventDestroy(ev_starts[s]);
      cudaEventDestroy(ev_ends[s]);
      cudaEventDestroy(ev_h2d_starts[s]);
      cudaEventDestroy(ev_h2d_ends[s]);
    }
    cudaEventDestroy(comm_done);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_copy);
#ifdef RING_USE_NCCL
    ncclCommDestroy(nccl_comm);
#else
    cudaFreeHost(K_send_h);
    cudaFreeHost(V_send_h);
    cudaFreeHost(K_recv_h);
    cudaFreeHost(V_recv_h);
#endif
    RingResult probe_res;
    probe_res.total_ms = -1.0;  // sentinel: probe, not a timed run
    return probe_res;
  }

  // Pinned host fp16 buffers for the initial K/V state. They must be page-locked:
  // `cudaMemcpyAsync` from pageable memory silently serializes (internal staging)
  // and would defeat the overlap with `stream_compute`. We stage fp32→fp16 once
  // here, outside the timed loop.
  __half *k_init_h16 = nullptr, *v_init_h16 = nullptr;
  cudaHostAlloc(&k_init_h16, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&v_init_h16, kv_bytes, cudaHostAllocDefault);
  for (std::size_t i = 0; i < kv_local_elem; ++i) {
    k_init_h16[i] = __float2half(k_h_init[i]);
    v_init_h16[i] = __float2half(v_h_init[i]);
  }

  auto one_pass = [&]() -> std::tuple<double, double, double> {
    cudaMemcpyAsync(K_a.data(), k_init_h16, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
    cudaMemcpyAsync(V_a.data(), v_init_h16, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
    cudaEventRecord(comm_done, stream_copy);

    __half* K_cur = K_a.data();
    __half* V_cur = V_a.data();
    __half* K_recv = K_b.data();
    __half* V_recv = V_b.data();

    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), init_shape, m_count,
                          stream_compute);

    double comm_acc = 0.0, wait_acc = 0.0;

    for (int step = 0; step < P; ++step) {
      cudaStreamWaitEvent(stream_compute, comm_done, 0);

      cudaEventRecord(ev_starts[step], stream_compute);
      for (int sg_q = 0; sg_q < nsg; ++sg_q) {
        for (int sg_k = 0; sg_k < nsg; ++sg_k) {
          const int q_off_sg = part.q_offset(sg_q);
          const int k_off_sg = part.k_offset_for_step(step, sg_k);
          if (cfg.causal && k_off_sg > q_off_sg + Sl - 1) continue;
          launch_attention_step_fp16(q_d.data() + sg_q * q_sg_elem, K_cur + sg_k * kv_sg_elem,
                                     V_cur + sg_k * kv_sg_elem, out_d.data() + sg_q * q_sg_elem,
                                     m_d.data() + sg_q * m_sg, l_d.data() + sg_q * m_sg, sg_shape,
                                     q_off_sg, k_off_sg, cfg.causal, stream_compute);
        }
      }
      cudaEventRecord(ev_ends[step], stream_compute);

      if (step < P - 1) {
#ifdef RING_USE_NCCL
        if (step >= 1) cudaStreamWaitEvent(stream_copy, ev_ends[step - 1], 0);

        cudaEventRecord(ev_h2d_starts[step], stream_copy);
        ncclGroupStart();
        NCCL_CHECK(ncclSend(K_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclRecv(K_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclSend(V_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclRecv(V_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclGroupEnd());
        cudaEventRecord(ev_h2d_ends[step], stream_copy);
        cudaEventRecord(comm_done, stream_copy);
#else
        const double t_post0 = MPI_Wtime();
        cudaMemcpyAsync(K_send_h, K_cur, kv_bytes, cudaMemcpyDeviceToHost, stream_copy);
        cudaMemcpyAsync(V_send_h, V_cur, kv_bytes, cudaMemcpyDeviceToHost, stream_copy);
        cudaStreamSynchronize(stream_copy);
        MPI_Request reqs[4];
        const int n = mpi_int_count(kv_bytes, "run_ring_overlap_fp16/MPI_Isend|Irecv");
        MPI_Isend(K_send_h, n, MPI_BYTE, next_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(K_recv_h, n, MPI_BYTE, prev_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(V_send_h, n, MPI_BYTE, next_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(V_recv_h, n, MPI_BYTE, prev_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[3]);
        const double t_post1 = MPI_Wtime();
        comm_acc += (t_post1 - t_post0) * 1e3;

        const double t_wait0 = MPI_Wtime();
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;

        if (step >= 1) cudaStreamWaitEvent(stream_copy, ev_ends[step - 1], 0);

        cudaEventRecord(ev_h2d_starts[step], stream_copy);
        cudaMemcpyAsync(K_recv, K_recv_h, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
        cudaMemcpyAsync(V_recv, V_recv_h, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
        cudaEventRecord(ev_h2d_ends[step], stream_copy);
        cudaEventRecord(comm_done, stream_copy);
#endif

        std::swap(K_cur, K_recv);
        std::swap(V_cur, V_recv);
      }
    }

    cudaStreamSynchronize(stream_compute);
    for (int sg = 0; sg < nsg; ++sg)
      launch_attention_finalize(out_d.data() + sg * q_sg_elem, l_d.data() + sg * m_sg, sg_shape,
                                stream_compute);
    cudaStreamSynchronize(stream_compute);

    double comp_acc = 0.0;
    for (int s = 0; s < P; ++s) {
      float t = 0.f;
      cudaEventElapsedTime(&t, ev_starts[s], ev_ends[s]);
      comp_acc += t;
    }
    for (int s = 0; s < P - 1; ++s) {
      float t = 0.f;
      cudaEventElapsedTime(&t, ev_h2d_starts[s], ev_h2d_ends[s]);
      comm_acc += t;
    }
    return {comm_acc, comp_acc, wait_acc};
  };

  RingResult res;
  detail::run_timed_passes(cfg, one_pass, res);

  if (cfg.verify) {
    one_pass();
    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);
    std::vector<int> q_offsets(nsg);
    for (int sg = 0; sg < nsg; ++sg) q_offsets[sg] = part.q_offset(sg);
    res.max_err =
        detail::verify_local_output(cfg, B, H, S, D, kv_H, Sl, q_sg_elem, q_offsets, dev_out_h);
  }

  for (int s = 0; s < P; ++s) {
    cudaEventDestroy(ev_starts[s]);
    cudaEventDestroy(ev_ends[s]);
    cudaEventDestroy(ev_h2d_starts[s]);
    cudaEventDestroy(ev_h2d_ends[s]);
  }
  cudaEventDestroy(comm_done);
  cudaStreamDestroy(stream_compute);
  cudaStreamDestroy(stream_copy);
#ifdef RING_USE_NCCL
  ncclCommDestroy(nccl_comm);
#else
  cudaFreeHost(K_send_h);
  cudaFreeHost(V_send_h);
  cudaFreeHost(K_recv_h);
  cudaFreeHost(V_recv_h);
#endif
  cudaFreeHost(k_init_h16);
  cudaFreeHost(v_init_h16);
  return res;
}

}  // namespace

namespace ring_attention {

RingResult run_ring_attention_fp16(const RingConfig& cfg) {
  if (cfg.zigzag && cfg.mode == RingMode::AllGather)
    mpi_die("--zigzag is not supported with --mode=allgather (use ring-blocking or ring-overlap)");
  switch (cfg.mode) {
    case RingMode::AllGather:
      return run_allgather_fp16(cfg);
    case RingMode::RingBlocking:
      return run_ring_blocking_fp16(cfg);
    case RingMode::RingOverlap:
      return run_ring_overlap_fp16(cfg);
    case RingMode::Ring2D:
      mpi_die("--mode=ring-2d has no fp16 path yet (use --dtype fp32)");
  }
  mpi_die("unreachable");
}

}  // namespace ring_attention
