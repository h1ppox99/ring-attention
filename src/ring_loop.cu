/// @file
/// Host-side ring-attention orchestrator.
/// Implements all three modes: allgather, ring-blocking, ring-overlap.

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
#include "ring_partition.hpp"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

/// MPI-safe abort so a single rank's failure tears down all ranks.
[[noreturn]] void mpi_die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  std::exit(EXIT_FAILURE);  // satisfy [[noreturn]]; MPI_Abort does not return
}

/// Validate that `count` fits in the `int` count used by MPI-3 point-to-point
/// and collective calls. NVHPC 24.1 ships OpenMPI 4.1.7 (MPI-3.1) so we cannot
/// use `MPI_Count` / the `_c` variants here. On overflow we abort with a clear
/// message rather than silently truncating the transfer size.
int mpi_int_count(std::size_t count, const char* where) {
  if (count > static_cast<std::size_t>(INT_MAX)) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "MPI count overflow in %s: %zu elements exceeds INT_MAX=%d — "
                  "reduce per-rank tensor size or chunk the transfer.",
                  where, count, INT_MAX);
    mpi_die(buf);
  }
  return static_cast<int>(count);
}

/// Fill a host vector of shape (B, H, seq_len, D) using gen_elem.
/// global_seq_start is the first global sequence index in this slice.
void fill_host_tensor(std::vector<float>& buf, uint32_t seed, int tensor_id, int B, int H,
                      int seq_len, int D, int global_seq_start) {
  buf.resize(static_cast<std::size_t>(B) * H * seq_len * D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int s = 0; s < seq_len; ++s)
        for (int d = 0; d < D; ++d) {
          const std::size_t idx = (static_cast<std::size_t>(b * H + h) * seq_len + s) * D + d;
          buf[idx] = ring_attention::gen_elem(seed, tensor_id, b, h, global_seq_start + s, d);
        }
}

// ---------------------------------------------------------------------------
// AllGather baseline
// ---------------------------------------------------------------------------

/// Gather K/V from all ranks, run one attention_step with the full K/V.
///
/// Communication pattern:
///   MPI_Allgather(K,V)  →  rearrange (P,B,H,Sl,D) → (B,H,S,D)  →  H2D
///
/// Timing:
///   - comm_ms : MPI_Wtime around Allgather + rearrange + H2D, averaged over iters
///   - comp_ms : cudaEvent around attention_step, averaged over iters
///   - total_ms: MPI_Barrier-to-Barrier wall-clock / iters
///   All three are MPI_Reduce(MAX) across ranks before being stored in RingResult.
ring_attention::RingResult run_allgather(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;
  const int Sl = S / P;
  const int q_off = R * Sl;

  const std::size_t q_local_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t kv_local_elem = static_cast<std::size_t>(B) * kv_H * Sl * D;
  const std::size_t q_full_elem = static_cast<std::size_t>(B) * H * S * D;
  const std::size_t kv_full_elem = static_cast<std::size_t>(B) * kv_H * S * D;
  const std::size_t m_count = static_cast<std::size_t>(B) * H * Sl;

  // Fill local Q/K/V on host — constant across iterations.
  std::vector<float> q_h, k_h, v_h;
  fill_host_tensor(q_h, cfg.seed, 0, B, H, Sl, D, q_off);
  fill_host_tensor(k_h, cfg.seed, 1, B, kv_H, Sl, D, q_off);
  fill_host_tensor(v_h, cfg.seed, 2, B, kv_H, Sl, D, q_off);

  // H2D local Q — stays on device for all iterations.
  DeviceTensor<float> q_d(q_local_elem);
  q_d.copy_from_host(q_h);

  // Reusable buffers for gather + rearrange + device K/V.
  std::vector<float> k_gathered(kv_full_elem), v_gathered(kv_full_elem);
  std::vector<float> full_k_h(kv_full_elem), full_v_h(kv_full_elem);
  DeviceTensor<float> k_d(kv_full_elem), v_d(kv_full_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);
  AttentionShape shape{B, H, Sl, S, D};
  shape.kv_heads = kv_H;

  // One complete allgather + attention pass.
  // Returns {comm_ms, comp_ms} for that single pass.
  auto one_pass = [&]() -> std::pair<double, double> {
    // --- Communication: allgather K,V + rearrange + H2D ----------------------
    const double tc0 = MPI_Wtime();
    const int kv_local_elem_int = mpi_int_count(kv_local_elem, "run_allgather/MPI_Allgather");
    MPI_Allgather(k_h.data(), kv_local_elem_int, MPI_FLOAT, k_gathered.data(), kv_local_elem_int,
                  MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(v_h.data(), kv_local_elem_int, MPI_FLOAT, v_gathered.data(), kv_local_elem_int,
                  MPI_FLOAT, MPI_COMM_WORLD);

    // Rearrange from rank-major (P,B,kv_H,Sl,D) to (B,kv_H,S,D).
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

    k_d.copy_from_host(full_k_h);
    v_d.copy_from_host(full_v_h);
    const double tc1 = MPI_Wtime();

    // --- Compute: init → step → finalize -------------------------------------
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), shape, m_count);
    cudaEventRecord(ev0);
    launch_attention_step(q_d.data(), k_d.data(), v_d.data(), out_d.data(), m_d.data(), l_d.data(),
                          shape, q_off, /*k_offset=*/0, cfg.causal);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    launch_attention_finalize(out_d.data(), l_d.data(), shape);
    cudaDeviceSynchronize();
    float comp_float_ms = 0.f;
    cudaEventElapsedTime(&comp_float_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    return {(tc1 - tc0) * 1e3, static_cast<double>(comp_float_ms)};
  };

  // Warmup: one pass without timing to prime GPU/MPI state.
  one_pass();

  // Timed loop: iters passes, barrier-wrapped for wall-clock total.
  double sum_comm = 0.0, sum_comp = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  for (int i = 0; i < cfg.iters; ++i) {
    const auto [comm, comp] = one_pass();
    sum_comm += comm;
    sum_comp += comp;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  RingResult res;
  res.comm_ms = sum_comm / cfg.iters;
  res.comp_ms = sum_comp / cfg.iters;
  res.wait_ms = 0.0;
  res.total_ms = (t_end - t_start) * 1e3 / cfg.iters;

  // Reduce to rank 0: MAX across ranks = bottleneck rank's time.
  const double local_t[3] = {res.comm_ms, res.comp_ms, res.total_ms};
  double global_t[3] = {};
  MPI_Reduce(local_t, global_t, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (R == 0) {
    res.comm_ms = global_t[0];
    res.comp_ms = global_t[1];
    res.total_ms = global_t[2];
  }

  // Verification: run once after timing, compare against cpu_attention.
  float max_err = -1.f;
  if (cfg.verify) {
    std::vector<float> full_q_cpu(q_full_elem), full_k_cpu(kv_full_elem), full_v_cpu(kv_full_elem);
    fill_host_tensor(full_q_cpu, cfg.seed, 0, B, H, S, D, 0);
    fill_host_tensor(full_k_cpu, cfg.seed, 1, B, kv_H, S, D, 0);
    fill_host_tensor(full_v_cpu, cfg.seed, 2, B, kv_H, S, D, 0);

    AttentionShape ref_shape{B, H, S, S, D};
    ref_shape.kv_heads = kv_H;
    std::vector<float> cpu_out(q_full_elem);
    cpu_attention(full_q_cpu.data(), full_k_cpu.data(), full_v_cpu.data(), cpu_out.data(),
                  ref_shape, cfg.causal);

    // Re-run one pass so out_d holds a fresh result.
    one_pass();

    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);

    max_err = 0.f;
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h) {
        const float* cpu_slice =
            cpu_out.data() + (static_cast<std::size_t>(b * H + h) * S + q_off) * D;
        const float* dev_slice = dev_out_h.data() + static_cast<std::size_t>(b * H + h) * Sl * D;
        for (int e = 0; e < Sl * D; ++e)
          max_err = std::max(max_err, std::abs(cpu_slice[e] - dev_slice[e]));
      }

    float global_max;
    MPI_Reduce(&max_err, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    max_err = (R == 0) ? global_max : -1.f;
  }

  res.max_err = max_err;
  return res;
}

// ---------------------------------------------------------------------------
// Ring rotation, blocking style (`--mode=ring-blocking`)
// ---------------------------------------------------------------------------

/// Host-staged ring rotation with no compute/comm overlap.
///
/// Each rank holds one (K_chunk, V_chunk) at a time, double-buffered on the
/// device. Per step we:
///   D2H K_cur,V_cur  →  Isend/Irecv to next/prev ranks  →  attention_step on
///   K_cur,V_cur  →  cudaDeviceSync + MPI_Waitall  →  H2D recv buffers  →  swap.
///
/// Supports both Contiguous and Zigzag token assignment. In Zigzag mode the
/// buffer layout is [lo_tensor | hi_tensor]: two contiguous [B,H,chunk,D] blocks
/// of size sg_elem each. K_cur + sg*sg_elem is a valid kernel pointer for sub-group sg.
/// Per step, up to 4 launch_attention_step calls (sg_q × sg_k); causal pairs
/// where k_offset > q_offset + chunk - 1 are pruned.
///
/// Timing:
///   - comm_ms : host-staged path: D2H + H2D + Isend/Irecv post (unavoidable
///               staging cost). NCCL path: on-device NCCL transfer time,
///               captured via a dedicated event pair around ncclGroupEnd.
///   - wait_ms : host-staged path: MPI_Waitall time. NCCL path: ≈ 0 by
///               construction — NCCL and kernels share stream 0, so the
///               kernel-event sync also drains NCCL; the actual transfer
///               cost shows up in comm_ms, not here.
///   - comp_ms : sum of cudaEvent intervals around each step's kernel calls
ring_attention::RingResult run_ring_blocking(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;

  const RingPartition::Mode mode =
      cfg.zigzag ? RingPartition::Mode::Zigzag : RingPartition::Mode::Contiguous;
  RingPartition part(P, R, S, mode);
  const int nsg = part.num_sub_groups();  // 1 (contiguous) or 2 (zigzag)
  const int Sl = part.local_chunk_len();  // per-sub-group rows: S/P or S/(2P)
  const int Sl_local = Sl * nsg;          // total local rows per rank = S/P
  const std::size_t q_sg_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t q_local_elem = static_cast<std::size_t>(nsg) * q_sg_elem;
  const std::size_t kv_sg_elem = static_cast<std::size_t>(B) * kv_H * Sl * D;
  const std::size_t kv_local_elem = static_cast<std::size_t>(nsg) * kv_sg_elem;
  const std::size_t m_sg = static_cast<std::size_t>(B) * H * Sl;
  const std::size_t m_count = static_cast<std::size_t>(nsg) * m_sg;
  const std::size_t kv_bytes = kv_local_elem * sizeof(float);

  const int next_rank = part.next_rank();
  const int prev_rank = part.prev_rank();

  // Fill into the [lo|hi] layout: sub-group sg occupies [sg*sg_elem, (sg+1)*sg_elem).
  // The stride within each (b,h) head is Sl (not Sl_local), so the kernel sees a
  // valid [B,H,Sl,D] tensor at buf + sg*q_sg_elem (Q) or buf + sg*kv_sg_elem (K/V).
  auto fill_q_into = [&](std::vector<float>& buf, int tid, std::size_t off, int gss) {
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h)
        for (int s = 0; s < Sl; ++s)
          for (int d = 0; d < D; ++d)
            buf[off + (static_cast<std::size_t>(b * H + h) * Sl + s) * D + d] =
                gen_elem(cfg.seed, tid, b, h, gss + s, d);
  };

  auto fill_kv_into = [&](std::vector<float>& buf, int tid, std::size_t off, int gss) {
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < kv_H; ++h)
        for (int s = 0; s < Sl; ++s)
          for (int d = 0; d < D; ++d)
            buf[off + (static_cast<std::size_t>(b * kv_H + h) * Sl + s) * D + d] =
                gen_elem(cfg.seed, tid, b, h, gss + s, d);
  };

  std::vector<float> q_h(q_local_elem), k_h_init(kv_local_elem), v_h_init(kv_local_elem);
  for (int sg = 0; sg < nsg; ++sg) {
    fill_q_into(q_h, 0, sg * q_sg_elem, part.q_offset(sg));
    fill_kv_into(k_h_init, 1, sg * kv_sg_elem, part.k_offset_for_step(0, sg));
    fill_kv_into(v_h_init, 2, sg * kv_sg_elem, part.k_offset_for_step(0, sg));
  }

  // Device buffers: Q stays put; K/V double-buffered (K_a/K_b, V_a/V_b).
  DeviceTensor<float> q_d(q_local_elem);
  q_d.copy_from_host(q_h);
  DeviceTensor<float> K_a(kv_local_elem), K_b(kv_local_elem);
  DeviceTensor<float> V_a(kv_local_elem), V_b(kv_local_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);
#ifdef RING_USE_NCCL
  // FP16 transit buffers: the NCCL ring moves __half (half the bytes); the
  // kernel still reads FP32. See KERNEL_OPTIMIZATIONS.md Round 14. (Round 16
  // tried INT8 here — ~1.5-1.9× less comm but breached tol=1e-3; reverted.)
  DeviceTensor<__half> K_ha(kv_local_elem), K_hb(kv_local_elem);
  DeviceTensor<__half> V_ha(kv_local_elem), V_hb(kv_local_elem);
#endif

  // init_shape seq_q = Sl_local initialises all sub-groups in one call;
  // sg_shape seq_q = Sl is used per (sg_q, sg_k) kernel call.
  const AttentionShape init_shape{B, H, Sl_local, S, D};
  AttentionShape sg_shape{B, H, Sl, Sl, D};
  sg_shape.kv_heads = kv_H;

#ifdef RING_USE_NCCL
  // NCCL communicator: GPU-to-GPU direct, no host staging needed.
  ncclComm_t nccl_comm = ring_attention::nccl_init(R, P);
#else
  // Pinned host staging — page-locked so D2H/H2D can be async without copies.
  float *K_send_h = nullptr, *V_send_h = nullptr, *K_recv_h = nullptr, *V_recv_h = nullptr;
  cudaHostAlloc(&K_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&K_recv_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_recv_h, kv_bytes, cudaHostAllocDefault);
#endif

  // One full ring pass — returns (comm_ms, comp_ms, wait_ms) for that pass.
  auto one_pass = [&]() -> std::tuple<double, double, double> {
    // Reset K/V to local chunk (timed iterations need a clean starting state).
    cudaMemcpy(K_a.data(), k_h_init.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(V_a.data(), v_h_init.data(), kv_bytes, cudaMemcpyHostToDevice);
    float* K_cur = K_a.data();
    float* V_cur = V_a.data();
    float* K_recv = K_b.data();
    float* V_recv = V_b.data();
#ifdef RING_USE_NCCL
    // Narrow the local chunk to FP16 once; received FP16 chunks are forwarded
    // as-is next step (no re-narrowing).
    __half* K_h_cur = K_ha.data();
    __half* V_h_cur = V_ha.data();
    __half* K_h_recv = K_hb.data();
    __half* V_h_recv = V_hb.data();
    launch_float_to_half(K_cur, K_h_cur, kv_local_elem, 0);
    launch_float_to_half(V_cur, V_h_cur, kv_local_elem, 0);
#endif

    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), init_shape, m_count);

    double comm_acc = 0.0, comp_acc = 0.0, wait_acc = 0.0;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
#ifdef RING_USE_NCCL
    // Extra event pair to capture on-device NCCL transfer time. NCCL ops run
    // on stream 0 before the kernel events, so without these the transfer
    // would be invisible to every sub-metric: comm_acc only sees host-enqueue
    // cost, ev0→ev1 only spans kernel time, and the post-kernel sync is a
    // no-op because the stream is already drained.
    cudaEvent_t ev_nccl0, ev_nccl1;
    cudaEventCreate(&ev_nccl0);
    cudaEventCreate(&ev_nccl1);
#endif

    for (int step = 0; step < P; ++step) {
#ifndef RING_USE_NCCL
      MPI_Request reqs[4];
      int n_req = 0;
#endif

      // (1) Post ring transfer for next step's K/V chunk. Skipped on the last step.
      const double t_post0 = MPI_Wtime();
      if (step < P - 1) {
#ifdef RING_USE_NCCL
        // Direct GPU-to-GPU via NCCL — no host staging, no D2H/H2D.
        // ncclGroupEnd enqueues all ops to stream 0; the actual transfer
        // runs on the GPU and is bracketed by ev_nccl0/ev_nccl1 so its
        // on-device cost is captured in comm_acc (otherwise it would be
        // hidden inside cudaEventSynchronize(ev1) below).
        cudaEventRecord(ev_nccl0, 0);
        ncclGroupStart();
        NCCL_CHECK(ncclSend(K_h_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, 0));
        NCCL_CHECK(ncclRecv(K_h_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, 0));
        NCCL_CHECK(ncclSend(V_h_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, 0));
        NCCL_CHECK(ncclRecv(V_h_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, 0));
        NCCL_CHECK(ncclGroupEnd());
        cudaEventRecord(ev_nccl1, 0);
        // Widen received FP16 → FP32 for the next step's kernel.
        launch_half_to_float(K_h_recv, K_recv, kv_local_elem, 0);
        launch_half_to_float(V_h_recv, V_recv, kv_local_elem, 0);
#else
        // (1) D2H stage current K/V → host pinned buffer + (2) post MPI exchange.
        cudaMemcpy(K_send_h, K_cur, kv_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(V_send_h, V_cur, kv_bytes, cudaMemcpyDeviceToHost);
        const int n = mpi_int_count(kv_local_elem, "run_ring_blocking/MPI_Isend|Irecv");
        MPI_Isend(K_send_h, n, MPI_FLOAT, next_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Irecv(K_recv_h, n, MPI_FLOAT, prev_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Isend(V_send_h, n, MPI_FLOAT, next_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Irecv(V_recv_h, n, MPI_FLOAT, prev_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[n_req++]);
#endif
      }
      const double t_post1 = MPI_Wtime();
      comm_acc += (t_post1 - t_post0) * 1e3;

      // (3) Compute: one kernel call per surviving (sg_q, sg_k) pair.
      //     Bracket all calls in the step with a single event pair so that
      //     comp_ms accumulates total GPU time across all sub-groups.
      cudaEventRecord(ev0);
      for (int sg_q = 0; sg_q < nsg; ++sg_q) {
        for (int sg_k = 0; sg_k < nsg; ++sg_k) {
          const int q_off_sg = part.q_offset(sg_q);
          const int k_off_sg = part.k_offset_for_step(step, sg_k);
          // Causal prune: skip if every key position is strictly after every
          // query position in this block (the entire block contributes zero).
          if (cfg.causal && k_off_sg > q_off_sg + Sl - 1) continue;
          launch_attention_step(q_d.data() + sg_q * q_sg_elem, K_cur + sg_k * kv_sg_elem,
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

      // (4) Wait for the transfer posted in (1) to complete, then promote buffers.
      if (step < P - 1) {
#ifdef RING_USE_NCCL
        // Stream 0 was already drained by cudaEventSynchronize(ev1) above
        // (NCCL and kernels share stream 0, so the kernel sync also waits on
        // NCCL). This sync is a no-op kept for symmetry; wait_ms ≈ 0 in
        // blocking mode by construction — the actual NCCL cost is attributed
        // to comm_ms via the ev_nccl0/ev_nccl1 elapsed time below.
        const double t_wait0 = MPI_Wtime();
        cudaStreamSynchronize(0);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;
        float nccl_ms = 0.f;
        cudaEventElapsedTime(&nccl_ms, ev_nccl0, ev_nccl1);
        comm_acc += nccl_ms;
        // K_recv already holds the received data on the device — no H2D needed.
#else
        // wait_ms = MPI_Waitall only (the headline "unhidden comm" metric);
        // H2D is folded into comm_ms.
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
        // (5) Promote received buffers to "current"; the old current slot will
        //     be reused as the next recv target.
        std::swap(K_cur, K_recv);
        std::swap(V_cur, V_recv);
#ifdef RING_USE_NCCL
        std::swap(K_h_cur, K_h_recv);
        std::swap(V_h_cur, V_h_recv);
#endif
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

  // Warmup — first call pays kernel-load and MPI-startup costs.
  one_pass();

  // Timed loop.
  double sum_comm = 0.0, sum_comp = 0.0, sum_wait = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  for (int i = 0; i < cfg.iters; ++i) {
    const auto [c, p_, w] = one_pass();
    sum_comm += c;
    sum_comp += p_;
    sum_wait += w;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  RingResult res;
  res.comm_ms = sum_comm / cfg.iters;
  res.comp_ms = sum_comp / cfg.iters;
  res.wait_ms = sum_wait / cfg.iters;
  res.total_ms = (t_end - t_start) * 1e3 / cfg.iters;

  // Reduce MAX across ranks: the slowest rank dictates wall-clock.
  const double local_t[4] = {res.comm_ms, res.comp_ms, res.wait_ms, res.total_ms};
  double global_t[4] = {};
  MPI_Reduce(local_t, global_t, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (R == 0) {
    res.comm_ms = global_t[0];
    res.comp_ms = global_t[1];
    res.wait_ms = global_t[2];
    res.total_ms = global_t[3];
  }

  // Verification: regenerate full Q/K/V, run cpu_attention, compare each sub-group.
  float max_err = -1.f;
  if (cfg.verify) {
    const std::size_t q_full_elem = static_cast<std::size_t>(B) * H * S * D;
    const std::size_t kv_full_elem = static_cast<std::size_t>(B) * kv_H * S * D;
    std::vector<float> full_q(q_full_elem), full_k(kv_full_elem), full_v(kv_full_elem);
    fill_host_tensor(full_q, cfg.seed, 0, B, H, S, D, 0);
    fill_host_tensor(full_k, cfg.seed, 1, B, kv_H, S, D, 0);
    fill_host_tensor(full_v, cfg.seed, 2, B, kv_H, S, D, 0);

    AttentionShape ref_shape{B, H, S, S, D};
    ref_shape.kv_heads = kv_H;
    std::vector<float> cpu_out(q_full_elem);
    cpu_attention(full_q.data(), full_k.data(), full_v.data(), cpu_out.data(), ref_shape,
                  cfg.causal);

    // Re-run one pass so out_d reflects a complete forward.
    one_pass();
    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);

    max_err = 0.f;
    for (int sg = 0; sg < nsg; ++sg) {
      const int q_off_sg = part.q_offset(sg);
      for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
          const float* cpu_slice =
              cpu_out.data() + (static_cast<std::size_t>(b * H + h) * S + q_off_sg) * D;
          const float* dev_slice =
              dev_out_h.data() + sg * q_sg_elem + static_cast<std::size_t>(b * H + h) * Sl * D;
          for (int e = 0; e < Sl * D; ++e)
            max_err = std::max(max_err, std::abs(cpu_slice[e] - dev_slice[e]));
        }
    }

    float global_max;
    MPI_Reduce(&max_err, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    max_err = (R == 0) ? global_max : -1.f;
  }
  res.max_err = max_err;

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
// Ring rotation with compute/comm overlap (`--mode=ring-overlap`)
// ---------------------------------------------------------------------------

/// Two-stream ring rotation that hides MPI behind the attention kernel.
///
/// Streams:
///   - stream_compute : kernel + init/finalize
///   - stream_copy    : D2H send-staging and H2D recv-staging
/// Cross-stream sync uses one persistent event `comm_done` recorded at the
/// end of each step's H2D; the next step's kernel calls
/// `cudaStreamWaitEvent(stream_compute, comm_done)` so it sees the freshly-
/// received chunk before reading it.
///
/// Per step:
///   (1) stream_compute waits on comm_done (prev step's H2D into K_recv).
///   (2) Kernel is queued on stream_compute (returns immediately).
///   (3) On stream_copy + host: D2H K_cur → pinned → sync → MPI_Isend/Irecv →
///       MPI_Waitall → H2D pinned → K_recv on stream_copy → record comm_done.
///       The host is blocked inside MPI_Waitall — but the GPU keeps running
///       the kernel from step (2). *That* is the overlap.
///   (4) Pointer swap (CPU bookkeeping only).
///
/// Timing: comp_ms is gathered from per-step `cudaEvent` pairs but elapsed
/// times are computed only after the loop's terminal sync, so the events
/// don't serialize the loop. comm_ms and wait_ms are MPI_Wtime around the
/// host-issued portions, same convention as run_ring_blocking.
ring_attention::RingResult run_ring_overlap(const ring_attention::RingConfig& cfg) {
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
  const std::size_t kv_bytes = kv_local_elem * sizeof(float);

  const int next_rank = part.next_rank();
  const int prev_rank = part.prev_rank();

  auto fill_q_into = [&](std::vector<float>& buf, int tid, std::size_t off, int gss) {
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h)
        for (int s = 0; s < Sl; ++s)
          for (int d = 0; d < D; ++d)
            buf[off + (static_cast<std::size_t>(b * H + h) * Sl + s) * D + d] =
                gen_elem(cfg.seed, tid, b, h, gss + s, d);
  };

  auto fill_kv_into = [&](std::vector<float>& buf, int tid, std::size_t off, int gss) {
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < kv_H; ++h)
        for (int s = 0; s < Sl; ++s)
          for (int d = 0; d < D; ++d)
            buf[off + (static_cast<std::size_t>(b * kv_H + h) * Sl + s) * D + d] =
                gen_elem(cfg.seed, tid, b, h, gss + s, d);
  };

  std::vector<float> q_h(q_local_elem), k_h_init(kv_local_elem), v_h_init(kv_local_elem);
  for (int sg = 0; sg < nsg; ++sg) {
    fill_q_into(q_h, 0, sg * q_sg_elem, part.q_offset(sg));
    fill_kv_into(k_h_init, 1, sg * kv_sg_elem, part.k_offset_for_step(0, sg));
    fill_kv_into(v_h_init, 2, sg * kv_sg_elem, part.k_offset_for_step(0, sg));
  }

  // Device buffers — Q stays put; K/V double-buffered with pointer swap.
  DeviceTensor<float> q_d(q_local_elem);
  q_d.copy_from_host(q_h);
  DeviceTensor<float> K_a(kv_local_elem), K_b(kv_local_elem);
  DeviceTensor<float> V_a(kv_local_elem), V_b(kv_local_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);
#ifdef RING_USE_NCCL
  // FP16 transit buffers — the NCCL ring moves __half (half the bytes); the
  // kernel reads FP32. See KERNEL_OPTIMIZATIONS.md Round 14. (Round 16 tried
  // INT8 here — breached tol=1e-3; reverted.)
  DeviceTensor<__half> K_ha(kv_local_elem), K_hb(kv_local_elem);
  DeviceTensor<__half> V_ha(kv_local_elem), V_hb(kv_local_elem);
#endif

  const AttentionShape init_shape{B, H, Sl_local, S, D};
  AttentionShape sg_shape{B, H, Sl, Sl, D};
  sg_shape.kv_heads = kv_H;

#ifdef RING_USE_NCCL
  // NCCL communicator: GPU-to-GPU direct, no host staging needed.
  ncclComm_t nccl_comm = ring_attention::nccl_init(R, P);
#else
  // Pinned host staging.
  float *K_send_h = nullptr, *V_send_h = nullptr, *K_recv_h = nullptr, *V_recv_h = nullptr;
  cudaHostAlloc(&K_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&K_recv_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_recv_h, kv_bytes, cudaHostAllocDefault);
#endif

  // Two CUDA streams + one reusable event for the producer/consumer handshake.
  cudaStream_t stream_compute = nullptr, stream_copy = nullptr;
  cudaStreamCreate(&stream_compute);
  cudaStreamCreate(&stream_copy);
  cudaEvent_t comm_done;
  cudaEventCreate(&comm_done);

  // Per-step kernel timing — queried only after the loop terminates so that
  // cudaEventElapsedTime never blocks inside the iteration body.
  std::vector<cudaEvent_t> ev_starts(P), ev_ends(P);
  // Per-step H2D timing on stream_copy. We can't sample H2D with MPI_Wtime
  // because cudaMemcpyAsync only enqueues; the actual transfer runs async on
  // the copy engine and we deliberately do NOT sync (that would kill the
  // overlap). Events let us measure the transfer time after the fact without
  // serializing the hot path. Only P-1 H2Ds actually fire (skipped on last
  // step); the trailing event stays unrecorded and is ignored when summing.
  std::vector<cudaEvent_t> ev_h2d_starts(P), ev_h2d_ends(P);
  for (int s = 0; s < P; ++s) {
    cudaEventCreate(&ev_starts[s]);
    cudaEventCreate(&ev_ends[s]);
    cudaEventCreate(&ev_h2d_starts[s]);
    cudaEventCreate(&ev_h2d_ends[s]);
  }

  auto one_pass = [&]() -> std::tuple<double, double, double> {
    // Initial H2D for step 0's K/V on stream_copy; record event so step 0's
    // kernel waits on it (uniform treatment of step 0 vs. step s>0).
    cudaMemcpyAsync(K_a.data(), k_h_init.data(), kv_bytes, cudaMemcpyHostToDevice, stream_copy);
    cudaMemcpyAsync(V_a.data(), v_h_init.data(), kv_bytes, cudaMemcpyHostToDevice, stream_copy);

    float* K_cur = K_a.data();
    float* V_cur = V_a.data();
    float* K_recv = K_b.data();
    float* V_recv = V_b.data();
#ifdef RING_USE_NCCL
    // Narrow the local FP32 chunk to the FP16 send buffer (on stream_copy, so
    // it is ordered before step 0's send). Received FP16 is forwarded as-is.
    __half* K_h_cur = K_ha.data();
    __half* V_h_cur = V_ha.data();
    __half* K_h_recv = K_hb.data();
    __half* V_h_recv = V_hb.data();
    launch_float_to_half(K_cur, K_h_cur, kv_local_elem, stream_copy);
    launch_float_to_half(V_cur, V_h_cur, kv_local_elem, stream_copy);
#endif
    cudaEventRecord(comm_done, stream_copy);

    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), init_shape, m_count,
                          stream_compute);

    double comm_acc = 0.0, wait_acc = 0.0;

    for (int step = 0; step < P; ++step) {
      // (1) Compute stream waits for the H2D into K_cur to complete.
      cudaStreamWaitEvent(stream_compute, comm_done, 0);

      // (2) Queue kernels for all surviving (sg_q, sg_k) pairs — returns immediately;
      //     GPU runs them asynchronously. Both events bracket the full set so that
      //     ev_ends[step] fires only after all sub-group calls for this step complete,
      //     which is the correct WAR fence point for the next step's H2D.
      cudaEventRecord(ev_starts[step], stream_compute);
      for (int sg_q = 0; sg_q < nsg; ++sg_q) {
        for (int sg_k = 0; sg_k < nsg; ++sg_k) {
          const int q_off_sg = part.q_offset(sg_q);
          const int k_off_sg = part.k_offset_for_step(step, sg_k);
          if (cfg.causal && k_off_sg > q_off_sg + Sl - 1) continue;
          launch_attention_step(q_d.data() + sg_q * q_sg_elem, K_cur + sg_k * kv_sg_elem,
                                V_cur + sg_k * kv_sg_elem, out_d.data() + sg_q * q_sg_elem,
                                m_d.data() + sg_q * m_sg, l_d.data() + sg_q * m_sg, sg_shape,
                                q_off_sg, k_off_sg, cfg.causal, stream_compute);
        }
      }
      cudaEventRecord(ev_ends[step], stream_compute);

      // (3) Overlap window: ship K_cur out and pull next chunk in while the
      //     kernel from (2) runs on stream_compute. Skipped on the last step
      //     because there is no next chunk to ingest.
      if (step < P - 1) {
#ifdef RING_USE_NCCL
        // WAR fence: K_recv is aliased from the previous step's K_cur.
        // The previous step's kernel (on stream_compute) was still reading it;
        // stream_copy must not write until stream_compute has passed ev_ends[step-1].
        if (step >= 1) cudaStreamWaitEvent(stream_copy, ev_ends[step - 1], 0);

        // NCCL send/recv on stream_copy — fully async, no host involvement.
        // The kernel on stream_compute continues in parallel (true GPU overlap).
        // comm_ms is captured via ev_h2d_starts/ends after the loop.
        cudaEventRecord(ev_h2d_starts[step], stream_copy);
        ncclGroupStart();
        NCCL_CHECK(ncclSend(K_h_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclRecv(K_h_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclSend(V_h_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclRecv(V_h_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclGroupEnd());
        cudaEventRecord(ev_h2d_ends[step], stream_copy);
        // Widen received FP16 → FP32 (on stream_copy, after the WAR fence at
        // the top of this block) so the next step's kernel reads FP32.
        // comm_done is recorded *after* the widen so the consuming kernel waits
        // for the widened buffer, not just the raw FP16 receive.
        launch_half_to_float(K_h_recv, K_recv, kv_local_elem, stream_copy);
        launch_half_to_float(V_h_recv, V_recv, kv_local_elem, stream_copy);
        cudaEventRecord(comm_done, stream_copy);
        // wait_acc remains 0: no blocking host wait exists in the NCCL path.
#else
        const double t_post0 = MPI_Wtime();
        // D2H reads K_cur — concurrent with the kernel reading K_cur (both
        // are reads, so there is no race; the copy engine and the SMs run
        // on independent hardware paths).
        cudaMemcpyAsync(K_send_h, K_cur, kv_bytes, cudaMemcpyDeviceToHost, stream_copy);
        cudaMemcpyAsync(V_send_h, V_cur, kv_bytes, cudaMemcpyDeviceToHost, stream_copy);
        // MPI must see the data on the host — block briefly on stream_copy.
        // stream_compute is untouched, so the kernel keeps running.
        cudaStreamSynchronize(stream_copy);
        MPI_Request reqs[4];
        const int n = mpi_int_count(kv_local_elem, "run_ring_overlap/MPI_Isend|Irecv");
        MPI_Isend(K_send_h, n, MPI_FLOAT, next_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(K_recv_h, n, MPI_FLOAT, prev_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(V_send_h, n, MPI_FLOAT, next_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(V_recv_h, n, MPI_FLOAT, prev_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[3]);
        const double t_post1 = MPI_Wtime();
        comm_acc += (t_post1 - t_post0) * 1e3;

        // Host blocks here; the GPU continues running the kernel — this is
        // the overlap window. wait_ms shrinks as the kernel takes longer.
        const double t_wait0 = MPI_Wtime();
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;

        // Write-after-read fence: K_recv aliases the previous step's K_cur,
        // which the previous step's kernel read. We must wait for that kernel
        // to finish before overwriting the buffer, or the H2D corrupts data
        // the kernel is still consuming. Skip at step 0 (no prior kernel).
        if (step >= 1) {
          cudaStreamWaitEvent(stream_copy, ev_ends[step - 1], 0);
        }

        // H2D into K_recv on stream_copy; record event for the next iter.
        // Time the actual transfer with cudaEvents (see note at allocation),
        // not MPI_Wtime — the wall-clock here would only see enqueue cost
        // and underreport comm_ms vs. ring-blocking.
        cudaEventRecord(ev_h2d_starts[step], stream_copy);
        cudaMemcpyAsync(K_recv, K_recv_h, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
        cudaMemcpyAsync(V_recv, V_recv_h, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
        cudaEventRecord(ev_h2d_ends[step], stream_copy);
        cudaEventRecord(comm_done, stream_copy);
#endif

        // (4) Promote received buffers. Pointer-only swap — the actual write
        //     is still in flight on stream_copy and is gated by comm_done.
        std::swap(K_cur, K_recv);
        std::swap(V_cur, V_recv);
        std::swap(K_h_cur, K_h_recv);
        std::swap(V_h_cur, V_h_recv);
      }
    }

    // Drain stream_compute before finalize, then sync once more.
    cudaStreamSynchronize(stream_compute);
    for (int sg = 0; sg < nsg; ++sg)
      launch_attention_finalize(out_d.data() + sg * q_sg_elem, l_d.data() + sg * m_sg, sg_shape,
                                stream_compute);
    cudaStreamSynchronize(stream_compute);

    // Now all kernel + H2D events have fired — safe to query elapsed times.
    double comp_acc = 0.0;
    for (int s = 0; s < P; ++s) {
      float t = 0.f;
      cudaEventElapsedTime(&t, ev_starts[s], ev_ends[s]);
      comp_acc += t;
    }
    // Only P-1 H2D events were recorded (last step has no next chunk).
    for (int s = 0; s < P - 1; ++s) {
      float t = 0.f;
      cudaEventElapsedTime(&t, ev_h2d_starts[s], ev_h2d_ends[s]);
      comm_acc += t;
    }
    return {comm_acc, comp_acc, wait_acc};
  };

  one_pass();  // warmup

  double sum_comm = 0.0, sum_comp = 0.0, sum_wait = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  for (int i = 0; i < cfg.iters; ++i) {
    const auto [c, p_, w] = one_pass();
    sum_comm += c;
    sum_comp += p_;
    sum_wait += w;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  RingResult res;
  res.comm_ms = sum_comm / cfg.iters;
  res.comp_ms = sum_comp / cfg.iters;
  res.wait_ms = sum_wait / cfg.iters;
  res.total_ms = (t_end - t_start) * 1e3 / cfg.iters;

  const double local_t[4] = {res.comm_ms, res.comp_ms, res.wait_ms, res.total_ms};
  double global_t[4] = {};
  MPI_Reduce(local_t, global_t, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (R == 0) {
    res.comm_ms = global_t[0];
    res.comp_ms = global_t[1];
    res.wait_ms = global_t[2];
    res.total_ms = global_t[3];
  }

  float max_err = -1.f;
  if (cfg.verify) {
    const std::size_t q_full_elem = static_cast<std::size_t>(B) * H * S * D;
    const std::size_t kv_full_elem = static_cast<std::size_t>(B) * kv_H * S * D;
    std::vector<float> full_q(q_full_elem), full_k(kv_full_elem), full_v(kv_full_elem);
    fill_host_tensor(full_q, cfg.seed, 0, B, H, S, D, 0);
    fill_host_tensor(full_k, cfg.seed, 1, B, kv_H, S, D, 0);
    fill_host_tensor(full_v, cfg.seed, 2, B, kv_H, S, D, 0);

    AttentionShape ref_shape{B, H, S, S, D};
    ref_shape.kv_heads = kv_H;
    std::vector<float> cpu_out(q_full_elem);
    cpu_attention(full_q.data(), full_k.data(), full_v.data(), cpu_out.data(), ref_shape,
                  cfg.causal);

    one_pass();
    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);

    max_err = 0.f;
    for (int sg = 0; sg < nsg; ++sg) {
      const int q_off_sg = part.q_offset(sg);
      for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
          const float* cpu_slice =
              cpu_out.data() + (static_cast<std::size_t>(b * H + h) * S + q_off_sg) * D;
          const float* dev_slice =
              dev_out_h.data() + sg * q_sg_elem + static_cast<std::size_t>(b * H + h) * Sl * D;
          for (int e = 0; e < Sl * D; ++e)
            max_err = std::max(max_err, std::abs(cpu_slice[e] - dev_slice[e]));
        }
    }

    float global_max;
    MPI_Reduce(&max_err, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    max_err = (R == 0) ? global_max : -1.f;
  }
  res.max_err = max_err;

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

  return res;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

namespace ring_attention {

RingMode mode_from_string(const std::string& s) {
  if (s == "allgather") return RingMode::AllGather;
  if (s == "ring-blocking") return RingMode::RingBlocking;
  if (s == "ring-overlap") return RingMode::RingOverlap;
  char msg[128];
  std::snprintf(msg, sizeof(msg), "Unknown --mode=%s (valid: allgather ring-blocking ring-overlap)",
                s.c_str());
  mpi_die(msg);
}

RingDtype dtype_from_string(const std::string& s) {
  if (s == "fp32" || s == "float") return RingDtype::Float;
  if (s == "fp16" || s == "half") return RingDtype::Half;
  char msg[128];
  std::snprintf(msg, sizeof(msg), "Unknown --dtype=%s (valid: fp32 fp16)", s.c_str());
  mpi_die(msg);
}

// FP16 path lives in ring_loop_fp16.cu.
RingResult run_ring_attention_fp16(const RingConfig& cfg);

RingResult run_ring_attention(const RingConfig& cfg) {
  if (cfg.zigzag && cfg.mode == RingMode::AllGather)
    mpi_die("--zigzag is not supported with --mode=allgather (use ring-blocking or ring-overlap)");
  if (cfg.dtype == RingDtype::Half) return run_ring_attention_fp16(cfg);
  switch (cfg.mode) {
    case RingMode::AllGather:
      return run_allgather(cfg);
    case RingMode::RingBlocking:
      return run_ring_blocking(cfg);
    case RingMode::RingOverlap:
      return run_ring_overlap(cfg);
  }
  mpi_die("unreachable");
}

}  // namespace ring_attention
