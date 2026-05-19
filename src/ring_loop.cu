/// @file
/// Host-side ring-attention orchestrator.
/// Implements allgather baseline and ring-blocking; ring-overlap is stubbed.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "attention.hpp"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"
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
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;
  const int Sl = S / P;
  const int q_off = R * Sl;

  const std::size_t local_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t full_elem = static_cast<std::size_t>(B) * H * S * D;
  const std::size_t m_count = static_cast<std::size_t>(B) * H * Sl;

  // Fill local Q/K/V on host — constant across iterations.
  std::vector<float> q_h, k_h, v_h;
  fill_host_tensor(q_h, cfg.seed, 0, B, H, Sl, D, q_off);
  fill_host_tensor(k_h, cfg.seed, 1, B, H, Sl, D, q_off);
  fill_host_tensor(v_h, cfg.seed, 2, B, H, Sl, D, q_off);

  // H2D local Q — stays on device for all iterations.
  DeviceTensor<float> q_d(local_elem);
  q_d.copy_from_host(q_h);

  // Reusable buffers for gather + rearrange + device K/V.
  std::vector<float> k_gathered(full_elem), v_gathered(full_elem);
  std::vector<float> full_k_h(full_elem), full_v_h(full_elem);
  DeviceTensor<float> k_d(full_elem), v_d(full_elem);
  DeviceTensor<float> out_d(local_elem), m_d(m_count), l_d(m_count);
  const AttentionShape shape{B, H, Sl, S, D};

  // One complete allgather + attention pass.
  // Returns {comm_ms, comp_ms} for that single pass.
  auto one_pass = [&]() -> std::pair<double, double> {
    // --- Communication: allgather K,V + rearrange + H2D ----------------------
    const double tc0 = MPI_Wtime();
    MPI_Allgather(k_h.data(), static_cast<int>(local_elem), MPI_FLOAT, k_gathered.data(),
                  static_cast<int>(local_elem), MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Allgather(v_h.data(), static_cast<int>(local_elem), MPI_FLOAT, v_gathered.data(),
                  static_cast<int>(local_elem), MPI_FLOAT, MPI_COMM_WORLD);

    // Rearrange from rank-major (P,B,H,Sl,D) to (B,H,S,D).
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h)
        for (int p = 0; p < P; ++p) {
          const std::size_t src = static_cast<std::size_t>(p) * local_elem +
                                  static_cast<std::size_t>(b * H + h) * Sl * D;
          const std::size_t dst = (static_cast<std::size_t>(b * H + h) * S + p * Sl) * D;
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
    std::vector<float> full_q_cpu(full_elem), full_k_cpu(full_elem), full_v_cpu(full_elem);
    fill_host_tensor(full_q_cpu, cfg.seed, 0, B, H, S, D, 0);
    fill_host_tensor(full_k_cpu, cfg.seed, 1, B, H, S, D, 0);
    fill_host_tensor(full_v_cpu, cfg.seed, 2, B, H, S, D, 0);

    std::vector<float> cpu_out(full_elem);
    cpu_attention(full_q_cpu.data(), full_k_cpu.data(), full_v_cpu.data(), cpu_out.data(),
                  AttentionShape{B, H, S, S, D}, cfg.causal);

    // Re-run one pass so out_d holds a fresh result.
    one_pass();

    std::vector<float> dev_out_h(local_elem);
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
/// Both the GPU step and the host-side MPI exchange run on the default stream
/// and on the host respectively, so they happen to overlap incidentally, but
/// the trailing `cudaDeviceSync` + `MPI_Waitall` block until both are complete
/// before the next step — that's the "blocking" part. Real overlap (separate
/// streams + events) comes in P6.
///
/// Timing:
///   - comm_ms : D2H + H2D + Isend/Irecv post (the unavoidable staging cost)
///   - wait_ms : MPI_Waitall time = unhidden communication latency
///   - comp_ms : sum of cudaEvent intervals around each attention_step call
ring_attention::RingResult run_ring_blocking(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;
  const int Sl = S / P;
  const int q_off = R * Sl;

  const std::size_t local_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t m_count = static_cast<std::size_t>(B) * H * Sl;
  const std::size_t bytes = local_elem * sizeof(float);

  // Partition supplies next/prev ranks and k_offset for each ring step.
  RingPartition part(P, R, S, RingPartition::Mode::Contiguous);
  const int next_rank = part.next_rank();
  const int prev_rank = part.prev_rank();

  // Local Q/K/V — generated once on the host, re-uploaded each iteration so
  // the ring loop always starts from a clean K/V state.
  std::vector<float> q_h, k_h_init, v_h_init;
  fill_host_tensor(q_h, cfg.seed, 0, B, H, Sl, D, q_off);
  fill_host_tensor(k_h_init, cfg.seed, 1, B, H, Sl, D, q_off);
  fill_host_tensor(v_h_init, cfg.seed, 2, B, H, Sl, D, q_off);

  // Device buffers: Q stays put; K/V double-buffered (K_a/K_b, V_a/V_b).
  DeviceTensor<float> q_d(local_elem);
  q_d.copy_from_host(q_h);
  DeviceTensor<float> K_a(local_elem), K_b(local_elem);
  DeviceTensor<float> V_a(local_elem), V_b(local_elem);
  DeviceTensor<float> out_d(local_elem), m_d(m_count), l_d(m_count);

  const AttentionShape full_shape{B, H, Sl, S, D};    // for init / finalize (uses seq_q)
  const AttentionShape chunk_shape{B, H, Sl, Sl, D};  // each ring step processes one chunk

  // Pinned host staging — page-locked so D2H/H2D can be async without copies.
  float *K_send_h = nullptr, *V_send_h = nullptr, *K_recv_h = nullptr, *V_recv_h = nullptr;
  cudaHostAlloc(&K_send_h, bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_send_h, bytes, cudaHostAllocDefault);
  cudaHostAlloc(&K_recv_h, bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_recv_h, bytes, cudaHostAllocDefault);

  // One full ring pass — returns (comm_ms, comp_ms, wait_ms) for that pass.
  auto one_pass = [&]() -> std::tuple<double, double, double> {
    // Reset K/V to local chunk (timed iterations need a clean starting state).
    cudaMemcpy(K_a.data(), k_h_init.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(V_a.data(), v_h_init.data(), bytes, cudaMemcpyHostToDevice);
    float* K_cur = K_a.data();
    float* V_cur = V_a.data();
    float* K_recv = K_b.data();
    float* V_recv = V_b.data();

    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), full_shape, m_count);

    double comm_acc = 0.0, comp_acc = 0.0, wait_acc = 0.0;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    for (int step = 0; step < P; ++step) {
      MPI_Request reqs[4];
      int n_req = 0;

      // (1) D2H stage current K/V → host pinned buffer + (2) post MPI exchange.
      //     Skipped on the last step (no further chunk needed).
      const double t_post0 = MPI_Wtime();
      if (step < P - 1) {
        cudaMemcpy(K_send_h, K_cur, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(V_send_h, V_cur, bytes, cudaMemcpyDeviceToHost);
        const int n = static_cast<int>(local_elem);
        MPI_Isend(K_send_h, n, MPI_FLOAT, next_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Irecv(K_recv_h, n, MPI_FLOAT, prev_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Isend(V_send_h, n, MPI_FLOAT, next_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Irecv(V_recv_h, n, MPI_FLOAT, prev_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[n_req++]);
      }
      const double t_post1 = MPI_Wtime();
      comm_acc += (t_post1 - t_post0) * 1e3;

      // (3) Compute the step on the current K/V chunk.
      //     k_offset for step s is the global token offset of the source rank.
      const int k_off = part.k_offset_for_step(step);
      cudaEventRecord(ev0);
      launch_attention_step(q_d.data(), K_cur, V_cur, out_d.data(), m_d.data(), l_d.data(),
                            chunk_shape, q_off, k_off, cfg.causal);
      cudaEventRecord(ev1);
      cudaEventSynchronize(ev1);
      float comp_float_ms = 0.f;
      cudaEventElapsedTime(&comp_float_ms, ev0, ev1);
      comp_acc += comp_float_ms;

      // (4) Wait for the comm we posted, then H2D the received chunk.
      //     wait_ms = MPI_Waitall only (the headline "unhidden comm" metric);
      //     H2D is folded into comm_ms.
      if (step < P - 1) {
        const double t_wait0 = MPI_Wtime();
        MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;

        const double t_h2d0 = MPI_Wtime();
        cudaMemcpy(K_recv, K_recv_h, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(V_recv, V_recv_h, bytes, cudaMemcpyHostToDevice);
        const double t_h2d1 = MPI_Wtime();
        comm_acc += (t_h2d1 - t_h2d0) * 1e3;

        // (5) Promote received buffers to "current"; the old current slot will
        //     be reused as the next recv target.
        std::swap(K_cur, K_recv);
        std::swap(V_cur, V_recv);
      }
    }

    launch_attention_finalize(out_d.data(), l_d.data(), full_shape);
    cudaDeviceSynchronize();
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
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

  // Verification: regenerate full Q/K/V, run cpu_attention, compare this rank's slice.
  float max_err = -1.f;
  if (cfg.verify) {
    const std::size_t full_elem = static_cast<std::size_t>(B) * H * S * D;
    std::vector<float> full_q(full_elem), full_k(full_elem), full_v(full_elem);
    fill_host_tensor(full_q, cfg.seed, 0, B, H, S, D, 0);
    fill_host_tensor(full_k, cfg.seed, 1, B, H, S, D, 0);
    fill_host_tensor(full_v, cfg.seed, 2, B, H, S, D, 0);

    std::vector<float> cpu_out(full_elem);
    cpu_attention(full_q.data(), full_k.data(), full_v.data(), cpu_out.data(),
                  AttentionShape{B, H, S, S, D}, cfg.causal);

    // Re-run one pass so out_d reflects a complete forward.
    one_pass();
    std::vector<float> dev_out_h(local_elem);
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

  cudaFreeHost(K_send_h);
  cudaFreeHost(V_send_h);
  cudaFreeHost(K_recv_h);
  cudaFreeHost(V_recv_h);

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

RingResult run_ring_attention(const RingConfig& cfg) {
  switch (cfg.mode) {
    case RingMode::AllGather:
      return run_allgather(cfg);
    case RingMode::RingBlocking:
      return run_ring_blocking(cfg);
    case RingMode::RingOverlap:
      mpi_die("ring-overlap not yet implemented");
  }
  mpi_die("unreachable");
}

}  // namespace ring_attention
