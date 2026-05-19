/// @file
/// Host-side ring-attention orchestrator.
/// Implements allgather baseline; ring-blocking and ring-overlap are stubbed.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "attention.hpp"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"
#include "ring_gen.hpp"
#include "ring_loop.hpp"

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
      mpi_die("ring-blocking not yet implemented");
    case RingMode::RingOverlap:
      mpi_die("ring-overlap not yet implemented");
  }
  mpi_die("unreachable");
}

}  // namespace ring_attention
