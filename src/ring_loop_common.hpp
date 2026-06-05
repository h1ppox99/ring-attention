#pragma once

/// @file
/// Shared host-side helpers for the ring-attention orchestrators.
///
/// Both the FP32 path (`ring_loop.cu`) and the FP16 path (`ring_loop_fp16.cu`),
/// and the allgather / ring-blocking / ring-overlap / ring-2d modes within each,
/// used to carry private copies of the MPI abort/count guards, the gen_elem
/// tensor fills, the warmup+timed-loop driver, and the CPU-reference
/// verification. They are collected here as `inline` helpers so there is one
/// copy. Everything operates on host fp32 data plus `RingConfig` / `RingResult`,
/// so it is dtype-agnostic (the FP16 kernels still produce an fp32 output, which
/// is what `verify_local_output` compares).

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "attention.hpp"
#include "cpu_attention.hpp"
#include "ring_gen.hpp"
#include "ring_loop.hpp"

namespace ring_attention {
namespace detail {

/// MPI-safe abort so a single rank's failure tears down all ranks.
[[noreturn]] inline void mpi_die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  std::exit(EXIT_FAILURE);  // satisfy [[noreturn]]; MPI_Abort does not return
}

/// Validate that `count` fits the `int` used by MPI-3 point-to-point and
/// collective calls. NVHPC 24.1 ships OpenMPI 4.1.7 (MPI-3.1), so we cannot use
/// `MPI_Count` / the `_c` variants. `count` is an element count on the FP32 path
/// and a byte count on the FP16 path; `where` disambiguates in the message.
/// Aborts on overflow rather than silently truncating the transfer size.
inline int mpi_int_count(std::size_t count, const char* where) {
  if (count > static_cast<std::size_t>(INT_MAX)) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "MPI count overflow in %s: %zu exceeds INT_MAX=%d — "
                  "reduce per-rank tensor size or chunk the transfer.",
                  where, count, INT_MAX);
    mpi_die(buf);
  }
  return static_cast<int>(count);
}

/// Fill `buf[off, off + B*H*seq*D)` with gen_elem values for a (B, H, seq, D)
/// block whose first global sequence index is `gss`. `buf` must already be sized.
inline void fill_region(std::vector<float>& buf, std::size_t off, uint32_t seed, int tensor_id,
                        int B, int H, int seq, int D, int gss) {
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int s = 0; s < seq; ++s)
        for (int d = 0; d < D; ++d)
          buf[off + (static_cast<std::size_t>(b * H + h) * seq + s) * D + d] =
              gen_elem(seed, tensor_id, b, h, gss + s, d);
}

/// Resize `buf` to a full (B, H, seq, D) tensor and fill it from gen_elem.
/// `gss` is the first global sequence index in this slice.
inline void fill_host_tensor(std::vector<float>& buf, uint32_t seed, int tensor_id, int B, int H,
                             int seq, int D, int gss) {
  buf.resize(static_cast<std::size_t>(B) * H * seq * D);
  fill_region(buf, 0, seed, tensor_id, B, H, seq, D, gss);
}

/// MPI_Reduce(MAX) the four timing fields to rank 0 — the slowest rank dictates
/// the wall-clock. Other ranks keep their local values.
inline void reduce_timings_max(int rank, RingResult& res) {
  const double local_t[4] = {res.comm_ms, res.comp_ms, res.wait_ms, res.total_ms};
  double global_t[4] = {};
  MPI_Reduce(local_t, global_t, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    res.comm_ms = global_t[0];
    res.comp_ms = global_t[1];
    res.wait_ms = global_t[2];
    res.total_ms = global_t[3];
  }
}

/// One warmup pass, then `cfg.iters` timed passes wrapped barrier-to-barrier.
/// `one_pass()` returns `{comm_ms, comp_ms, wait_ms}` for a single forward pass
/// (modes with no host wait return 0 for the third). Fills the (averaged) timing
/// fields of `res` and reduces them across ranks. `max_err` is left untouched.
template <typename PassFn>
inline void run_timed_passes(const RingConfig& cfg, const PassFn& one_pass, RingResult& res) {
  one_pass();  // warmup — pays kernel-load / MPI-startup costs

  double sum_comm = 0.0, sum_comp = 0.0, sum_wait = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  for (int i = 0; i < cfg.iters; ++i) {
    const auto [c, p, w] = one_pass();
    sum_comm += c;
    sum_comp += p;
    sum_wait += w;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  res.comm_ms = sum_comm / cfg.iters;
  res.comp_ms = sum_comp / cfg.iters;
  res.wait_ms = sum_wait / cfg.iters;
  res.total_ms = (t_end - t_start) * 1e3 / cfg.iters;
  reduce_timings_max(cfg.rank, res);
}

/// Compare this rank's local device output against the CPU reference.
///
/// Regenerates the full (B,H,S,D) Q/K/V from `cfg.seed`, runs `cpu_attention`,
/// and walks each local sub-group. `q_offsets[sg]` is the global query offset of
/// sub-group sg; in `dev_out_h` that sub-group lives at
/// `sg*q_sg_elem + (b*H+h)*Sl*D`. (Allgather is the single-sub-group case:
/// `q_offsets = {rank*Sl}`, `q_sg_elem = q_local_elem`.) Returns the global max
/// absolute error on rank 0, -1 on every other rank.
inline float verify_local_output(const RingConfig& cfg, int B, int H, int S, int D, int kv_H,
                                 int Sl, std::size_t q_sg_elem, const std::vector<int>& q_offsets,
                                 const std::vector<float>& dev_out_h) {
  const std::size_t q_full_elem = static_cast<std::size_t>(B) * H * S * D;
  const std::size_t kv_full_elem = static_cast<std::size_t>(B) * kv_H * S * D;
  std::vector<float> full_q(q_full_elem), full_k(kv_full_elem), full_v(kv_full_elem);
  fill_host_tensor(full_q, cfg.seed, 0, B, H, S, D, 0);
  fill_host_tensor(full_k, cfg.seed, 1, B, kv_H, S, D, 0);
  fill_host_tensor(full_v, cfg.seed, 2, B, kv_H, S, D, 0);

  AttentionShape ref_shape{B, H, S, S, D};
  ref_shape.kv_heads = kv_H;
  std::vector<float> cpu_out(q_full_elem);
  cpu_attention(full_q.data(), full_k.data(), full_v.data(), cpu_out.data(), ref_shape, cfg.causal);

  float max_err = 0.f;
  for (std::size_t sg = 0; sg < q_offsets.size(); ++sg) {
    const int q_off_sg = q_offsets[sg];
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

  float global_max = 0.f;
  MPI_Reduce(&max_err, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  return (cfg.rank == 0) ? global_max : -1.f;
}

}  // namespace detail
}  // namespace ring_attention
