#pragma once

/// @file
/// Host-side orchestrator for multi-GPU ring attention.
///
/// One `run_ring_attention` call drives a full attention forward pass across
/// all MPI ranks for a given mode (allgather / ring-blocking / ring-overlap).
/// Timing fields in `RingResult` are populated by the inner loops; they are
/// reduced across ranks by the caller if a summary is needed.

#include <cstdint>
#include <string>

namespace ring_attention {

enum class RingMode { AllGather, RingBlocking, RingOverlap };

/// Compute / transport dtype. FP16 routes through the wmma-based kernel and
/// sends K/V over MPI as half-precision; FP32 is the original path.
enum class RingDtype { Float, Half };

/// Parse "--mode=<string>" into a RingMode. Aborts (MPI_Abort) on unknown input.
RingMode mode_from_string(const std::string& s);

/// Parse "--dtype=<string>" into a RingDtype (`fp32`/`float` or `fp16`/`half`).
RingDtype dtype_from_string(const std::string& s);

/// Per-iteration timing and correctness results (all ranks report their own
/// values; caller does MPI_Reduce(MPI_MAX) to get global summary).
struct RingResult {
  double comm_ms = 0.0;   ///< D2H staging + MPI post time.
  double comp_ms = 0.0;   ///< Sum of cudaEvent intervals around attention_step.
  double wait_ms = 0.0;   ///< Time blocked in MPI_Waitall (unhidden comm).
  double total_ms = 0.0;  ///< Wall-clock from barrier to barrier.
  float max_err = -1.f;   ///< Max absolute error vs CPU reference; -1 if not verified.
};

/// Configuration passed from the CLI driver to the ring loop.
struct RingConfig {
  int rank;
  int cp_size;
  int batch;
  int heads;
  int seq;  ///< Global sequence length (must be divisible by cp_size).
  int head_dim;
  bool causal;
  bool zigzag;
  bool verify;
  bool csv;
  RingMode mode;
  int iters;
  uint32_t seed = 42u;
  RingDtype dtype = RingDtype::Float;
};

/// Run one complete ring-attention forward pass (all modes, all phases).
///
/// Allocates device buffers internally. Q/K/V are filled from `gen_elem`
/// using `cfg.seed`. If `cfg.verify`, also runs `cpu_attention` and
/// reports max absolute error in `RingResult::max_err`.
RingResult run_ring_attention(const RingConfig& cfg);

}  // namespace ring_attention
