#pragma once

/// @file
/// Dense CPU reference attention used as ground truth for CUDA kernel tests.

#include <cstdint>
#include <vector>

#include "attention.hpp"

namespace ring_attention {

/// Dense scaled-dot-product attention on the host, fp32.
///
/// Computes `O = softmax(Q K^T / sqrt(head_dim)) V` with optional causal
/// masking (lower-triangular). Inputs are row-major `(batch, heads, seq, head_dim)`.
///
/// @param q       Query tensor, length `batch*heads*seq_q*head_dim`.
/// @param k       Key tensor,   length `batch*heads*seq_k*head_dim`.
/// @param v       Value tensor, length `batch*heads*seq_k*head_dim`.
/// @param out     Output tensor, length `batch*heads*seq_q*head_dim` (overwritten).
/// @param shape   Problem shape.
/// @param causal  Mask key positions `j > i + (seq_k - seq_q)` (aligned to the
///                end so seq_q == seq_k gives the standard lower-triangular mask).
void cpu_attention(const float* q, const float* k, const float* v, float* out,
                   const AttentionShape& shape, bool causal);

/// Deterministic xorshift32-based RNG to produce reproducible test inputs.
class XorShift32 {
 public:
  explicit XorShift32(std::uint32_t seed = 0x9E3779B9u) : state_(seed ? seed : 0x9E3779B9u) {}

  /// Next raw 32-bit value.
  std::uint32_t next_u32();
  /// Uniform float in [-1, 1).
  float next_uniform();

  /// Fill `out` with uniform [-1, 1) floats.
  void fill_uniform(std::vector<float>& out);

 private:
  std::uint32_t state_;
};

}  // namespace ring_attention
