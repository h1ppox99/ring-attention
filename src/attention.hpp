#pragma once

/// @file
/// Public host-side API for the ring-attention CUDA kernels.
///
/// Tensor layout convention: row-major (batch, heads, seq, head_dim), contiguous.
/// All host functions launch kernels on the default stream and return after
/// launch (no implicit synchronization). Callers must synchronize before
/// reading device results.

#include <cstddef>

namespace ring_attention {

/// Problem shape for one attention call.
struct AttentionShape {
  int batch;
  int heads;
  int seq_q;     ///< Number of query rows.
  int seq_k;     ///< Number of key/value rows.
  int head_dim;  ///< Per-head feature dimension.
};

/// Total element count for a (batch, heads, seq, head_dim) tensor.
inline std::size_t tensor_numel(int batch, int heads, int seq, int head_dim) {
  return static_cast<std::size_t>(batch) * heads * seq * head_dim;
}

// Kernel-launching entry points will be declared in subsequent milestones.

}  // namespace ring_attention
