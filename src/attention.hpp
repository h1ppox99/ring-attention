#pragma once

/// @file
/// Public host-side API for the ring-attention CUDA kernels.
///
/// Tensor layout convention: row-major (batch, heads, seq, head_dim), contiguous.
/// All host functions launch kernels on the default stream and return after
/// launch (no implicit synchronization). Callers must synchronize before
/// reading device results.

#include <cuda_runtime.h>

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

/// Naive dense attention on the GPU: materializes the per-row score vector in
/// shared memory and computes `O = softmax(QK^T / sqrt(D)) V`.
///
/// One CUDA block per `(batch, head, query-row)`. Requires
/// `seq_k * sizeof(float)` of dynamic shared memory plus two scalars, so
/// `seq_k` is capped by the device's per-block shared-memory limit.
///
/// All pointers refer to device memory; the tensors are row-major
/// `(batch, heads, seq, head_dim)`. Launches on `stream` (default 0).
void launch_naive_attention(const float* q, const float* k, const float* v, float* out,
                            const AttentionShape& shape, bool causal, cudaStream_t stream = 0);

}  // namespace ring_attention
