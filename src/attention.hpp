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

/// Tiled FlashAttention-style kernel with online softmax — never materializes
/// the full score matrix.
///
/// Supported `head_dim` values: 32, 64, 128 (other sizes throw via assertion).
/// Causal masking is end-aligned (key `j` visible to query `i` iff
/// `j <= i + (seq_k - seq_q)`), matching `cpu_attention`.
void launch_flash_attention(const float* q, const float* k, const float* v, float* out,
                            const AttentionShape& shape, bool causal, cudaStream_t stream = 0);

/// Reset the persistent online-softmax state used by `launch_attention_step`.
///
/// `out` is the per-row output accumulator with the same layout as the full
/// output tensor. `m` and `l` are per-row scalars, each of shape
/// `(batch, heads, seq_q)` and counted in `m_count = batch*heads*seq_q`.
/// Sets `out = 0`, `l = 0`, `m = -inf`.
void launch_attention_init(float* out, float* m, float* l, const AttentionShape& shape,
                           std::size_t m_count, cudaStream_t stream = 0);

/// One ring step: process a (K_chunk, V_chunk) against the queries, updating
/// the persistent `(out, m, l)` state in place using the FlashAttention online
/// softmax recurrence. `shape.seq_k` describes the chunk only.
///
/// Global token positions are `(q_offset + i)` for queries and
/// `(k_offset + j)` for keys; causal masking visibility is
/// `k_offset + j <= q_offset + i`.
void launch_attention_step(const float* q, const float* k, const float* v, float* out, float* m,
                           float* l, const AttentionShape& shape, int q_offset, int k_offset,
                           bool causal, cudaStream_t stream = 0);

/// Finalize the ring: divide `out` by the per-row sum `l` (no-op rows where
/// `l == 0` are zeroed). After this call, `out` contains the final attention
/// output and `m`, `l` are no longer needed.
void launch_attention_finalize(float* out, const float* l, const AttentionShape& shape,
                               std::size_t m_count, cudaStream_t stream = 0);

}  // namespace ring_attention
