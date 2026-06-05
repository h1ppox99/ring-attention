#pragma once

/// @file
/// Public host-side API for the ring-attention CUDA kernels.
///
/// Tensor layout convention: row-major (batch, heads, seq, head_dim), contiguous.
/// All host functions launch kernels on the default stream and return after
/// launch (no implicit synchronization). Callers must synchronize before
/// reading device results.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace ring_attention {

/// Problem shape for one attention call.
struct AttentionShape {
  int batch;
  int heads;
  int seq_q;        ///< Number of query rows.
  int seq_k;        ///< Number of key/value rows.
  int head_dim;     ///< Per-head feature dimension.
  int kv_heads{0};  ///< KV head count for GQA/MQA; 0 means same as heads (MHA).
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

/// FP16 / Tensor-Core variant of `launch_flash_attention`.
///
/// Same row-major `(batch, heads, seq, head_dim)` layout and same `float*`
/// API as the FP32 kernel — Q/K/V/O are stored in FP32 and converted to
/// `__half` inside the kernel. Both matmuls (Q·Kᵀ and P·V) run on Tensor
/// Cores (sm_75 `wmma 16x16x16`) with FP32 accumulation; softmax stays in
/// FP32. Numerical tolerance vs. the CPU reference is ~1e-2 (vs. 1e-4 for
/// the FP32 path).
///
/// Supported `head_dim` values: 32, 64, 128. Requires sm_70+.
void launch_flash_attention_fp16(const float* q, const float* k, const float* v, float* out,
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
/// Global token positions are `(q_offset + i*pos_stride)` for queries and
/// `(k_offset + j*pos_stride)` for keys; causal masking visibility is
/// `k_offset + j*pos_stride <= q_offset + i*pos_stride`. `pos_stride` defaults
/// to 1 (contiguous / zig-zag chunks, where consecutive local rows are
/// consecutive global positions); pass `cp_size` for striped partitioning,
/// where rank r owns global positions r, r+cp_size, r+2*cp_size, …
///
/// When `shape.seq_q == 1` this dispatches automatically to the
/// `launch_attention_decode_step` kernel, which is structurally different
/// (one warp per (B,H) row, lane-split D, no shared memory) — see
/// `src/attention_decode.cu` and KERNEL_OPTIMIZATIONS.md Round 5. Decode is a
/// single query row, so `pos_stride` does not apply there.
void launch_attention_step(const float* q, const float* k, const float* v, float* out, float* m,
                           float* l, const AttentionShape& shape, int q_offset, int k_offset,
                           bool causal, cudaStream_t stream = 0, int pos_stride = 1);

/// Piecewise-affine position map for the segmented attention step.
///
/// The local shard is `n_seg` equal-length contiguous segments; local row `i`
/// sits at global position `base[i / seg_len] + (i % seg_len)` (stride 1 *within*
/// a segment, arbitrary jump *between* segments). This is exactly a fine zig-zag
/// assignment expressed as one shard instead of `n_seg` separate sub-groups.
struct SegMap {
  static constexpr int kMaxSeg = 16;  ///< up to 8 zig-zag passes (2N sub-groups).
  int n_seg{1};                       ///< number of segments; seq == n_seg * seg_len.
  int seg_len{0};                     ///< rows per segment (equal across segments).
  int base[kMaxSeg]{};                ///< global position of each segment's first row.
};

/// Single-launch segmented attention step (FP32). Same online-softmax
/// `(O, m, l)` persistent-state semantics as `launch_attention_step`, but query
/// and key global positions come from a piecewise-affine `SegMap` instead of an
/// affine `(offset, stride)`. This lets a fine zig-zag assignment run in ONE
/// kernel launch over the whole local shard, rather than the `n_seg^2` affine
/// calls the sub-group loop in `ring_loop.cu` issues. Supported `head_dim`:
/// 32, 64, 128, 256.
void launch_attention_step_segmented(const float* q, const float* k, const float* v, float* out,
                                     float* m, float* l, const AttentionShape& shape,
                                     const SegMap& qmap, const SegMap& kmap, bool causal,
                                     cudaStream_t stream = 0);

/// Decode-specialized step kernel — `seq_q = 1`, one warp per `(batch, head)`.
/// Same `(M, L, O)` persistent-state semantics as `launch_attention_step` so
/// the ring decode loop can call it once per ring step. Supports
/// `head_dim ∈ {32, 64, 128, 256}` (must be a multiple of 32).
/// `kv_row_stride` is the per-head row stride of the K/V buffers: 0 (default)
/// means contiguous (stride == `shape.seq_k`); pass the KV cache's `S_max` to
/// read the cache in place without a de-strided pack copy.
void launch_attention_decode_step(const float* q, const float* k, const float* v, float* out,
                                  float* m, float* l, const AttentionShape& shape, int q_offset,
                                  int k_offset, bool causal, cudaStream_t stream = 0,
                                  int kv_row_stride = 0);

/// Finalize the ring: divide `out` by the per-row sum `l` (no-op rows where
/// `l == 0` are zeroed). After this call, `out` contains the final attention
/// output and `m`, `l` are no longer needed.
void launch_attention_finalize(float* out, const float* l, const AttentionShape& shape,
                               cudaStream_t stream = 0);

/// FP16 / Tensor-Core ring step. Same semantics as `launch_attention_step` but
/// Q/K/V live in `__half` (so MPI traffic between ranks can be FP16) and the
/// two matmuls run on Tensor Cores. The persistent state `(out, m, l)` stays
/// FP32 — these accumulators benefit from the extra precision and are written
/// back at every step. `launch_attention_init` / `launch_attention_finalize`
/// are reused unchanged.
///
/// Supported `head_dim`: 32, 64, 128. Requires sm_70+.
void launch_attention_step_fp16(const __half* q, const __half* k, const __half* v, float* out,
                                float* m, float* l, const AttentionShape& shape, int q_offset,
                                int k_offset, bool causal, cudaStream_t stream = 0,
                                int pos_stride = 1);

/// Element-wise FP32 → FP16 cast on the device. Used to stage Q (and K/V in
/// tests) into the FP16 path without a CPU round-trip.
void launch_float_to_half(const float* src, __half* dst, std::size_t n, cudaStream_t stream = 0);

/// Element-wise FP16 → FP32 cast on the device. Inverse of
/// `launch_float_to_half`; used to widen FP16 transit buffers back to FP32
/// before FP32 kernels read them.
void launch_half_to_float(const __half* src, float* dst, std::size_t n, cudaStream_t stream = 0);

/// Element-wise FP32 ↔ INT8 casts with a fixed symmetric scale of 127.
/// Kept for experimentation with quantized KV transit. Valid for inputs in
/// [-1, 1) (the project's synthetic data range); out-of-range values are clamped.
void launch_float_to_int8(const float* src, signed char* dst, std::size_t n,
                          cudaStream_t stream = 0);
void launch_int8_to_float(const signed char* src, float* dst, std::size_t n,
                          cudaStream_t stream = 0);

}  // namespace ring_attention
