#pragma once

/// @file
/// Single-rank decode operation: attend a single query token against the
/// rank-local `DeviceKVCache`.
///
/// Internally packs the cache's populated rows into a contiguous tile so the
/// existing `launch_attention_step` (or its fp16 variant) sees a tensor with
/// stride `seq_k = current_len`, not `s_max`. Online-softmax state `(m, ℓ)`
/// is allocated and finalized internally.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kv_cache.hpp"

namespace ring_attention {

/// Decode one token against the local KV cache (fp32).
///
/// @param q              Query, shape `(B, heads, 1, D)`, device pointer.
/// @param cache          Rank-local KV cache; only rows `[0, current_len)`
///                       are read.
/// @param heads          Number of Q heads (cache holds `kv_heads` internally).
/// @param q_pos_global   Absolute position of the decoded token; used by the
///                       causal mask predicate.
/// @param cache_k_offset Global position of `cache.k_data()[0]`. For a single
///                       rank with the cache built by appending tokens in
///                       canonical order, pass `0`.
/// @param causal         Apply causal mask (key `j` visible iff
///                       `cache_k_offset + j <= q_pos_global`).
/// @param out            Output, shape `(B, heads, 1, D)`, device pointer.
/// @param stream         CUDA stream for all kernels and the cache-pack copy.
void run_local_decode_step(const float* q, const DeviceKVCache<float>& cache, int heads,
                           int q_pos_global, int cache_k_offset, bool causal, float* out,
                           cudaStream_t stream = 0);

/// fp16 variant. Q and the cache are `__half`; `out` stays fp32 (matching the
/// existing fp16 ring kernel's accumulator-precision convention).
void run_local_decode_step_fp16(const __half* q, const DeviceKVCache<__half>& cache, int heads,
                                int q_pos_global, int cache_k_offset, bool causal, float* out,
                                cudaStream_t stream = 0);

}  // namespace ring_attention
