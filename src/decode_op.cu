/// @file
/// Single-rank decode op implementation.
///
/// Drive the flash-attention step kernel with seq_q=1, seq_k=current_len.
///
/// The fp32 path reads the cache in place: the decode kernel accepts a per-head
/// row stride (`kv_row_stride = s_max`), so no de-stride copy is needed. The
/// fp16 / Tensor-Core step kernel has no strided-read mode, so the fp16 path
/// still packs the populated rows into a contiguous (B, kv_H, current_len, D)
/// tile via `cudaMemcpy2DAsync` (handling the `S_max` → `current_len` stride
/// mismatch) before the kernel call.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "attention.hpp"
#include "common.cuh"
#include "decode_op.hpp"
#include "device_tensor.hpp"
#include "kv_cache.hpp"

namespace ring_attention {

namespace {

template <typename T>
void pack_cache(const DeviceKVCache<T>& cache, T* k_packed, T* v_packed, int Sk,
                cudaStream_t stream) {
  const int B = cache.batch();
  const int kv_H = cache.kv_heads();
  const int D = cache.head_dim();
  const int S_max = cache.s_max();
  const std::size_t width = static_cast<std::size_t>(Sk) * D * sizeof(T);
  const std::size_t dpitch = width;
  const std::size_t spitch = static_cast<std::size_t>(S_max) * D * sizeof(T);
  const std::size_t height = static_cast<std::size_t>(B) * kv_H;
  cudaCheck(cudaMemcpy2DAsync(k_packed, dpitch, cache.k_data(), spitch, width, height,
                              cudaMemcpyDeviceToDevice, stream));
  cudaCheck(cudaMemcpy2DAsync(v_packed, dpitch, cache.v_data(), spitch, width, height,
                              cudaMemcpyDeviceToDevice, stream));
}

}  // namespace

void run_local_decode_step(const float* q, const DeviceKVCache<float>& cache, int heads,
                           int q_pos_global, int cache_k_offset, bool causal, float* out,
                           cudaStream_t stream) {
  const int B = cache.batch();
  const int kv_H = cache.kv_heads();
  const int D = cache.head_dim();
  const int Sk = cache.current_len();

  AttentionShape shape{B, heads, 1, Sk, D};
  shape.kv_heads = kv_H;
  const std::size_t m_count = static_cast<std::size_t>(B) * heads;
  DeviceTensor<float> m_d(m_count), l_d(m_count);

  launch_attention_init(out, m_d.data(), l_d.data(), shape, m_count, stream);
  // Read the cache in place (per-head row stride = s_max) instead of packing the
  // populated rows into a contiguous tile first — the fp32 decode kernel handles
  // the stride directly, the same path the ring decode loop uses (Round 17).
  launch_attention_decode_step(q, cache.k_data(), cache.v_data(), out, m_d.data(), l_d.data(),
                               shape, q_pos_global, cache_k_offset, causal, stream,
                               /*kv_row_stride=*/cache.s_max());
  launch_attention_finalize(out, l_d.data(), shape, stream);
}

void run_local_decode_step_fp16(const __half* q, const DeviceKVCache<__half>& cache, int heads,
                                int q_pos_global, int cache_k_offset, bool causal, float* out,
                                cudaStream_t stream) {
  const int B = cache.batch();
  const int kv_H = cache.kv_heads();
  const int D = cache.head_dim();
  const int Sk = cache.current_len();

  const std::size_t packed_count = static_cast<std::size_t>(B) * kv_H * Sk * D;
  DeviceTensor<__half> k_packed(packed_count);
  DeviceTensor<__half> v_packed(packed_count);
  pack_cache(cache, k_packed.data(), v_packed.data(), Sk, stream);

  AttentionShape shape{B, heads, 1, Sk, D};
  shape.kv_heads = kv_H;
  const std::size_t m_count = static_cast<std::size_t>(B) * heads;
  DeviceTensor<float> m_d(m_count), l_d(m_count);

  launch_attention_init(out, m_d.data(), l_d.data(), shape, m_count, stream);
  launch_attention_step_fp16(q, k_packed.data(), v_packed.data(), out, m_d.data(), l_d.data(),
                             shape, q_pos_global, cache_k_offset, causal, stream);
  launch_attention_finalize(out, l_d.data(), shape, stream);
}

}  // namespace ring_attention
