/// @file
/// DeviceKVCache template instantiations.
///
/// `copy_prefill` and `append` both use `cudaMemcpy2DAsync` to handle the
/// strided write pattern: the destination buffer is `(B*H_kv, S_max, D)` in
/// memory, but we write only `(B*H_kv, prefill_len_or_1, D)` rows at an
/// offset within each `S_max * D` destination row.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "common.cuh"
#include "kv_cache.hpp"

namespace ring_attention {

template <typename T>
DeviceKVCache<T>::DeviceKVCache(int batch, int kv_heads, int s_max, int head_dim)
    : k_(static_cast<std::size_t>(batch) * kv_heads * s_max * head_dim),
      v_(static_cast<std::size_t>(batch) * kv_heads * s_max * head_dim),
      batch_(batch),
      kv_heads_(kv_heads),
      s_max_(s_max),
      head_dim_(head_dim) {}

template <typename T>
void DeviceKVCache<T>::copy_prefill(const T* k_src, const T* v_src, int prefill_len,
                                    cudaStream_t stream) {
  if (prefill_len > s_max_) {
    fprintf(stderr, "DeviceKVCache::copy_prefill: prefill_len=%d > s_max=%d\n", prefill_len,
            s_max_);
    std::abort();
  }
  const std::size_t dst_pitch = static_cast<std::size_t>(s_max_) * head_dim_ * sizeof(T);
  const std::size_t src_pitch = static_cast<std::size_t>(prefill_len) * head_dim_ * sizeof(T);
  const std::size_t width = src_pitch;
  const std::size_t height = static_cast<std::size_t>(batch_) * kv_heads_;
  cudaCheck(cudaMemcpy2DAsync(k_.data(), dst_pitch, k_src, src_pitch, width, height,
                              cudaMemcpyDeviceToDevice, stream));
  cudaCheck(cudaMemcpy2DAsync(v_.data(), dst_pitch, v_src, src_pitch, width, height,
                              cudaMemcpyDeviceToDevice, stream));
  current_len_ = prefill_len;
}

template <typename T>
void DeviceKVCache<T>::append(const T* k_step, const T* v_step, cudaStream_t stream) {
  if (current_len_ >= s_max_) {
    fprintf(stderr, "DeviceKVCache::append: cache full at current_len=%d == s_max=%d\n",
            current_len_, s_max_);
    std::abort();
  }
  const std::size_t dst_pitch = static_cast<std::size_t>(s_max_) * head_dim_ * sizeof(T);
  const std::size_t src_pitch = static_cast<std::size_t>(head_dim_) * sizeof(T);
  const std::size_t width = src_pitch;
  const std::size_t height = static_cast<std::size_t>(batch_) * kv_heads_;
  const std::size_t dst_offset = static_cast<std::size_t>(current_len_) * head_dim_;
  cudaCheck(cudaMemcpy2DAsync(k_.data() + dst_offset, dst_pitch, k_step, src_pitch, width, height,
                              cudaMemcpyDeviceToDevice, stream));
  cudaCheck(cudaMemcpy2DAsync(v_.data() + dst_offset, dst_pitch, v_step, src_pitch, width, height,
                              cudaMemcpyDeviceToDevice, stream));
  ++current_len_;
}

template class DeviceKVCache<float>;
template class DeviceKVCache<__half>;

}  // namespace ring_attention
