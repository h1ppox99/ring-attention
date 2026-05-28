#pragma once

/// @file
/// Per-rank distributed K/V cache for ring-attention decode.
///
/// Owns contiguous device buffers of shape `(batch, kv_heads, s_max, head_dim)`
/// for K and V. `copy_prefill` bulk-loads the prefill K/V at the front of the
/// cache; `append` writes one token's K/V row at slot `current_len` and bumps
/// the counter.
///
/// One cache lives on each MPI rank; cross-rank K/V rotation is handled by the
/// existing ring loop (NCCL or MPI), which reads from `k_data()` / `v_data()`
/// after `current_len()` rows have been populated.

#include <cstddef>

#include "device_tensor.hpp"

namespace ring_attention {

/// Contiguous growing K/V cache. Non-copyable, movable.
template <typename T>
class DeviceKVCache {
 public:
  DeviceKVCache(int batch, int kv_heads, int s_max, int head_dim);

  DeviceKVCache(const DeviceKVCache&) = delete;
  DeviceKVCache& operator=(const DeviceKVCache&) = delete;
  DeviceKVCache(DeviceKVCache&&) noexcept = default;
  DeviceKVCache& operator=(DeviceKVCache&&) noexcept = default;

  /// Number of populated KV rows (advances on append).
  int current_len() const noexcept { return current_len_; }
  int s_max() const noexcept { return s_max_; }
  int batch() const noexcept { return batch_; }
  int kv_heads() const noexcept { return kv_heads_; }
  int head_dim() const noexcept { return head_dim_; }

  /// Per-tensor element count, i.e. `batch * kv_heads * s_max * head_dim`.
  std::size_t numel() const noexcept {
    return static_cast<std::size_t>(batch_) * kv_heads_ * s_max_ * head_dim_;
  }

  /// Raw device pointer to K. Same row-major layout as the rest of the
  /// codebase: `(batch, kv_heads, s_max, head_dim)`.
  T* k_data() noexcept { return k_.data(); }
  T* v_data() noexcept { return v_.data(); }
  const T* k_data() const noexcept { return k_.data(); }
  const T* v_data() const noexcept { return v_.data(); }

  /// Bulk-load the prefill K/V into rows `[0, prefill_len)` of the cache and
  /// set `current_len = prefill_len`. Source layout:
  /// `(batch, kv_heads, prefill_len, head_dim)`. Aborts if
  /// `prefill_len > s_max`.
  void copy_prefill(const T* k_src, const T* v_src, int prefill_len, cudaStream_t stream = 0);

  /// Append one token's K/V row at slot `current_len_` and bump the counter.
  /// `k_step` and `v_step` layout: `(batch, kv_heads, 1, head_dim)`. Aborts
  /// if the cache is full.
  void append(const T* k_step, const T* v_step, cudaStream_t stream = 0);

  /// Reset to empty. Does not free or clear device memory.
  void reset() noexcept { current_len_ = 0; }

 private:
  DeviceTensor<T> k_;
  DeviceTensor<T> v_;
  int batch_;
  int kv_heads_;
  int s_max_;
  int head_dim_;
  int current_len_ = 0;
};

}  // namespace ring_attention
