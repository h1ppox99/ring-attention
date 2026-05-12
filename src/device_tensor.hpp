#pragma once

/// @file
/// RAII device-memory buffer with simple host<->device transfer helpers.

#include <cstddef>
#include <vector>

namespace ring_attention {

/// Owning device buffer for POD types. Non-copyable, movable.
template <typename T>
class DeviceTensor {
 public:
  DeviceTensor() = default;
  explicit DeviceTensor(std::size_t count);
  ~DeviceTensor();

  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  DeviceTensor(DeviceTensor&& other) noexcept;
  DeviceTensor& operator=(DeviceTensor&& other) noexcept;

  /// Number of elements.
  std::size_t size() const noexcept { return count_; }
  /// Size in bytes.
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }
  /// Raw device pointer.
  T* data() noexcept { return ptr_; }
  const T* data() const noexcept { return ptr_; }

  /// Copy `count_` elements from host pointer to device.
  void copy_from_host(const T* host);
  /// Copy `count_` elements from device to host pointer.
  void copy_to_host(T* host) const;

  /// Convenience: copy from std::vector (must be same size).
  void copy_from_host(const std::vector<T>& host);
  /// Convenience: copy to std::vector (resized to count_).
  void copy_to_host(std::vector<T>& host) const;

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace ring_attention
