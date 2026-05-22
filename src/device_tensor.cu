/// @file
/// DeviceTensor template instantiations.

#include <cuda_fp16.h>

#include <cassert>
#include <utility>

#include "common.cuh"
#include "device_tensor.hpp"

namespace ring_attention {

template <typename T>
DeviceTensor<T>::DeviceTensor(std::size_t count) : count_(count) {
  if (count_ > 0) {
    cudaCheck(cudaMalloc(&ptr_, count_ * sizeof(T)));
  }
}

template <typename T>
DeviceTensor<T>::~DeviceTensor() {
  if (ptr_ != nullptr) {
    cudaFree(ptr_);
  }
}

template <typename T>
DeviceTensor<T>::DeviceTensor(DeviceTensor&& other) noexcept
    : ptr_(other.ptr_), count_(other.count_) {
  other.ptr_ = nullptr;
  other.count_ = 0;
}

template <typename T>
DeviceTensor<T>& DeviceTensor<T>::operator=(DeviceTensor&& other) noexcept {
  if (this != &other) {
    if (ptr_ != nullptr) cudaFree(ptr_);
    ptr_ = other.ptr_;
    count_ = other.count_;
    other.ptr_ = nullptr;
    other.count_ = 0;
  }
  return *this;
}

template <typename T>
void DeviceTensor<T>::copy_from_host(const T* host) {
  cudaCheck(cudaMemcpy(ptr_, host, count_ * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void DeviceTensor<T>::copy_to_host(T* host) const {
  cudaCheck(cudaMemcpy(host, ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void DeviceTensor<T>::copy_from_host(const std::vector<T>& host) {
  assert(host.size() == count_);
  copy_from_host(host.data());
}

template <typename T>
void DeviceTensor<T>::copy_to_host(std::vector<T>& host) const {
  host.resize(count_);
  copy_to_host(host.data());
}

template class DeviceTensor<float>;
template class DeviceTensor<int>;
template class DeviceTensor<unsigned int>;
template class DeviceTensor<__half>;

}  // namespace ring_attention
