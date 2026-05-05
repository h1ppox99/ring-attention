/// @file
/// Smoke test: device buffer round-trip + trivial kernel launch.

#include <cstdio>
#include <vector>

#include "common.cuh"
#include "device_tensor.hpp"

namespace {

__global__ void scale_kernel(float* x, float factor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= factor;
}

}  // namespace

int main() {
  using namespace ring_attention;

  constexpr int N = 1024;
  std::vector<float> host(N);
  for (int i = 0; i < N; ++i) host[i] = static_cast<float>(i);

  DeviceTensor<float> d(N);
  d.copy_from_host(host);

  constexpr int kBlock = 128;
  scale_kernel<<<ceil_div(N, kBlock), kBlock>>>(d.data(), 2.0f, N);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> result;
  d.copy_to_host(result);

  for (int i = 0; i < N; ++i) {
    float expected = 2.0f * static_cast<float>(i);
    if (result[i] != expected) {
      fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, result[i], expected);
      return 1;
    }
  }
  printf("smoke OK (N=%d)\n", N);
  return 0;
}
