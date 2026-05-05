#pragma once

/// @file
/// Common CUDA utilities: error checking macro and dtype aliases.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ring_attention {

/// Check a CUDA runtime call; abort with diagnostic on failure.
#define cudaCheck(expr)                                                                       \
  do {                                                                                        \
    cudaError_t _err = (expr);                                                                \
    if (_err != cudaSuccess) {                                                                \
      fprintf(stderr, "CUDA error %s at %s:%d in '%s'\n", cudaGetErrorString(_err), __FILE__, \
              __LINE__, #expr);                                                               \
      std::abort();                                                                           \
    }                                                                                         \
  } while (0)

/// Integer ceiling division.
__host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace ring_attention
