#pragma once

/// @file
/// Reproducible hash-based element generator shared by the ring loop and the CLI driver.

#include <cstdint>

namespace ring_attention {

/// Returns a reproducible float in [-1, 1) for global position (tensor_id, b, h, s, d).
///
/// Each element is an independent hash of its coordinates, so any rank can
/// generate any position in O(1) without iterating past elements it doesn't own.
///
/// @param seed       Per-run seed for reproducibility.
/// @param tensor_id  0 = Q, 1 = K, 2 = V.
/// @param b, h, s, d Tensor coordinates (global sequence index for s).
inline float gen_elem(uint32_t seed, int tensor_id, int b, int h, int s, int d) {
  uint32_t v = seed;
  v ^= static_cast<uint32_t>(tensor_id) * 2654435761u;
  v ^= static_cast<uint32_t>(b) * 2246822519u;
  v ^= static_cast<uint32_t>(h) * 3266489917u;
  v ^= static_cast<uint32_t>(s) * 668265263u;
  v ^= static_cast<uint32_t>(d) * 374761393u;
  v ^= v << 13u;
  v ^= v >> 17u;
  v ^= v << 5u;
  return static_cast<float>(static_cast<int32_t>(v)) * (1.0f / 2147483648.0f);
}

}  // namespace ring_attention
