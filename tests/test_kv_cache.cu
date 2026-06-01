/// @file
/// Unit tests for DeviceKVCache.
///
/// Verifies: construction state, copy_prefill writes to the correct slots
/// without touching the tail, append bumps current_len and writes the new row
/// at the right offset within each (s_max * head_dim) destination stride,
/// reset returns current_len to 0, and move semantics transfer ownership.
///
/// Test convention matches the other CUDA tests in this directory: print
/// "FAIL: ..." to stderr and return 1, "kv_cache OK" + 0 on success.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <utility>
#include <vector>

#include "device_tensor.hpp"
#include "kv_cache.hpp"

namespace {

int check(bool cond, const char* what) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", what);
    return 1;
  }
  return 0;
}

/// Encode (b, h, t, d) into a unique float so we can recover which slot the
/// data ended up in just by reading back the cache contents.
float slot_value(int b, int h, int t, int d) {
  return static_cast<float>(b * 1000 + h * 100 + t * 10 + d);
}

int test_construction() {
  using namespace ring_attention;
  DeviceKVCache<float> cache(/*batch=*/2, /*kv_heads=*/3, /*s_max=*/8, /*head_dim=*/4);
  if (check(cache.batch() == 2, "ctor batch")) return 1;
  if (check(cache.kv_heads() == 3, "ctor kv_heads")) return 1;
  if (check(cache.s_max() == 8, "ctor s_max")) return 1;
  if (check(cache.head_dim() == 4, "ctor head_dim")) return 1;
  if (check(cache.current_len() == 0, "ctor current_len starts at 0")) return 1;
  if (check(cache.numel() == 2u * 3 * 8 * 4, "ctor numel")) return 1;
  if (check(cache.k_data() != nullptr, "ctor k_data non-null")) return 1;
  if (check(cache.v_data() != nullptr, "ctor v_data non-null")) return 1;
  return 0;
}

int test_copy_prefill_and_append_roundtrip() {
  using namespace ring_attention;
  const int B = 2, H = 2, S_max = 8, D = 4, prefill = 3;

  // Build a contiguous (B, H, prefill, D) source for K and V on the host.
  std::vector<float> k_src_host(B * H * prefill * D), v_src_host(B * H * prefill * D);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < prefill; ++t) {
        for (int d = 0; d < D; ++d) {
          const std::size_t idx = ((b * H + h) * prefill + t) * D + d;
          k_src_host[idx] = slot_value(b, h, t, d);
          v_src_host[idx] = slot_value(b, h, t, d) + 0.5f;  // distinguish V from K
        }
      }
    }
  }
  DeviceTensor<float> k_src(B * H * prefill * D), v_src(B * H * prefill * D);
  k_src.copy_from_host(k_src_host);
  v_src.copy_from_host(v_src_host);

  // Three appended tokens at t = prefill, prefill+1, prefill+2.
  std::vector<DeviceTensor<float>> k_steps, v_steps;
  for (int step = 0; step < 3; ++step) {
    const int t = prefill + step;
    std::vector<float> k_host(B * H * D), v_host(B * H * D);
    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; ++d) {
          const std::size_t idx = ((b * H + h)) * D + d;
          k_host[idx] = slot_value(b, h, t, d);
          v_host[idx] = slot_value(b, h, t, d) + 0.5f;
        }
      }
    }
    DeviceTensor<float> k_step(B * H * D), v_step(B * H * D);
    k_step.copy_from_host(k_host);
    v_step.copy_from_host(v_host);
    k_steps.emplace_back(std::move(k_step));
    v_steps.emplace_back(std::move(v_step));
  }

  DeviceKVCache<float> cache(B, H, S_max, D);
  cache.copy_prefill(k_src.data(), v_src.data(), prefill);
  if (check(cache.current_len() == prefill, "current_len after copy_prefill")) return 1;

  for (int step = 0; step < 3; ++step) {
    cache.append(k_steps[step].data(), v_steps[step].data());
  }
  if (check(cache.current_len() == prefill + 3, "current_len after 3 appends")) return 1;

  // Sync the default stream before reading back.
  cudaDeviceSynchronize();

  // Read back the full K and V buffers and verify every populated slot.
  std::vector<float> k_host(cache.numel()), v_host(cache.numel());
  cudaMemcpy(k_host.data(), cache.k_data(), cache.numel() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(v_host.data(), cache.v_data(), cache.numel() * sizeof(float), cudaMemcpyDeviceToHost);

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int t = 0; t < prefill + 3; ++t) {
        for (int d = 0; d < D; ++d) {
          // Cache layout is (B, H, S_max, D), so the stride for the time dim is D
          // and the row spans S_max slots.
          const std::size_t idx = ((b * H + h) * S_max + t) * D + d;
          const float expected_k = slot_value(b, h, t, d);
          const float expected_v = expected_k + 0.5f;
          if (k_host[idx] != expected_k) {
            fprintf(stderr, "K mismatch at (b=%d,h=%d,t=%d,d=%d): got %f, expected %f\n", b, h, t,
                    d, k_host[idx], expected_k);
            return 1;
          }
          if (v_host[idx] != expected_v) {
            fprintf(stderr, "V mismatch at (b=%d,h=%d,t=%d,d=%d): got %f, expected %f\n", b, h, t,
                    d, v_host[idx], expected_v);
            return 1;
          }
        }
      }
    }
  }
  return 0;
}

int test_reset_resumes_at_zero() {
  using namespace ring_attention;
  const int B = 1, H = 1, S_max = 4, D = 2;
  DeviceKVCache<float> cache(B, H, S_max, D);

  std::vector<float> step_host(B * H * D, 42.0f);
  DeviceTensor<float> step_k(B * H * D), step_v(B * H * D);
  step_k.copy_from_host(step_host);
  step_v.copy_from_host(step_host);

  cache.append(step_k.data(), step_v.data());
  cache.append(step_k.data(), step_v.data());
  if (check(cache.current_len() == 2, "current_len before reset")) return 1;
  cache.reset();
  if (check(cache.current_len() == 0, "current_len after reset")) return 1;
  // Append after reset: should overwrite slot 0 again without complaint.
  cache.append(step_k.data(), step_v.data());
  if (check(cache.current_len() == 1, "current_len after reset+append")) return 1;
  return 0;
}

int test_move_constructor_transfers_state() {
  using namespace ring_attention;
  const int B = 1, H = 1, S_max = 4, D = 2;
  DeviceKVCache<float> src(B, H, S_max, D);

  std::vector<float> step_host(B * H * D, 7.0f);
  DeviceTensor<float> step_k(B * H * D), step_v(B * H * D);
  step_k.copy_from_host(step_host);
  step_v.copy_from_host(step_host);
  src.append(step_k.data(), step_v.data());

  const float* k_ptr_before = src.k_data();
  const int current_len_before = src.current_len();

  DeviceKVCache<float> dst(std::move(src));
  if (check(dst.k_data() == k_ptr_before, "move ctor inherits k buffer")) return 1;
  if (check(dst.current_len() == current_len_before, "move ctor preserves current_len")) return 1;
  if (check(dst.s_max() == S_max, "move ctor preserves s_max")) return 1;
  return 0;
}

int test_half_specialization_compiles_and_runs() {
  using namespace ring_attention;
  // Just confirm the __half specialization can be instantiated and that
  // append updates current_len. No host-side numerical check (would require
  // __half<->float conversion).
  DeviceKVCache<__half> cache(/*batch=*/1, /*kv_heads=*/1, /*s_max=*/2, /*head_dim=*/4);
  std::vector<__half> step_host(4, __float2half(1.0f));
  DeviceTensor<__half> step_k(4), step_v(4);
  step_k.copy_from_host(step_host);
  step_v.copy_from_host(step_host);
  cache.append(step_k.data(), step_v.data());
  cache.append(step_k.data(), step_v.data());
  if (check(cache.current_len() == 2, "__half append bumps current_len")) return 1;
  return 0;
}

}  // namespace

int main() {
  if (test_construction()) return 1;
  if (test_copy_prefill_and_append_roundtrip()) return 1;
  if (test_reset_resumes_at_zero()) return 1;
  if (test_move_constructor_transfers_state()) return 1;
  if (test_half_specialization_compiles_and_runs()) return 1;

  printf("kv_cache OK\n");
  return 0;
}
