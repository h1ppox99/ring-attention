/// @file
/// Tests for run_local_decode_step (single-rank decode against a KV cache).
///
/// Strategy: generate a full (B, H, S, D) Q/K/V, take the last query row as
/// the "decoded token", populate a KV cache with the first S rows of K/V, and
/// compare the decode output against `cpu_attention`'s row S-1.
///
/// fp16 path uses launch_float_to_half to stage Q + cache contents into the
/// __half buffers.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "cpu_attention.hpp"
#include "decode_op.hpp"
#include "device_tensor.hpp"
#include "kv_cache.hpp"

using namespace ring_attention;

namespace {

/// Compare two flat vectors, return 0 on success, 1 on failure.
int compare_close(const std::vector<float>& got, const std::vector<float>& expected,
                  const char* tag, float tol) {
  float max_diff = 0.f;
  int bad = -1;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float d = std::fabs(got[i] - expected[i]);
    if (d > max_diff) max_diff = d;
    if (d > tol && bad < 0) bad = static_cast<int>(i);
  }
  if (bad >= 0) {
    fprintf(stderr, "FAIL %s: idx=%d got=%g expected=%g max_diff=%g tol=%g\n", tag, bad, got[bad],
            expected[bad], max_diff, tol);
    return 1;
  }
  printf("OK %s (max_diff=%g)\n", tag, max_diff);
  return 0;
}

/// fp32 single-decode-token test against the dense cpu_attention oracle.
int test_decode_fp32(int B, int H, int S, int D, bool causal, std::uint32_t seed, const char* tag) {
  XorShift32 rng(seed);
  const std::size_t qn = static_cast<std::size_t>(B) * H * S * D;
  std::vector<float> q_full(qn), k_full(qn), v_full(qn);
  rng.fill_uniform(q_full);
  rng.fill_uniform(k_full);
  rng.fill_uniform(v_full);

  const AttentionShape full_shape{B, H, S, S, D};
  std::vector<float> o_full(qn);
  cpu_attention(q_full.data(), k_full.data(), v_full.data(), o_full.data(), full_shape, causal);

  // Decode token = last row. Cache holds all S rows (including the decoded
  // token's own K/V — exactly what a real decode step would have after append).
  const int decode_pos = S - 1;

  // Pack a (B, H, S, D) source tensor for copy_prefill.
  DeviceTensor<float> k_src(qn), v_src(qn);
  k_src.copy_from_host(k_full);
  v_src.copy_from_host(v_full);

  DeviceKVCache<float> cache(B, /*kv_heads=*/H, /*s_max=*/S, D);
  cache.copy_prefill(k_src.data(), v_src.data(), S);

  // Extract the last query row into a (B, H, 1, D) device buffer.
  std::vector<float> q_h(static_cast<std::size_t>(B) * H * D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d) {
        const std::size_t src = ((static_cast<std::size_t>(b) * H + h) * S + decode_pos) * D + d;
        const std::size_t dst = (static_cast<std::size_t>(b) * H + h) * D + d;
        q_h[dst] = q_full[src];
      }
  DeviceTensor<float> q_d(static_cast<std::size_t>(B) * H * D);
  q_d.copy_from_host(q_h);

  DeviceTensor<float> out_d(static_cast<std::size_t>(B) * H * D);

  run_local_decode_step(q_d.data(), cache, H, decode_pos, /*cache_k_offset=*/0, causal,
                        out_d.data());
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> out_h;
  out_d.copy_to_host(out_h);

  // Expected = cpu_attention's row at decode_pos.
  std::vector<float> expected(static_cast<std::size_t>(B) * H * D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d) {
        const std::size_t src = ((static_cast<std::size_t>(b) * H + h) * S + decode_pos) * D + d;
        const std::size_t dst = (static_cast<std::size_t>(b) * H + h) * D + d;
        expected[dst] = o_full[src];
      }

  return compare_close(out_h, expected, tag, /*tol=*/1e-3f);
}

/// fp16 single-decode-token test. Tolerance is much wider — Tensor-Core fp16
/// matmuls accumulate ~1e-2 round-off vs. the fp32 CPU oracle.
int test_decode_fp16(int B, int H, int S, int D, bool causal, std::uint32_t seed, const char* tag) {
  XorShift32 rng(seed);
  const std::size_t qn = static_cast<std::size_t>(B) * H * S * D;
  std::vector<float> q_full(qn), k_full(qn), v_full(qn);
  rng.fill_uniform(q_full);
  rng.fill_uniform(k_full);
  rng.fill_uniform(v_full);

  const AttentionShape full_shape{B, H, S, S, D};
  std::vector<float> o_full(qn);
  cpu_attention(q_full.data(), k_full.data(), v_full.data(), o_full.data(), full_shape, causal);

  const int decode_pos = S - 1;

  // Stage fp32 K/V into device, cast to fp16.
  DeviceTensor<float> k_src_f32(qn), v_src_f32(qn);
  k_src_f32.copy_from_host(k_full);
  v_src_f32.copy_from_host(v_full);
  DeviceTensor<__half> k_src_h(qn), v_src_h(qn);
  launch_float_to_half(k_src_f32.data(), k_src_h.data(), qn);
  launch_float_to_half(v_src_f32.data(), v_src_h.data(), qn);

  DeviceKVCache<__half> cache(B, /*kv_heads=*/H, /*s_max=*/S, D);
  cache.copy_prefill(k_src_h.data(), v_src_h.data(), S);

  // Extract + cast the last query row.
  std::vector<float> q_h(static_cast<std::size_t>(B) * H * D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d) {
        const std::size_t src = ((static_cast<std::size_t>(b) * H + h) * S + decode_pos) * D + d;
        const std::size_t dst = (static_cast<std::size_t>(b) * H + h) * D + d;
        q_h[dst] = q_full[src];
      }
  DeviceTensor<float> q_f32(static_cast<std::size_t>(B) * H * D);
  q_f32.copy_from_host(q_h);
  DeviceTensor<__half> q_h_dev(static_cast<std::size_t>(B) * H * D);
  launch_float_to_half(q_f32.data(), q_h_dev.data(), q_f32.size());

  DeviceTensor<float> out_d(static_cast<std::size_t>(B) * H * D);

  run_local_decode_step_fp16(q_h_dev.data(), cache, H, decode_pos, /*cache_k_offset=*/0, causal,
                             out_d.data());
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> out_h;
  out_d.copy_to_host(out_h);

  std::vector<float> expected(static_cast<std::size_t>(B) * H * D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < D; ++d) {
        const std::size_t src = ((static_cast<std::size_t>(b) * H + h) * S + decode_pos) * D + d;
        const std::size_t dst = (static_cast<std::size_t>(b) * H + h) * D + d;
        expected[dst] = o_full[src];
      }

  return compare_close(out_h, expected, tag, /*tol=*/5e-2f);
}

}  // namespace

int main() {
  int rc = 0;

  // fp32 sweep: causal/non-causal, multiple head_dim, batched, GQA.
  rc |= test_decode_fp32(1, 1, 32, 32, false, 1u, "fp32 D=32 S=32 non-causal");
  rc |= test_decode_fp32(1, 1, 32, 32, true, 2u, "fp32 D=32 S=32 causal");
  rc |= test_decode_fp32(1, 2, 64, 64, true, 3u, "fp32 D=64 S=64 causal H=2");
  rc |= test_decode_fp32(2, 4, 128, 128, true, 4u, "fp32 D=128 S=128 batched causal");
  rc |= test_decode_fp32(1, 1, 64, 256, true, 5u, "fp32 D=256 S=64 causal");

  // fp16 sweep.
  rc |= test_decode_fp16(1, 1, 32, 32, false, 6u, "fp16 D=32 S=32 non-causal");
  rc |= test_decode_fp16(1, 2, 64, 64, true, 7u, "fp16 D=64 S=64 causal H=2");
  rc |= test_decode_fp16(2, 4, 128, 128, true, 8u, "fp16 D=128 S=128 batched causal");

  if (rc == 0) printf("decode_op OK\n");
  return rc;
}
