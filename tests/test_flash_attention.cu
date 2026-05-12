/// @file
/// Compare the tiled FlashAttention-style kernel against the CPU reference.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"

using ring_attention::AttentionShape;
using ring_attention::cpu_attention;
using ring_attention::DeviceTensor;
using ring_attention::launch_flash_attention;
using ring_attention::XorShift32;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-4f;

bool close(float a, float b) {
  const float diff = std::fabs(a - b);
  const float thresh = kAtol + kRtol * std::max(std::fabs(a), std::fabs(b));
  return diff <= thresh;
}

int compare(const std::vector<float>& gpu, const std::vector<float>& cpu, const char* tag) {
  float max_diff = 0.0f;
  int bad = -1;
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    const float d = std::fabs(gpu[i] - cpu[i]);
    if (d > max_diff) max_diff = d;
    if (!close(gpu[i], cpu[i]) && bad < 0) bad = static_cast<int>(i);
  }
  if (bad >= 0) {
    fprintf(stderr, "FAIL %s: idx=%d gpu=%g cpu=%g (max_diff=%g)\n", tag, bad, gpu[bad], cpu[bad],
            max_diff);
    return 1;
  }
  printf("OK %s (max_diff=%g)\n", tag, max_diff);
  return 0;
}

int run_case(const AttentionShape& s, bool causal, std::uint32_t seed, const char* tag) {
  const std::size_t qn = (std::size_t)s.batch * s.heads * s.seq_q * s.head_dim;
  const std::size_t kn = (std::size_t)s.batch * s.heads * s.seq_k * s.head_dim;

  std::vector<float> q(qn), k(kn), v(kn);
  XorShift32 rng(seed);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);

  std::vector<float> o_cpu(qn, 0.0f);
  cpu_attention(q.data(), k.data(), v.data(), o_cpu.data(), s, causal);

  DeviceTensor<float> dq(qn), dk(kn), dv(kn), dout(qn);
  dq.copy_from_host(q);
  dk.copy_from_host(k);
  dv.copy_from_host(v);

  launch_flash_attention(dq.data(), dk.data(), dv.data(), dout.data(), s, causal);
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> o_gpu;
  dout.copy_to_host(o_gpu);
  return compare(o_gpu, o_cpu, tag);
}

}  // namespace

int main() {
  int rc = 0;

  // head_dim = 64 — first stage.
  rc |= run_case({1, 1, 32, 32, 64}, false, 1u, "d=64 32x32 non-causal");
  rc |= run_case({1, 1, 32, 32, 64}, true, 2u, "d=64 32x32 causal");
  rc |= run_case({2, 4, 128, 128, 64}, false, 3u, "d=64 128x128 non-causal");
  rc |= run_case({2, 4, 128, 128, 64}, true, 4u, "d=64 128x128 causal");

  // Non-multiple of tile dims (Sq=33 doesn't divide BR=32; Sk=47 doesn't divide BC=32).
  rc |= run_case({1, 2, 33, 47, 64}, false, 5u, "d=64 33x47 non-causal (ragged)");
  rc |= run_case({1, 2, 33, 47, 64}, true, 6u, "d=64 33x47 causal (ragged)");

  // Sq != Sk causal alignment (cross-attention shape).
  rc |= run_case({1, 1, 16, 48, 64}, true, 7u, "d=64 Sq<Sk causal");

  // head_dim = 32.
  rc |= run_case({1, 2, 64, 64, 32}, true, 8u, "d=32 64x64 causal");

  // head_dim = 128.
  rc |= run_case({1, 1, 32, 64, 128}, false, 9u, "d=128 32x64 non-causal");
  rc |= run_case({1, 1, 64, 64, 128}, true, 10u, "d=128 64x64 causal");

  if (rc == 0) printf("flash_attention OK\n");
  return rc;
}
