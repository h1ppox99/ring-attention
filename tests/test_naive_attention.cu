/// @file
/// Compare the naive CUDA attention kernel against the CPU reference.

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
using ring_attention::launch_naive_attention;
using ring_attention::XorShift32;

namespace {

constexpr float kAtol = 1e-5f;
constexpr float kRtol = 1e-5f;

bool close(float a, float b) {
  const float diff = std::fabs(a - b);
  const float thresh = kAtol + kRtol * std::max(std::fabs(a), std::fabs(b));
  return diff <= thresh;
}

int max_err_or_fail(const std::vector<float>& gpu, const std::vector<float>& cpu, const char* tag) {
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
  const int kv_H = (s.kv_heads > 0) ? s.kv_heads : s.heads;
  const std::size_t qn = (std::size_t)s.batch * s.heads * s.seq_q * s.head_dim;
  const std::size_t kn = (std::size_t)s.batch * kv_H * s.seq_k * s.head_dim;

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

  launch_naive_attention(dq.data(), dk.data(), dv.data(), dout.data(), s, causal);
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> o_gpu;
  dout.copy_to_host(o_gpu);

  return max_err_or_fail(o_gpu, o_cpu, tag);
}

}  // namespace

int main() {
  int rc = 0;
  rc |= run_case({1, 1, 8, 8, 16}, false, 1u, "1x1x8x8x16 non-causal");
  rc |= run_case({1, 1, 8, 8, 16}, true, 2u, "1x1x8x8x16 causal");
  rc |= run_case({2, 4, 32, 32, 32}, false, 3u, "2x4x32x32x32 non-causal");
  rc |= run_case({2, 4, 32, 32, 32}, true, 4u, "2x4x32x32x32 causal");
  rc |= run_case({1, 2, 16, 48, 24}, false, 5u, "Sq<Sk non-causal");
  rc |= run_case({1, 2, 16, 48, 24}, true, 6u, "Sq<Sk causal-aligned");
  rc |= run_case({1, 1, 64, 64, 64}, true, 7u, "64x64x64 causal");

  // GQA: 8 Q heads, 2 KV heads.
  {
    AttentionShape s{1, 8, 32, 32, 32};
    s.kv_heads = 2;
    rc |= run_case(s, false, 8u, "GQA H=8 kv=2 non-causal");
    rc |= run_case(s, true, 9u, "GQA H=8 kv=2 causal");
  }
  // MQA: 4 Q heads, 1 KV head.
  {
    AttentionShape s{1, 4, 32, 32, 32};
    s.kv_heads = 1;
    rc |= run_case(s, false, 10u, "MQA H=4 kv=1 non-causal");
    rc |= run_case(s, true, 11u, "MQA H=4 kv=1 causal");
  }

  if (rc == 0) printf("naive_attention OK\n");
  return rc;
}
