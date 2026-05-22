/// @file
/// Performance sweep across cpu_attention, launch_naive_attention, and
/// launch_flash_attention. Emits one CSV row per (kernel, config). Skips
/// kernels that are infeasible at a given size (CPU too slow, naive smem cap).
///
/// CSV columns:
///   kernel,batch,heads,seq,head_dim,causal,iters,time_ms,gflops,gbps
///
/// CLI:
///   --bh B H              override batch and heads (default 1 8)
///   --only seq d causal   run only this single (seq, head_dim, causal) config
///                         (useful for Nsight Compute)
///   --kernels list        comma-separated subset of {cpu,naive,flash,flash_fp16}
///                         (default: all)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"

using ring_attention::AttentionShape;
using ring_attention::cpu_attention;
using ring_attention::DeviceTensor;
using ring_attention::launch_flash_attention;
using ring_attention::launch_flash_attention_fp16;
using ring_attention::launch_naive_attention;
using ring_attention::XorShift32;

namespace {

struct BenchResult {
  float ms;
  double gflops;
  double gbps;
};

double flops_per_call(const AttentionShape& s, bool causal) {
  const double bhsqsk = (double)s.batch * s.heads * s.seq_q * s.seq_k * s.head_dim;
  double f = 4.0 * bhsqsk;
  if (causal) f *= 0.5;
  return f;
}

double bytes_per_call(const AttentionShape& s) {
  const double q = (double)s.batch * s.heads * s.seq_q * s.head_dim;
  const double kv = (double)s.batch * s.heads * s.seq_k * s.head_dim;
  return ((q + 2.0 * kv + q) * sizeof(float));
}

using LaunchFn = void (*)(const float*, const float*, const float*, float*, const AttentionShape&,
                          bool, cudaStream_t);

BenchResult bench_gpu(LaunchFn launch, const AttentionShape& s, bool causal, int warmup,
                      int iters) {
  const std::size_t qn = (std::size_t)s.batch * s.heads * s.seq_q * s.head_dim;
  const std::size_t kn = (std::size_t)s.batch * s.heads * s.seq_k * s.head_dim;
  std::vector<float> q(qn), k(kn), v(kn);
  XorShift32 rng(0xBEEFu);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);

  DeviceTensor<float> dq(qn), dk(kn), dv(kn), dout(qn);
  dq.copy_from_host(q);
  dk.copy_from_host(k);
  dv.copy_from_host(v);

  for (int i = 0; i < warmup; ++i)
    launch(dq.data(), dk.data(), dv.data(), dout.data(), s, causal, 0);
  cudaCheck(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
  cudaCheck(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    launch(dq.data(), dk.data(), dv.data(), dout.data(), s, causal, 0);
  cudaCheck(cudaEventRecord(stop));
  cudaCheck(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  cudaCheck(cudaEventElapsedTime(&total_ms, start, stop));
  cudaCheck(cudaEventDestroy(start));
  cudaCheck(cudaEventDestroy(stop));

  const float ms = total_ms / iters;
  const double secs = ms * 1e-3;
  return {ms, flops_per_call(s, causal) / secs / 1e9, bytes_per_call(s) / secs / 1e9};
}

BenchResult bench_cpu(const AttentionShape& s, bool causal, int iters) {
  const std::size_t qn = (std::size_t)s.batch * s.heads * s.seq_q * s.head_dim;
  const std::size_t kn = (std::size_t)s.batch * s.heads * s.seq_k * s.head_dim;
  std::vector<float> q(qn), k(kn), v(kn), o(qn);
  XorShift32 rng(0xBEEFu);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);

  // One untimed warmup (caches).
  cpu_attention(q.data(), k.data(), v.data(), o.data(), s, causal);

  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  for (int i = 0; i < iters; ++i) cpu_attention(q.data(), k.data(), v.data(), o.data(), s, causal);
  const auto t1 = clk::now();
  const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const float ms = static_cast<float>(total_ms / iters);
  const double secs = ms * 1e-3;
  return {ms, flops_per_call(s, causal) / secs / 1e9, bytes_per_call(s) / secs / 1e9};
}

bool kernel_enabled(const std::vector<std::string>& kernels, const char* name) {
  if (kernels.empty()) return true;
  for (auto& k : kernels)
    if (k == name) return true;
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  // Defaults.
  int B = 1, H = 8;
  int only_seq = -1, only_d = -1, only_causal = -1;
  std::vector<std::string> kernels;

  // Parse.
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--bh" && i + 2 < argc) {
      B = std::atoi(argv[i + 1]);
      H = std::atoi(argv[i + 2]);
      i += 2;
    } else if (a == "--only" && i + 3 < argc) {
      only_seq = std::atoi(argv[i + 1]);
      only_d = std::atoi(argv[i + 2]);
      only_causal = std::atoi(argv[i + 3]);
      i += 3;
    } else if (a == "--kernels" && i + 1 < argc) {
      std::string list = argv[i + 1];
      std::size_t p = 0;
      while (p < list.size()) {
        std::size_t q = list.find(',', p);
        if (q == std::string::npos) q = list.size();
        kernels.emplace_back(list.substr(p, q - p));
        p = q + 1;
      }
      ++i;
    }
  }

  const std::vector<int> all_seqs{512, 1024, 2048, 4096, 8192, 16384};
  const std::vector<int> all_dims{32, 64, 128};

  // Caps tuned to keep overnight runtime sane:
  //   - cpu: O(N^2 D) single-thread → cap by total work, not just seq.
  //   - naive: shared-memory budget bounds Sk (~8k on sm_75 default).
  const std::size_t kCpuFlopBudget = 1.5e10;  // ~15 GFLOPS-budget per call
  const int kNaiveSeqCap = 8192;

  const int warmup_gpu = 3;
  const int iters_gpu = 20;
  const int iters_cpu = 1;

  printf("kernel,batch,heads,seq,head_dim,causal,iters,time_ms,gflops,gbps\n");

  for (int d : all_dims) {
    for (int n : all_seqs) {
      for (int causal_flag : {0, 1}) {
        if (only_seq >= 0 && (n != only_seq || d != only_d || causal_flag != only_causal)) continue;
        const bool causal = (causal_flag != 0);
        AttentionShape s{B, H, n, n, d};
        const double work = flops_per_call(s, causal);

        if (kernel_enabled(kernels, "cpu") && work <= kCpuFlopBudget) {
          BenchResult r = bench_cpu(s, causal, iters_cpu);
          printf("cpu,%d,%d,%d,%d,%d,%d,%.4f,%.2f,%.2f\n", B, H, n, d, causal_flag, iters_cpu, r.ms,
                 r.gflops, r.gbps);
          fflush(stdout);
        }
        if (kernel_enabled(kernels, "naive") && n <= kNaiveSeqCap) {
          BenchResult r = bench_gpu(launch_naive_attention, s, causal, warmup_gpu, iters_gpu);
          printf("naive,%d,%d,%d,%d,%d,%d,%.4f,%.2f,%.2f\n", B, H, n, d, causal_flag, iters_gpu,
                 r.ms, r.gflops, r.gbps);
          fflush(stdout);
        }
        if (kernel_enabled(kernels, "flash")) {
          BenchResult r = bench_gpu(launch_flash_attention, s, causal, warmup_gpu, iters_gpu);
          printf("flash,%d,%d,%d,%d,%d,%d,%.4f,%.2f,%.2f\n", B, H, n, d, causal_flag, iters_gpu,
                 r.ms, r.gflops, r.gbps);
          fflush(stdout);
        }
        if (kernel_enabled(kernels, "flash_fp16")) {
          BenchResult r = bench_gpu(launch_flash_attention_fp16, s, causal, warmup_gpu, iters_gpu);
          printf("flash_fp16,%d,%d,%d,%d,%d,%d,%.4f,%.2f,%.2f\n", B, H, n, d, causal_flag,
                 iters_gpu, r.ms, r.gflops, r.gbps);
          fflush(stdout);
        }
      }
    }
  }
  return 0;
}
