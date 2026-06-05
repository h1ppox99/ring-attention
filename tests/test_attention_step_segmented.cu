/// @file
/// Segmented (single-launch fine zig-zag) attention-step tests + micro-benchmark.
///
/// Correctness: a single `launch_attention_step_segmented` call over a whole
/// local shard of `n_seg` contiguous sub-blocks must produce *exactly* what the
/// `n_seg^2` affine `launch_attention_step` sub-group loop produces (the kernels
/// process keys in the same order, so the online-softmax accumulation matches).
///
/// Benchmark: isolate the cost of the piecewise-affine segment lookup by timing
/// the affine kernel (striped/contiguous) vs. the segmented kernel on the *same*
/// single-launch work, plus the n_seg^2 affine sub-group loop for reference.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"
#include "ring_partition.hpp"

using ring_attention::AttentionShape;
using ring_attention::DeviceTensor;
using ring_attention::launch_attention_finalize;
using ring_attention::launch_attention_init;
using ring_attention::launch_attention_step;
using ring_attention::launch_attention_step_segmented;
using ring_attention::RingPartition;
using ring_attention::SegMap;
using ring_attention::XorShift32;

namespace {

constexpr float kAtol = 1e-4f;

/// Build the per-rank segment bases for a fine zig-zag of `n_seg` sub-groups.
/// Returns a SegMap whose base[s] = q_offset(s) for this rank, seg_len = S/(n*P).
SegMap make_qmap(const RingPartition& part, int n_seg, int seg_len) {
  SegMap m;
  m.n_seg = n_seg;
  m.seg_len = seg_len;
  for (int s = 0; s < n_seg; ++s) m.base[s] = part.q_offset(s);
  return m;
}

/// Correctness: segmented single launch == affine n_seg^2 sub-group loop.
/// Uses B=H=1 so each sub-group's rows are contiguous in the [seq, D] buffer
/// and can be sliced by a plain pointer offset for the reference path.
int test_equivalence(int cp_size, int n_seg, int seg_len, int D, bool causal, std::uint32_t seed,
                     const char* tag) {
  const int seq = n_seg * seg_len;  // this rank's local shard length
  const int global_seq = n_seg * cp_size * seg_len;
  RingPartition part(cp_size, /*rank=*/0, global_seq, RingPartition::Mode::Zigzag, n_seg);
  const SegMap qmap = make_qmap(part, n_seg, seg_len);
  const SegMap kmap = qmap;  // step 0: this rank attends to its own K/V

  const std::size_t n = (std::size_t)seq * D;
  std::vector<float> q(n), k(n), v(n);
  XorShift32 rng(seed);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);

  DeviceTensor<float> dq(n), dk(n), dv(n);
  dq.copy_from_host(q);
  dk.copy_from_host(k);
  dv.copy_from_host(v);

  const std::size_t m_count = seq;  // B=H=1

  // --- Reference: n_seg^2 affine sub-group calls (the current ring pattern). ---
  DeviceTensor<float> o_ref(n), m_ref(m_count), l_ref(m_count);
  AttentionShape full{1, 1, seq, seq, D};
  launch_attention_init(o_ref.data(), m_ref.data(), l_ref.data(), full, m_count);
  AttentionShape sub{1, 1, seg_len, seg_len, D};
  for (int sgq = 0; sgq < n_seg; ++sgq) {
    for (int sgk = 0; sgk < n_seg; ++sgk) {
      launch_attention_step(
          dq.data() + (std::size_t)sgq * seg_len * D, dk.data() + (std::size_t)sgk * seg_len * D,
          dv.data() + (std::size_t)sgk * seg_len * D, o_ref.data() + (std::size_t)sgq * seg_len * D,
          m_ref.data() + (std::size_t)sgq * seg_len, l_ref.data() + (std::size_t)sgq * seg_len, sub,
          qmap.base[sgq], kmap.base[sgk], causal, /*stream=*/0, /*pos_stride=*/1);
    }
  }
  launch_attention_finalize(o_ref.data(), l_ref.data(), full);

  // --- Segmented: one launch over the whole shard. ---
  DeviceTensor<float> o_seg(n), m_seg(m_count), l_seg(m_count);
  launch_attention_init(o_seg.data(), m_seg.data(), l_seg.data(), full, m_count);
  launch_attention_step_segmented(dq.data(), dk.data(), dv.data(), o_seg.data(), m_seg.data(),
                                  l_seg.data(), full, qmap, kmap, causal);
  launch_attention_finalize(o_seg.data(), l_seg.data(), full);
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> h_ref, h_seg;
  o_ref.copy_to_host(h_ref);
  o_seg.copy_to_host(h_seg);

  float max_diff = 0.0f;
  for (std::size_t i = 0; i < n; ++i) max_diff = std::max(max_diff, std::fabs(h_ref[i] - h_seg[i]));
  if (max_diff > kAtol) {
    fprintf(stderr, "FAIL %s: max_diff=%g > %g\n", tag, max_diff, kAtol);
    return 1;
  }
  printf("OK %s (max_diff=%g)\n", tag, max_diff);
  return 0;
}

float time_ms(const std::function<void()>& body, int iters) {
  cudaEvent_t a, b;
  cudaCheck(cudaEventCreate(&a));
  cudaCheck(cudaEventCreate(&b));
  body();  // warmup
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) body();
  cudaCheck(cudaEventRecord(b));
  cudaCheck(cudaEventSynchronize(b));
  float ms = 0.0f;
  cudaCheck(cudaEventElapsedTime(&ms, a, b));
  cudaCheck(cudaEventDestroy(a));
  cudaCheck(cudaEventDestroy(b));
  return ms / iters;
}

/// Micro-benchmark: one ring step's compute over a full local shard.
///   affine    : 1 launch, S_local x S_local, in-register affine positions (striped).
///   segmented : 1 launch, S_local x S_local, piecewise-affine segment lookup.
///   subloop   : n_seg^2 affine launches of seg_len x seg_len (the zig-zag pattern),
///               skipping fully-masked sub-group pairs.
void benchmark(int cp_size, int n_seg, int seg_len, int H, int D, int iters) {
  const int S = n_seg * seg_len;  // local shard length
  const int global_seq = n_seg * cp_size * seg_len;
  RingPartition part(cp_size, /*rank=*/0, global_seq, RingPartition::Mode::Zigzag, n_seg);
  const SegMap qmap = make_qmap(part, n_seg, seg_len);

  const std::size_t n = (std::size_t)H * S * D;
  std::vector<float> host(n);
  XorShift32 rng(7u);
  rng.fill_uniform(host);

  DeviceTensor<float> dq(n), dk(n), dv(n), dout(n);
  DeviceTensor<float> dm((std::size_t)H * S), dl((std::size_t)H * S);
  dq.copy_from_host(host);
  dk.copy_from_host(host);
  dv.copy_from_host(host);

  AttentionShape full{1, H, S, S, D};
  launch_attention_init(dout.data(), dm.data(), dl.data(), full, (std::size_t)H * S);

  // affine (striped-style, stride = cp_size) — one full-shard launch.
  const float t_affine = time_ms(
      [&] {
        launch_attention_step(dq.data(), dk.data(), dv.data(), dout.data(), dm.data(), dl.data(),
                              full, /*q_offset=*/0, /*k_offset=*/0, /*causal=*/true, /*stream=*/0,
                              /*pos_stride=*/cp_size);
      },
      iters);

  // segmented — one full-shard launch, same work, segment lookup.
  const float t_seg = time_ms(
      [&] {
        launch_attention_step_segmented(dq.data(), dk.data(), dv.data(), dout.data(), dm.data(),
                                        dl.data(), full, qmap, qmap, /*causal=*/true);
      },
      iters);

  // sub-group loop (zig-zag pattern) — n_seg^2 affine launches, masked pairs skipped.
  AttentionShape sub{1, H, seg_len, seg_len, D};
  DeviceTensor<float> qs(n), ks(n), vs(n), os(n);
  DeviceTensor<float> ms((std::size_t)H * S), ls((std::size_t)H * S);
  qs.copy_from_host(host);
  ks.copy_from_host(host);
  vs.copy_from_host(host);
  const float t_sub = time_ms(
      [&] {
        for (int sgq = 0; sgq < n_seg; ++sgq) {
          for (int sgk = 0; sgk < n_seg; ++sgk) {
            const int q_off = qmap.base[sgq];
            const int k_off = qmap.base[sgk];
            if (k_off > q_off + seg_len - 1) continue;  // fully masked under causal
            launch_attention_step(qs.data() + (std::size_t)sgq * H * seg_len * D,
                                  ks.data() + (std::size_t)sgk * H * seg_len * D,
                                  vs.data() + (std::size_t)sgk * H * seg_len * D,
                                  os.data() + (std::size_t)sgq * H * seg_len * D,
                                  ms.data() + (std::size_t)sgq * H * seg_len,
                                  ls.data() + (std::size_t)sgq * H * seg_len, sub, q_off, k_off,
                                  /*causal=*/true, /*stream=*/0, /*pos_stride=*/1);
          }
        }
      },
      iters);
  cudaCheck(cudaDeviceSynchronize());

  printf("\n=== bench  H=%d D=%d  local_seq=%d  (n_seg=%d seg_len=%d, cp=%d)  iters=%d ===\n", H, D,
         S, n_seg, seg_len, cp_size, iters);
  printf("  affine    (striped, 1 launch)      : %8.3f ms\n", t_affine);
  printf("  segmented (1 launch)               : %8.3f ms   (%.2fx vs affine)\n", t_seg,
         t_seg / t_affine);
  printf("  sub-loop  (zig-zag, n_seg^2 launch): %8.3f ms   (%.2fx vs affine)\n", t_sub,
         t_sub / t_affine);
}

}  // namespace

int main() {
  int rc = 0;
  rc |= test_equivalence(4, 2, 64, 64, false, 1u, "n_seg=2 D=64 non-causal");
  rc |= test_equivalence(4, 2, 64, 64, true, 2u, "n_seg=2 D=64 causal");
  rc |= test_equivalence(4, 4, 64, 64, true, 3u, "n_seg=4 D=64 causal");
  rc |= test_equivalence(2, 4, 32, 128, true, 4u, "n_seg=4 D=128 causal");
  rc |= test_equivalence(4, 8, 16, 64, true, 5u, "n_seg=8 D=64 causal");
  if (rc != 0) {
    fprintf(stderr, "segmented equivalence tests FAILED\n");
    return rc;
  }
  printf("attention_step_segmented equivalence OK\n");

  // Representative ring-step shape: cp=4 -> local_seq = 4096, head_dim 128.
  benchmark(/*cp=*/4, /*n_seg=*/4, /*seg_len=*/1024, /*H=*/4, /*D=*/128, /*iters=*/50);
  return 0;
}
