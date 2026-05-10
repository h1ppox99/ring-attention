/// @file
/// Single-GPU simulation of the ring: split K/V into P chunks, loop the
/// online-softmax step over each chunk, finalize, and check that the result
/// matches a full cpu_attention call on the unsplit tensors. This is the
/// milestone-4 dry run.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"

using ring_attention::AttentionShape;
using ring_attention::cpu_attention;
using ring_attention::DeviceTensor;
using ring_attention::launch_attention_finalize;
using ring_attention::launch_attention_init;
using ring_attention::launch_attention_step;
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

/// Run the ring simulation on one GPU. Q is the full queries; K, V get split
/// into `num_chunks` equal pieces; the step kernel is called once per chunk
/// with the appropriate `k_offset`. Returns 0 on success.
int run_ring(const AttentionShape& full, int num_chunks, bool causal, std::uint32_t seed,
             const char* tag) {
  if (full.seq_k % num_chunks != 0) {
    fprintf(stderr, "BAD TEST SETUP %s: seq_k=%d not divisible by num_chunks=%d\n", tag, full.seq_k,
            num_chunks);
    return 1;
  }
  const int chunk_k = full.seq_k / num_chunks;

  const std::size_t qn = (std::size_t)full.batch * full.heads * full.seq_q * full.head_dim;
  const std::size_t kn_total = (std::size_t)full.batch * full.heads * full.seq_k * full.head_dim;
  const std::size_t kn_chunk = (std::size_t)full.batch * full.heads * chunk_k * full.head_dim;
  const std::size_t m_count = (std::size_t)full.batch * full.heads * full.seq_q;

  std::vector<float> q(qn), k(kn_total), v(kn_total);
  XorShift32 rng(seed);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);

  // CPU oracle on the full unsplit problem.
  std::vector<float> o_cpu(qn, 0.0f);
  cpu_attention(q.data(), k.data(), v.data(), o_cpu.data(), full, causal);

  // Device buffers.
  DeviceTensor<float> dq(qn);
  DeviceTensor<float> dk_chunk(kn_chunk);
  DeviceTensor<float> dv_chunk(kn_chunk);
  DeviceTensor<float> dout(qn);
  DeviceTensor<float> dm(m_count);
  DeviceTensor<float> dl(m_count);
  dq.copy_from_host(q);

  // K/V live as one big device buffer that we slice into chunks per call.
  // (In milestone 4 these chunks come from MPI; here we just re-upload contiguous
  // slices from the host buffer.)
  AttentionShape chunk_shape = full;
  chunk_shape.seq_k = chunk_k;

  launch_attention_init(dout.data(), dm.data(), dl.data(), full, m_count);

  // Per-head, per-batch stride into K/V (full sequence dim).
  // The Kp slice for chunk p covers seq positions [p*chunk_k, (p+1)*chunk_k)
  // *interleaved* across (batch, heads). Rather than gather on the host, we
  // build a small contiguous chunk buffer per call.
  std::vector<float> k_chunk_host(kn_chunk);
  std::vector<float> v_chunk_host(kn_chunk);

  for (int p = 0; p < num_chunks; ++p) {
    const int k_off = p * chunk_k;
    // Pack chunk: for each (b, h), copy chunk_k rows starting at offset k_off.
    for (int b = 0; b < full.batch; ++b) {
      for (int h = 0; h < full.heads; ++h) {
        const std::size_t src_head =
            (((std::size_t)b * full.heads) + h) * full.seq_k * full.head_dim;
        const std::size_t dst_head = (((std::size_t)b * full.heads) + h) * chunk_k * full.head_dim;
        const std::size_t src_off = src_head + (std::size_t)k_off * full.head_dim;
        const std::size_t bytes = (std::size_t)chunk_k * full.head_dim * sizeof(float);
        std::memcpy(k_chunk_host.data() + dst_head, k.data() + src_off, bytes);
        std::memcpy(v_chunk_host.data() + dst_head, v.data() + src_off, bytes);
      }
    }
    dk_chunk.copy_from_host(k_chunk_host);
    dv_chunk.copy_from_host(v_chunk_host);

    launch_attention_step(dq.data(), dk_chunk.data(), dv_chunk.data(), dout.data(), dm.data(),
                          dl.data(), chunk_shape, /*q_offset*/ 0, /*k_offset*/ k_off, causal);
  }

  launch_attention_finalize(dout.data(), dl.data(), full, m_count);
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> o_gpu;
  dout.copy_to_host(o_gpu);
  return compare(o_gpu, o_cpu, tag);
}

}  // namespace

int main() {
  int rc = 0;

  // Non-causal: order of chunks should not matter for correctness.
  rc |= run_ring({1, 1, 32, 64, 64}, /*chunks*/ 2, false, 1u, "d=64 Sq=32 Sk=64 P=2 non-causal");
  rc |= run_ring({1, 1, 32, 64, 64}, 4, false, 2u, "d=64 Sq=32 Sk=64 P=4 non-causal");
  rc |= run_ring({2, 4, 32, 128, 64}, 4, false, 3u, "d=64 batched P=4 non-causal");

  // Causal Sq == Sk: chunks correspond to contiguous key ranges.
  rc |= run_ring({1, 1, 64, 64, 64}, 2, true, 4u, "d=64 Sq=Sk=64 P=2 causal");
  rc |= run_ring({1, 1, 64, 64, 64}, 4, true, 5u, "d=64 Sq=Sk=64 P=4 causal");
  rc |= run_ring({2, 2, 128, 128, 64}, 4, true, 6u, "d=64 batched Sq=Sk=128 P=4 causal");

  // Different head dims (Sq == Sk so end-aligned and natural-causal coincide).
  rc |= run_ring({1, 1, 64, 64, 32}, 4, true, 7u, "d=32 Sq=Sk=64 P=4 causal");
  rc |= run_ring({1, 1, 64, 64, 128}, 2, true, 8u, "d=128 Sq=Sk=64 P=2 causal");

  // Single chunk == should match a flat flash call.
  rc |= run_ring({1, 2, 32, 64, 64}, 1, false, 9u, "P=1 sanity non-causal");
  rc |= run_ring({1, 2, 64, 64, 64}, 1, true, 10u, "P=1 sanity causal");

  if (rc == 0) printf("attention_step OK\n");
  return rc;
}
