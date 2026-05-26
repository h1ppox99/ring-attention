/// @file
/// Tests for TODO 3 (attn_bias) and TODO 4 (segment_ids) on the FP32 ring step.
///
/// Strategy: simulate the ring on a single GPU by splitting K/V into chunks,
/// then compare the GPU result against a CPU oracle that applies the same
/// bias / segment masking.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"

using ring_attention::AttentionShape;
using ring_attention::DeviceTensor;
using ring_attention::launch_attention_finalize;
using ring_attention::launch_attention_init;
using ring_attention::launch_attention_step;
using ring_attention::XorShift32;

namespace {

constexpr float kAtol = 1e-4f;
constexpr float kRtol = 1e-4f;

bool allclose(float a, float b) {
  return std::fabs(a - b) <= kAtol + kRtol * std::max(std::fabs(a), std::fabs(b));
}

int compare(const std::vector<float>& gpu, const std::vector<float>& cpu, const char* tag) {
  float max_diff = 0.0f;
  int bad = -1;
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    const float d = std::fabs(gpu[i] - cpu[i]);
    if (d > max_diff) max_diff = d;
    if (!allclose(gpu[i], cpu[i]) && bad < 0) bad = static_cast<int>(i);
  }
  if (bad >= 0) {
    fprintf(stderr, "FAIL %s: idx=%d gpu=%g cpu=%g (max_diff=%g)\n", tag, bad, gpu[bad], cpu[bad],
            max_diff);
    return 1;
  }
  printf("OK %s (max_diff=%g)\n", tag, max_diff);
  return 0;
}

/// CPU oracle: scaled-dot-product attention with an additive bias.
/// bias shape: [B, H, Sq, Sk] (full sequence K dimension).
void cpu_attention_biased(const float* q, const float* k, const float* v, const float* bias,
                          float* out, const AttentionShape& shape, bool causal) {
  const int B = shape.batch, H = shape.heads, Sq = shape.seq_q, Sk = shape.seq_k,
            D = shape.head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int causal_shift = Sk - Sq;
  std::vector<float> scores(static_cast<std::size_t>(Sk));

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const float* qh = q + (static_cast<std::size_t>(b * H + h)) * Sq * D;
      const float* kh = k + (static_cast<std::size_t>(b * H + h)) * Sk * D;
      const float* vh = v + (static_cast<std::size_t>(b * H + h)) * Sk * D;
      const float* bh = bias + (static_cast<std::size_t>(b * H + h)) * Sq * Sk;
      float* oh = out + (static_cast<std::size_t>(b * H + h)) * Sq * D;

      for (int i = 0; i < Sq; ++i) {
        const float* qi = qh + static_cast<std::size_t>(i) * D;
        const int j_max = causal ? std::min(Sk, i + causal_shift + 1) : Sk;

        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < j_max; ++j) {
          const float* kj = kh + static_cast<std::size_t>(j) * D;
          float s = 0.0f;
          for (int d = 0; d < D; ++d) s += qi[d] * kj[d];
          s = s * scale + bh[static_cast<std::size_t>(i) * Sk + j];
          scores[j] = s;
          if (s > row_max) row_max = s;
        }

        float denom = 0.0f;
        for (int j = 0; j < j_max; ++j) {
          scores[j] = std::exp(scores[j] - row_max);
          denom += scores[j];
        }
        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

        float* oi = oh + static_cast<std::size_t>(i) * D;
        for (int d = 0; d < D; ++d) oi[d] = 0.0f;
        for (int j = 0; j < j_max; ++j) {
          const float p = scores[j] * inv;
          const float* vj = vh + static_cast<std::size_t>(j) * D;
          for (int d = 0; d < D; ++d) oi[d] += p * vj[d];
        }
      }
    }
  }
}

/// CPU oracle: attention with segment masking. Tokens from different segments
/// cannot attend to each other (score forced to -inf before softmax).
/// segment_ids shape: [B, S_global]. seq_global == Sk for a full-sequence call.
void cpu_attention_segmented(const float* q, const float* k, const float* v, const int* segment_ids,
                             int seq_global, float* out, const AttentionShape& shape, bool causal) {
  const int B = shape.batch, H = shape.heads, Sq = shape.seq_q, Sk = shape.seq_k,
            D = shape.head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int causal_shift = Sk - Sq;
  std::vector<float> scores(static_cast<std::size_t>(Sk));

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const float* qh = q + (static_cast<std::size_t>(b * H + h)) * Sq * D;
      const float* kh = k + (static_cast<std::size_t>(b * H + h)) * Sk * D;
      const float* vh = v + (static_cast<std::size_t>(b * H + h)) * Sk * D;
      float* oh = out + (static_cast<std::size_t>(b * H + h)) * Sq * D;
      const int* seg_b = segment_ids + b * seq_global;

      for (int i = 0; i < Sq; ++i) {
        const float* qi = qh + static_cast<std::size_t>(i) * D;
        const int j_max = causal ? std::min(Sk, i + causal_shift + 1) : Sk;
        // Global query position = i (q_offset == 0 in these tests).
        const int qi_seg = seg_b[i];

        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < j_max; ++j) {
          const float* kj = kh + static_cast<std::size_t>(j) * D;
          float s;
          if (seg_b[j] != qi_seg) {
            s = -std::numeric_limits<float>::infinity();
          } else {
            s = 0.0f;
            for (int d = 0; d < D; ++d) s += qi[d] * kj[d];
            s *= scale;
          }
          scores[j] = s;
          if (s > row_max) row_max = s;
        }

        // If all scores are -inf the row output is zero (no valid key).
        if (row_max == -std::numeric_limits<float>::infinity()) {
          float* oi = oh + static_cast<std::size_t>(i) * D;
          for (int d = 0; d < D; ++d) oi[d] = 0.0f;
          continue;
        }

        float denom = 0.0f;
        for (int j = 0; j < j_max; ++j) {
          scores[j] = (scores[j] == -std::numeric_limits<float>::infinity())
                          ? 0.0f
                          : std::exp(scores[j] - row_max);
          denom += scores[j];
        }
        const float inv = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

        float* oi = oh + static_cast<std::size_t>(i) * D;
        for (int d = 0; d < D; ++d) oi[d] = 0.0f;
        for (int j = 0; j < j_max; ++j) {
          const float p = scores[j] * inv;
          const float* vj = vh + static_cast<std::size_t>(j) * D;
          for (int d = 0; d < D; ++d) oi[d] += p * vj[d];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Ring simulation helpers (pack a contiguous chunk from a full host tensor)
// ---------------------------------------------------------------------------

void pack_kv_chunk(const std::vector<float>& full, std::vector<float>& chunk, int B, int H,
                   int seq_full, int chunk_k, int k_off, int D) {
  const std::size_t chunk_elem = static_cast<std::size_t>(B) * H * chunk_k * D;
  chunk.resize(chunk_elem);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const std::size_t src = (static_cast<std::size_t>(b * H + h) * seq_full + k_off) * D;
      const std::size_t dst = static_cast<std::size_t>(b * H + h) * chunk_k * D;
      std::memcpy(chunk.data() + dst, full.data() + src,
                  static_cast<std::size_t>(chunk_k) * D * sizeof(float));
    }
}

/// Pack a bias chunk: full bias is [B, H, Sq, S_k_total]; chunk is [B, H, Sq, chunk_k].
void pack_bias_chunk(const std::vector<float>& full_bias, std::vector<float>& bias_chunk, int B,
                     int H, int Sq, int S_k_total, int chunk_k, int k_off) {
  const std::size_t chunk_elem = static_cast<std::size_t>(B) * H * Sq * chunk_k;
  bias_chunk.resize(chunk_elem);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < Sq; ++i) {
        const std::size_t src = (static_cast<std::size_t>(b * H + h) * Sq + i) * S_k_total + k_off;
        const std::size_t dst = (static_cast<std::size_t>(b * H + h) * Sq + i) * chunk_k;
        std::memcpy(bias_chunk.data() + dst, full_bias.data() + src,
                    static_cast<std::size_t>(chunk_k) * sizeof(float));
      }
}

// ---------------------------------------------------------------------------
// TODO 3: attention bias
// ---------------------------------------------------------------------------

/// Ring simulation with attn_bias. The full bias tensor [B, H, Sq, Sk] is
/// chunked along the Sk dimension to match how K/V are chunked.
int run_ring_with_bias(const AttentionShape& full, int num_chunks, bool causal, std::uint32_t seed,
                       const char* tag) {
  if (full.seq_k % num_chunks != 0) {
    fprintf(stderr, "BAD TEST SETUP %s: seq_k=%d not divisible by num_chunks=%d\n", tag, full.seq_k,
            num_chunks);
    return 1;
  }
  const int B = full.batch, H = full.heads, Sq = full.seq_q, Sk_total = full.seq_k;
  const int D = full.head_dim;
  const int chunk_k = Sk_total / num_chunks;

  const std::size_t qn = static_cast<std::size_t>(B) * H * Sq * D;
  const std::size_t kn_total = static_cast<std::size_t>(B) * H * Sk_total * D;
  const std::size_t kn_chunk = static_cast<std::size_t>(B) * H * chunk_k * D;
  const std::size_t m_count = static_cast<std::size_t>(B) * H * Sq;
  const std::size_t bias_total = static_cast<std::size_t>(B) * H * Sq * Sk_total;
  const std::size_t bias_chunk_n = static_cast<std::size_t>(B) * H * Sq * chunk_k;

  std::vector<float> q(qn), k(kn_total), v(kn_total), bias_full(bias_total);
  XorShift32 rng(seed);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);
  // Bias values: small random values so they shift softmax meaningfully.
  for (auto& x : bias_full) x = rng.next_uniform() * 0.5f;

  // CPU oracle.
  std::vector<float> o_cpu(qn, 0.0f);
  cpu_attention_biased(q.data(), k.data(), v.data(), bias_full.data(), o_cpu.data(), full, causal);

  // GPU ring simulation.
  DeviceTensor<float> dq(qn), dk(kn_chunk), dv(kn_chunk), dbias(bias_chunk_n);
  DeviceTensor<float> dout(qn), dm(m_count), dl(m_count);
  dq.copy_from_host(q);

  AttentionShape chunk_shape = full;
  chunk_shape.seq_k = chunk_k;

  launch_attention_init(dout.data(), dm.data(), dl.data(), full, m_count);

  std::vector<float> k_chunk_h, v_chunk_h, bias_chunk_h;
  for (int p = 0; p < num_chunks; ++p) {
    const int k_off = p * chunk_k;
    pack_kv_chunk(k, k_chunk_h, B, H, Sk_total, chunk_k, k_off, D);
    pack_kv_chunk(v, v_chunk_h, B, H, Sk_total, chunk_k, k_off, D);
    pack_bias_chunk(bias_full, bias_chunk_h, B, H, Sq, Sk_total, chunk_k, k_off);
    dk.copy_from_host(k_chunk_h);
    dv.copy_from_host(v_chunk_h);
    dbias.copy_from_host(bias_chunk_h);

    launch_attention_step(dq.data(), dk.data(), dv.data(), dout.data(), dm.data(), dl.data(),
                          chunk_shape, /*q_offset=*/0, k_off, causal,
                          /*stream=*/0, dbias.data());
  }

  launch_attention_finalize(dout.data(), dl.data(), full);
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> o_gpu;
  dout.copy_to_host(o_gpu);
  return compare(o_gpu, o_cpu, tag);
}

// ---------------------------------------------------------------------------
// TODO 4: segment IDs
// ---------------------------------------------------------------------------

/// Ring simulation with segment_ids. Tokens are split into two segments:
/// positions [0, S/2) → segment 0, positions [S/2, S) → segment 1.
int run_ring_with_segments(const AttentionShape& full, int num_chunks, bool causal,
                           std::uint32_t seed, const char* tag) {
  if (full.seq_k % num_chunks != 0) {
    fprintf(stderr, "BAD TEST SETUP %s: seq_k=%d not divisible by num_chunks=%d\n", tag, full.seq_k,
            num_chunks);
    return 1;
  }
  const int B = full.batch, H = full.heads, Sq = full.seq_q, Sk_total = full.seq_k;
  const int D = full.head_dim;
  const int chunk_k = Sk_total / num_chunks;
  // seq_global: the larger of Sq and Sk_total (covers all global positions).
  const int S_global = std::max(Sq, Sk_total);

  const std::size_t qn = static_cast<std::size_t>(B) * H * Sq * D;
  const std::size_t kn_total = static_cast<std::size_t>(B) * H * Sk_total * D;
  const std::size_t kn_chunk = static_cast<std::size_t>(B) * H * chunk_k * D;
  const std::size_t m_count = static_cast<std::size_t>(B) * H * Sq;

  std::vector<float> q(qn), k(kn_total), v(kn_total);
  XorShift32 rng(seed);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);

  // Segment labels: first half in seg 0, second half in seg 1.
  // For a square (Sq == Sk_total == S_global) this gives the natural split.
  std::vector<int> seg_ids_h(static_cast<std::size_t>(B) * S_global);
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < S_global; ++s) seg_ids_h[b * S_global + s] = (s < S_global / 2) ? 0 : 1;

  // CPU oracle.
  std::vector<float> o_cpu(qn, 0.0f);
  cpu_attention_segmented(q.data(), k.data(), v.data(), seg_ids_h.data(), S_global, o_cpu.data(),
                          full, causal);

  // GPU ring simulation.
  DeviceTensor<float> dq(qn), dk(kn_chunk), dv(kn_chunk);
  DeviceTensor<float> dout(qn), dm(m_count), dl(m_count);
  // Segment IDs on device: shape [B, S_global].
  DeviceTensor<int> dseg(static_cast<std::size_t>(B) * S_global);
  dq.copy_from_host(q);
  dseg.copy_from_host(seg_ids_h);

  AttentionShape chunk_shape = full;
  chunk_shape.seq_k = chunk_k;

  launch_attention_init(dout.data(), dm.data(), dl.data(), full, m_count);

  std::vector<float> k_chunk_h, v_chunk_h;
  for (int p = 0; p < num_chunks; ++p) {
    const int k_off = p * chunk_k;
    pack_kv_chunk(k, k_chunk_h, B, H, Sk_total, chunk_k, k_off, D);
    pack_kv_chunk(v, v_chunk_h, B, H, Sk_total, chunk_k, k_off, D);
    dk.copy_from_host(k_chunk_h);
    dv.copy_from_host(v_chunk_h);

    launch_attention_step(dq.data(), dk.data(), dv.data(), dout.data(), dm.data(), dl.data(),
                          chunk_shape, /*q_offset=*/0, k_off, causal,
                          /*stream=*/0, /*attn_bias=*/nullptr, dseg.data(), S_global);
  }

  launch_attention_finalize(dout.data(), dl.data(), full);
  cudaCheck(cudaDeviceSynchronize());

  std::vector<float> o_gpu;
  dout.copy_to_host(o_gpu);
  return compare(o_gpu, o_cpu, tag);
}

}  // namespace

int main() {
  int rc = 0;

  // --- TODO 3: attn_bias ---
  // Non-causal, single chunk (simplest sanity check).
  rc |= run_ring_with_bias({1, 1, 32, 32, 64}, 1, false, 10u, "bias d=64 P=1 non-causal");
  // Non-causal, multiple chunks.
  rc |= run_ring_with_bias({1, 1, 32, 64, 64}, 2, false, 11u, "bias d=64 P=2 non-causal");
  rc |= run_ring_with_bias({2, 2, 32, 64, 64}, 4, false, 12u, "bias d=64 batch P=4 non-causal");
  // Causal.
  rc |= run_ring_with_bias({1, 1, 64, 64, 64}, 2, true, 13u, "bias d=64 P=2 causal");
  rc |= run_ring_with_bias({2, 2, 64, 64, 128}, 2, true, 14u, "bias d=128 batch P=2 causal");
  // D=32 edge case.
  rc |= run_ring_with_bias({1, 1, 64, 64, 32}, 4, false, 15u, "bias d=32 P=4 non-causal");

  // --- TODO 4: segment_ids ---
  // Non-causal, single chunk.
  rc |= run_ring_with_segments({1, 1, 32, 32, 64}, 1, false, 20u, "segs d=64 P=1 non-causal");
  // Non-causal, multiple chunks.
  rc |= run_ring_with_segments({1, 1, 64, 64, 64}, 2, false, 21u, "segs d=64 P=2 non-causal");
  rc |= run_ring_with_segments({2, 2, 64, 64, 64}, 4, false, 22u, "segs d=64 batch P=4 non-causal");
  // Causal.
  rc |= run_ring_with_segments({1, 1, 64, 64, 64}, 2, true, 23u, "segs d=64 P=2 causal");
  rc |= run_ring_with_segments({2, 2, 64, 64, 128}, 2, true, 24u, "segs d=128 batch P=2 causal");
  // D=32.
  rc |= run_ring_with_segments({1, 2, 64, 64, 32}, 4, false, 25u, "segs d=32 P=4 non-causal");

  if (rc == 0) printf("attention_step_bias_segs OK\n");
  return rc;
}
