/// @file
/// Host-side reference attention.

#include "cpu_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ring_attention {

namespace {

/// Offset of the (b, h) head slice within a (batch, heads, seq, head_dim) tensor.
inline std::size_t head_offset(int b, int h, int heads, int seq, int head_dim) {
  return ((static_cast<std::size_t>(b) * heads) + h) * seq * head_dim;
}

}  // namespace

void cpu_attention(const float* q, const float* k, const float* v, float* out,
                   const AttentionShape& shape, bool causal) {
  const int B = shape.batch;
  const int H = shape.heads;
  const int kv_H = (shape.kv_heads > 0) ? shape.kv_heads : shape.heads;
  const int Sq = shape.seq_q;
  const int Sk = shape.seq_k;
  const int D = shape.head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  // Causal mask aligned to the end: key j is visible to query i iff
  //   j <= i + (Sk - Sq).
  const int causal_shift = Sk - Sq;

  std::vector<float> scores(static_cast<std::size_t>(Sk));

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const float* q_h = q + head_offset(b, h, H, Sq, D);
      const float* k_h = k + head_offset(b, h % kv_H, kv_H, Sk, D);
      const float* v_h = v + head_offset(b, h % kv_H, kv_H, Sk, D);
      float* o_h = out + head_offset(b, h, H, Sq, D);

      for (int i = 0; i < Sq; ++i) {
        const float* q_row = q_h + static_cast<std::size_t>(i) * D;
        const int j_max = causal ? std::min(Sk, i + causal_shift + 1) : Sk;

        // Score row = Q_i · K_j^T * scale.
        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < j_max; ++j) {
          const float* k_row = k_h + static_cast<std::size_t>(j) * D;
          float s = 0.0f;
          for (int d = 0; d < D; ++d) s += q_row[d] * k_row[d];
          s *= scale;
          scores[j] = s;
          if (s > row_max) row_max = s;
        }

        // Softmax (numerically stable).
        float denom = 0.0f;
        for (int j = 0; j < j_max; ++j) {
          scores[j] = std::exp(scores[j] - row_max);
          denom += scores[j];
        }
        const float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

        // O_i = sum_j p_j * V_j.
        float* o_row = o_h + static_cast<std::size_t>(i) * D;
        for (int d = 0; d < D; ++d) o_row[d] = 0.0f;
        for (int j = 0; j < j_max; ++j) {
          const float p = scores[j] * inv_denom;
          const float* v_row = v_h + static_cast<std::size_t>(j) * D;
          for (int d = 0; d < D; ++d) o_row[d] += p * v_row[d];
        }
      }
    }
  }
}

std::uint32_t XorShift32::next_u32() {
  std::uint32_t x = state_;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state_ = x;
  return x;
}

float XorShift32::next_uniform() {
  // 24-bit mantissa → uniform [0, 1), then shift to [-1, 1).
  const std::uint32_t m = next_u32() >> 8;
  const float u = static_cast<float>(m) * (1.0f / 16777216.0f);
  return 2.0f * u - 1.0f;
}

void XorShift32::fill_uniform(std::vector<float>& out) {
  for (auto& x : out) x = next_uniform();
}

}  // namespace ring_attention
