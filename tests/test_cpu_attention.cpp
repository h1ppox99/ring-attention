/// @file
/// Analytical sanity checks for the CPU reference attention.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "attention.hpp"
#include "cpu_attention.hpp"

using ring_attention::AttentionShape;
using ring_attention::cpu_attention;
using ring_attention::XorShift32;

namespace {

constexpr float kTol = 1e-5f;

bool approx_equal(float a, float b, float tol = kTol) {
  return std::fabs(a - b) <= tol * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

#define EXPECT(cond, msg)                                           \
  do {                                                              \
    if (!(cond)) {                                                  \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
      return 1;                                                     \
    }                                                               \
  } while (0)

/// Sq = Sk = 1: O must equal V (softmax of a single score is 1).
int test_single_token() {
  AttentionShape s{/*batch*/ 1, /*heads*/ 1, /*seq_q*/ 1, /*seq_k*/ 1, /*head_dim*/ 4};
  std::vector<float> q{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> k{0.5f, -0.5f, 0.25f, -0.25f};
  std::vector<float> v{7.0f, -1.0f, 0.0f, 42.0f};
  std::vector<float> o(4, 0.0f);
  cpu_attention(q.data(), k.data(), v.data(), o.data(), s, /*causal*/ false);
  for (int d = 0; d < 4; ++d) EXPECT(approx_equal(o[d], v[d]), "single-token output != V");
  return 0;
}

/// All V rows identical → O must equal that common row, regardless of Q, K.
int test_uniform_values() {
  AttentionShape s{1, 2, 5, 7, 8};
  const std::size_t qn = static_cast<std::size_t>(s.batch) * s.heads * s.seq_q * s.head_dim;
  const std::size_t kn = static_cast<std::size_t>(s.batch) * s.heads * s.seq_k * s.head_dim;
  std::vector<float> q(qn), k(kn), v(kn), o(qn, 0.0f);
  XorShift32 rng(0xC0FFEEu);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  // Each row of V (per head) is the constant value 3.14159.
  for (int b = 0; b < s.batch; ++b) {
    for (int h = 0; h < s.heads; ++h) {
      const float val = 3.14159f + h;
      for (int j = 0; j < s.seq_k; ++j) {
        for (int d = 0; d < s.head_dim; ++d) {
          const std::size_t idx =
              (((static_cast<std::size_t>(b) * s.heads) + h) * s.seq_k + j) * s.head_dim + d;
          v[idx] = val;
        }
      }
    }
  }
  cpu_attention(q.data(), k.data(), v.data(), o.data(), s, /*causal*/ false);
  for (int b = 0; b < s.batch; ++b) {
    for (int h = 0; h < s.heads; ++h) {
      const float val = 3.14159f + h;
      for (int i = 0; i < s.seq_q; ++i) {
        for (int d = 0; d < s.head_dim; ++d) {
          const std::size_t idx =
              (((static_cast<std::size_t>(b) * s.heads) + h) * s.seq_q + i) * s.head_dim + d;
          EXPECT(approx_equal(o[idx], val), "uniform-V output != V");
        }
      }
    }
  }
  return 0;
}

/// Causal Sq=Sk: query row 0 only sees key 0 → O[0] must equal V[0].
int test_causal_first_row() {
  AttentionShape s{1, 1, 4, 4, 3};
  std::vector<float> q(12), k(12), v(12), o(12, 0.0f);
  XorShift32 rng(42u);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);
  cpu_attention(q.data(), k.data(), v.data(), o.data(), s, /*causal*/ true);
  for (int d = 0; d < 3; ++d) EXPECT(approx_equal(o[d], v[d]), "causal O[0] != V[0]");
  return 0;
}

/// Hand-computed 2x1 case: head_dim=1, Sq=Sk=2, all entries = 1.
/// Scores = QK^T/sqrt(1) = [[1,1],[1,1]]; softmax row = [0.5, 0.5];
/// O = [0.5*v0 + 0.5*v1, ...].
int test_hand_computed_2x2() {
  AttentionShape s{1, 1, 2, 2, 1};
  std::vector<float> q{1.0f, 1.0f};
  std::vector<float> k{1.0f, 1.0f};
  std::vector<float> v{2.0f, 6.0f};
  std::vector<float> o(2, 0.0f);
  cpu_attention(q.data(), k.data(), v.data(), o.data(), s, /*causal*/ false);
  EXPECT(approx_equal(o[0], 4.0f), "non-causal 2x2 O[0]");
  EXPECT(approx_equal(o[1], 4.0f), "non-causal 2x2 O[1]");

  cpu_attention(q.data(), k.data(), v.data(), o.data(), s, /*causal*/ true);
  EXPECT(approx_equal(o[0], 2.0f), "causal 2x2 O[0]");  // sees only v[0]
  EXPECT(approx_equal(o[1], 4.0f), "causal 2x2 O[1]");  // sees v[0] and v[1] equally
  return 0;
}

/// GQA: verify each Q head attends to the correct KV head (h % kv_H).
/// Build a GQA case where each KV head has a distinctive V pattern; check
/// that Q heads sharing the same KV head produce identical output.
int test_gqa_head_sharing() {
  // H=4 Q heads, kv_H=2 KV heads: Q heads {0,2} → KV head 0; {1,3} → KV head 1.
  AttentionShape s{1, 4, 8, 8, 4};
  s.kv_heads = 2;
  const int kv_H = 2;

  const std::size_t qn = (std::size_t)s.batch * s.heads * s.seq_q * s.head_dim;
  const std::size_t kn = (std::size_t)s.batch * kv_H * s.seq_k * s.head_dim;
  std::vector<float> q(qn), k(kn), v(kn), o(qn, 0.0f);
  XorShift32 rng(0xABCDu);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);

  cpu_attention(q.data(), k.data(), v.data(), o.data(), s, /*causal*/ false);

  // Q heads 0 and 2 use KV head 0; their outputs must differ (different Q) but
  // we can verify they used the same K/V by running single-head MHA and comparing.
  for (int pair = 0; pair < 2; ++pair) {
    const int h0 = pair;      // Q head mapping to KV head `pair`
    const int h1 = pair + 2;  // Q head also mapping to KV head `pair`
    // Extract the output rows for both Q heads and verify they are not identical
    // (they have different Q, so outputs differ).  The real check is that swapping
    // which KV head they attend to would change the result — we confirm by running
    // two 1-head MHA calls with the correct KV head and comparing to cpu_attention.
    AttentionShape s1{1, 1, s.seq_q, s.seq_k, s.head_dim};

    // Build single-head Q tensors for h0 and h1.
    const std::size_t hstride_q = (std::size_t)s.seq_q * s.head_dim;
    const std::size_t hstride_kv = (std::size_t)s.seq_k * s.head_dim;
    std::vector<float> q0(hstride_q), q1(hstride_q);
    std::vector<float> kv(hstride_kv), vv(hstride_kv);
    std::copy(q.begin() + h0 * hstride_q, q.begin() + (h0 + 1) * hstride_q, q0.begin());
    std::copy(q.begin() + h1 * hstride_q, q.begin() + (h1 + 1) * hstride_q, q1.begin());
    std::copy(k.begin() + pair * hstride_kv, k.begin() + (pair + 1) * hstride_kv, kv.begin());
    std::copy(v.begin() + pair * hstride_kv, v.begin() + (pair + 1) * hstride_kv, vv.begin());

    std::vector<float> o0(hstride_q, 0.0f), o1(hstride_q, 0.0f);
    cpu_attention(q0.data(), kv.data(), vv.data(), o0.data(), s1, false);
    cpu_attention(q1.data(), kv.data(), vv.data(), o1.data(), s1, false);

    for (std::size_t i = 0; i < hstride_q; ++i) {
      EXPECT(approx_equal(o[h0 * hstride_q + i], o0[i]), "GQA head h0 output mismatch");
      EXPECT(approx_equal(o[h1 * hstride_q + i], o1[i]), "GQA head h1 output mismatch");
    }
  }
  return 0;
}

/// MQA: single KV head shared by all Q heads.
int test_mqa_single_kv() {
  AttentionShape s{1, 4, 8, 8, 4};
  s.kv_heads = 1;

  const std::size_t qn = (std::size_t)s.batch * s.heads * s.seq_q * s.head_dim;
  const std::size_t kn = (std::size_t)s.batch * 1 * s.seq_k * s.head_dim;
  std::vector<float> q(qn), k(kn), v(kn), o(qn, 0.0f);
  XorShift32 rng(0xDEADu);
  rng.fill_uniform(q);
  rng.fill_uniform(k);
  rng.fill_uniform(v);

  cpu_attention(q.data(), k.data(), v.data(), o.data(), s, /*causal*/ false);

  // Verify each Q head against a 1-head MHA call with the single KV head.
  const std::size_t hstride_q = (std::size_t)s.seq_q * s.head_dim;
  const std::size_t hstride_kv = (std::size_t)s.seq_k * s.head_dim;
  AttentionShape s1{1, 1, s.seq_q, s.seq_k, s.head_dim};
  for (int h = 0; h < s.heads; ++h) {
    std::vector<float> qh(hstride_q), oh(hstride_q, 0.0f);
    std::copy(q.begin() + h * hstride_q, q.begin() + (h + 1) * hstride_q, qh.begin());
    cpu_attention(qh.data(), k.data(), v.data(), oh.data(), s1, false);
    for (std::size_t i = 0; i < hstride_q; ++i)
      EXPECT(approx_equal(o[h * hstride_q + i], oh[i]), "MQA head output mismatch");
  }
  return 0;
}

/// Cross-attention shape Sq != Sk with causal alignment shift Sk - Sq.
/// Sq=1, Sk=3, query "i=0" sees keys j <= 0 + (3-1) = 2 → all keys; should match non-causal.
int test_causal_alignment_cross() {
  AttentionShape s{1, 1, 1, 3, 2};
  std::vector<float> q{0.1f, -0.2f};
  std::vector<float> k{1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f};
  std::vector<float> v{1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f};
  std::vector<float> o_c(2, 0.0f), o_nc(2, 0.0f);
  cpu_attention(q.data(), k.data(), v.data(), o_nc.data(), s, /*causal*/ false);
  cpu_attention(q.data(), k.data(), v.data(), o_c.data(), s, /*causal*/ true);
  for (int d = 0; d < 2; ++d)
    EXPECT(approx_equal(o_c[d], o_nc[d]), "causal-aligned Sq<Sk should match non-causal");
  return 0;
}

}  // namespace

int main() {
  int rc = 0;
  rc |= test_single_token();
  rc |= test_uniform_values();
  rc |= test_causal_first_row();
  rc |= test_hand_computed_2x2();
  rc |= test_causal_alignment_cross();
  rc |= test_gqa_head_sharing();
  rc |= test_mqa_single_kv();
  if (rc == 0) printf("cpu_attention OK\n");
  return rc;
}
