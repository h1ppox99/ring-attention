/// @file
/// Naive dense GPU attention kernel: materializes per-row scores in shared
/// memory. Serves as the correctness baseline for the tiled FlashAttention-style
/// kernel that lands in S4.

#include <cmath>
#include <cstdio>

#include "attention.hpp"
#include "common.cuh"

namespace ring_attention {

namespace {

/// One block per (batch, head, query row). Threads cooperatively compute the
/// score vector, then thread 0 does the (small) row reduction, then threads
/// parallelize the V projection across head_dim.
__global__ void naive_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                       const float* __restrict__ V, float* __restrict__ O, int H,
                                       int kv_H, int Sq, int Sk, int D, float scale, bool causal,
                                       int causal_shift) {
  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const long head_q = ((long)b * H + h) * Sq * D;
  const long head_k = ((long)b * kv_H + (h % kv_H)) * Sk * D;

  const float* q_row = Q + head_q + (long)i * D;
  const float* k_h = K + head_k;
  const float* v_h = V + head_k;
  float* o_row = O + head_q + (long)i * D;

  // Dynamic shared memory layout: [scores: Sk floats][row_max: 1][row_sum: 1].
  extern __shared__ float smem[];
  float* scores = smem;
  float* shared_max = smem + Sk;
  float* shared_sum = smem + Sk + 1;

  const int j_max = causal ? min(Sk, i + causal_shift + 1) : Sk;

  // 1) Scores: scores[j] = (q_row . k_h[j]) * scale, or -inf if masked out.
  for (int j = tid; j < Sk; j += nthreads) {
    if (j < j_max) {
      float s = 0.0f;
      const float* k_row = k_h + (long)j * D;
      for (int d = 0; d < D; ++d) s += q_row[d] * k_row[d];
      scores[j] = s * scale;
    } else {
      scores[j] = -INFINITY;
    }
  }
  __syncthreads();

  // 2) & 3) Row max + exp + sum, done by a single thread (naive: O(Sk) serial).
  if (tid == 0) {
    float m = -INFINITY;
    for (int j = 0; j < j_max; ++j)
      if (scores[j] > m) m = scores[j];
    *shared_max = m;
    float sum = 0.0f;
    for (int j = 0; j < j_max; ++j) {
      float e = expf(scores[j] - m);
      scores[j] = e;
      sum += e;
    }
    *shared_sum = sum;
  }
  __syncthreads();

  const float inv_denom = (*shared_sum > 0.0f) ? (1.0f / *shared_sum) : 0.0f;

  // 4) O[d] = sum_j scores[j] * V[j, d] / denom — parallel across d.
  for (int d = tid; d < D; d += nthreads) {
    float acc = 0.0f;
    for (int j = 0; j < j_max; ++j) {
      acc += scores[j] * v_h[(long)j * D + d];
    }
    o_row[d] = acc * inv_denom;
  }
}

}  // namespace

void launch_naive_attention(const float* q, const float* k, const float* v, float* out,
                            const AttentionShape& shape, bool causal, cudaStream_t stream) {
  const int B = shape.batch;
  const int H = shape.heads;
  const int kv_H = (shape.kv_heads > 0) ? shape.kv_heads : shape.heads;
  const int Sq = shape.seq_q;
  const int Sk = shape.seq_k;
  const int D = shape.head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int causal_shift = Sk - Sq;

  // Block size: enough threads to amortize over Sk and D. 128 is a safe default
  // for the small problems this kernel targets.
  const int block = 128;
  const dim3 grid(Sq, H, B);
  const std::size_t smem_bytes = (static_cast<std::size_t>(Sk) + 2) * sizeof(float);

  naive_attention_kernel<<<grid, block, smem_bytes, stream>>>(q, k, v, out, H, kv_H, Sq, Sk, D,
                                                              scale, causal, causal_shift);
  cudaCheck(cudaGetLastError());
}

}  // namespace ring_attention
