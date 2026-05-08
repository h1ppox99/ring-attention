/// @file
/// Tiled FlashAttention-style kernel: online softmax across K/V tiles, no
/// materialization of the full score matrix.
///
/// Layout per block:
///   - One block computes BR consecutive query rows for one (batch, head).
///   - Thread `t` owns query row `q_tile_base + t`.
///   - Shared memory holds the Q tile (BR×D), and the current K/V tiles (BC×D each).
///   - Each thread keeps its own (m_i, l_i, O_i[D]) in registers.
///
/// The inner online-softmax update follows the FlashAttention paper:
///   m_new   = max(m_i, max_j s_ij)
///   alpha   = exp(m_i - m_new)        (zero when m_i = -inf and m_new finite)
///   O_i    <- alpha * O_i + sum_j exp(s_ij - m_new) * V_j
///   l_i    <- alpha * l_i + sum_j exp(s_ij - m_new)
///   m_i    <- m_new
/// Final output: O_i / l_i.

#include <cassert>
#include <cmath>
#include <cstdio>

#include "attention.hpp"
#include "common.cuh"

namespace ring_attention {

namespace {

template <int BR, int BC, int D>
__global__ void flash_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                       const float* __restrict__ V, float* __restrict__ O, int H,
                                       int Sq, int Sk, float scale, bool causal, int causal_shift) {
  const int tid = threadIdx.x;  // 0 .. BR-1
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  const int i_global = q_tile * BR + tid;
  const bool active = (i_global < Sq);

  const long head_q = ((long)b * H + h) * Sq * D;
  const long head_k = ((long)b * H + h) * Sk * D;

  __shared__ float Q_tile[BR * D];
  __shared__ float K_tile[BC * D];
  __shared__ float V_tile[BC * D];

  // Load Q tile (one row per thread; pad with zeros for inactive rows).
  if (active) {
    const float* q_src = Q + head_q + (long)i_global * D;
    for (int d = 0; d < D; ++d) Q_tile[tid * D + d] = q_src[d];
  } else {
    for (int d = 0; d < D; ++d) Q_tile[tid * D + d] = 0.0f;
  }

  // Per-row online softmax state.
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float O_i[D];
#pragma unroll
  for (int d = 0; d < D; ++d) O_i[d] = 0.0f;

  const int num_k_tiles = (Sk + BC - 1) / BC;
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    __syncthreads();
    const int j_base = kt * BC;

    // Cooperative K/V load (BR threads loading BC*D elements each).
    for (int idx = tid; idx < BC * D; idx += BR) {
      const int j_local = idx / D;
      const int d = idx % D;
      const int j_global = j_base + j_local;
      if (j_global < Sk) {
        K_tile[idx] = K[head_k + (long)j_global * D + d];
        V_tile[idx] = V[head_k + (long)j_global * D + d];
      } else {
        K_tile[idx] = 0.0f;
        V_tile[idx] = 0.0f;
      }
    }
    __syncthreads();

    if (!active) continue;

    // Compute s[j] = Q_i · K_j * scale and find this-tile row max.
    float s[BC];
    float m_new = m_i;
#pragma unroll
    for (int j = 0; j < BC; ++j) {
      const int j_global = j_base + j;
      const bool visible = (j_global < Sk) && (!causal || (j_global <= i_global + causal_shift));
      if (visible) {
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < D; ++d) dot += Q_tile[tid * D + d] * K_tile[j * D + d];
        const float v = dot * scale;
        s[j] = v;
        if (v > m_new) m_new = v;
      } else {
        s[j] = -INFINITY;
      }
    }

    // If every score in this tile (and previous tiles) is still -inf, skip:
    // the standard recurrence would produce NaN from exp(-inf - -inf).
    if (m_new == -INFINITY) continue;

    const float alpha = expf(m_i - m_new);  // 0 when m_i == -inf and m_new finite
    float row_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < BC; ++j) {
      s[j] = (s[j] == -INFINITY) ? 0.0f : expf(s[j] - m_new);
      row_sum += s[j];
    }

    // O_i <- alpha * O_i + sum_j s[j] * V_j
#pragma unroll
    for (int d = 0; d < D; ++d) {
      float acc = alpha * O_i[d];
#pragma unroll
      for (int j = 0; j < BC; ++j) {
        acc += s[j] * V_tile[j * D + d];
      }
      O_i[d] = acc;
    }
    l_i = alpha * l_i + row_sum;
    m_i = m_new;
  }

  if (active) {
    float* o_dst = O + head_q + (long)i_global * D;
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    for (int d = 0; d < D; ++d) o_dst[d] = O_i[d] * inv_l;
  }
}

template <int BR, int BC, int D>
void launch_typed(const float* q, const float* k, const float* v, float* out,
                  const AttentionShape& shape, bool causal, cudaStream_t stream) {
  const dim3 grid(ceil_div(shape.seq_q, BR), shape.heads, shape.batch);
  const dim3 block(BR);
  const int causal_shift = shape.seq_k - shape.seq_q;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  flash_attention_kernel<BR, BC, D><<<grid, block, 0, stream>>>(
      q, k, v, out, shape.heads, shape.seq_q, shape.seq_k, scale, causal, causal_shift);
  cudaCheck(cudaGetLastError());
}

}  // namespace

void launch_flash_attention(const float* q, const float* k, const float* v, float* out,
                            const AttentionShape& shape, bool causal, cudaStream_t stream) {
  // Tile sizes chosen to stay under the 48 KB per-block shared-memory budget
  // on sm_75: (BR + 2*BC) * D * 4 bytes.
  switch (shape.head_dim) {
    case 32:
      launch_typed<64, 64, 32>(q, k, v, out, shape, causal, stream);  // 12 KB smem
      break;
    case 64:
      launch_typed<32, 32, 64>(q, k, v, out, shape, causal, stream);  // 24 KB smem
      break;
    case 128:
      launch_typed<32, 16, 128>(q, k, v, out, shape, causal, stream);  // 32 KB smem
      break;
    default:
      fprintf(stderr, "flash_attention: unsupported head_dim=%d (supported: 32, 64, 128)\n",
              shape.head_dim);
      std::abort();
  }
}

}  // namespace ring_attention
