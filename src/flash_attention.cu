/// @file
/// Tiled FlashAttention-style kernel: online softmax across K/V tiles, no
/// materialization of the full score matrix.
///
/// Layout per block:
///   - One block computes BR consecutive query rows for one (batch, head).
///   - Thread `t` owns query row `q_tile_base + t`.
///   - Shared memory holds the Q tile (BR x (D+1) padded — see below) and the
///     current K/V tiles (BC x D each).
///   - Each thread keeps its own (m_i, l_i, O_i[D]) in registers.
///
/// Shared-memory access notes (driven by Nsight Compute bank-conflict /
/// MIO-stall feedback):
///   1. Q_tile rows are padded by one float (stride = D+1). With stride = D
///      the 32 warp lanes all land in the same bank during `Q_tile[tid*D+d]`,
///      giving a 32-way conflict on every inner-matmul load. With stride
///      D+1 the bank index becomes `(tid + d) % 32`, so the 32 lanes hit
///      32 distinct banks — no conflict.
///   2. The Q*K^T loop is iterated as `for d { q = Q[d]; for j { s[j] += q*K[j][d]; } }`
///      so each Q element is loaded once per d and reused across all BC j's,
///      cutting shared-load instructions by a factor of BC. Masking/scaling
///      runs as a small post-pass.
///   3. K/V cooperative loads use float4 (LDS.128 / STS.128) which issues 4x
///      fewer instructions and reduces MIO-queue pressure.
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
                                       int kv_H, int Sq, int Sk, float scale, bool causal,
                                       int causal_shift) {
  static_assert(D % 32 == 0, "D must be a multiple of 32 for the current shared-mem layout.");
  static_assert((BC * D) % (BR * 4) == 0,
                "Cooperative float4 K/V load assumes BC*D/4 divides evenly across BR threads.");

  // Pad each Q row so the 32 warp lanes hit 32 distinct banks instead of one.
  constexpr int kQStride = D + 1;

  const int tid = threadIdx.x;  // 0 .. BR-1
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  const int i_global = q_tile * BR + tid;
  const bool active = (i_global < Sq);

  const long head_q = ((long)b * H + h) * Sq * D;
  const long head_k = ((long)b * kv_H + (h % kv_H)) * Sk * D;

  __shared__ float Q_tile[BR * kQStride];
  __shared__ float K_tile[BC * D];
  __shared__ float V_tile[BC * D];

  // Load Q tile (one row per thread; pad with zeros for inactive rows).
  if (active) {
    const float* q_src = Q + head_q + (long)i_global * D;
#pragma unroll
    for (int d = 0; d < D; ++d) Q_tile[tid * kQStride + d] = q_src[d];
  } else {
#pragma unroll
    for (int d = 0; d < D; ++d) Q_tile[tid * kQStride + d] = 0.0f;
  }

  // Per-row online softmax state.
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float O_i[D];
#pragma unroll
  for (int d = 0; d < D; ++d) O_i[d] = 0.0f;

  // Each float4 iteration moves 4 contiguous floats of K (and V) from global
  // to shared. With our (BR, BC, D) instantiations the loop count divides
  // evenly across BR threads.
  constexpr int kTileVec4 = (BC * D) / 4;

  const int num_k_tiles = (Sk + BC - 1) / BC;
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    __syncthreads();
    const int j_base = kt * BC;

    // Cooperative K/V load with float4 (LDS.128 / STS.128). 4x fewer shared
    // store instructions than a per-float loop, easing MIO pressure.
    for (int idx4 = tid; idx4 < kTileVec4; idx4 += BR) {
      const int idx = idx4 * 4;
      const int j_local = idx / D;
      const int d = idx - j_local * D;
      const int j_global = j_base + j_local;
      float4 k4;
      float4 v4;
      if (j_global < Sk) {
        k4 = *reinterpret_cast<const float4*>(K + head_k + (long)j_global * D + d);
        v4 = *reinterpret_cast<const float4*>(V + head_k + (long)j_global * D + d);
      } else {
        k4 = float4{0.0f, 0.0f, 0.0f, 0.0f};
        v4 = float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      *reinterpret_cast<float4*>(K_tile + idx) = k4;
      *reinterpret_cast<float4*>(V_tile + idx) = v4;
    }
    __syncthreads();

    if (!active) continue;

    // Q . K^T matmul. Loop order is (d outer, j inner): each Q element is
    // loaded once per d and reused across all BC j's. Combined with the
    // D+1 padding above, the Q shared loads are conflict-free single-banked
    // accesses and the K shared loads are warp-wide broadcasts.
    float s[BC];
#pragma unroll
    for (int j = 0; j < BC; ++j) s[j] = 0.0f;

    for (int d = 0; d < D; ++d) {
      const float q = Q_tile[tid * kQStride + d];
#pragma unroll
      for (int j = 0; j < BC; ++j) {
        s[j] += q * K_tile[j * D + d];
      }
    }

    // Apply scale + bounds/causal mask, find tile-local row max.
    // K_tile rows for j_global >= Sk were zeroed in the cooperative load, so
    // s[j] for those rows is 0; we still overwrite with -INFINITY here.
    float m_new = m_i;
#pragma unroll
    for (int j = 0; j < BC; ++j) {
      const int j_global = j_base + j;
      const bool visible = (j_global < Sk) && (!causal || (j_global <= i_global + causal_shift));
      if (visible) {
        s[j] *= scale;
        if (s[j] > m_new) m_new = s[j];
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
#pragma unroll
    for (int d = 0; d < D; ++d) o_dst[d] = O_i[d] * inv_l;
  }
}

template <int BR, int BC, int D>
void launch_typed(const float* q, const float* k, const float* v, float* out,
                  const AttentionShape& shape, bool causal, cudaStream_t stream) {
  const dim3 grid(ceil_div(shape.seq_q, BR), shape.heads, shape.batch);
  const dim3 block(BR);
  const int kv_H = (shape.kv_heads > 0) ? shape.kv_heads : shape.heads;
  const int causal_shift = shape.seq_k - shape.seq_q;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  flash_attention_kernel<BR, BC, D><<<grid, block, 0, stream>>>(
      q, k, v, out, shape.heads, kv_H, shape.seq_q, shape.seq_k, scale, causal, causal_shift);
  cudaCheck(cudaGetLastError());
}

}  // namespace

void launch_flash_attention(const float* q, const float* k, const float* v, float* out,
                            const AttentionShape& shape, bool causal, cudaStream_t stream) {
  // Tile sizes chosen to stay under the 48 KB per-block shared-memory budget
  // on sm_75. Smem = (BR*(D+1) + 2*BC*D) * 4 bytes after the D+1 Q padding.
  switch (shape.head_dim) {
    case 32:
      launch_typed<64, 64, 32>(q, k, v, out, shape, causal, stream);  // ~24.3 KB smem
      break;
    case 64:
      launch_typed<32, 32, 64>(q, k, v, out, shape, causal, stream);  // ~24.1 KB smem
      break;
    case 128:
      launch_typed<32, 16, 128>(q, k, v, out, shape, causal, stream);  // ~32.1 KB smem
      break;
    case 256:
      launch_typed<16, 8, 256>(q, k, v, out, shape, causal, stream);  // ~32.1 KB smem
      break;
    default:
      fprintf(stderr, "flash_attention: unsupported head_dim=%d (supported: 32, 64, 128, 256)\n",
              shape.head_dim);
      std::abort();
  }
}

}  // namespace ring_attention
