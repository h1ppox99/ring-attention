/// @file
/// FP16 / Tensor-Core variant of the FlashAttention-style kernel.
///
/// Mirrors the FP32 reference kernel in `flash_attention.cu`, but performs the
/// two matmuls (Q·Kᵀ and P·V) with `nvcuda::wmma` Tensor Core ops at FP16
/// precision and FP32 accumulation. The public API still takes `float*` so it
/// is a drop-in for `launch_flash_attention` and bench/test paths can compare
/// the two side-by-side without changing surrounding code.
///
/// Tiling:
///   - One warp (32 threads) per block.
///   - One block computes BR=16 consecutive query rows for one (batch, head).
///   - K/V is consumed in BC=16-key tiles.
///   - The head-dim D is processed in 16-wide inner slices (D must be a
///     multiple of 16). Supported: 32 / 64 / 128.
///
/// Per-block shared memory:
///   - Q_h[BR][D]   fp16   (one-time load)
///   - K_h[BC][D]   fp16   (per K-tile load, converted from fp32)
///   - V_h[BC][D]   fp16   (per K-tile load)
///   - S_f[BR][BC]  fp32   (Q·Kᵀ result, then softmax workspace)
///   - P_h[BR][BC]  fp16   (softmaxed S, fed to P·V)
///   - O_f[BR][D]   fp32   (running output accumulator, in smem so we can
///                          rescale it row-wise on each tile without poking
///                          into wmma fragment internals)
///   - m_s[BR]      fp32   (running per-row max)
///   - l_s[BR]      fp32   (running per-row sum-exp)
/// Worst-case at D=128: ~22 KB, comfortably under the 48 KB sm_75 budget.

#include <cuda_fp16.h>
#include <mma.h>

#include <cassert>
#include <cmath>
#include <cstdio>

#include "attention.hpp"
#include "common.cuh"

namespace ring_attention {

namespace {

using namespace nvcuda;

constexpr int kBR = 16;
constexpr int kBC = 16;
constexpr int kMmaK = 16;

// Convert a float to half with round-to-nearest-even.
__device__ __forceinline__ __half f2h(float x) { return __float2half_rn(x); }

/// FlashAttention with wmma 16×16×16 Tensor Core matmuls.
///
/// @tparam D Head dimension; must be a multiple of 16 and one of {32, 64, 128}.
template <int D>
__global__ void flash_attention_fp16_kernel(const float* __restrict__ Q,
                                            const float* __restrict__ K,
                                            const float* __restrict__ V, float* __restrict__ O,
                                            int H, int Sq, int Sk, float scale, bool causal,
                                            int causal_shift) {
  static_assert(D % kMmaK == 0, "D must be a multiple of 16 for wmma path");
  constexpr int kDSlices = D / kMmaK;

  const int tid = threadIdx.x;  // 0..31 (one warp per block)
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  const long head_q = ((long)b * H + h) * Sq * D;
  const long head_k = ((long)b * H + h) * Sk * D;

  __shared__ __half Q_h[kBR * D];
  __shared__ __half K_h[kBC * D];
  __shared__ __half V_h[kBC * D];
  __shared__ float S_f[kBR * kBC];
  __shared__ __half P_h[kBR * kBC];
  __shared__ float O_f[kBR * D];
  __shared__ float m_s[kBR];
  __shared__ float l_s[kBR];

  // ---- Initialisation -----------------------------------------------------
  // O = 0, m = -inf, l = 0.  All 32 threads cooperate.
  for (int i = tid; i < kBR; i += blockDim.x) {
    m_s[i] = -INFINITY;
    l_s[i] = 0.0f;
  }
  for (int i = tid; i < kBR * D; i += blockDim.x) O_f[i] = 0.0f;

  // ---- Load Q tile (fp32 → fp16) -----------------------------------------
  for (int idx = tid; idx < kBR * D; idx += blockDim.x) {
    const int r = idx / D;
    const int d = idx % D;
    const int i_global = q_tile * kBR + r;
    if (i_global < Sq) {
      Q_h[idx] = f2h(Q[head_q + (long)i_global * D + d]);
    } else {
      Q_h[idx] = f2h(0.0f);
    }
  }
  __syncthreads();

  // ---- Iterate over K/V tiles --------------------------------------------
  const int num_k_tiles = (Sk + kBC - 1) / kBC;
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    const int j_base = kt * kBC;

    // Cooperative K/V load (fp32 → fp16).  Out-of-range keys padded with 0.
    for (int idx = tid; idx < kBC * D; idx += blockDim.x) {
      const int j_local = idx / D;
      const int d = idx % D;
      const int j_global = j_base + j_local;
      if (j_global < Sk) {
        K_h[idx] = f2h(K[head_k + (long)j_global * D + d]);
        V_h[idx] = f2h(V[head_k + (long)j_global * D + d]);
      } else {
        K_h[idx] = f2h(0.0f);
        V_h[idx] = f2h(0.0f);
      }
    }
    __syncthreads();

    // ---- Matmul 1:  S = (Q · Kᵀ) * scale  -------------------------------
    // S is 16x16 fp32.  Accumulate over D/16 inner-K slices.
    // A = Q (row_major, 16xD slice).  B = K viewed as col_major to realize Kᵀ
    // — element (r, c) of the col_major fragment is K_h[c*D + r], i.e. K[c][r]
    // which is Kᵀ[r][c].  Both fragments load 16x16 = one mma per inner step.
    {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
      wmma::fill_fragment(s_frag, 0.0f);

#pragma unroll
      for (int kk = 0; kk < kDSlices; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
        wmma::load_matrix_sync(q_frag, Q_h + kk * kMmaK, /*stride=*/D);
        wmma::load_matrix_sync(k_frag, K_h + kk * kMmaK, /*stride=*/D);
        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
      }
      // Apply scale and stash to smem so the softmax can run as plain CUDA.
#pragma unroll
      for (int i = 0; i < s_frag.num_elements; ++i) s_frag.x[i] *= scale;
      wmma::store_matrix_sync(S_f, s_frag, /*stride=*/kBC, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- Per-row online softmax ----------------------------------------
    // First 16 threads each own one query row.  Compute m_new, alpha, exp,
    // row_sum; write P_h; rescale O_f row by alpha; update m_s, l_s.
    if (tid < kBR) {
      const int r = tid;
      const int i_global = q_tile * kBR + r;
      const bool row_active = (i_global < Sq);

      // Find tile-local max with causal/range masking baked in.
      float m_prev = m_s[r];
      float m_new = m_prev;
      if (row_active) {
#pragma unroll
        for (int j = 0; j < kBC; ++j) {
          const int j_global = j_base + j;
          const bool visible =
              (j_global < Sk) && (!causal || (j_global <= i_global + causal_shift));
          const float s = visible ? S_f[r * kBC + j] : -INFINITY;
          S_f[r * kBC + j] = s;  // overwrite with masked value
          if (s > m_new) m_new = s;
        }
      }

      // If the row has not yet seen any valid key (m_new still -inf), the
      // standard recurrence produces NaN from exp(-inf - -inf).  Skip the
      // update entirely; the row state stays (m=-inf, l=0, O=0).
      const bool can_update = row_active && (m_new != -INFINITY);

      float alpha = 0.0f;
      float row_sum = 0.0f;
      if (can_update) {
        // alpha = 0 when m_prev = -inf and m_new is finite (first valid tile).
        alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
#pragma unroll
        for (int j = 0; j < kBC; ++j) {
          const float s = S_f[r * kBC + j];
          const float p = (s == -INFINITY) ? 0.0f : expf(s - m_new);
          P_h[r * kBC + j] = f2h(p);
          row_sum += p;
        }
        l_s[r] = alpha * l_s[r] + row_sum;
        m_s[r] = m_new;
      } else {
        // Zero out P so the P·V matmul contributes nothing for this row.
#pragma unroll
        for (int j = 0; j < kBC; ++j) P_h[r * kBC + j] = f2h(0.0f);
      }

      // Rescale running output for this row by alpha (works for both can_update
      // and !can_update because alpha stays 0 in the latter, leaving O_f
      // untouched only when it was zero to begin with — which it is, since
      // every prior tile also failed to update).
      if (can_update) {
        const float a = alpha;
#pragma unroll
        for (int d = 0; d < D; ++d) O_f[r * D + d] *= a;
      }
    }
    __syncthreads();

    // ---- Matmul 2:  O += P · V  ----------------------------------------
    // For each 16-wide slice of D, accumulate into the O_f tile.
#pragma unroll
    for (int ds = 0; ds < kDSlices; ++ds) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_frag;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag;
      wmma::load_matrix_sync(p_frag, P_h, /*stride=*/kBC);
      wmma::load_matrix_sync(v_frag, V_h + ds * kMmaK, /*stride=*/D);
      wmma::load_matrix_sync(o_frag, O_f + ds * kMmaK, /*stride=*/D, wmma::mem_row_major);
      wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
      wmma::store_matrix_sync(O_f + ds * kMmaK, o_frag, /*stride=*/D, wmma::mem_row_major);
    }
    __syncthreads();
  }

  // ---- Finalize: write O / l back to global as fp32 -----------------------
  for (int idx = tid; idx < kBR * D; idx += blockDim.x) {
    const int r = idx / D;
    const int d = idx % D;
    const int i_global = q_tile * kBR + r;
    if (i_global >= Sq) continue;
    const float l = l_s[r];
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    O[head_q + (long)i_global * D + d] = O_f[r * D + d] * inv_l;
  }
}

template <int D>
void launch_typed(const float* q, const float* k, const float* v, float* out,
                  const AttentionShape& shape, bool causal, cudaStream_t stream) {
  const dim3 grid(ceil_div(shape.seq_q, kBR), shape.heads, shape.batch);
  const dim3 block(32);  // one warp per block
  const int causal_shift = shape.seq_k - shape.seq_q;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  flash_attention_fp16_kernel<D><<<grid, block, 0, stream>>>(
      q, k, v, out, shape.heads, shape.seq_q, shape.seq_k, scale, causal, causal_shift);
  cudaCheck(cudaGetLastError());
}

}  // namespace

void launch_flash_attention_fp16(const float* q, const float* k, const float* v, float* out,
                                 const AttentionShape& shape, bool causal, cudaStream_t stream) {
  switch (shape.head_dim) {
    case 32:
      launch_typed<32>(q, k, v, out, shape, causal, stream);
      break;
    case 64:
      launch_typed<64>(q, k, v, out, shape, causal, stream);
      break;
    case 128:
      launch_typed<128>(q, k, v, out, shape, causal, stream);
      break;
    default:
      fprintf(stderr, "flash_attention_fp16: unsupported head_dim=%d (supported: 32, 64, 128)\n",
              shape.head_dim);
      std::abort();
  }
}

}  // namespace ring_attention
