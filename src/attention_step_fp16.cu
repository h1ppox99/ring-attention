/// @file
/// FP16 / Tensor-Core persistent-state attention step. Mirrors
/// `attention_step.cu` (the building block of the ring loop) but consumes Q/K/V
/// in `__half` and performs the two matmuls (Q·Kᵀ and P·V) with
/// `nvcuda::wmma 16x16x16` Tensor Core ops. Running state (O, m, l) stays in
/// FP32 — these are sums-of-exponentials that need the extra precision and are
/// rewritten to global memory at every ring step.
///
/// Same `(BR=16, BC=16)` tiling as `flash_attention_fp16.cu`, so the smem
/// budget at D=128 is ~22 KB (well under the 48 KB sm_75 limit). One warp per
/// block; one block per `(q_tile, head, batch)` triple.
///
/// The FP32 path (`attention_step.cu`) is untouched — both can be selected at
/// runtime from `ring_loop` via `RingConfig::dtype`.

#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "attention.hpp"
#include "common.cuh"

namespace ring_attention {

namespace {

using namespace nvcuda;

constexpr int kBR = 16;
constexpr int kBC = 16;
constexpr int kMmaK = 16;

/// One ring-step kernel: read persistent (O, m, l) from global, fold in one
/// (K, V) chunk, write the updated state back. After the final chunk
/// `launch_attention_finalize` divides O by l (unchanged, FP32 path).
///
/// @tparam D Head dimension; multiple of 16. Supported: 32, 64, 128, 256.
template <int D>
__global__ void attention_step_fp16_kernel(const __half* __restrict__ Q,
                                           const __half* __restrict__ K,
                                           const __half* __restrict__ V, float* __restrict__ O,
                                           float* __restrict__ M, float* __restrict__ L, int H,
                                           int kv_H, int Sq, int Sk, float scale, bool causal,
                                           int q_offset, int k_offset) {
  static_assert(D % kMmaK == 0, "D must be a multiple of 16 for wmma path");
  constexpr int kDSlices = D / kMmaK;

  const int tid = threadIdx.x;  // 0..31, one warp per block
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  const long head_q = ((long)b * H + h) * Sq * D;
  const long head_k = ((long)b * kv_H + (h % kv_H)) * Sk * D;
  const long base_row = ((long)b * H + h) * Sq;

  __shared__ __half Q_h[kBR * D];
  __shared__ __half K_h[kBC * D];
  __shared__ __half V_h[kBC * D];
  __shared__ float S_f[kBR * kBC];
  __shared__ __half P_h[kBR * kBC];
  __shared__ float O_f[kBR * D];
  __shared__ float m_s[kBR];
  __shared__ float l_s[kBR];

  // ---- Load Q tile (already FP16 in global) ------------------------------
  for (int idx = tid; idx < kBR * D; idx += blockDim.x) {
    const int r = idx / D;
    const int d = idx % D;
    const int i_local = q_tile * kBR + r;
    Q_h[idx] = (i_local < Sq) ? Q[head_q + (long)i_local * D + d] : __float2half_rn(0.0f);
  }

  // ---- Load persistent (O, m, l) state from global -----------------------
  if (tid < kBR) {
    const int r = tid;
    const int i_local = q_tile * kBR + r;
    if (i_local < Sq) {
      m_s[r] = M[base_row + i_local];
      l_s[r] = L[base_row + i_local];
    } else {
      m_s[r] = -INFINITY;
      l_s[r] = 0.0f;
    }
  }
  for (int idx = tid; idx < kBR * D; idx += blockDim.x) {
    const int r = idx / D;
    const int d = idx % D;
    const int i_local = q_tile * kBR + r;
    O_f[idx] = (i_local < Sq) ? O[head_q + (long)i_local * D + d] : 0.0f;
  }
  __syncthreads();

  // ---- Iterate K/V tiles within this chunk -------------------------------
  const int num_k_tiles = (Sk + kBC - 1) / kBC;
  for (int kt = 0; kt < num_k_tiles; ++kt) {
    const int j_base = kt * kBC;

    for (int idx = tid; idx < kBC * D; idx += blockDim.x) {
      const int j_local = idx / D;
      const int d = idx % D;
      const int j_local_in_chunk = j_base + j_local;
      if (j_local_in_chunk < Sk) {
        K_h[idx] = K[head_k + (long)j_local_in_chunk * D + d];
        V_h[idx] = V[head_k + (long)j_local_in_chunk * D + d];
      } else {
        K_h[idx] = __float2half_rn(0.0f);
        V_h[idx] = __float2half_rn(0.0f);
      }
    }
    __syncthreads();

    // ---- Matmul 1: S = (Q · Kᵀ) * scale  -------------------------------
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
#pragma unroll
      for (int i = 0; i < s_frag.num_elements; ++i) s_frag.x[i] *= scale;
      wmma::store_matrix_sync(S_f, s_frag, /*stride=*/kBC, wmma::mem_row_major);
    }
    __syncthreads();

    // ---- Per-row online softmax ----------------------------------------
    if (tid < kBR) {
      const int r = tid;
      const int i_local = q_tile * kBR + r;
      const int i_global = q_offset + i_local;
      const bool row_active = (i_local < Sq);

      float m_prev = m_s[r];
      float m_new = m_prev;
      if (row_active) {
#pragma unroll
        for (int j = 0; j < kBC; ++j) {
          const int j_local_in_chunk = j_base + j;
          const int j_global = k_offset + j_local_in_chunk;
          const bool visible = (j_local_in_chunk < Sk) && (!causal || (j_global <= i_global));
          const float s = visible ? S_f[r * kBC + j] : -INFINITY;
          S_f[r * kBC + j] = s;
          if (s > m_new) m_new = s;
        }
      }

      const bool can_update = row_active && (m_new != -INFINITY);
      float alpha = 0.0f;
      if (can_update) {
        alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_new);
        float row_sum = 0.0f;
#pragma unroll
        for (int j = 0; j < kBC; ++j) {
          const float s = S_f[r * kBC + j];
          const float p = (s == -INFINITY) ? 0.0f : expf(s - m_new);
          P_h[r * kBC + j] = __float2half_rn(p);
          row_sum += p;
        }
        l_s[r] = alpha * l_s[r] + row_sum;
        m_s[r] = m_new;
      } else {
#pragma unroll
        for (int j = 0; j < kBC; ++j) P_h[r * kBC + j] = __float2half_rn(0.0f);
      }

      if (can_update) {
        const float a = alpha;
#pragma unroll
        for (int d = 0; d < D; ++d) O_f[r * D + d] *= a;
      }
    }
    __syncthreads();

    // ---- Matmul 2: O += P · V  -----------------------------------------
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

  // ---- Persist state back to global --------------------------------------
  if (tid < kBR) {
    const int r = tid;
    const int i_local = q_tile * kBR + r;
    if (i_local < Sq) {
      M[base_row + i_local] = m_s[r];
      L[base_row + i_local] = l_s[r];
    }
  }
  for (int idx = tid; idx < kBR * D; idx += blockDim.x) {
    const int r = idx / D;
    const int d = idx % D;
    const int i_local = q_tile * kBR + r;
    if (i_local < Sq) O[head_q + (long)i_local * D + d] = O_f[idx];
  }
}

/// Cast a float buffer to half element-wise. Used by ring_loop_fp16 to convert
/// Q once at startup, and by tests to feed FP32 reference inputs into the FP16
/// path.
__global__ void float_to_half_kernel(const float* __restrict__ src, __half* __restrict__ dst,
                                     std::size_t n) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (std::size_t)gridDim.x * blockDim.x) {
    dst[i] = __float2half_rn(src[i]);
  }
}

/// Cast a half buffer back to float element-wise. Inverse of
/// `float_to_half_kernel`; widens an FP16 KV chunk received over NCCL.
__global__ void half_to_float_kernel(const __half* __restrict__ src, float* __restrict__ dst,
                                     std::size_t n) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (std::size_t)gridDim.x * blockDim.x) {
    dst[i] = __half2float(src[i]);
  }
}

// Symmetric INT8 quant with a fixed scale of 127. The ring-attention inputs
// are bounded in [-1, 1) (gen_elem / XorShift32::next_uniform), so 127 maps
// the full range with no clipping; the clamp guards stray out-of-range values.
__global__ void float_to_int8_kernel(const float* __restrict__ src, signed char* __restrict__ dst,
                                     std::size_t n) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (std::size_t)gridDim.x * blockDim.x) {
    int q = __float2int_rn(src[i] * 127.0f);
    q = max(-127, min(127, q));
    dst[i] = static_cast<signed char>(q);
  }
}

__global__ void int8_to_float_kernel(const signed char* __restrict__ src, float* __restrict__ dst,
                                     std::size_t n) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (std::size_t)gridDim.x * blockDim.x) {
    dst[i] = static_cast<float>(src[i]) * (1.0f / 127.0f);
  }
}

template <int D>
void launch_step_fp16_typed(const __half* q, const __half* k, const __half* v, float* o, float* m,
                            float* l, const AttentionShape& shape, int q_offset, int k_offset,
                            bool causal, cudaStream_t stream) {
  const dim3 grid(ceil_div(shape.seq_q, kBR), shape.heads, shape.batch);
  const dim3 block(32);
  const int kv_H = (shape.kv_heads > 0) ? shape.kv_heads : shape.heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  attention_step_fp16_kernel<D><<<grid, block, 0, stream>>>(q, k, v, o, m, l, shape.heads, kv_H,
                                                            shape.seq_q, shape.seq_k, scale, causal,
                                                            q_offset, k_offset);
  cudaCheck(cudaGetLastError());
}

}  // namespace

void launch_attention_step_fp16(const __half* q, const __half* k, const __half* v, float* o,
                                float* m, float* l, const AttentionShape& shape, int q_offset,
                                int k_offset, bool causal, cudaStream_t stream) {
  switch (shape.head_dim) {
    case 32:
      launch_step_fp16_typed<32>(q, k, v, o, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 64:
      launch_step_fp16_typed<64>(q, k, v, o, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 128:
      launch_step_fp16_typed<128>(q, k, v, o, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 256:
      launch_step_fp16_typed<256>(q, k, v, o, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    default:
      fprintf(stderr,
              "attention_step_fp16: unsupported head_dim=%d (supported: 32, 64, 128, 256)\n",
              shape.head_dim);
      std::abort();
  }
}

void launch_float_to_half(const float* src, __half* dst, std::size_t n, cudaStream_t stream) {
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  float_to_half_kernel<<<grid, block, 0, stream>>>(src, dst, n);
  cudaCheck(cudaGetLastError());
}

void launch_half_to_float(const __half* src, float* dst, std::size_t n, cudaStream_t stream) {
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  half_to_float_kernel<<<grid, block, 0, stream>>>(src, dst, n);
  cudaCheck(cudaGetLastError());
}

void launch_float_to_int8(const float* src, signed char* dst, std::size_t n, cudaStream_t stream) {
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  float_to_int8_kernel<<<grid, block, 0, stream>>>(src, dst, n);
  cudaCheck(cudaGetLastError());
}

void launch_int8_to_float(const signed char* src, float* dst, std::size_t n, cudaStream_t stream) {
  const int block = 256;
  const int grid = static_cast<int>((n + block - 1) / block);
  int8_to_float_kernel<<<grid, block, 0, stream>>>(src, dst, n);
  cudaCheck(cudaGetLastError());
}

}  // namespace ring_attention
