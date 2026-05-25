/// @file
/// Persistent-state online-softmax kernel: the building block called once per
/// ring step. The kernel reads (O, m, l) from global memory, processes one
/// K/V chunk, and writes the updated state back. A separate finalize divides
/// O by l after all chunks have been consumed.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "attention.hpp"
#include "common.cuh"

namespace ring_attention {

namespace {

/// Initialize per-row state. m_count = batch*heads*seq_q.
__global__ void init_kernel(float* O, float* m, float* l, std::size_t out_count,
                            std::size_t m_count) {
  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < out_count;
       idx += (std::size_t)gridDim.x * blockDim.x) {
    O[idx] = 0.0f;
  }
  for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < m_count;
       idx += (std::size_t)gridDim.x * blockDim.x) {
    m[idx] = -INFINITY;
    l[idx] = 0.0f;
  }
}

/// Final normalization: out_i <- out_i / l_i (or 0 if l_i == 0).
__global__ void finalize_kernel(float* O, const float* L, int H, int Sq, int D) {
  const int i = blockIdx.x;  // query row within (b, h)
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;

  const long row = ((long)b * H + h) * Sq + i;
  const float l = L[row];
  const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  float* o = O + row * D;
  for (int d = tid; d < D; d += blockDim.x) {
    o[d] *= inv;
  }
}

/// One ring-step of the online-softmax (flash-attention) recurrence.
///
/// Same recurrence as `flash_attention_kernel`, but instead of finalizing
/// `O <- O / l` inside the kernel, the running state `(O, m, l)` is persisted
/// to global memory so a subsequent call can resume with the next K/V chunk.
/// After all chunks have been consumed, `launch_attention_finalize` performs
/// the single division.
///
/// Grid / block layout:
///   - grid = (ceil(Sq / BR), H, B), block = (BR)
///   - Each thread owns one query row of the BR-row tile; threads in a block
///     cooperatively load BC keys / values into shared memory per K-tile.
///
/// Template parameters:
///   - BR : query rows per block (one thread per row).
///   - BC : K/V columns loaded per shared-memory tile.
///   - D  : head dimension (compile-time so per-thread `O_i[D]` lives in regs).
///
/// @param Q         Queries, shape [B, H, Sq, D], row-major, device pointer.
/// @param K         Keys for *this* chunk, shape [B, H, Sk, D].
/// @param V         Values for *this* chunk, shape [B, H, Sk, D].
/// @param O         In/out running output accumulator, shape [B, H, Sq, D].
///                  Read at entry, updated, written back. Must be zero-init
///                  before the first ring step (see `init_kernel`).
/// @param M         In/out per-row running max, shape [B, H, Sq]. Init to -inf.
/// @param L         In/out per-row running sum-of-exp, shape [B, H, Sq]. Init to 0.
/// @param H         Number of heads.
/// @param Sq        Local query sequence length (this rank's slice).
/// @param Sk        Chunk key/value length (this step's K/V tile, not the full Sk).
/// @param scale     Softmax scale, typically 1 / sqrt(D).
/// @param causal    If true, mask entries where `j_global > i_global`.
/// @param q_offset  Global position of the first local query row. Used only
///                  for causal masking; identifies where this Q slice sits
///                  in the full (pre-zigzag) sequence.
/// @param k_offset  Global position of the first key in this chunk. Combined
///                  with the local `j` index to recover `j_global` for the
///                  causal predicate. Lets the ring pass non-contiguous chunks
///                  (e.g. zig-zag) without changing the kernel.
template <int BR, int BC, int D>
__global__ void attention_step_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                      const float* __restrict__ V, float* __restrict__ O,
                                      float* __restrict__ M, float* __restrict__ L, int H, int kv_H,
                                      int Sq, int Sk, float scale, bool causal, int q_offset,
                                      int k_offset) {
  const int tid = threadIdx.x;
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  const int i_local = q_tile * BR + tid;
  const bool active = (i_local < Sq);

  const long head_q = ((long)b * H + h) * Sq * D;
  const long head_k = ((long)b * kv_H + (h % kv_H)) * Sk * D;
  const long row_idx = ((long)b * H + h) * Sq + i_local;

  __shared__ float Q_tile[BR * D];
  __shared__ float K_tile[BC * D];
  __shared__ float V_tile[BC * D];

  if (active) {
    const float* q_src = Q + head_q + (long)i_local * D;
    for (int d = 0; d < D; ++d) Q_tile[tid * D + d] = q_src[d];
  } else {
    for (int d = 0; d < D; ++d) Q_tile[tid * D + d] = 0.0f;
  }

  // Load persistent state for this row.
  float m_i = active ? M[row_idx] : -INFINITY;
  float l_i = active ? L[row_idx] : 0.0f;
  float O_i[D];
  if (active) {
    const float* o_src = O + head_q + (long)i_local * D;
#pragma unroll
    for (int d = 0; d < D; ++d) O_i[d] = o_src[d];
  } else {
#pragma unroll
    for (int d = 0; d < D; ++d) O_i[d] = 0.0f;
  }

  const int i_global = q_offset + i_local;
  const int num_k_tiles = (Sk + BC - 1) / BC;

  for (int kt = 0; kt < num_k_tiles; ++kt) {
    __syncthreads();
    const int j_base = kt * BC;

    for (int idx = tid; idx < BC * D; idx += BR) {
      const int j_local = idx / D;
      const int d = idx % D;
      const int j_local_in_chunk = j_base + j_local;
      if (j_local_in_chunk < Sk) {
        K_tile[idx] = K[head_k + (long)j_local_in_chunk * D + d];
        V_tile[idx] = V[head_k + (long)j_local_in_chunk * D + d];
      } else {
        K_tile[idx] = 0.0f;
        V_tile[idx] = 0.0f;
      }
    }
    __syncthreads();

    if (!active) continue;

    float s[BC];
    float m_new = m_i;
#pragma unroll
    for (int j = 0; j < BC; ++j) {
      const int j_local_in_chunk = j_base + j;
      const int j_global = k_offset + j_local_in_chunk;
      const bool visible = (j_local_in_chunk < Sk) && (!causal || (j_global <= i_global));
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

    if (m_new == -INFINITY) continue;

    const float alpha = expf(m_i - m_new);
    float row_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < BC; ++j) {
      s[j] = (s[j] == -INFINITY) ? 0.0f : expf(s[j] - m_new);
      row_sum += s[j];
    }

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
    float* o_dst = O + head_q + (long)i_local * D;
#pragma unroll
    for (int d = 0; d < D; ++d) o_dst[d] = O_i[d];
    M[row_idx] = m_i;
    L[row_idx] = l_i;
  }
}

template <int BR, int BC, int D>
void launch_step_typed(const float* q, const float* k, const float* v, float* out, float* m,
                       float* l, const AttentionShape& shape, int q_offset, int k_offset,
                       bool causal, cudaStream_t stream) {
  const dim3 grid(ceil_div(shape.seq_q, BR), shape.heads, shape.batch);
  const dim3 block(BR);
  const int kv_H = (shape.kv_heads > 0) ? shape.kv_heads : shape.heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  attention_step_kernel<BR, BC, D><<<grid, block, 0, stream>>>(q, k, v, out, m, l, shape.heads,
                                                               kv_H, shape.seq_q, shape.seq_k,
                                                               scale, causal, q_offset, k_offset);
  cudaCheck(cudaGetLastError());
}

}  // namespace

void launch_attention_init(float* out, float* m, float* l, const AttentionShape& shape,
                           std::size_t m_count, cudaStream_t stream) {
  const std::size_t out_count = tensor_numel(shape.batch, shape.heads, shape.seq_q, shape.head_dim);
  const std::size_t work = (out_count > m_count) ? out_count : m_count;
  const int block = 256;
  const int grid = static_cast<int>((work + block - 1) / block);
  init_kernel<<<grid, block, 0, stream>>>(out, m, l, out_count, m_count);
  cudaCheck(cudaGetLastError());
}

void launch_attention_step(const float* q, const float* k, const float* v, float* out, float* m,
                           float* l, const AttentionShape& shape, int q_offset, int k_offset,
                           bool causal, cudaStream_t stream) {
  switch (shape.head_dim) {
    case 32:
      launch_step_typed<64, 64, 32>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 64:
      launch_step_typed<32, 32, 64>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 128:
      launch_step_typed<32, 16, 128>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 256:
      launch_step_typed<16, 8, 256>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    default:
      fprintf(stderr, "attention_step: unsupported head_dim=%d (supported: 32, 64, 128, 256)\n",
              shape.head_dim);
      std::abort();
  }
}

void launch_attention_finalize(float* out, const float* l, const AttentionShape& shape,
                               cudaStream_t stream) {
  const dim3 grid(shape.seq_q, shape.heads, shape.batch);
  const dim3 block(128);
  finalize_kernel<<<grid, block, 0, stream>>>(out, l, shape.heads, shape.seq_q, shape.head_dim);
  cudaCheck(cudaGetLastError());
}

}  // namespace ring_attention
