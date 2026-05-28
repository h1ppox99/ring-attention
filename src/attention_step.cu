/// @file
/// Persistent-state online-softmax kernel: the building block called once per
/// ring step. The kernel reads (O, m, l) from global memory, processes one
/// K/V chunk, and writes the updated state back. A separate finalize divides
/// O by l after all chunks have been consumed.
///
/// Shared-memory / occupancy notes (driven by Nsight Compute feedback on
/// bank conflicts, MIO stalls, and SM occupancy):
///   1. Q lives in per-thread *registers* (`Q_reg[D]`), not shared memory.
///      Q is per-query-row and never shared across threads, so shared was the
///      wrong tier. `#pragma unroll` on the d loop keeps `Q_reg[d]` register-
///      indexed so nvcc does not lower the array to local memory. With Q out
///      of shared, per-block smem is just `K_tile + V_tile` = 16 KB
///      (independent of BR), so bumping BR adds warps "for free" — see (4).
///   2. The Q*K^T loop is iterated as `for d { q = Q_reg[d]; for j { s[j] += q*K[j][d]; } }`,
///      so Q comes from registers (no bank conflicts possible) and K is a
///      warp-wide broadcast (also conflict-free). Masking/scaling runs as a
///      small post-pass.
///   3. The inner matmul and the output accumulation read K_tile / V_tile via
///      float4 (LDS.128). Each access is still a warp-wide broadcast (1
///      wavefront), but the issued *instruction* count drops 4x, which
///      directly relieves the Short Scoreboard / MIO-queue stalls.
///   4. BR is sized so each block has 2-4 warps (was 1) — same smem footprint,
///      more resident warps per SM, lifting theoretical occupancy from 12.5%
///      to 25-37.5% on Turing for the D=64 / D=128 configs.
///   5. K/V cooperative loads use float4 (LDS.128 / STS.128) which issues 4x
///      fewer instructions and reduces MIO-queue pressure on the load path
///      too.

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
  static_assert(D % 4 == 0, "D must be a multiple of 4 for float4 K/V loads.");
  static_assert((BC * D) % (BR * 4) == 0,
                "Cooperative float4 K/V load assumes BC*D/4 divides evenly across BR threads.");

  const int tid = threadIdx.x;
  const int q_tile = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;

  const int i_local = q_tile * BR + tid;
  const bool active = (i_local < Sq);

  const long head_q = ((long)b * H + h) * Sq * D;
  const long head_k = ((long)b * kv_H + (h % kv_H)) * Sk * D;
  const long row_idx = ((long)b * H + h) * Sq + i_local;

  __shared__ float K_tile[BC * D];
  __shared__ float V_tile[BC * D];

  // Q lives in registers, one row per thread. With the d loop fully unrolled
  // below, `Q_reg[d]` is compile-time indexed and nvcc keeps the array in
  // registers (no local-memory spill).
  float Q_reg[D];
  if (active) {
    const float* q_src = Q + head_q + (long)i_local * D;
#pragma unroll
    for (int d = 0; d < D; ++d) Q_reg[d] = q_src[d];
  } else {
#pragma unroll
    for (int d = 0; d < D; ++d) Q_reg[d] = 0.0f;
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

  // Each float4 iteration moves 4 contiguous floats of K (and V) from global
  // to shared. With our (BR, BC, D) instantiations the loop count divides
  // evenly across BR threads.
  constexpr int kTileVec4 = (BC * D) / 4;

  for (int kt = 0; kt < num_k_tiles; ++kt) {
    __syncthreads();
    const int j_base = kt * BC;

    // Cooperative K/V load with float4 (LDS.128 / STS.128). 4x fewer shared
    // store instructions than a per-float loop, easing MIO pressure.
    for (int idx4 = tid; idx4 < kTileVec4; idx4 += BR) {
      const int idx = idx4 * 4;
      const int j_local = idx / D;
      const int d = idx - j_local * D;
      const int j_local_in_chunk = j_base + j_local;
      float4 k4;
      float4 v4;
      if (j_local_in_chunk < Sk) {
        k4 = *reinterpret_cast<const float4*>(K + head_k + (long)j_local_in_chunk * D + d);
        v4 = *reinterpret_cast<const float4*>(V + head_k + (long)j_local_in_chunk * D + d);
      } else {
        k4 = float4{0.0f, 0.0f, 0.0f, 0.0f};
        v4 = float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
      *reinterpret_cast<float4*>(K_tile + idx) = k4;
      *reinterpret_cast<float4*>(V_tile + idx) = v4;
    }
    __syncthreads();

    if (!active) continue;

    // Q . K^T matmul. Q comes from registers (`Q_reg[d]`); K is a warp-wide
    // broadcast load from shared. We walk d in groups of 4 and read K_tile via
    // float4 (LDS.128) — same broadcast wavefront count, 4x fewer issued
    // shared-load instructions, which is what the Short Scoreboard waits on.
    float s[BC];
#pragma unroll
    for (int j = 0; j < BC; ++j) s[j] = 0.0f;

#pragma unroll
    for (int d4 = 0; d4 < D / 4; ++d4) {
      const float q0 = Q_reg[d4 * 4 + 0];
      const float q1 = Q_reg[d4 * 4 + 1];
      const float q2 = Q_reg[d4 * 4 + 2];
      const float q3 = Q_reg[d4 * 4 + 3];
#pragma unroll
      for (int j = 0; j < BC; ++j) {
        const float4 k4 = *reinterpret_cast<const float4*>(&K_tile[j * D + d4 * 4]);
        s[j] += q0 * k4.x + q1 * k4.y + q2 * k4.z + q3 * k4.w;
      }
    }

    // Apply scale + bounds/causal mask, find tile-local row max.
    // K_tile rows for j_local_in_chunk >= Sk were zeroed in the cooperative
    // load, so s[j] for those rows is 0; we still overwrite with -INFINITY.
    float m_new = m_i;
#pragma unroll
    for (int j = 0; j < BC; ++j) {
      const int j_local_in_chunk = j_base + j;
      const int j_global = k_offset + j_local_in_chunk;
      const bool visible = (j_local_in_chunk < Sk) && (!causal || (j_global <= i_global));
      if (visible) {
        s[j] *= scale;
        if (s[j] > m_new) m_new = s[j];
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

    // Output accumulation: O_i <- alpha * O_i + sum_j s[j] * V_j.
    // V_tile broadcast read via float4 — same idea as the inner matmul: 4x
    // fewer issued shared-load instructions.
#pragma unroll
    for (int d4 = 0; d4 < D / 4; ++d4) {
      float4 acc;
      acc.x = alpha * O_i[d4 * 4 + 0];
      acc.y = alpha * O_i[d4 * 4 + 1];
      acc.z = alpha * O_i[d4 * 4 + 2];
      acc.w = alpha * O_i[d4 * 4 + 3];
#pragma unroll
      for (int j = 0; j < BC; ++j) {
        const float4 v4 = *reinterpret_cast<const float4*>(&V_tile[j * D + d4 * 4]);
        acc.x += s[j] * v4.x;
        acc.y += s[j] * v4.y;
        acc.z += s[j] * v4.z;
        acc.w += s[j] * v4.w;
      }
      O_i[d4 * 4 + 0] = acc.x;
      O_i[d4 * 4 + 1] = acc.y;
      O_i[d4 * 4 + 2] = acc.z;
      O_i[d4 * 4 + 3] = acc.w;
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
  // BR is sized so each block has 2-4 warps; per-block smem is just K+V
  // (16 KB) regardless of BR, so packing more warps per block lifts occupancy
  // without changing the smem footprint.
  // D=256 keeps BR=16 because Q_reg[256] already spills heavily; bigger BR
  // would multiply local-memory traffic across more threads.
  switch (shape.head_dim) {
    case 32:
      launch_step_typed<128, 64, 32>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 64:
      launch_step_typed<64, 32, 64>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 128:
      launch_step_typed<64, 16, 128>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
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
