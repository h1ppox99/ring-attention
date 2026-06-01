/// @file
/// Decode-specialized step kernel (seq_q = 1) with split-K + reduce.
///
/// Round 9 splits the K dimension across blocks. Grid expands from
/// `(H, B)` = 8 blocks (Round 8) to `(K_SPLIT, H, B)` = 64 blocks, where
/// each block processes `Sk / K_SPLIT` rows of K with the same 4-warp /
/// register-prefetch pipeline as Round 8 and writes a per-block partial
/// `(m, ℓ, O)` to global memory. A small reduce kernel then folds the
/// `K_SPLIT` partials into the persistent `(M, L, O)` state.
///
/// Per block (split kernel):
///   - 4 warps × 32 lanes; intra-block K loop partitioned across warps.
///   - Q in registers (D/32 floats / lane).
///   - K/V streamed via warp-coalesced LDG with 2-slot register prefetch
///     (Round 6 scheme).
///   - Online softmax accumulated in each warp's own (m_w, ℓ_w, O_w).
///   - Intra-block 4→1 merge in shared memory, then warp 0 writes the
///     per-block partial (m_b, ℓ_b, O_b) to the workspace.
///
/// Per block (reduce kernel):
///   - 1 warp = 32 lanes. Grid `(H, B)` = one block per (b, h) row.
///   - Reads existing persistent (M, L, O), folds in K_SPLIT partials
///     sequentially, writes (M, L, O) back.
///
/// The reduce kernel preserves the same persistent-state semantics
/// `launch_attention_step` provides — multi-step ring decode accumulates
/// across calls without API changes.
///
/// History: Round 5 introduced the warp-cooperative D reduction (lifted
/// avg active threads 2.5 → 32 / warp). Round 6 added 2-slot register
/// prefetch. Round 8 added multi-warp blocks (4× SM warp residency at
/// fixed grid). Round 9 adds split-K — the structural lever for the
/// 64-of-72-SMs-idle problem Round 8 left in place.

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "attention.hpp"
#include "common.cuh"

namespace ring_attention {

namespace {

constexpr int K_SPLIT = 8;
constexpr int NUM_WARPS = 4;

// Process-static decode workspace. Holds [M_partial | L_partial | O_partial]
// contiguously. Grows on demand; never freed (released by the CUDA driver at
// program exit). Single-stream / single-thread use only — see Round 10 in
// KERNEL_OPTIMIZATIONS.md for the trade-off and the path to a session-level
// allocator.
float* g_decode_workspace = nullptr;
std::size_t g_decode_workspace_floats = 0;

float* ensure_decode_workspace(std::size_t needed_floats) {
  if (needed_floats > g_decode_workspace_floats) {
    if (g_decode_workspace != nullptr) cudaCheck(cudaFree(g_decode_workspace));
    cudaCheck(cudaMalloc(&g_decode_workspace, needed_floats * sizeof(float)));
    g_decode_workspace_floats = needed_floats;
  }
  return g_decode_workspace;
}

/// Per-block split-K compute kernel.
///
/// Grid: `(K_SPLIT, H, B)`. Block: 4 warps × 32 lanes = 128 threads.
/// Each block processes K rows `[split_id · Sk/K_SPLIT, (split_id+1) · Sk/K_SPLIT)`
/// (clipped by Sk and the causal cutoff) and writes one partial
/// `(m, ℓ, O)` tuple at `[split_id, b, h]` of the workspace buffers.
template <int D>
__global__ void attention_decode_split_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ M_partial, float* __restrict__ L_partial, float* __restrict__ O_partial,
    int H, int kv_H, int Sk, int kv_stride, float scale, bool causal, int q_offset, int k_offset) {
  static_assert(D % 32 == 0, "D must be a multiple of warp size (32).");
  constexpr int VPL = D / 32;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int split_id = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int B = gridDim.z;
  const int h_kv = h % kv_H;

  const long head_q = ((long)b * H + h) * D;
  // kv_stride is the per-head row stride of the K/V layout: Sk for a contiguous
  // packed tile, or the cache's S_max when reading the KV cache in place
  // (Round 17 — skips the per-token de-stride copy).
  const long head_k = ((long)b * kv_H + h_kv) * (long)kv_stride * D;
  const long partial_row = ((long)split_id * B + b) * H + h;
  const long partial_head = partial_row * D;

  // Q in registers — all 4 warps load the same Q.
  const float* q_ptr = Q + head_q;
  float Q_reg[VPL];
#pragma unroll
  for (int v = 0; v < VPL; ++v) Q_reg[v] = q_ptr[v * 32 + lane];

  // K range for this split, with causal pruning applied at the split boundary.
  const int k_per_split = (Sk + K_SPLIT - 1) / K_SPLIT;
  const int split_start = split_id * k_per_split;
  const int split_end_raw = split_start + k_per_split;
  const int split_end = (split_end_raw < Sk) ? split_end_raw : Sk;
  int split_eff_end = split_end;
  if (causal) {
    const int max_visible = q_offset - k_offset + 1;
    if (max_visible <= split_start)
      split_eff_end = split_start;  // entire split masked
    else if (max_visible < split_end)
      split_eff_end = max_visible;
  }

  // Partition the split's K range across the 4 warps.
  const int Sk_split = split_eff_end - split_start;
  const int Sk_per_warp = (Sk_split + NUM_WARPS - 1) / NUM_WARPS;
  const int warp_k_start = split_start + warp_id * Sk_per_warp;
  const int warp_k_end_raw = warp_k_start + Sk_per_warp;
  const int warp_k_end = (warp_k_end_raw < split_eff_end) ? warp_k_end_raw : split_eff_end;

  float m_part = -INFINITY;
  float l_part = 0.0f;
  float O_part[VPL];
#pragma unroll
  for (int v = 0; v < VPL; ++v) O_part[v] = 0.0f;

  if (warp_k_end > warp_k_start) {
    const float* k_base = K + head_k;
    const float* v_base = V + head_k;

    // 2-slot register prefetch (Round 6 scheme).
    float K_buf[2][VPL];
    float V_buf[2][VPL];
#pragma unroll
    for (int v = 0; v < VPL; ++v) {
      K_buf[0][v] = k_base[(long)warp_k_start * D + v * 32 + lane];
      V_buf[0][v] = v_base[(long)warp_k_start * D + v * 32 + lane];
    }
    int cur = 0;

    for (int j = warp_k_start; j < warp_k_end; ++j) {
      if (j + 1 < warp_k_end) {
        const int nxt = cur ^ 1;
        const float* k_next = k_base + (long)(j + 1) * D;
        const float* v_next = v_base + (long)(j + 1) * D;
#pragma unroll
        for (int v = 0; v < VPL; ++v) {
          K_buf[nxt][v] = k_next[v * 32 + lane];
          V_buf[nxt][v] = v_next[v * 32 + lane];
        }
      }

      float partial = 0.0f;
#pragma unroll
      for (int v = 0; v < VPL; ++v) partial += Q_reg[v] * K_buf[cur][v];

#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
      const float s_j = partial * scale;

      const float m_new = (s_j > m_part) ? s_j : m_part;
      const float alpha = expf(m_part - m_new);
      const float p_j = expf(s_j - m_new);
      l_part = alpha * l_part + p_j;
      m_part = m_new;

#pragma unroll
      for (int v = 0; v < VPL; ++v) O_part[v] = alpha * O_part[v] + p_j * V_buf[cur][v];

      cur ^= 1;
    }
  }

  // Intra-block merge: 4 warps → 1 partial. Warp 0 writes to the workspace.
  __shared__ float m_shared[NUM_WARPS];
  __shared__ float l_shared[NUM_WARPS];
  __shared__ float O_shared[NUM_WARPS][D];

  if (lane == 0) {
    m_shared[warp_id] = m_part;
    l_shared[warp_id] = l_part;
  }
#pragma unroll
  for (int v = 0; v < VPL; ++v) O_shared[warp_id][v * 32 + lane] = O_part[v];

  __syncthreads();

  if (warp_id == 0) {
    float m_combined = -INFINITY;
    float l_combined = 0.0f;
    float O_combined[VPL];
#pragma unroll
    for (int v = 0; v < VPL; ++v) O_combined[v] = 0.0f;

#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      const float m_w = m_shared[w];
      if (m_w == -INFINITY) continue;
      const float l_w = l_shared[w];
      const float m_new = (m_w > m_combined) ? m_w : m_combined;
      const float alpha_c = expf(m_combined - m_new);
      const float alpha_w = expf(m_w - m_new);
      l_combined = alpha_c * l_combined + alpha_w * l_w;
#pragma unroll
      for (int v = 0; v < VPL; ++v)
        O_combined[v] = alpha_c * O_combined[v] + alpha_w * O_shared[w][v * 32 + lane];
      m_combined = m_new;
    }

#pragma unroll
    for (int v = 0; v < VPL; ++v) O_partial[partial_head + v * 32 + lane] = O_combined[v];
    if (lane == 0) {
      M_partial[partial_row] = m_combined;
      L_partial[partial_row] = l_combined;
    }
  }
}

/// Reduce kernel: folds K_SPLIT partials into the existing (M, L, O).
///
/// Grid: `(H, B)`. Block: 1 warp = 32 lanes. Each block owns one (b, h) row;
/// the 32 lanes split D the same way as the split kernel.
template <int D>
__global__ void attention_decode_reduce_kernel(const float* __restrict__ M_partial,
                                               const float* __restrict__ L_partial,
                                               const float* __restrict__ O_partial,
                                               float* __restrict__ M, float* __restrict__ L,
                                               float* __restrict__ O, int H) {
  static_assert(D % 32 == 0, "D must be a multiple of warp size (32).");
  constexpr int VPL = D / 32;

  const int lane = threadIdx.x;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int B = gridDim.y;

  const long row_idx = (long)b * H + h;
  const long head_idx = row_idx * D;
  float* o_ptr = O + head_idx;

  // Read existing persistent state.
  float m_combined = M[row_idx];
  float l_combined = L[row_idx];
  float O_combined[VPL];
#pragma unroll
  for (int v = 0; v < VPL; ++v) O_combined[v] = o_ptr[v * 32 + lane];

    // Fold in K_SPLIT partials sequentially.
#pragma unroll
  for (int s = 0; s < K_SPLIT; ++s) {
    const long partial_row = ((long)s * B + b) * H + h;
    const long partial_head = partial_row * D;
    const float m_s = M_partial[partial_row];
    if (m_s == -INFINITY) continue;
    const float l_s = L_partial[partial_row];
    const float m_new = (m_s > m_combined) ? m_s : m_combined;
    const float alpha_c = expf(m_combined - m_new);
    const float alpha_s = expf(m_s - m_new);
    l_combined = alpha_c * l_combined + alpha_s * l_s;
#pragma unroll
    for (int v = 0; v < VPL; ++v)
      O_combined[v] = alpha_c * O_combined[v] + alpha_s * O_partial[partial_head + v * 32 + lane];
    m_combined = m_new;
  }

#pragma unroll
  for (int v = 0; v < VPL; ++v) o_ptr[v * 32 + lane] = O_combined[v];
  if (lane == 0) {
    M[row_idx] = m_combined;
    L[row_idx] = l_combined;
  }
}

template <int D>
void launch_decode_typed(const float* q, const float* k, const float* v, float* out, float* m,
                         float* l, const AttentionShape& shape, int q_offset, int k_offset,
                         bool causal, int kv_row_stride, cudaStream_t stream) {
  const int kv_H = (shape.kv_heads > 0) ? shape.kv_heads : shape.heads;
  // 0 ⇒ contiguous (stride == row count); else the caller's row stride (e.g.
  // the KV cache's S_max so the kernel can read the cache in place).
  const int kv_stride = (kv_row_stride > 0) ? kv_row_stride : shape.seq_k;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int B = shape.batch;
  const int H = shape.heads;

  // Workspace layout: [M_partial | L_partial | O_partial], all contiguous.
  // Production shape (K_SPLIT=8, B=1, H=8, D=128) wants 8 320 floats = 32.5
  // KB. The first call cudaMallocs; subsequent calls at the same or smaller
  // shape reuse the buffer for free.
  const std::size_t partial_count = static_cast<std::size_t>(K_SPLIT) * B * H;
  const std::size_t needed = 2 * partial_count + partial_count * D;
  float* ws = ensure_decode_workspace(needed);
  float* M_partial = ws;
  float* L_partial = ws + partial_count;
  float* O_partial = ws + 2 * partial_count;

  // Compute: split-K kernel.
  const dim3 grid_split(K_SPLIT, H, B);
  const dim3 block_split(128);  // 4 warps × 32 lanes.
  attention_decode_split_kernel<D><<<grid_split, block_split, 0, stream>>>(
      q, k, v, M_partial, L_partial, O_partial, H, kv_H, shape.seq_k, kv_stride, scale, causal,
      q_offset, k_offset);
  cudaCheck(cudaGetLastError());

  // Reduce: fold K_SPLIT partials into persistent state.
  const dim3 grid_reduce(H, B);
  const dim3 block_reduce(32);
  attention_decode_reduce_kernel<D>
      <<<grid_reduce, block_reduce, 0, stream>>>(M_partial, L_partial, O_partial, m, l, out, H);
  cudaCheck(cudaGetLastError());
}

}  // namespace

void launch_attention_decode_step(const float* q, const float* k, const float* v, float* out,
                                  float* m, float* l, const AttentionShape& shape, int q_offset,
                                  int k_offset, bool causal, cudaStream_t stream,
                                  int kv_row_stride) {
  switch (shape.head_dim) {
    case 32:
      launch_decode_typed<32>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, kv_row_stride,
                              stream);
      break;
    case 64:
      launch_decode_typed<64>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, kv_row_stride,
                              stream);
      break;
    case 128:
      launch_decode_typed<128>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, kv_row_stride,
                               stream);
      break;
    case 256:
      launch_decode_typed<256>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, kv_row_stride,
                               stream);
      break;
    default:
      fprintf(stderr,
              "attention_decode_step: unsupported head_dim=%d (supported: 32, 64, 128, 256)\n",
              shape.head_dim);
      std::abort();
  }
}

}  // namespace ring_attention
