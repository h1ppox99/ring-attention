/// @file
/// Decode-specialized step kernel (seq_q = 1).
///
/// Block = 4 warps = 128 threads. Each block still owns one (batch, head)
/// row, but the K loop is now block-partitioned across the 4 warps — warp w
/// streams K rows `[w·Sk/4, (w+1)·Sk/4)` with its own independent online-
/// softmax accumulator. After the K loop each warp posts its partial
/// `(m, ℓ, O)` to shared memory; warp 0 reads all four, folds them into the
/// existing persistent global state, and writes back. The 32 lanes inside a
/// warp still split the D dimension (Q in registers, K/V streamed via warp-
/// coalesced LDGs, dot product via `__shfl_xor_sync`).
///
/// Why multi-warp: Round 5 left grid size at `B × H` (= 8 blocks for the
/// production shape), so only 8 of 72 Turing SMs saw any work. Round 8 keeps
/// the grid but quadruples the warps per block, raising per-SM warp residency
/// from 1 → 4 — the same 4× memory parallelism a split-K design would have
/// provided across blocks, achieved here without a separate reduction kernel.
///
/// API matches `launch_attention_step`: same (M, L, O) persistent state, so
/// the ring decode loop can call it once per ring step without changing the
/// init / finalize pair.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "attention.hpp"
#include "common.cuh"

namespace ring_attention {

namespace {

/// One warp per (batch, head) row. Each lane holds D/32 elements of Q and O.
template <int D>
__global__ void attention_decode_step_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V, float* __restrict__ O,
                                             float* __restrict__ M, float* __restrict__ L, int H,
                                             int kv_H, int Sk, float scale, bool causal,
                                             int q_offset, int k_offset) {
  static_assert(D % 32 == 0, "D must be a multiple of warp size (32).");
  constexpr int VPL = D / 32;   // values per lane
  constexpr int NUM_WARPS = 4;  // warps per block

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp_id = tid >> 5;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int h_kv = h % kv_H;

  const long head_q = ((long)b * H + h) * D;  // seq_q=1, so no per-row stride
  const long head_k = ((long)b * kv_H + h_kv) * Sk * D;
  const long row_idx = (long)b * H + h;
  float* o_ptr = O + head_q;

  // Q in registers: lane k holds Q[k], Q[k+32], Q[k+64], ...
  // Every warp loads the same Q — the broadcast is cheap and avoids a
  // shared-memory shuffle here.
  const float* q_ptr = Q + head_q;
  float Q_reg[VPL];
#pragma unroll
  for (int v = 0; v < VPL; ++v) Q_reg[v] = q_ptr[v * 32 + lane];

  // Determine the effective Sk after causal pruning. q_offset/k_offset are
  // block-uniform, so the predicate `k_offset + j > q_offset` resolves to a
  // single cutoff for the entire block.
  const int i_global = q_offset;
  int Sk_eff = Sk;
  if (causal) {
    const int max_visible = i_global - k_offset + 1;
    if (max_visible < 0)
      Sk_eff = 0;
    else if (max_visible < Sk)
      Sk_eff = max_visible;
  }

  // Block-partition: warp w handles K rows [k_warp_start, k_warp_end).
  // Each warp has its own independent online-softmax accumulator over its
  // K range; the four partials get merged after the loop.
  const int Sk_per_warp = (Sk_eff + NUM_WARPS - 1) / NUM_WARPS;
  const int k_warp_start = warp_id * Sk_per_warp;
  const int k_warp_end =
      (k_warp_start + Sk_per_warp < Sk_eff) ? (k_warp_start + Sk_per_warp) : Sk_eff;

  float m_part = -INFINITY;
  float l_part = 0.0f;
  float O_part[VPL];
#pragma unroll
  for (int v = 0; v < VPL; ++v) O_part[v] = 0.0f;

  if (k_warp_end > k_warp_start) {
    const float* k_base = K + head_k;
    const float* v_base = V + head_k;

    // Register-level double-buffered prefetch — same scheme as Round 6, now
    // run independently inside each warp on its K sub-range.
    float K_buf[2][VPL];
    float V_buf[2][VPL];
#pragma unroll
    for (int v = 0; v < VPL; ++v) {
      K_buf[0][v] = k_base[(long)k_warp_start * D + v * 32 + lane];
      V_buf[0][v] = v_base[(long)k_warp_start * D + v * 32 + lane];
    }
    int cur = 0;

    for (int j = k_warp_start; j < k_warp_end; ++j) {
      if (j + 1 < k_warp_end) {
        const int nxt = cur ^ 1;
        const float* k_next = k_base + (long)(j + 1) * D;
        const float* v_next = v_base + (long)(j + 1) * D;
#pragma unroll
        for (int v = 0; v < VPL; ++v) {
          K_buf[nxt][v] = k_next[v * 32 + lane];
          V_buf[nxt][v] = v_next[v * 32 + lane];
        }
      }

      // Q · K[j] partial: each lane multiplies its VPL elements.
      float partial = 0.0f;
#pragma unroll
      for (int v = 0; v < VPL; ++v) partial += Q_reg[v] * K_buf[cur][v];

        // Warp XOR shuffle reduction.
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
      const float s_j = partial * scale;

      // Online softmax update on this warp's partial.
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

  // Stage partials in shared memory, then warp 0 folds them into the
  // existing persistent (M, L, O) state and writes back. Cost: one
  // __syncthreads + a 4-iteration serial merge — negligible vs the K loop.
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
    float m_combined = M[row_idx];
    float l_combined = L[row_idx];
    float O_combined[VPL];
#pragma unroll
    for (int v = 0; v < VPL; ++v) O_combined[v] = o_ptr[v * 32 + lane];

#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      const float m_w = m_shared[w];
      // Skip empty partials (warps with k_warp_end == k_warp_start). Without
      // this guard, m_combined == m_w == -inf produces NaN in the rescale.
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
    for (int v = 0; v < VPL; ++v) o_ptr[v * 32 + lane] = O_combined[v];
    if (lane == 0) {
      M[row_idx] = m_combined;
      L[row_idx] = l_combined;
    }
  }
}

template <int D>
void launch_decode_typed(const float* q, const float* k, const float* v, float* out, float* m,
                         float* l, const AttentionShape& shape, int q_offset, int k_offset,
                         bool causal, cudaStream_t stream) {
  const int kv_H = (shape.kv_heads > 0) ? shape.kv_heads : shape.heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const dim3 grid(shape.heads, shape.batch);
  const dim3 block(128);  // 4 warps × 32 lanes — see file header.
  attention_decode_step_kernel<D><<<grid, block, 0, stream>>>(
      q, k, v, out, m, l, shape.heads, kv_H, shape.seq_k, scale, causal, q_offset, k_offset);
  cudaCheck(cudaGetLastError());
}

}  // namespace

void launch_attention_decode_step(const float* q, const float* k, const float* v, float* out,
                                  float* m, float* l, const AttentionShape& shape, int q_offset,
                                  int k_offset, bool causal, cudaStream_t stream) {
  switch (shape.head_dim) {
    case 32:
      launch_decode_typed<32>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 64:
      launch_decode_typed<64>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 128:
      launch_decode_typed<128>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    case 256:
      launch_decode_typed<256>(q, k, v, out, m, l, shape, q_offset, k_offset, causal, stream);
      break;
    default:
      fprintf(stderr,
              "attention_decode_step: unsupported head_dim=%d (supported: 32, 64, 128, 256)\n",
              shape.head_dim);
      std::abort();
  }
}

}  // namespace ring_attention
