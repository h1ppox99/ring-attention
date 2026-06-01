#pragma once

/// @file
/// Multi-rank ring decode: one query token vs. KV history distributed across
/// the ring. Each rank holds its own `DeviceKVCache`; the ring rotates K/V
/// shards while Q stays put.
///
/// Causal masking is implicit. The cache is constructed to only ever hold
/// tokens at positions ≤ the current decoded position, so attending to every
/// cached row is correct — kernels run with `causal=false`. This matches the
/// Python reference (`ring_attention_ref.inference.decode_step`).
///
/// Requires NCCL (build with `USE_NCCL=ON`). The MPI-only build path is not
/// supported for decode; the function aborts at runtime if NCCL is disabled.

#include <cstdint>
#include <vector>

#include "kv_cache.hpp"
#include "ring_loop.hpp"  // RingResult timing struct

#ifdef RING_USE_NCCL
#include <nccl.h>
#endif

namespace ring_attention {

/// Per-call config for `run_ring_decode_step`. Intentionally minimal — the
/// caller owns the cache, the comm, and the per-rank length array, and reuses
/// them across decode steps to avoid per-step setup cost.
struct RingDecodeConfig {
  int rank;
  int cp_size;
  int batch;
  int heads;
#ifdef RING_USE_NCCL
  ncclComm_t nccl_comm;  ///< Pre-initialized (e.g., via `nccl_init`). Reused across steps.
#endif
};

/// One ring decode step (fp32).
///
/// Steps performed:
///   1. If `cfg.rank == owner_rank`: append `(k_new, v_new)` to `cache`.
///   2. Initialize online-softmax state for one query row.
///   3. Pack the local cache into a transit buffer.
///   4. cp_size ring rotations: NCCL Send/Recv of K/V chunks, kernel call
///      per chunk against the replicated query.
///   5. Finalize (divide output by accumulated sum).
///
/// `current_len_per_rank` must reflect the cache size on every rank *after*
/// this step's append. The caller computes this deterministically given the
/// owner_rank policy (typically round-robin).
///
/// @param cfg                    Per-call config (rank, cp_size, B, H, NCCL comm).
/// @param q                      Query, `(B, H, 1, D)`, replicated on every rank.
/// @param k_new, v_new           New K/V row, `(B, kv_H, 1, D)`. Only read on
///                               `owner_rank`; pass any valid pointer elsewhere.
/// @param owner_rank             Which rank owns the new token's KV.
/// @param cache                  Mutable cache; appended on owner only.
/// @param current_len_per_rank   Size `cp_size`, post-append.
/// @param out                    Output, `(B, H, 1, D)`. After return, every
///                               rank holds the same value.
///
/// @return Per-call timings (`max_err` is left at -1; correctness is the
///         caller's responsibility).
RingResult run_ring_decode_step(const RingDecodeConfig& cfg, const float* q, const float* k_new,
                                const float* v_new, int owner_rank, DeviceKVCache<float>& cache,
                                const std::vector<int>& current_len_per_rank, float* out);

}  // namespace ring_attention
