/// @file
/// Multi-rank ring decode test. Runs under `mpirun -n N`.
///
/// Strategy:
///   1. Generate a full causal Q/K/V of length `prompt_len + n_decode` from a
///      shared seed on every rank.
///   2. On rank 0, compute the CPU oracle (`cpu_attention`, causal=true) over
///      the full sequence — this is the truth we compare against.
///   3. Sequentially partition the prompt K/V across ranks and bulk-load each
///      rank's `DeviceKVCache` via `copy_prefill`.
///   4. For each of `n_decode` autoregressive steps, choose a round-robin
///      owner rank, replicate Q on every rank, hand the new K/V to the owner,
///      and call `run_ring_decode_step`.
///   5. Every rank's output for the decode token must equal the same value
///      (ring rotation means each rank ultimately accumulates over every
///      cache shard). Rank 0 compares its result against the corresponding
///      row of the CPU oracle.
///
/// Pass criterion (printed to stdout for the ctest regex):
///     "ring_decode max_err=... PASS"

#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"
#include "kv_cache.hpp"
#include "ring_decode.hpp"

#ifdef RING_USE_NCCL
#include "nccl_utils.hpp"
#endif

using namespace ring_attention;

namespace {

/// Bind this rank to the right GPU using node-local rank (robust to launcher
/// rank-distribution policy).
int bind_gpu(int world_rank) {
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);
  int local_rank = 0;
  MPI_Comm_rank(node_comm, &local_rank);
  MPI_Comm_free(&node_comm);
  int n = 1;
  cudaGetDeviceCount(&n);
  const int device = local_rank % n;
  cudaSetDevice(device);
  return device;
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, cp_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &cp_size);
  bind_gpu(rank);

  // Defaults exercise enough variety to catch the obvious bugs. Override via
  // argv to widen the matrix in ad-hoc runs.
  int prompt_len = 32;
  int n_decode = 4;
  int B = 1, H = 2, D = 32;
  for (int i = 1; i + 1 < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--prompt_len")
      prompt_len = std::atoi(argv[++i]);
    else if (a == "--n_decode")
      n_decode = std::atoi(argv[++i]);
    else if (a == "--batch")
      B = std::atoi(argv[++i]);
    else if (a == "--heads")
      H = std::atoi(argv[++i]);
    else if (a == "--head_dim")
      D = std::atoi(argv[++i]);
  }
  if (prompt_len % cp_size != 0) {
    if (rank == 0)
      fprintf(stderr, "prompt_len=%d must be divisible by cp_size=%d\n", prompt_len, cp_size);
    MPI_Finalize();
    return 1;
  }
  const int S_total = prompt_len + n_decode;
  const int S_max = S_total + cp_size;  // headroom

#ifdef RING_USE_NCCL
  ncclComm_t comm = nccl_init(rank, cp_size);
#else
  if (rank == 0) fprintf(stderr, "Test requires NCCL build\n");
  MPI_Finalize();
  return 1;
#endif

  // 1. Generate full Q/K/V (identical on every rank).
  XorShift32 rng(42u);
  const std::size_t qn = static_cast<std::size_t>(B) * H * S_total * D;
  std::vector<float> q_full(qn), k_full(qn), v_full(qn);
  rng.fill_uniform(q_full);
  rng.fill_uniform(k_full);
  rng.fill_uniform(v_full);

  // 2. CPU oracle: full causal attention over the concatenated sequence.
  std::vector<float> o_full(qn);
  AttentionShape ref_sh{B, H, S_total, S_total, D};
  cpu_attention(q_full.data(), k_full.data(), v_full.data(), o_full.data(), ref_sh,
                /*causal=*/true);

  // 3. Build this rank's prefill K/V slice (sequential partition).
  const int Sp_per_rank = prompt_len / cp_size;
  std::vector<float> k_pref_h(static_cast<std::size_t>(B) * H * Sp_per_rank * D);
  std::vector<float> v_pref_h(static_cast<std::size_t>(B) * H * Sp_per_rank * D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t = 0; t < Sp_per_rank; ++t) {
        const int gpos = rank * Sp_per_rank + t;
        for (int d = 0; d < D; ++d) {
          const std::size_t dst = ((static_cast<std::size_t>(b) * H + h) * Sp_per_rank + t) * D + d;
          const std::size_t src = ((static_cast<std::size_t>(b) * H + h) * S_total + gpos) * D + d;
          k_pref_h[dst] = k_full[src];
          v_pref_h[dst] = v_full[src];
        }
      }
  DeviceTensor<float> k_pref_d(k_pref_h.size()), v_pref_d(v_pref_h.size());
  k_pref_d.copy_from_host(k_pref_h);
  v_pref_d.copy_from_host(v_pref_h);

  DeviceKVCache<float> cache(B, /*kv_heads=*/H, S_max, D);
  cache.copy_prefill(k_pref_d.data(), v_pref_d.data(), Sp_per_rank);

  std::vector<int> current_len(cp_size, Sp_per_rank);

  RingDecodeConfig cfg;
  cfg.rank = rank;
  cfg.cp_size = cp_size;
  cfg.batch = B;
  cfg.heads = H;
#ifdef RING_USE_NCCL
  cfg.nccl_comm = comm;
#endif

  // 4. Decode loop.
  float max_err = 0.f;
  const std::size_t row_elem = static_cast<std::size_t>(B) * H * D;
  for (int d_idx = 0; d_idx < n_decode; ++d_idx) {
    const int gpos = prompt_len + d_idx;
    const int owner = d_idx % cp_size;

    // Build Q for this token (replicated on every rank).
    std::vector<float> q_h(row_elem), kv_h(row_elem), kv_v_h(row_elem);
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h)
        for (int d = 0; d < D; ++d) {
          const std::size_t dst = (static_cast<std::size_t>(b) * H + h) * D + d;
          const std::size_t src = ((static_cast<std::size_t>(b) * H + h) * S_total + gpos) * D + d;
          q_h[dst] = q_full[src];
          kv_h[dst] = k_full[src];
          kv_v_h[dst] = v_full[src];
        }
    DeviceTensor<float> q_d(row_elem), k_new_d(row_elem), v_new_d(row_elem), out_d(row_elem);
    q_d.copy_from_host(q_h);
    if (rank == owner) {
      k_new_d.copy_from_host(kv_h);
      v_new_d.copy_from_host(kv_v_h);
    }

    current_len[owner] += 1;

    run_ring_decode_step(cfg, q_d.data(), k_new_d.data(), v_new_d.data(), owner, cache, current_len,
                         out_d.data());

    if (rank == 0) {
      std::vector<float> out_h;
      out_d.copy_to_host(out_h);
      for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h)
          for (int d = 0; d < D; ++d) {
            const std::size_t got_i = (static_cast<std::size_t>(b) * H + h) * D + d;
            const std::size_t exp_i =
                ((static_cast<std::size_t>(b) * H + h) * S_total + gpos) * D + d;
            max_err = std::max(max_err, std::fabs(out_h[got_i] - o_full[exp_i]));
          }
    }
  }

#ifdef RING_USE_NCCL
  ncclCommDestroy(comm);
#endif

  const float tol = 1e-3f;
  if (rank == 0) {
    const bool pass = (max_err < tol);
    printf("ring_decode max_err=%.3e tol=%.0e %s\n", max_err, tol, pass ? "PASS" : "FAIL");
    fflush(stdout);
    MPI_Finalize();
    return pass ? 0 : 1;
  }
  MPI_Finalize();
  return 0;
}
