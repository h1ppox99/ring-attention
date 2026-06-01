/// @file
/// Ring-attention distributed driver. One MPI rank per GPU.

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "attention.hpp"
#include "common.cuh"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"
#include "kv_cache.hpp"
#include "ring_decode.hpp"
#include "ring_loop.hpp"

#ifdef RING_USE_NCCL
#include "nccl_utils.hpp"
#endif

// ---------------------------------------------------------------------------
// Error macros — MPI-aware so any failure tears down all ranks, not just one.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                                        \
  do {                                                                                          \
    cudaError_t _e = (expr);                                                                    \
    if (_e != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                                  \
    }                                                                                           \
  } while (0)

#define MPI_CHECK(expr)                                                      \
  do {                                                                       \
    int _e = (expr);                                                         \
    if (_e != MPI_SUCCESS) {                                                 \
      char _msg[MPI_MAX_ERROR_STRING];                                       \
      int _len;                                                              \
      MPI_Error_string(_e, _msg, &_len);                                     \
      fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, _msg); \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                               \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
  int batch = 1;
  int heads = 4;
  int seq = 512;  // total sequence length; must be divisible by cp_size
  int head_dim = 64;
  int kv_heads = 0;  // 0 = MHA; set for GQA/MQA
  bool causal = false;
  bool zigzag = false;
  // Default to the only mode worth running in production. The other two are
  // kept as baselines for the KERNEL_OPTIMIZATIONS.md comparison story; explicit
  // opt-in via --mode is required to select them.
  std::string mode = "ring-overlap";
  std::string dtype = "fp32";  // fp32 | fp16
  int iters = 10;
  bool verify = false;
  bool csv = false;
  bool csv_header = false;  // emit header row only (paired with --csv or alone)
  // Decode mode (--run decode): synthetic decode benchmark. The KV cache is
  // pre-filled with random data; correctness is validated by the dedicated
  // test_ring_decode executable, not by this benchmark driver.
  std::string run = "prefill";  // prefill | decode
  int prompt_len = 256;         // for --run decode: cache size at decode time
  int decode_tokens = 8;        // for --run decode: how many tokens to generate
};

namespace {

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const bool nxt = (i + 1 < argc);
    if (!std::strcmp(argv[i], "--batch") && nxt)
      cfg.batch = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--heads") && nxt)
      cfg.heads = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seq") && nxt)
      cfg.seq = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--head_dim") && nxt)
      cfg.head_dim = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--kv_heads") && nxt)
      cfg.kv_heads = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--causal") && nxt)
      cfg.causal = std::atoi(argv[++i]) != 0;
    else if (!std::strcmp(argv[i], "--zigzag") && nxt)
      cfg.zigzag = std::atoi(argv[++i]) != 0;
    else if (!std::strcmp(argv[i], "--mode") && nxt)
      cfg.mode = argv[++i];
    else if (!std::strcmp(argv[i], "--dtype") && nxt)
      cfg.dtype = argv[++i];
    else if (!std::strcmp(argv[i], "--iters") && nxt)
      cfg.iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--verify"))
      cfg.verify = true;
    else if (!std::strcmp(argv[i], "--csv"))
      cfg.csv = true;
    else if (!std::strcmp(argv[i], "--csv-header"))
      cfg.csv_header = true;
    else if (!std::strcmp(argv[i], "--run") && nxt)
      cfg.run = argv[++i];
    else if (!std::strcmp(argv[i], "--prompt-len") && nxt)
      cfg.prompt_len = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--decode-tokens") && nxt)
      cfg.decode_tokens = std::atoi(argv[++i]);
  }
  return cfg;
}

/// Synthetic ring-decode benchmark.
///
/// Pre-fills every rank's KV cache with random data sized to `prompt_len / cp_size`,
/// then runs `decode_tokens` decode steps with round-robin owner. Reports
/// per-token timings averaged over the run. Correctness is *not* checked here
/// (the test_ring_decode executable owns that); this is a timing tool.
///
/// Output: one CSV row per token if `--csv` is set, plus a summary line on
/// rank 0. CSV columns:
///     run,cp_size,prompt_len,decode_token_idx,context_len,
///     comm_ms,comp_ms,total_ms
///
/// CSV header (when paired with --csv-header):
///     run,cp_size,prompt_len,decode_token_idx,context_len,comm_ms,comp_ms,total_ms
void run_decode_benchmark(const Config& cfg, int rank, int cp_size) {
#ifndef RING_USE_NCCL
  if (rank == 0) fprintf(stderr, "--run decode requires NCCL build\n");
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
  using namespace ring_attention;

  if (cfg.prompt_len % cp_size != 0) {
    if (rank == 0)
      fprintf(stderr, "ERROR: prompt-len=%d not divisible by cp_size=%d\n", cfg.prompt_len,
              cp_size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int Sp_per_rank = cfg.prompt_len / cp_size;
  const int S_max = cfg.prompt_len + cfg.decode_tokens + cp_size;
  const std::size_t row_elem = static_cast<std::size_t>(B) * H * D;
  const std::size_t kv_row_elem = static_cast<std::size_t>(B) * kv_H * D;
  const std::size_t kv_pref_elem = static_cast<std::size_t>(B) * kv_H * Sp_per_rank * D;

  // Random data for the synthetic cache + per-step Q/K/V. Same RNG seed on
  // every rank so the prefill section is consistent across ranks.
  XorShift32 rng(cfg.batch * 31u + rank * 17u + 1u);
  std::vector<float> pref_host(kv_pref_elem);
  rng.fill_uniform(pref_host);
  DeviceTensor<float> pref_d(kv_pref_elem);
  pref_d.copy_from_host(pref_host);

  DeviceKVCache<float> cache(B, kv_H, S_max, D);
  cache.copy_prefill(pref_d.data(), pref_d.data(), Sp_per_rank);

  ncclComm_t comm = nccl_init(rank, cp_size);
  RingDecodeConfig dcfg;
  dcfg.rank = rank;
  dcfg.cp_size = cp_size;
  dcfg.batch = B;
  dcfg.heads = H;
  dcfg.nccl_comm = comm;

  std::vector<int> current_len(cp_size, Sp_per_rank);

  // Pre-allocate per-step buffers — Q replicated, K/V used only on owner.
  DeviceTensor<float> q_d(row_elem), k_new_d(kv_row_elem), v_new_d(kv_row_elem), out_d(row_elem);
  std::vector<float> q_h(row_elem), k_h(kv_row_elem), v_h(kv_row_elem);

  // Warmup step so cudaMalloc and NCCL setup don't pollute the first timing.
  {
    rng.fill_uniform(q_h);
    rng.fill_uniform(k_h);
    rng.fill_uniform(v_h);
    q_d.copy_from_host(q_h);
    k_new_d.copy_from_host(k_h);
    v_new_d.copy_from_host(v_h);
    std::vector<int> tmp_lens = current_len;
    tmp_lens[0] += 1;
    DeviceKVCache<float> tmp_cache(B, kv_H, S_max, D);
    tmp_cache.copy_prefill(pref_d.data(), pref_d.data(), Sp_per_rank);
    run_ring_decode_step(dcfg, q_d.data(), k_new_d.data(), v_new_d.data(), /*owner=*/0, tmp_cache,
                         tmp_lens, out_d.data());
  }

  if (rank == 0 && cfg.csv_header) {
    printf("run,cp_size,prompt_len,decode_token_idx,context_len,comm_ms,comp_ms,total_ms\n");
  }

  cudaProfilerStart();
  double sum_total = 0.0, sum_comp = 0.0, sum_comm = 0.0;
  for (int t = 0; t < cfg.decode_tokens; ++t) {
    rng.fill_uniform(q_h);
    rng.fill_uniform(k_h);
    rng.fill_uniform(v_h);
    q_d.copy_from_host(q_h);
    const int owner = t % cp_size;
    if (rank == owner) {
      k_new_d.copy_from_host(k_h);
      v_new_d.copy_from_host(v_h);
    }
    current_len[owner] += 1;
    const RingResult res = run_ring_decode_step(dcfg, q_d.data(), k_new_d.data(), v_new_d.data(),
                                                owner, cache, current_len, out_d.data());
    sum_total += res.total_ms;
    sum_comp += res.comp_ms;
    sum_comm += res.comm_ms;

    if (rank == 0 && cfg.csv) {
      const int ctx_len = std::accumulate(current_len.begin(), current_len.end(), 0);
      printf("decode,%d,%d,%d,%d,%.4f,%.4f,%.4f\n", cp_size, cfg.prompt_len, t, ctx_len,
             res.comm_ms, res.comp_ms, res.total_ms);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  cudaProfilerStop();
  ncclCommDestroy(comm);

  if (rank == 0) {
    const double mean_total = sum_total / cfg.decode_tokens;
    const double mean_comp = sum_comp / cfg.decode_tokens;
    const double mean_comm = sum_comm / cfg.decode_tokens;
    printf(
        "decode summary  prompt=%d tokens=%d cp_size=%d  mean_total=%.3fms  mean_comp=%.3fms  "
        "mean_comm=%.3fms\n",
        cfg.prompt_len, cfg.decode_tokens, cp_size, mean_total, mean_comp, mean_comm);
  }
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, cp_size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &cp_size));

  const Config cfg = parse_args(argc, argv);

  // Bind rank to GPU using a node-local rank. MPI_COMM_TYPE_SHARED splits
  // MPI_COMM_WORLD by shared-memory domain (== same node), which is robust to
  // launcher rank-distribution policy (block vs. cyclic vs. round-robin). A
  // plain `rank % num_devices` would oversubscribe under cyclic distribution.
  MPI_Comm node_comm;
  MPI_CHECK(
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm));
  int local_rank;
  MPI_CHECK(MPI_Comm_rank(node_comm, &local_rank));
  MPI_CHECK(MPI_Comm_free(&node_comm));

  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  const int device = local_rank % num_devices;
  CUDA_CHECK(cudaSetDevice(device));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // Validate divisibility before doing any work.
  if (cfg.seq % cp_size != 0) {
    if (rank == 0) fprintf(stderr, "ERROR: seq=%d not divisible by cp_size=%d\n", cfg.seq, cp_size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  if (cfg.zigzag && cfg.seq % (2 * cp_size) != 0) {
    if (rank == 0)
      fprintf(stderr, "ERROR: zigzag requires seq=%d divisible by 2*cp_size=%d\n", cfg.seq,
              2 * cp_size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Decode mode short-circuits prefill: it runs its own synthetic benchmark
  // against a pre-populated cache. Correctness is owned by test_ring_decode;
  // this path measures latency only.
  if (cfg.run == "decode") {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0)
      printf("decode mode: B=%d H=%d D=%d prompt=%d tokens=%d cp_size=%d\n", cfg.batch, cfg.heads,
             cfg.head_dim, cfg.prompt_len, cfg.decode_tokens, cp_size);
    fflush(stdout);
    run_decode_benchmark(cfg, rank, cp_size);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return 0;
  }

  const int local_seq = cfg.seq / cp_size;

  // Barrier before printing so output from all ranks arrives in one burst.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  printf(
      "rank %d/%d  local_rank %d  gpu %d  %-24s  local_shape=(B=%d H=%d Sq=%d D=%d)  "
      "mode=%-14s  dtype=%-4s  causal=%d  zigzag=%d\n",
      rank, cp_size, local_rank, device, prop.name, cfg.batch, cfg.heads, local_seq, cfg.head_dim,
      cfg.mode.c_str(), cfg.dtype.c_str(), static_cast<int>(cfg.causal),
      static_cast<int>(cfg.zigzag));
  fflush(stdout);

  // Run the attention (all modes dispatch through run_ring_attention).
  ring_attention::RingConfig rcfg;
  rcfg.rank = rank;
  rcfg.cp_size = cp_size;
  rcfg.batch = cfg.batch;
  rcfg.heads = cfg.heads;
  rcfg.seq = cfg.seq;
  rcfg.head_dim = cfg.head_dim;
  rcfg.kv_heads = cfg.kv_heads;
  rcfg.causal = cfg.causal;
  rcfg.zigzag = cfg.zigzag;
  rcfg.verify = cfg.verify;
  rcfg.csv = cfg.csv;
  rcfg.mode = ring_attention::mode_from_string(cfg.mode);
  rcfg.dtype = ring_attention::dtype_from_string(cfg.dtype);
  rcfg.iters = cfg.iters;
  rcfg.seed = 42u;

  const ring_attention::RingResult res = ring_attention::run_ring_attention(rcfg);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (rank == 0) {
    if (cfg.verify) {
      // FP16 paths accumulate ~1e-2 round-off vs. the FP32 CPU oracle; the
      // FP32 path stays well under 1e-3.
      const float tol = (rcfg.dtype == ring_attention::RingDtype::Half) ? 5e-2f : 1e-3f;
      const bool pass = (res.max_err >= 0.f && res.max_err < tol);
      printf("verify  max_err=%.2e  tol=%.0e  %s\n", res.max_err, tol, pass ? "PASS" : "FAIL");
    }
    // Header is opt-in so that appending many --csv runs to one file produces
    // a single header at the top. First run: `--csv-header --csv`; subsequent
    // runs: `--csv` only.
    if (cfg.csv_header) {
      printf(
          "mode,cp_size,batch,heads,seq,head_dim,causal,zigzag,"
          "iters,comm_ms,comp_ms,wait_ms,total_ms,max_err\n");
    }
    if (cfg.csv) {
      printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.2e\n", cfg.mode.c_str(), cp_size,
             cfg.batch, cfg.heads, cfg.seq, cfg.head_dim, static_cast<int>(cfg.causal),
             static_cast<int>(cfg.zigzag), cfg.iters, res.comm_ms, res.comp_ms, res.wait_ms,
             res.total_ms, res.max_err);
    }
  }

  MPI_CHECK(MPI_Finalize());
  return 0;
}
