/// @file
/// Ring-attention distributed driver. One MPI rank per GPU.

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ring_loop.hpp"

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
  bool causal = false;
  bool zigzag = false;
  std::string mode = "allgather";  // allgather | ring-blocking | ring-overlap
  int iters = 10;
  bool verify = false;
  bool csv = false;
  bool csv_header = false;  // emit header row only (paired with --csv or alone)
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
    else if (!std::strcmp(argv[i], "--causal") && nxt)
      cfg.causal = std::atoi(argv[++i]) != 0;
    else if (!std::strcmp(argv[i], "--zigzag") && nxt)
      cfg.zigzag = std::atoi(argv[++i]) != 0;
    else if (!std::strcmp(argv[i], "--mode") && nxt)
      cfg.mode = argv[++i];
    else if (!std::strcmp(argv[i], "--iters") && nxt)
      cfg.iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--verify"))
      cfg.verify = true;
    else if (!std::strcmp(argv[i], "--csv"))
      cfg.csv = true;
    else if (!std::strcmp(argv[i], "--csv-header"))
      cfg.csv_header = true;
  }
  return cfg;
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

  const int local_seq = cfg.seq / cp_size;

  // Barrier before printing so output from all ranks arrives in one burst.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  printf(
      "rank %d/%d  local_rank %d  gpu %d  %-24s  local_shape=(B=%d H=%d Sq=%d D=%d)  "
      "mode=%-14s  causal=%d  zigzag=%d\n",
      rank, cp_size, local_rank, device, prop.name, cfg.batch, cfg.heads, local_seq, cfg.head_dim,
      cfg.mode.c_str(), static_cast<int>(cfg.causal), static_cast<int>(cfg.zigzag));
  fflush(stdout);

  // Run the attention (all modes dispatch through run_ring_attention).
  ring_attention::RingConfig rcfg;
  rcfg.rank = rank;
  rcfg.cp_size = cp_size;
  rcfg.batch = cfg.batch;
  rcfg.heads = cfg.heads;
  rcfg.seq = cfg.seq;
  rcfg.head_dim = cfg.head_dim;
  rcfg.causal = cfg.causal;
  rcfg.zigzag = cfg.zigzag;
  rcfg.verify = cfg.verify;
  rcfg.csv = cfg.csv;
  rcfg.mode = ring_attention::mode_from_string(cfg.mode);
  rcfg.iters = cfg.iters;
  rcfg.seed = 42u;

  const ring_attention::RingResult res = ring_attention::run_ring_attention(rcfg);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (rank == 0) {
    if (cfg.verify) {
      const bool pass = (res.max_err >= 0.f && res.max_err < 1e-3f);
      printf("verify  max_err=%.2e  %s\n", res.max_err, pass ? "PASS" : "FAIL");
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
