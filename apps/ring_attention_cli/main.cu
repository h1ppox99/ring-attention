/// @file
/// Ring-attention distributed driver. One MPI rank per GPU.
///
/// In P1 (scaffolding) the binary parses arguments, binds each rank to a GPU,
/// and prints per-rank diagnostic info. Later phases fill in the attention modes.

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
  }
  return cfg;
}

/// Reproducible float in [-1, 1) for global position (tensor_id, b, h, s, d).
/// Hash-based so each rank can generate any position in O(1) without skipping
/// over elements it doesn't own (unlike a sequential RNG). tensor_id: 0=Q 1=K 2=V.
[[maybe_unused]] float gen_elem(uint32_t seed, int tensor_id, int b, int h, int s, int d) {
  uint32_t v = seed;
  v ^= static_cast<uint32_t>(tensor_id) * 2654435761u;
  v ^= static_cast<uint32_t>(b) * 2246822519u;
  v ^= static_cast<uint32_t>(h) * 3266489917u;
  v ^= static_cast<uint32_t>(s) * 668265263u;
  v ^= static_cast<uint32_t>(d) * 374761393u;
  v ^= v << 13;
  v ^= v >> 17;
  v ^= v << 5;
  return static_cast<float>(static_cast<int32_t>(v)) * (1.0f / 2147483648.0f);
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

  // Bind rank to GPU: with 4 GPUs per node ranks {0,1,2,3} → GPUs {0,1,2,3},
  // ranks {4,5,6,7} on the next node again use {0,1,2,3}.
  int num_devices;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  const int device = rank % num_devices;
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
      "rank %d/%d  gpu %d  %-24s  local_shape=(B=%d H=%d Sq=%d D=%d)  "
      "mode=%-14s  causal=%d  zigzag=%d\n",
      rank, cp_size, device, prop.name, cfg.batch, cfg.heads, local_seq, cfg.head_dim,
      cfg.mode.c_str(), static_cast<int>(cfg.causal), static_cast<int>(cfg.zigzag));
  fflush(stdout);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  MPI_CHECK(MPI_Finalize());
  return 0;
}
