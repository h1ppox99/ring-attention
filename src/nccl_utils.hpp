#pragma once

/// @file
/// NCCL helpers used by ring_loop.cu and ring_loop_fp16.cu.
/// Compiled only when RING_USE_NCCL is defined by CMake.

#ifdef RING_USE_NCCL

#include <mpi.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>

/// Abort all MPI ranks on any NCCL error.
#define NCCL_CHECK(expr)                                                                        \
  do {                                                                                          \
    ncclResult_t _r = (expr);                                                                   \
    if (_r != ncclSuccess) {                                                                    \
      fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(_r)); \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                                  \
    }                                                                                           \
  } while (0)

namespace ring_attention {

/// Bootstrap an NCCL communicator from existing MPI rank/size.
/// Broadcasts the unique ID from rank 0 via MPI so no out-of-band rendezvous
/// is needed. Called once per run function; destroy with ncclCommDestroy.
inline ncclComm_t nccl_init(int rank, int size) {
  ncclUniqueId id;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));
  return comm;
}

}  // namespace ring_attention

#endif  // RING_USE_NCCL
