#include <cuda_runtime.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>

#define cudaCheck(expr)                                                                      \
  do {                                                                                       \
    cudaError_t _e = (expr);                                                                 \
    if (_e != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                               \
    }                                                                                        \
  } while (0)

#define MPI_CHECK(expr)                                                   \
  do {                                                                    \
    int _e = (expr);                                                      \
    if (_e != MPI_SUCCESS) {                                              \
      char _msg[MPI_MAX_ERROR_STRING];                                    \
      int _len;                                                           \
      MPI_Error_string(_e, _msg, &_len);                                  \
      fprintf(stderr, "MPI error %s:%d: %s\n", __FILE__, __LINE__, _msg); \
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                            \
    }                                                                     \
  } while (0)

__global__ void probe_kernel(int* out) { *out = threadIdx.x + blockIdx.x; }

int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));

  int rank, size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  int num_devices;
  cudaCheck(cudaGetDeviceCount(&num_devices));

  int device = rank % num_devices;
  cudaCheck(cudaSetDevice(device));

  cudaDeviceProp prop;
  cudaCheck(cudaGetDeviceProperties(&prop, device));

  // Launch a trivial kernel to confirm the device is usable.
  int* d_out;
  cudaCheck(cudaMalloc(&d_out, sizeof(int)));
  probe_kernel<<<1, 1>>>(d_out);
  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaFree(d_out));

  printf("rank %d/%d  gpu %d  %-24s  %.1f GB  sm_%d%d  OK\n", rank, size, device, prop.name,
         static_cast<double>(prop.totalGlobalMem) / 1e9, prop.major, prop.minor);
  fflush(stdout);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  MPI_CHECK(MPI_Finalize());
  return 0;
}
