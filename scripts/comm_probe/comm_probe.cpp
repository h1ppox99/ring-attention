// comm_probe.cpp — MPI ping-pong bandwidth/latency probe for the performance model.
//
// Measures the (alpha + beta*n) communication cost of a single point-to-point
// link between rank 0 and rank 1, where the placement (intra-node vs
// inter-node) is chosen by the mpirun mapping in the launching sbatch. Host
// buffers are used deliberately: on this cluster GPUDirect RDMA is disabled
// (NIC gdr=0) and the NCCL-off path host-stages, so the host<->host wire
// bandwidth over eth0 is exactly the quantity that bounds inter-node transfers.
//
// Output: per-size one-way latency and bandwidth, then a two-parameter
// least-squares fit  t_oneway(n) = alpha + beta*n  over the large-message
// regime, giving alpha [us] (latency) and 1/beta [MB/s] (asymptotic bandwidth)
// for direct use as the model constants (alpha, beta) / (alpha', beta').
//
// Build (see comm_probe.sbatch):  mpic++ -O2 -std=c++17 comm_probe.cpp -o comm_probe

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (world < 2) {
    if (rank == 0) std::fprintf(stderr, "need >= 2 ranks\n");
    MPI_Finalize();
    return 1;
  }

  // Self-document the link under test: print the two endpoint hostnames.
  char name[MPI_MAX_PROCESSOR_NAME];
  int name_len = 0;
  MPI_Get_processor_name(name, &name_len);
  if (rank <= 1) {
    char peer[MPI_MAX_PROCESSOR_NAME];
    int tag = 99;
    if (rank == 0) {
      MPI_Recv(peer, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::printf("# link: rank0=%s  <-->  rank1=%s\n", name, peer);
    } else {
      MPI_Send(name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
    }
  }

  // Size sweep: 8 B .. 64 MB by powers of two.
  std::vector<size_t> sizes;
  for (size_t n = 8; n <= (64ull << 20); n <<= 1) sizes.push_back(n);

  const size_t max_bytes = sizes.back();
  std::vector<char> buf(max_bytes, 1);

  if (rank == 0) {
    std::printf("# %12s %14s %14s\n", "bytes", "oneway_us", "BW_MB/s");
  }

  // Collected (n, t_oneway_seconds) for the LS fit on rank 0.
  std::vector<double> fit_n, fit_t;

  for (size_t n : sizes) {
    // Pick iteration count so each measured size runs long enough to be stable.
    int iters = static_cast<int>(std::max<size_t>(20, (1ull << 24) / n));
    if (iters > 2000) iters = 2000;
    const int warmup = std::max(5, iters / 10);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = 0.0;
    for (int it = 0; it < iters + warmup; ++it) {
      if (it == warmup) {
        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
      }
      if (rank == 0) {
        MPI_Send(buf.data(), static_cast<int>(n), MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buf.data(), static_cast<int>(n), MPI_CHAR, 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      } else if (rank == 1) {
        MPI_Recv(buf.data(), static_cast<int>(n), MPI_CHAR, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Send(buf.data(), static_cast<int>(n), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
      }
    }
    double t1 = MPI_Wtime();

    if (rank == 0) {
      double rtt = (t1 - t0) / iters;  // full round trip
      double oneway = rtt / 2.0;       // one direction
      double bw = n / oneway / 1.0e6;  // MB/s (10^6 bytes)
      std::printf("  %12zu %14.3f %14.2f\n", n, oneway * 1e6, bw);
      fit_n.push_back(static_cast<double>(n));
      fit_t.push_back(oneway);
    }
  }

  if (rank == 0) {
    // Two-parameter least squares t = alpha + beta*n over the bandwidth-bound
    // tail (>= 64 KB), so the small-message latency floor does not bias beta.
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int m = 0;
    for (size_t i = 0; i < fit_n.size(); ++i) {
      if (fit_n[i] < (64ull << 10)) continue;
      sx += fit_n[i];
      sy += fit_t[i];
      sxx += fit_n[i] * fit_n[i];
      sxy += fit_n[i] * fit_t[i];
      ++m;
    }
    double beta = (m * sxy - sx * sy) / (m * sxx - sx * sx);  // s/byte
    double alpha = (sy - beta * sx) / m;                      // s

    // Small-message latency: best (min) half-RTT over the smallest sizes.
    double lat_floor = fit_t.front();
    for (size_t i = 0; i < fit_n.size() && fit_n[i] <= 1024; ++i)
      lat_floor = std::min(lat_floor, fit_t[i]);

    std::printf("\n# === alpha + beta*n fit (tail n >= 64 KiB) ===\n");
    std::printf("# alpha (latency intercept) = %.2f us\n", alpha * 1e6);
    std::printf("# small-message latency floor = %.2f us\n", lat_floor * 1e6);
    std::printf("# beta = %.4e s/byte  ->  1/beta = %.2f MB/s  (%.3f Gbit/s)\n", beta,
                1.0 / beta / 1.0e6, 8.0 / beta / 1.0e9);
  }

  MPI_Finalize();
  return 0;
}
