/// @file
/// FP16 / Tensor-Core ring-attention orchestrator. Sibling of `ring_loop.cu`.
/// Same three modes (allgather, ring-blocking, ring-overlap) and same timing
/// instrumentation; differences vs the FP32 path:
///   - K/V live in device `__half` buffers (half the bytes per element).
///   - MPI transports the raw bytes (`MPI_BYTE`) — halves wire traffic.
///   - Q is uploaded once as fp32 then cast to `__half` on the device.
///   - Persistent state (O, m, l) stays fp32 — the online-softmax accumulators
///     need the range; `launch_attention_init` / `launch_attention_finalize`
///     are reused unchanged.
///   - Kernel calls dispatch to `launch_attention_step_fp16`.
///
/// The FP32 path is bit-for-bit untouched; both can be selected at runtime
/// from `run_ring_attention` via `RingConfig::dtype`.

#include <cuda_fp16.h>
#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "attention.hpp"
#include "cpu_attention.hpp"
#include "device_tensor.hpp"
#include "nccl_utils.hpp"
#include "ring_gen.hpp"
#include "ring_loop.hpp"
#include "ring_partition.hpp"

namespace {

[[noreturn]] void mpi_die(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  std::exit(EXIT_FAILURE);
}

/// Validate that `bytes` fits in the `int` count used by MPI-3 point-to-point
/// and collective calls. NVHPC 24.1 ships OpenMPI 4.1.7 (MPI-3.1) so we cannot
/// use `MPI_Count` / the `_c` variants here. On overflow we abort with a clear
/// message rather than silently truncating the count.
int mpi_int_count(std::size_t bytes, const char* where) {
  if (bytes > static_cast<std::size_t>(INT_MAX)) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "MPI count overflow in %s: %zu bytes exceeds INT_MAX=%d — "
                  "reduce per-rank tensor size or chunk the transfer.",
                  where, bytes, INT_MAX);
    mpi_die(buf);
  }
  return static_cast<int>(bytes);
}

/// Fill a host vector of shape (B, H, seq_len, D) using gen_elem (returns fp32).
void fill_host_tensor(std::vector<float>& buf, uint32_t seed, int tensor_id, int B, int H,
                      int seq_len, int D, int global_seq_start) {
  buf.resize(static_cast<std::size_t>(B) * H * seq_len * D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int s = 0; s < seq_len; ++s)
        for (int d = 0; d < D; ++d) {
          const std::size_t idx = (static_cast<std::size_t>(b * H + h) * seq_len + s) * D + d;
          buf[idx] = ring_attention::gen_elem(seed, tensor_id, b, h, global_seq_start + s, d);
        }
}

/// Upload a host fp32 buffer and cast it on the device into a __half buffer.
/// Scratch is reused across calls to amortise allocation; `scratch_count` must
/// be ≥ `host.size()`.
void upload_as_half(const std::vector<float>& host, ring_attention::DeviceTensor<float>& scratch,
                    __half* dst, std::size_t n) {
  scratch.copy_from_host(host);
  ring_attention::launch_float_to_half(scratch.data(), dst, n);
}

// ---------------------------------------------------------------------------
// AllGather baseline (fp16)
// ---------------------------------------------------------------------------

ring_attention::RingResult run_allgather_fp16(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;
  const int Sl = S / P;
  const int q_off = R * Sl;

  const std::size_t q_local_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t kv_local_elem = static_cast<std::size_t>(B) * kv_H * Sl * D;
  const std::size_t q_full_elem = static_cast<std::size_t>(B) * H * S * D;
  const std::size_t kv_full_elem = static_cast<std::size_t>(B) * kv_H * S * D;
  const std::size_t m_count = static_cast<std::size_t>(B) * H * Sl;
  const std::size_t kv_local_bytes = kv_local_elem * sizeof(__half);  // MPI byte count

  // Host-side fill (fp32) — constant across iterations.
  std::vector<float> q_h, k_h, v_h;
  fill_host_tensor(q_h, cfg.seed, 0, B, H, Sl, D, q_off);
  fill_host_tensor(k_h, cfg.seed, 1, B, kv_H, Sl, D, q_off);
  fill_host_tensor(v_h, cfg.seed, 2, B, kv_H, Sl, D, q_off);

  // Convert K/V to fp16 once (payload that rides MPI).
  std::vector<__half> k_h16(kv_local_elem), v_h16(kv_local_elem);
  for (std::size_t i = 0; i < kv_local_elem; ++i) {
    k_h16[i] = __float2half(k_h[i]);
    v_h16[i] = __float2half(v_h[i]);
  }

  // Q stays put as __half on the device.
  DeviceTensor<float> q_scratch_f(q_local_elem);
  DeviceTensor<__half> q_d(q_local_elem);
  upload_as_half(q_h, q_scratch_f, q_d.data(), q_local_elem);

  std::vector<__half> k_gathered(kv_full_elem), v_gathered(kv_full_elem);
  std::vector<__half> full_k_h(kv_full_elem), full_v_h(kv_full_elem);
  DeviceTensor<__half> k_d(kv_full_elem), v_d(kv_full_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);
  AttentionShape shape{B, H, Sl, S, D};
  shape.kv_heads = kv_H;

  const int kv_local_bytes_int = mpi_int_count(kv_local_bytes, "run_allgather_fp16/MPI_Allgather");

  auto one_pass = [&]() -> std::pair<double, double> {
    const double tc0 = MPI_Wtime();
    MPI_Allgather(k_h16.data(), kv_local_bytes_int, MPI_BYTE, k_gathered.data(), kv_local_bytes_int,
                  MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(v_h16.data(), kv_local_bytes_int, MPI_BYTE, v_gathered.data(), kv_local_bytes_int,
                  MPI_BYTE, MPI_COMM_WORLD);

    // Rearrange (P,B,kv_H,Sl,D) → (B,kv_H,S,D).
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < kv_H; ++h)
        for (int p = 0; p < P; ++p) {
          const std::size_t src = static_cast<std::size_t>(p) * kv_local_elem +
                                  static_cast<std::size_t>(b * kv_H + h) * Sl * D;
          const std::size_t dst = (static_cast<std::size_t>(b * kv_H + h) * S + p * Sl) * D;
          std::copy(k_gathered.data() + src, k_gathered.data() + src + Sl * D,
                    full_k_h.data() + dst);
          std::copy(v_gathered.data() + src, v_gathered.data() + src + Sl * D,
                    full_v_h.data() + dst);
        }

    cudaMemcpy(k_d.data(), full_k_h.data(), kv_full_elem * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(v_d.data(), full_v_h.data(), kv_full_elem * sizeof(__half), cudaMemcpyHostToDevice);
    const double tc1 = MPI_Wtime();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), shape, m_count);
    cudaEventRecord(ev0);
    launch_attention_step_fp16(q_d.data(), k_d.data(), v_d.data(), out_d.data(), m_d.data(),
                               l_d.data(), shape, q_off, /*k_offset=*/0, cfg.causal);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    launch_attention_finalize(out_d.data(), l_d.data(), shape);
    cudaDeviceSynchronize();
    float comp_float_ms = 0.f;
    cudaEventElapsedTime(&comp_float_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return {(tc1 - tc0) * 1e3, static_cast<double>(comp_float_ms)};
  };

  one_pass();  // warmup

  double sum_comm = 0.0, sum_comp = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  for (int i = 0; i < cfg.iters; ++i) {
    const auto [comm, comp] = one_pass();
    sum_comm += comm;
    sum_comp += comp;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  RingResult res;
  res.comm_ms = sum_comm / cfg.iters;
  res.comp_ms = sum_comp / cfg.iters;
  res.wait_ms = 0.0;
  res.total_ms = (t_end - t_start) * 1e3 / cfg.iters;

  const double local_t[3] = {res.comm_ms, res.comp_ms, res.total_ms};
  double global_t[3] = {};
  MPI_Reduce(local_t, global_t, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (R == 0) {
    res.comm_ms = global_t[0];
    res.comp_ms = global_t[1];
    res.total_ms = global_t[2];
  }

  float max_err = -1.f;
  if (cfg.verify) {
    std::vector<float> full_q_cpu(q_full_elem), full_k_cpu(kv_full_elem), full_v_cpu(kv_full_elem);
    fill_host_tensor(full_q_cpu, cfg.seed, 0, B, H, S, D, 0);
    fill_host_tensor(full_k_cpu, cfg.seed, 1, B, kv_H, S, D, 0);
    fill_host_tensor(full_v_cpu, cfg.seed, 2, B, kv_H, S, D, 0);

    AttentionShape ref_shape{B, H, S, S, D};
    ref_shape.kv_heads = kv_H;
    std::vector<float> cpu_out(q_full_elem);
    cpu_attention(full_q_cpu.data(), full_k_cpu.data(), full_v_cpu.data(), cpu_out.data(),
                  ref_shape, cfg.causal);

    one_pass();
    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);

    max_err = 0.f;
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h) {
        const float* cpu_slice =
            cpu_out.data() + (static_cast<std::size_t>(b * H + h) * S + q_off) * D;
        const float* dev_slice = dev_out_h.data() + static_cast<std::size_t>(b * H + h) * Sl * D;
        for (int e = 0; e < Sl * D; ++e)
          max_err = std::max(max_err, std::abs(cpu_slice[e] - dev_slice[e]));
      }

    float global_max;
    MPI_Reduce(&max_err, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    max_err = (R == 0) ? global_max : -1.f;
  }
  res.max_err = max_err;
  return res;
}

// ---------------------------------------------------------------------------
// Ring rotation, blocking style (fp16)
// ---------------------------------------------------------------------------

ring_attention::RingResult run_ring_blocking_fp16(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;

  const RingPartition::Mode mode =
      cfg.zigzag ? RingPartition::Mode::Zigzag : RingPartition::Mode::Contiguous;
  RingPartition part(P, R, S, mode);
  const int nsg = part.num_sub_groups();
  const int Sl = part.local_chunk_len();
  const int Sl_local = Sl * nsg;
  const std::size_t q_sg_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t q_local_elem = static_cast<std::size_t>(nsg) * q_sg_elem;
  const std::size_t kv_sg_elem = static_cast<std::size_t>(B) * kv_H * Sl * D;
  const std::size_t kv_local_elem = static_cast<std::size_t>(nsg) * kv_sg_elem;
  const std::size_t m_sg = static_cast<std::size_t>(B) * H * Sl;
  const std::size_t m_count = static_cast<std::size_t>(nsg) * m_sg;
  // Used only on the MPI path; the NCCL build never references it.
  [[maybe_unused]] const std::size_t kv_bytes = kv_local_elem * sizeof(__half);

  const int next_rank = part.next_rank();
  const int prev_rank = part.prev_rank();

  auto fill_q_into = [&](std::vector<float>& buf, int tid, std::size_t off, int gss) {
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h)
        for (int s = 0; s < Sl; ++s)
          for (int d = 0; d < D; ++d)
            buf[off + (static_cast<std::size_t>(b * H + h) * Sl + s) * D + d] =
                gen_elem(cfg.seed, tid, b, h, gss + s, d);
  };

  auto fill_kv_into = [&](std::vector<float>& buf, int tid, std::size_t off, int gss) {
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < kv_H; ++h)
        for (int s = 0; s < Sl; ++s)
          for (int d = 0; d < D; ++d)
            buf[off + (static_cast<std::size_t>(b * kv_H + h) * Sl + s) * D + d] =
                gen_elem(cfg.seed, tid, b, h, gss + s, d);
  };

  std::vector<float> q_h(q_local_elem), k_h_init(kv_local_elem), v_h_init(kv_local_elem);
  for (int sg = 0; sg < nsg; ++sg) {
    fill_q_into(q_h, 0, sg * q_sg_elem, part.q_offset(sg));
    fill_kv_into(k_h_init, 1, sg * kv_sg_elem, part.k_offset_for_step(0, sg));
    fill_kv_into(v_h_init, 2, sg * kv_sg_elem, part.k_offset_for_step(0, sg));
  }

  // Q (fp16 on device) — uploaded once. Scratch is reused below to stage K/V,
  // so size it to fit the largest payload (kv_local_elem can exceed q_local_elem
  // if kv_heads > heads).
  DeviceTensor<float> scratch_f(std::max(q_local_elem, kv_local_elem));
  DeviceTensor<__half> q_d(q_local_elem);
  upload_as_half(q_h, scratch_f, q_d.data(), q_local_elem);

  // Double-buffered K/V (fp16 on device), sized by kv_H.
  DeviceTensor<__half> K_a(kv_local_elem), K_b(kv_local_elem);
  DeviceTensor<__half> V_a(kv_local_elem), V_b(kv_local_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);

  const AttentionShape init_shape{B, H, Sl_local, S, D};
  AttentionShape sg_shape{B, H, Sl, Sl, D};
  sg_shape.kv_heads = kv_H;

#ifdef RING_USE_NCCL
  // NCCL communicator: GPU-to-GPU direct, no host staging needed.
  ncclComm_t nccl_comm = ring_attention::nccl_init(R, P);
#else
  // Pinned host stagers, sized in fp16 bytes for K/V.
  __half *K_send_h = nullptr, *V_send_h = nullptr, *K_recv_h = nullptr, *V_recv_h = nullptr;
  cudaHostAlloc(&K_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&K_recv_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_recv_h, kv_bytes, cudaHostAllocDefault);
#endif

  auto one_pass = [&]() -> std::tuple<double, double, double> {
    // Reset K/V to local chunk. scratch_f was sized to max(q_local_elem, kv_local_elem)
    // so it is large enough to serve as the fp32 staging buffer for any valid kv_H.
    upload_as_half(k_h_init, scratch_f, K_a.data(), kv_local_elem);
    upload_as_half(v_h_init, scratch_f, V_a.data(), kv_local_elem);
    __half* K_cur = K_a.data();
    __half* V_cur = V_a.data();
    __half* K_recv = K_b.data();
    __half* V_recv = V_b.data();

    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), init_shape, m_count);

    double comm_acc = 0.0, comp_acc = 0.0, wait_acc = 0.0;
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    for (int step = 0; step < P; ++step) {
#ifndef RING_USE_NCCL
      MPI_Request reqs[4];
      int n_req = 0;
#endif

      const double t_post0 = MPI_Wtime();
      if (step < P - 1) {
#ifdef RING_USE_NCCL
        ncclGroupStart();
        NCCL_CHECK(ncclSend(K_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, 0));
        NCCL_CHECK(ncclRecv(K_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, 0));
        NCCL_CHECK(ncclSend(V_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, 0));
        NCCL_CHECK(ncclRecv(V_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, 0));
        NCCL_CHECK(ncclGroupEnd());
#else
        cudaMemcpy(K_send_h, K_cur, kv_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(V_send_h, V_cur, kv_bytes, cudaMemcpyDeviceToHost);
        const int n = mpi_int_count(kv_bytes, "run_ring_blocking_fp16/MPI_Isend|Irecv");
        MPI_Isend(K_send_h, n, MPI_BYTE, next_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Irecv(K_recv_h, n, MPI_BYTE, prev_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Isend(V_send_h, n, MPI_BYTE, next_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[n_req++]);
        MPI_Irecv(V_recv_h, n, MPI_BYTE, prev_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[n_req++]);
#endif
      }
      const double t_post1 = MPI_Wtime();
      comm_acc += (t_post1 - t_post0) * 1e3;

      cudaEventRecord(ev0);
      for (int sg_q = 0; sg_q < nsg; ++sg_q) {
        for (int sg_k = 0; sg_k < nsg; ++sg_k) {
          const int q_off_sg = part.q_offset(sg_q);
          const int k_off_sg = part.k_offset_for_step(step, sg_k);
          if (cfg.causal && k_off_sg > q_off_sg + Sl - 1) continue;
          launch_attention_step_fp16(q_d.data() + sg_q * q_sg_elem, K_cur + sg_k * kv_sg_elem,
                                     V_cur + sg_k * kv_sg_elem, out_d.data() + sg_q * q_sg_elem,
                                     m_d.data() + sg_q * m_sg, l_d.data() + sg_q * m_sg, sg_shape,
                                     q_off_sg, k_off_sg, cfg.causal);
        }
      }
      cudaEventRecord(ev1);
      cudaEventSynchronize(ev1);
      float comp_float_ms = 0.f;
      cudaEventElapsedTime(&comp_float_ms, ev0, ev1);
      comp_acc += comp_float_ms;

      if (step < P - 1) {
#ifdef RING_USE_NCCL
        const double t_wait0 = MPI_Wtime();
        cudaStreamSynchronize(0);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;
#else
        const double t_wait0 = MPI_Wtime();
        MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;

        const double t_h2d0 = MPI_Wtime();
        cudaMemcpy(K_recv, K_recv_h, kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(V_recv, V_recv_h, kv_bytes, cudaMemcpyHostToDevice);
        const double t_h2d1 = MPI_Wtime();
        comm_acc += (t_h2d1 - t_h2d0) * 1e3;
#endif
        std::swap(K_cur, K_recv);
        std::swap(V_cur, V_recv);
      }
    }

    for (int sg = 0; sg < nsg; ++sg)
      launch_attention_finalize(out_d.data() + sg * q_sg_elem, l_d.data() + sg * m_sg, sg_shape);
    cudaDeviceSynchronize();
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return {comm_acc, comp_acc, wait_acc};
  };

  one_pass();

  double sum_comm = 0.0, sum_comp = 0.0, sum_wait = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  for (int i = 0; i < cfg.iters; ++i) {
    const auto [c, p_, w] = one_pass();
    sum_comm += c;
    sum_comp += p_;
    sum_wait += w;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  RingResult res;
  res.comm_ms = sum_comm / cfg.iters;
  res.comp_ms = sum_comp / cfg.iters;
  res.wait_ms = sum_wait / cfg.iters;
  res.total_ms = (t_end - t_start) * 1e3 / cfg.iters;

  const double local_t[4] = {res.comm_ms, res.comp_ms, res.wait_ms, res.total_ms};
  double global_t[4] = {};
  MPI_Reduce(local_t, global_t, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (R == 0) {
    res.comm_ms = global_t[0];
    res.comp_ms = global_t[1];
    res.wait_ms = global_t[2];
    res.total_ms = global_t[3];
  }

  float max_err = -1.f;
  if (cfg.verify) {
    const std::size_t q_full_elem = static_cast<std::size_t>(B) * H * S * D;
    const std::size_t kv_full_elem = static_cast<std::size_t>(B) * kv_H * S * D;
    std::vector<float> full_q(q_full_elem), full_k(kv_full_elem), full_v(kv_full_elem);
    fill_host_tensor(full_q, cfg.seed, 0, B, H, S, D, 0);
    fill_host_tensor(full_k, cfg.seed, 1, B, kv_H, S, D, 0);
    fill_host_tensor(full_v, cfg.seed, 2, B, kv_H, S, D, 0);

    AttentionShape ref_shape{B, H, S, S, D};
    ref_shape.kv_heads = kv_H;
    std::vector<float> cpu_out(q_full_elem);
    cpu_attention(full_q.data(), full_k.data(), full_v.data(), cpu_out.data(), ref_shape,
                  cfg.causal);

    one_pass();
    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);

    max_err = 0.f;
    for (int sg = 0; sg < nsg; ++sg) {
      const int q_off_sg = part.q_offset(sg);
      for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
          const float* cpu_slice =
              cpu_out.data() + (static_cast<std::size_t>(b * H + h) * S + q_off_sg) * D;
          const float* dev_slice =
              dev_out_h.data() + sg * q_sg_elem + static_cast<std::size_t>(b * H + h) * Sl * D;
          for (int e = 0; e < Sl * D; ++e)
            max_err = std::max(max_err, std::abs(cpu_slice[e] - dev_slice[e]));
        }
    }

    float global_max;
    MPI_Reduce(&max_err, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    max_err = (R == 0) ? global_max : -1.f;
  }
  res.max_err = max_err;

#ifdef RING_USE_NCCL
  ncclCommDestroy(nccl_comm);
#else
  cudaFreeHost(K_send_h);
  cudaFreeHost(V_send_h);
  cudaFreeHost(K_recv_h);
  cudaFreeHost(V_recv_h);
#endif
  return res;
}

// ---------------------------------------------------------------------------
// Ring rotation with compute/comm overlap (fp16)
// ---------------------------------------------------------------------------

ring_attention::RingResult run_ring_overlap_fp16(const ring_attention::RingConfig& cfg) {
  using namespace ring_attention;

  const int B = cfg.batch, H = cfg.heads, D = cfg.head_dim;
  const int kv_H = (cfg.kv_heads > 0) ? cfg.kv_heads : cfg.heads;
  const int S = cfg.seq, P = cfg.cp_size, R = cfg.rank;

  const RingPartition::Mode mode =
      cfg.zigzag ? RingPartition::Mode::Zigzag : RingPartition::Mode::Contiguous;
  RingPartition part(P, R, S, mode);
  const int nsg = part.num_sub_groups();
  const int Sl = part.local_chunk_len();
  const int Sl_local = Sl * nsg;
  const std::size_t q_sg_elem = static_cast<std::size_t>(B) * H * Sl * D;
  const std::size_t q_local_elem = static_cast<std::size_t>(nsg) * q_sg_elem;
  const std::size_t kv_sg_elem = static_cast<std::size_t>(B) * kv_H * Sl * D;
  const std::size_t kv_local_elem = static_cast<std::size_t>(nsg) * kv_sg_elem;
  const std::size_t m_sg = static_cast<std::size_t>(B) * H * Sl;
  const std::size_t m_count = static_cast<std::size_t>(nsg) * m_sg;
  const std::size_t kv_bytes = kv_local_elem * sizeof(__half);

  const int next_rank = part.next_rank();
  const int prev_rank = part.prev_rank();

  auto fill_q_into = [&](std::vector<float>& buf, int tid, std::size_t off, int gss) {
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h)
        for (int s = 0; s < Sl; ++s)
          for (int d = 0; d < D; ++d)
            buf[off + (static_cast<std::size_t>(b * H + h) * Sl + s) * D + d] =
                gen_elem(cfg.seed, tid, b, h, gss + s, d);
  };

  auto fill_kv_into = [&](std::vector<float>& buf, int tid, std::size_t off, int gss) {
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < kv_H; ++h)
        for (int s = 0; s < Sl; ++s)
          for (int d = 0; d < D; ++d)
            buf[off + (static_cast<std::size_t>(b * kv_H + h) * Sl + s) * D + d] =
                gen_elem(cfg.seed, tid, b, h, gss + s, d);
  };

  std::vector<float> q_h(q_local_elem), k_h_init(kv_local_elem), v_h_init(kv_local_elem);
  for (int sg = 0; sg < nsg; ++sg) {
    fill_q_into(q_h, 0, sg * q_sg_elem, part.q_offset(sg));
    fill_kv_into(k_h_init, 1, sg * kv_sg_elem, part.k_offset_for_step(0, sg));
    fill_kv_into(v_h_init, 2, sg * kv_sg_elem, part.k_offset_for_step(0, sg));
  }

  DeviceTensor<float> scratch_f(q_local_elem);
  DeviceTensor<__half> q_d(q_local_elem);
  upload_as_half(q_h, scratch_f, q_d.data(), q_local_elem);

  DeviceTensor<__half> K_a(kv_local_elem), K_b(kv_local_elem);
  DeviceTensor<__half> V_a(kv_local_elem), V_b(kv_local_elem);
  DeviceTensor<float> out_d(q_local_elem), m_d(m_count), l_d(m_count);

  const AttentionShape init_shape{B, H, Sl_local, S, D};
  AttentionShape sg_shape{B, H, Sl, Sl, D};
  sg_shape.kv_heads = kv_H;

#ifdef RING_USE_NCCL
  ncclComm_t nccl_comm = ring_attention::nccl_init(R, P);
#else
  __half *K_send_h = nullptr, *V_send_h = nullptr, *K_recv_h = nullptr, *V_recv_h = nullptr;
  cudaHostAlloc(&K_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_send_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&K_recv_h, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&V_recv_h, kv_bytes, cudaHostAllocDefault);
#endif

  cudaStream_t stream_compute = nullptr, stream_copy = nullptr;
  cudaStreamCreate(&stream_compute);
  cudaStreamCreate(&stream_copy);
  cudaEvent_t comm_done;
  cudaEventCreate(&comm_done);

  std::vector<cudaEvent_t> ev_starts(P), ev_ends(P);
  std::vector<cudaEvent_t> ev_h2d_starts(P), ev_h2d_ends(P);
  for (int s = 0; s < P; ++s) {
    cudaEventCreate(&ev_starts[s]);
    cudaEventCreate(&ev_ends[s]);
    cudaEventCreate(&ev_h2d_starts[s]);
    cudaEventCreate(&ev_h2d_ends[s]);
  }

  // Pinned host fp16 buffers for the initial K/V state. They must be page-locked:
  // `cudaMemcpyAsync` from pageable memory silently serializes (internal staging)
  // and would defeat the overlap with `stream_compute`. We stage fp32→fp16 once
  // here, outside the timed loop.
  __half *k_init_h16 = nullptr, *v_init_h16 = nullptr;
  cudaHostAlloc(&k_init_h16, kv_bytes, cudaHostAllocDefault);
  cudaHostAlloc(&v_init_h16, kv_bytes, cudaHostAllocDefault);
  for (std::size_t i = 0; i < kv_local_elem; ++i) {
    k_init_h16[i] = __float2half(k_h_init[i]);
    v_init_h16[i] = __float2half(v_h_init[i]);
  }

  auto one_pass = [&]() -> std::tuple<double, double, double> {
    cudaMemcpyAsync(K_a.data(), k_init_h16, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
    cudaMemcpyAsync(V_a.data(), v_init_h16, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
    cudaEventRecord(comm_done, stream_copy);

    __half* K_cur = K_a.data();
    __half* V_cur = V_a.data();
    __half* K_recv = K_b.data();
    __half* V_recv = V_b.data();

    launch_attention_init(out_d.data(), m_d.data(), l_d.data(), init_shape, m_count,
                          stream_compute);

    double comm_acc = 0.0, wait_acc = 0.0;

    for (int step = 0; step < P; ++step) {
      cudaStreamWaitEvent(stream_compute, comm_done, 0);

      cudaEventRecord(ev_starts[step], stream_compute);
      for (int sg_q = 0; sg_q < nsg; ++sg_q) {
        for (int sg_k = 0; sg_k < nsg; ++sg_k) {
          const int q_off_sg = part.q_offset(sg_q);
          const int k_off_sg = part.k_offset_for_step(step, sg_k);
          if (cfg.causal && k_off_sg > q_off_sg + Sl - 1) continue;
          launch_attention_step_fp16(q_d.data() + sg_q * q_sg_elem, K_cur + sg_k * kv_sg_elem,
                                     V_cur + sg_k * kv_sg_elem, out_d.data() + sg_q * q_sg_elem,
                                     m_d.data() + sg_q * m_sg, l_d.data() + sg_q * m_sg, sg_shape,
                                     q_off_sg, k_off_sg, cfg.causal, stream_compute);
        }
      }
      cudaEventRecord(ev_ends[step], stream_compute);

      if (step < P - 1) {
#ifdef RING_USE_NCCL
        if (step >= 1) cudaStreamWaitEvent(stream_copy, ev_ends[step - 1], 0);

        cudaEventRecord(ev_h2d_starts[step], stream_copy);
        ncclGroupStart();
        NCCL_CHECK(ncclSend(K_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclRecv(K_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclSend(V_cur, kv_local_elem, ncclHalf, next_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclRecv(V_recv, kv_local_elem, ncclHalf, prev_rank, nccl_comm, stream_copy));
        NCCL_CHECK(ncclGroupEnd());
        cudaEventRecord(ev_h2d_ends[step], stream_copy);
        cudaEventRecord(comm_done, stream_copy);
#else
        const double t_post0 = MPI_Wtime();
        cudaMemcpyAsync(K_send_h, K_cur, kv_bytes, cudaMemcpyDeviceToHost, stream_copy);
        cudaMemcpyAsync(V_send_h, V_cur, kv_bytes, cudaMemcpyDeviceToHost, stream_copy);
        cudaStreamSynchronize(stream_copy);
        MPI_Request reqs[4];
        const int n = mpi_int_count(kv_bytes, "run_ring_overlap_fp16/MPI_Isend|Irecv");
        MPI_Isend(K_send_h, n, MPI_BYTE, next_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(K_recv_h, n, MPI_BYTE, prev_rank, /*tag=*/0, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(V_send_h, n, MPI_BYTE, next_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(V_recv_h, n, MPI_BYTE, prev_rank, /*tag=*/1, MPI_COMM_WORLD, &reqs[3]);
        const double t_post1 = MPI_Wtime();
        comm_acc += (t_post1 - t_post0) * 1e3;

        const double t_wait0 = MPI_Wtime();
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        const double t_wait1 = MPI_Wtime();
        wait_acc += (t_wait1 - t_wait0) * 1e3;

        if (step >= 1) cudaStreamWaitEvent(stream_copy, ev_ends[step - 1], 0);

        cudaEventRecord(ev_h2d_starts[step], stream_copy);
        cudaMemcpyAsync(K_recv, K_recv_h, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
        cudaMemcpyAsync(V_recv, V_recv_h, kv_bytes, cudaMemcpyHostToDevice, stream_copy);
        cudaEventRecord(ev_h2d_ends[step], stream_copy);
        cudaEventRecord(comm_done, stream_copy);
#endif

        std::swap(K_cur, K_recv);
        std::swap(V_cur, V_recv);
      }
    }

    cudaStreamSynchronize(stream_compute);
    for (int sg = 0; sg < nsg; ++sg)
      launch_attention_finalize(out_d.data() + sg * q_sg_elem, l_d.data() + sg * m_sg, sg_shape,
                                stream_compute);
    cudaStreamSynchronize(stream_compute);

    double comp_acc = 0.0;
    for (int s = 0; s < P; ++s) {
      float t = 0.f;
      cudaEventElapsedTime(&t, ev_starts[s], ev_ends[s]);
      comp_acc += t;
    }
    for (int s = 0; s < P - 1; ++s) {
      float t = 0.f;
      cudaEventElapsedTime(&t, ev_h2d_starts[s], ev_h2d_ends[s]);
      comm_acc += t;
    }
    return {comm_acc, comp_acc, wait_acc};
  };

  one_pass();

  double sum_comm = 0.0, sum_comp = 0.0, sum_wait = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();
  for (int i = 0; i < cfg.iters; ++i) {
    const auto [c, p_, w] = one_pass();
    sum_comm += c;
    sum_comp += p_;
    sum_wait += w;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const double t_end = MPI_Wtime();

  RingResult res;
  res.comm_ms = sum_comm / cfg.iters;
  res.comp_ms = sum_comp / cfg.iters;
  res.wait_ms = sum_wait / cfg.iters;
  res.total_ms = (t_end - t_start) * 1e3 / cfg.iters;

  const double local_t[4] = {res.comm_ms, res.comp_ms, res.wait_ms, res.total_ms};
  double global_t[4] = {};
  MPI_Reduce(local_t, global_t, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (R == 0) {
    res.comm_ms = global_t[0];
    res.comp_ms = global_t[1];
    res.wait_ms = global_t[2];
    res.total_ms = global_t[3];
  }

  float max_err = -1.f;
  if (cfg.verify) {
    const std::size_t q_full_elem = static_cast<std::size_t>(B) * H * S * D;
    const std::size_t kv_full_elem = static_cast<std::size_t>(B) * kv_H * S * D;
    std::vector<float> full_q(q_full_elem), full_k(kv_full_elem), full_v(kv_full_elem);
    fill_host_tensor(full_q, cfg.seed, 0, B, H, S, D, 0);
    fill_host_tensor(full_k, cfg.seed, 1, B, kv_H, S, D, 0);
    fill_host_tensor(full_v, cfg.seed, 2, B, kv_H, S, D, 0);

    AttentionShape ref_shape{B, H, S, S, D};
    ref_shape.kv_heads = kv_H;
    std::vector<float> cpu_out(q_full_elem);
    cpu_attention(full_q.data(), full_k.data(), full_v.data(), cpu_out.data(), ref_shape,
                  cfg.causal);

    one_pass();
    std::vector<float> dev_out_h(q_local_elem);
    out_d.copy_to_host(dev_out_h);

    max_err = 0.f;
    for (int sg = 0; sg < nsg; ++sg) {
      const int q_off_sg = part.q_offset(sg);
      for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
          const float* cpu_slice =
              cpu_out.data() + (static_cast<std::size_t>(b * H + h) * S + q_off_sg) * D;
          const float* dev_slice =
              dev_out_h.data() + sg * q_sg_elem + static_cast<std::size_t>(b * H + h) * Sl * D;
          for (int e = 0; e < Sl * D; ++e)
            max_err = std::max(max_err, std::abs(cpu_slice[e] - dev_slice[e]));
        }
    }

    float global_max;
    MPI_Reduce(&max_err, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    max_err = (R == 0) ? global_max : -1.f;
  }
  res.max_err = max_err;

  for (int s = 0; s < P; ++s) {
    cudaEventDestroy(ev_starts[s]);
    cudaEventDestroy(ev_ends[s]);
    cudaEventDestroy(ev_h2d_starts[s]);
    cudaEventDestroy(ev_h2d_ends[s]);
  }
  cudaEventDestroy(comm_done);
  cudaStreamDestroy(stream_compute);
  cudaStreamDestroy(stream_copy);
#ifdef RING_USE_NCCL
  ncclCommDestroy(nccl_comm);
#else
  cudaFreeHost(K_send_h);
  cudaFreeHost(V_send_h);
  cudaFreeHost(K_recv_h);
  cudaFreeHost(V_recv_h);
#endif
  cudaFreeHost(k_init_h16);
  cudaFreeHost(v_init_h16);
  return res;
}

}  // namespace

namespace ring_attention {

RingResult run_ring_attention_fp16(const RingConfig& cfg) {
  if (cfg.zigzag && cfg.mode == RingMode::AllGather)
    mpi_die("--zigzag is not supported with --mode=allgather (use ring-blocking or ring-overlap)");
  switch (cfg.mode) {
    case RingMode::AllGather:
      return run_allgather_fp16(cfg);
    case RingMode::RingBlocking:
      return run_ring_blocking_fp16(cfg);
    case RingMode::RingOverlap:
      return run_ring_overlap_fp16(cfg);
  }
  mpi_die("unreachable");
}

}  // namespace ring_attention
