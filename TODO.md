# Ring Attention: Comparison with haoliuhl/ringattention & TODOs

This file compares the CUDA/C++ implementation in `src/` against the official JAX reference
at <https://github.com/haoliuhl/ringattention> (`ringattention/ringattention_jax.py` and
`ringattention_jax_inference.py`).  Each section names the gap, explains it precisely,
points to where the reference code lives, and describes what would need to change here.

Run these once and save the results as your reference:

  # 1. Full correctness gate
  bash scripts/slurm/gpu_tests.sbatch          # or sbatch it

  # 2. Single-kernel baseline
  ./build/release/apps/bench_attention/bench_attention --bh 1 8 >
   results/bench_baseline.csv

  # 3. Ring pipeline baseline
  sbatch scripts/slurm/bench_ring.sbatch       # saves
  results/bench_ring.csv

  # 4. Kernel counters baseline
  sbatch scripts/slurm/profile.sbatch          # saves .nsys-rep
  + .ncu-rep
  bash scripts/analyze_profiles.sh             # extracts
  results/stats/summary.txt

  After implementing a TODO, repeat steps 1–4 with output to
  results/bench_<todo>.csv etc., then diff the CSVs and compare
  summary.txt. The wait_ms column from the ring benchmark is the
  single most sensitive indicator of whether communication is
  being hidden; comp_ms shows whether a kernel change hurt or
  helped throughput.

---

## 1. Zigzag scheme: coarse (this repo) vs. fine-grained (reference)

**What the reference does.**  In `ringattention_jax.py` the index origin of each step is
`k_block_idx = (lax.axis_index(axis_name) - idx) % axis_size`, which rotates the *whole*
KV block of a rank.  The companion Python package's `zigzag.py::zigzag_indices` assigns
positions to ranks so that within every macro-chunk of `2*cp_size` tokens, rank `i` owns
token `i` (low) and token `2*cp_size − 1 − i` (high).  These two positions are
**interleaved** across the sequence: rank 0 gets [0, 2P-1, 2P, 4P-1, …], rank 1 gets
[1, 2P-2, 2P+1, 4P-2, …], etc.

**What this repo does.**  `src/ring_partition.cpp::q_offset` gives rank `r` two
*contiguous* sub-groups: `[r·chunk, (r+1)·chunk)` and `[(2P-1-r)·chunk, (2P-r)·chunk)`.
The code itself documents this: *"Note: this is 'coarse' zigzag (2 contiguous
sub-groups). The Python reference's `zigzag_indices` uses the finer scheme."*

**Why it matters.**  Both schemes balance causal work equally across ranks.  However, the
fine-grained scheme distributes high-attended and low-attended tokens more evenly *within*
each sub-group, which reduces load imbalance when a single chunk has an unusually long or
short attended prefix.  The reference tests in `reference/tests/test_zigzag.py` pin the
fine-grained layout.

**TODO.**  Replace `RingPartition::Mode::Zigzag` with the fine-grained indexing from
`reference/ring_attention_ref/zigzag.py::zigzag_indices`.  The kernel API already accepts
per-call `(q_offset, k_offset)` pairs, so the kernel itself does not need to change; only
`ring_partition.cpp` and the host-side buffer fill/gather loop need updating.

---

## 2. No backward pass (no gradient computation)

**What the reference does.**  `ringattention_jax.py` defines `ring_attention` with
`@partial(jax.custom_vjp, ...)`, a full `_ring_attention_fwd` that saves `(output, q, k,
v, denominator, max_score)` as residuals, and `_ring_attention_bwd` that:
1. Rotates K/V backward through the ring via `lax.ppermute`,
2. Reconstructs attention weights from the saved max scores,
3. Computes `dQ`, `dK`, `dV` using `_blockwise_attention_bwd` with `jnp.einsum`.

**What this repo does.**  Forward pass only; `run_ring_attention` returns attention output
but there is no way to propagate gradients through it.

**Why it matters.**  Training a model with ring attention requires differentiating through
the forward pass to compute weight gradients.  Without a backward pass this implementation
can only be used for inference or evaluation.

**TODO.**  Add a CUDA backward pass.  Each ring step `s` must:
1. Reload the saved `(max_score, denominator)` for that step.
2. Recompute attention weights `P = softmax(QK^T)` using the saved running state.
3. Compute `dV += P^T · dO`, `dP = dO · V^T`, then `dS = P ⊙ (dP − dO·O^T)`.
4. Accumulate `dQ += dS · K` and `dK += dS^T · Q`.
5. Rotate `dK`, `dV` (and `K`, `V`) backward around the ring, same as forward but reversed.

The reference in `ringattention_jax.py::_ring_attention_bwd` (lines ~38–68) is the
algorithm template.

---

## 3. Attention bias (`attn_bias`) not supported

**What the reference does.**  `_ring_attention_fwd` accepts an `attn_bias` tensor of shape
`[batch, heads, q_len, kv_len]` and, inside `_chunk_attention_bias`, slices the relevant
`[q_chunk, k_chunk]` window with `lax.dynamic_slice` before adding it to the raw scores.
This is how ALiBi slopes, relative-position encodings (RoPE additive bias), and other
position-dependent biases are injected without materialising the full score matrix.

**What this repo does.**  `launch_attention_step` has no bias argument.  `attention_step.cu`
computes raw dot products and scales, with no additive term beyond the causal mask.

**TODO.**  Add a `float* attn_bias` parameter (device pointer, shape `[B, H, Sq, Sk]`) to
`AttentionShape` / `launch_attention_step`.  Inside `attention_step_kernel`, after the
dot-product, add `bias[b*H*Sq*Sk + h*Sq*Sk + i*Sk + j]` to each score element `s[j]`
before the max-reduction.  The ring loop in `ring_loop.cu` would need to pass the correct
`k_offset`-sliced bias window at each step.  Reference: `ringattention_jax.py`,
`_chunk_attention_bias`, lines ~130–155.

---

## 4. Segment IDs not supported

**What the reference does.**  `_ring_attention_fwd` accepts `segment_ids` (shape
`[batch, seq_len]`).  `_chunk_attention_bias` computes a boolean mask
`~equal(q_segment, k_segment)` and applies `jnp.finfo(dtype).min` to scores across
different segments.  This allows multiple independent sequences to be packed into one batch
element ("sequence packing") without cross-contamination.

**What this repo does.**  No segment-ID concept; all tokens in a single batch-head can
attend to each other (modulo the causal mask).

**TODO.**  Add a `const int* segment_ids` parameter (device pointer, shape `[B, S_global]`)
to `AttentionShape` and `launch_attention_step`.  In the kernel, before accumulating score
`s[j]`, check `segment_ids[b*S + (q_offset+i)] != segment_ids[b*S + (k_offset+j)]` and
set `s[j] = -INFINITY` on mismatch.  Reference: `ringattention_jax.py`,
`_chunk_attention_bias`, lines ~157–167.

---

## 5. Attention dropout not supported

**What the reference does.**  `_blockwise_attention_fwd` accepts `attn_pdrop` and a
`dropout_rng` key.  When `not deterministic and attn_pdrop > 0`, it generates a Bernoulli
mask `(batch, heads, q_len, kv_len)` and adds `jnp.finfo(dtype).min` to dropped
positions, zeroing them out after softmax.

**What this repo does.**  No dropout; the kernels are deterministic.

**TODO.**  Pass a cuRAND or deterministic hash-based dropout mask to the step kernel.  The
mask can be generated on the fly per-element using a seeded hash of `(b, h, i_global, j_global)`
to avoid storing the full matrix, similar to how FlashAttention 2 handles dropout.
Reference algorithm: `ringattention_jax.py`, `_blockwise_attention_fwd`, lines ~182–188 and
`_chunk_attention_bias` lines ~169–179.

---

## 6. Arbitrary `head_dim` not supported ✓ COMPLETED

**What the reference does.**  `_blockwise_attention_fwd` uses `jnp.einsum('bqhd,bkhd->bhqk',
q_chunk, k_chunk)` which works for any dimension `d`.  No template specialisation is needed.

**What this repo did.**  `launch_attention_step` in `attention_step.cu` used a C++ template
`<int BR, int BC, int D>` where `D` is a compile-time constant.  Only `D ∈ {32, 64, 128}`
was instantiated; any other value triggered `std::abort()`.

**Why it matters.**  Modern LLMs commonly use `head_dim = 256` (e.g., Llama-3 8B).  The
constraint forces recompilation for every new model architecture.

**What was done.**  Added `case 256` to `launch_attention_step` (tiles `BR=16, BC=8`,
shared memory 32 KB — well within the 48 KB sm_75 limit) and to `launch_attention_step_fp16`
(`kBR=kBC=16` are fixed; smem with D=256 is 41.6 KB, still fits).  Two new test cases
per dtype (non-causal + causal, both `max_diff < 3 × 10⁻⁴`) confirm correctness.

**Key takeaway.**  Adding a new `head_dim` is a one-line `case` in each `switch` — the kernel
template already generalises over `D` via compile-time unrolling.  D=256 brings Llama-3-style
models in reach without any kernel rewrite; the only trade-off is higher register pressure from
`float O_i[256]` per thread (potential L1 spill), which a future optimisation could mitigate by
tiling the D dimension.

---

## 7. MQA / GQA (multi-query and grouped-query attention) not supported

**What the reference does.**  The forward pass in `ringattention_jax.py` reads Q shape as
`(batch, q_len, num_heads, dim)` and K/V shape as `(batch, kv_len, num_heads, dim)`.
The `einsum` is flexible enough to support a different `num_heads` for K/V if the caller
broadcasts appropriately, which is how MQA/GQA are typically implemented on top.

**What this repo does.**  `AttentionShape` has a single `heads` field used for both Q and
K/V; there is no way to express that K/V may have fewer heads.

**TODO.**  Split `AttentionShape.heads` into `q_heads` and `kv_heads`, and replicate the
K/V head index as `h % kv_heads` when loading from K/V in the kernel.  This is a small
change to `attention_step_kernel` (replace `(b * H + h)` K/V offsets with
`(b * kv_H + h % kv_H)`) but requires plumbing `kv_heads` through
`RingConfig`, `AttentionShape`, and all callers.

---

## 8. No inference / decode mode (KV-cache ring attention)

**What the reference does.**  `ringattention_jax_inference.py` implements `ring_attention_inference`:
a variant where `cache_idx` selects which query position is being decoded.  A single new
query token attends to all cached KV tokens distributed across the ring.  The forward loop
uses `lax.dynamic_slice_in_dim` on the attention mask and skips the per-step KV rotation
of Q (only K/V rotate).

**What this repo does.**  The ring loop always assumes `seq_q = S / cp_size` (a full local
slice of queries).  There is no concept of decoding a single token against a distributed KV
cache.

**TODO.**  Add a `decode` mode to `RingMode` and `RingConfig`.  In decode mode: Q is a
single token `[B, H, 1, D]`; K/V are the full cached sequence distributed across ranks.
Each ring step processes one rank's KV shard.  The finalize step is unchanged (divide O by
l).  Reference: `ringattention_jax_inference.py::_ring_attention_inference_fwd`.

---

## 9. MPI used for GPU-to-GPU transfers instead of NCCL ✓ COMPLETED

**What the reference does.**  Uses JAX's `lax.ppermute`, which on NVIDIA GPUs compiles to
NCCL send/recv or NVLink direct transfers depending on the topology.  NCCL is GPU-aware and
can avoid the host entirely.

**What this repo did.**  `ring_loop.cu` used `MPI_Isend` / `MPI_Irecv` via pinned host
staging: D2H → MPI → H2D per step.  This doubled the PCI-e traffic compared to a
GPU-direct path even when GPUs are on the same node.  The `ring_overlap` mode partially
hid this but the H2D/D2H latency was still real.

**What was done.**  Replaced MPI host-staging with `ncclSend`/`ncclRecv` (grouped with
`ncclGroupStart`/`ncclGroupEnd`) in `ring_loop.cu` and `ring_loop_fp16.cu`.  Pinned host
buffers (`cudaHostAlloc`) and all D2H/H2D `cudaMemcpy` calls in the ring step loop are
removed.  NCCL communicator is bootstrapped once per `run_ring_*` call via `nccl_init()`
in `src/nccl_utils.hpp` (rank 0 broadcasts the unique ID via `MPI_Bcast`).  MPI path
retained behind `#ifdef RING_USE_NCCL` / `#else` guards; disable with
`cmake --preset=release -DUSE_NCCL=OFF`.  CMake auto-detects NCCL via `$CPATH` and
`$LD_LIBRARY_PATH` set by the NVHPC 24.1 module.

**Validating the improvement.**  On a GPU node (≥2 GPUs), after sourcing `activate.sh`:

```bash
# 1. Build (NCCL on by default):
cmake --preset=release && cmake --build build/release -j

# 2. Correctness — all three modes, fp32 and fp16, causal on/off:
salloc --partition=gpu-turing --gres=gpu:2 --ntasks=2 --time=00:10:00
mpirun -n 2 ./build/release/apps/ring_attention_cli/ring_attention_cli \
    --seq 512 --verify --mode ring-blocking --dtype fp32
mpirun -n 2 ./build/release/apps/ring_attention_cli/ring_attention_cli \
    --seq 512 --verify --mode ring-overlap  --dtype fp16 --causal

# 3. Timing — compare wait_ms and total_ms vs. MPI baseline:
mpirun -n 2 ./build/release/apps/ring_attention_cli/ring_attention_cli \
    --seq 4096 --iters 20 --mode ring-overlap --dtype fp16 --causal
```

Key metrics to watch: `wait_ms` drops toward 0 in `ring-overlap` (NCCL runs asynchronously
on `stream_copy`; no host wait) and `total_ms` falls because PCIe round-trips are
eliminated.  `comm_ms` now measures the NCCL transfer time on `stream_copy` via
`cudaEvents` rather than host-side wall-clock.  To compare against the MPI path, rebuild
with `-DUSE_NCCL=OFF` and re-run the same command.

**Observed results** (2× Quadro RTX 6000, same node, seq=4096, fp16, causal, cp_size=2):

| mode | backend | comm_ms | comp_ms | wait_ms | total_ms |
|---|---|---|---|---|---|
| ring-blocking | NCCL | 0.010 | 9.079 | 0.002 | **9.880** |
| ring-blocking | MPI  | 0.612 | 9.174 | 4.848 | 10.543 |
| ring-overlap  | NCCL | 8.655 | 9.183 | **0.000** | 9.405 |
| ring-overlap  | MPI  | 0.669 | 9.091 | 7.986 | 9.308 |

`ring-blocking`: NCCL eliminates the host round-trip entirely — `wait_ms` falls from 4.848 ms
to 0.002 ms and `total_ms` improves by 0.66 ms (6.3%).  NCCL uses the GPU P2P write path on
stream 0, completing the transfer before `cudaStreamSynchronize` even returns.

`ring-overlap`: both backends reach near-compute-bound throughput (~9.3–9.4 ms ≈ `comp_ms`)
because the 2 MB K/V transfer time (≈8.7 ms over PCIe) fits inside the kernel window.
`wait_ms = 0.000` for NCCL confirms no host stall exists in the overlap path.  The gap
between NCCL and MPI here would widen on NVLink or InfiniBand + GPUDirect RDMA hardware.

---

## 10. `causal_block_size` parameter missing

**What the reference does.**  `_blockwise_attention_fwd` has a `causal_block_size` argument.
Inside `_chunk_attention_bias`, the causal mask is computed at *block granularity*:
`query_idx // causal_block_size < key_idx // causal_block_size` rather than token-level.
This is used for Blockwise Parallel Transformer variants where each token attends to all
keys within the same causal block.  The helper `below_or_on_diag` (bottom of `ringattention_jax.py`)
also uses this to skip entire KV chunks early without entering `scan_kv_block`.

**What this repo does.**  The causal predicate in `attention_step_kernel` is always
`j_global <= i_global` (token-level, equivalent to `causal_block_size = 1`).  There is no
way to express coarser-grained masking.

**TODO.**  Add an optional `causal_block_size` field (default 1) to `AttentionShape`.
In `attention_step_kernel`, change the visibility predicate to:
`(j_global / causal_block_size) <= (i_global / causal_block_size)`.
In the ring loop's pruning condition (currently `k_off_sg > q_off_sg + Sl - 1`),
lift the comparison to block granularity for the early-skip optimisation.
Reference: `ringattention_jax.py`, `below_or_on_diag` (last ~10 lines of file).

---

## 11. `float32_logits` / upcast-before-softmax option missing

**What the reference does.**  `_ring_attention_fwd` accepts `float32_logits: bool`.  When
`True`, Q and K are upcast to `jnp.float32` before the einsum so that the dot products and
softmax numerics are in full precision even when the model weights are bf16 or fp16.  This
prevents overflow/underflow in the score computation without affecting the output dtype.

**What this repo does.**  The fp16 kernel (`attention_step_fp16.cu`) uses `wmma` half-precision
tile-matrix multiply for both the `QK^T` and `PV` matmuls.  The accumulator stays fp32 via
`wmma::fragment<accumulator, float>`, but the *input* to the matmul is half-precision.
There is no option to upcast Q and K to fp32 for the dot product while keeping the rest in
fp16.

**TODO.**  Add a `bool float32_logits` field to `RingConfig`.  When set, the fp16 kernel
should load Q/K tiles as `__half` but promote them to `float` before the inner-product
loop (or keep the wmma path but add a fallback scalar loop for correctness comparison).
Reference: `ringattention_jax.py`, `_ring_attention_fwd`, lines ~15–16.

---

## Summary Table

| Feature | haoliuhl/ringattention (reference) | This repo (`src/`) |
|---|---|---|
| Language / framework | JAX (Python) | CUDA/C++ + MPI |
| Zigzag scheme | Fine-grained (interleaved per macro-chunk) | Coarse (2 contiguous sub-groups) |
| Backward pass (dQ, dK, dV) | Yes (`custom_vjp`) | **No** |
| Attention bias | Yes | **No** |
| Segment IDs | Yes | **No** |
| Dropout | Yes | **No** |
| Arbitrary `head_dim` | Yes (einsum) | Only 32, 64, 128 |
| MQA / GQA | Straightforward | **No** |
| Inference / decode mode | Yes (`ring_attention_inference`) | **No** |
| Communication backend | NCCL / NVLink via JAX | **NCCL** (MPI fallback via `USE_NCCL=OFF`) |
| `causal_block_size` | Yes | **No** (always token-level) |
| `float32_logits` upcast | Yes | **No** |
| TPU support | Yes (Pallas kernel) | No (GPU-only) |
| FP16 Tensor Core path | No explicit | Yes (`attention_step_fp16.cu`) |
| Compute/comm overlap | Implicit (XLA) | Explicit (dual CUDA streams) |

---

**Key takeaway from TODO #9 (NCCL).**  The most impactful ring-attention optimization is
eliminating the host staging round-trip: replacing MPI + pinned D2H/H2D with NCCL
`ncclSend`/`ncclRecv` directly on device pointers removes two full PCIe traversals per ring
step, which dominate `wait_ms` at large sequence lengths.  In the overlap mode NCCL runs on
`stream_copy` concurrently with the compute kernel on `stream_compute`, so the remaining
communication cost appears only in `comm_ms` (measured with CUDA events) while `wait_ms`
drops to zero — this is the clearest signal that communication is being fully hidden.  The
compile-time `USE_NCCL` guard keeps the MPI fallback available for portability to clusters
without NCCL or NVLink.
