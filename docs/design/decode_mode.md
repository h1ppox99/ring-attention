# Decode Mode + Distributed KV Cache — Design

**Status**: design accepted; Phase 1 + Phase 2 in flight.
**Scope**: extend the ring-attention machinery to support autoregressive decoding
of a single token against a KV cache that is sharded across the ring.
**Out of scope (deliberately)**: paged KV cache, continuous batching, serving
stack, full-model integration (Llama/Gemma). Those remain follow-ups.

## Motivation

Today `run_ring_attention` always assumes `seq_q == seq_k / cp_size` (a balanced
prefill). It cannot decode: there is no notion of a query that is one token wide
against a KV history of arbitrary length distributed around the ring.

Adding a decode path is the natural next algorithmic extension. It is
self-contained inside `src/`, exercises the ring in a fundamentally new regime
(small Q, large K → latency-bound rather than throughput-bound), and produces a
new benchmark axis (per-token decode latency and prefill TTFT vs. `cp_size` and
context length) that the current pipeline cannot measure.

The Python reference (`ringattention_jax_inference.py::_ring_attention_inference_fwd`)
is the algorithmic template; this document adapts it to the C++/CUDA stack.

## Algorithm

### Prefill (existing path, unchanged)

`run_ring_attention(cfg)` with `seq = prompt_len` produces the full prefill
output as today and — this is the new behavior — leaves the per-rank K/V shard
sitting in the cache rather than discarding it.

After prefill, every rank `r` holds:

- a local KV cache of shape `(B, H_kv, S_r, D)` where `S_r = prompt_len / cp_size`
  under coarse zigzag (the current partitioning),
- the prefill output `O` of shape `(B, H, S_r, D)` (returned to the caller).

### Decode (new path)

To decode token `t = prompt_len + d` (for `d = 0, 1, 2, …`):

1. **Q is replicated on every rank**: the new query row has shape `(B, H, 1, D)`.
   For a single-rank-owned QKV projection upstream, this means rank 0 computes
   the projection from the previously-decoded token and broadcasts (1 token × H ×
   D is tiny — broadcast cost is negligible compared to KV traffic). For now, the
   reference and the CLI driver will assume Q is already replicated.
2. **K/V for the new token attach to one rank.** Under coarse-zigzag, token `t`
   belongs to a specific rank `r_t`. That rank appends its `K_step`, `V_step`
   row to its local cache; all other ranks leave their cache unchanged. The
   global cache length grows from `S` to `S + 1`.
3. **Ring loop**: K/V rotate around the ring exactly as in prefill, but Q does
   not. Each ring step has `seq_q = 1` and `seq_k = S_r` (or `S_r + 1` on the rank
   that just appended). Online-softmax accumulator `(m, ℓ, O)` is per-row, but
   "per-row" here is one row.
4. **Causal mask**: the decoded token attends to every cached position with
   `j_global <= t`. With contiguous prefill + append, this is trivially "all
   positions in cache" because the cache only ever holds positions `≤ t`.
5. **Finalize**: divide `O` by `ℓ` exactly as today.

The kernel work per ring step is `O(H · S_r · D)` (one query row × KV rows ×
head dim) — orders of magnitude smaller than prefill. This shifts the
bottleneck from compute (matmul) to communication (NCCL latency).

### Zigzag and decode

Coarse zigzag was designed for *prefill* causal load balance. In decode there is
only one query token, so load-balance arguments don't apply directly. However,
the *cache layout* still has to follow whatever partitioning was used at prefill
time — otherwise the cached `(K, V)` entries don't match the positions the new
query expects to attend to.

For now we use the same coarse-zigzag layout that prefill uses. Rank `r_t` for
token `t` follows the existing `q_offset(r, t, cp_size)` mapping. The reference
keeps the same `partition` / `unpartition` helpers.

## Data structures

### Python reference (`reference/ring_attention_ref/inference.py`)

Two new objects:

```python
@dataclass
class KVCache:
    k: torch.Tensor   # (cp_size, B, H_kv, S_max, D), per-rank shard
    v: torch.Tensor   # (cp_size, B, H_kv, S_max, D)
    current_len: torch.Tensor  # (cp_size,) int64, per-rank fill level

def prefill(q, k, v, *, cp_size, causal, zig_zag) -> tuple[torch.Tensor, KVCache]:
    """Run full prefill, return (per-rank output shards, populated cache)."""

def decode_step(q_new, k_new, v_new, cache, *, cp_size, zig_zag) -> tuple[torch.Tensor, KVCache]:
    """One autoregressive step. q_new/k_new/v_new are (B, H, 1, D).

    Returns the new attention output (B, H, 1, D) and the updated cache.
    """
```

Both functions are pure: `decode_step` returns a new cache rather than mutating
in place. This keeps the reference simple and easy to test (no fixture cleanup,
no shared state across parametrized tests).

Correctness oracle: prefill of length `S` then `N` decode steps must equal a
full prefill of length `S + N` on the concatenated Q/K/V, sliced to the last
`N + 1` rows (the new tokens plus the last prefill row if we want to spot-check).

### C++ side (`src/kv_cache.{hpp,cu}`)

```cpp
struct DeviceKVCache {
  DeviceTensor<float>  k;   // [B, H_kv, S_max, D] for fp32 path
  DeviceTensor<float>  v;
  // (fp16 specialization adds DeviceTensor<__half> mirrors; see below)
  int batch;
  int kv_heads;
  int s_max;
  int head_dim;
  int current_len = 0;      // number of populated KV rows

  // Append one token's K/V into slot `current_len` and bump the counter.
  // K_step / V_step are device pointers of shape [B, H_kv, 1, D].
  void append(const float* k_step, const float* v_step, cudaStream_t stream = 0);
};
```

Storage rationale:

- **Contiguous up-front allocation** (`S_max` reserved at construction). No
  paging, no block tables. This matches the "minimum viable" caching strategy
  the user signed off on and keeps the pointer arithmetic identical to the
  existing attention kernels — `seq_k` is just `current_len` at call time.
- **One cache per rank** (no global cache object). The cache lives next to the
  ring loop on each rank; cross-rank rotation is done by the existing
  send/recv plumbing.
- **fp32 + fp16 variants**: a templated `DeviceKVCache<T>` mirrors how
  `DeviceTensor<T>` already specializes for `float` and `__half`. The decode
  kernels then accept the matching pointer types.

### Decode kernel

A new entry point:

```cpp
void launch_attention_decode_step(
    const float* q,          // [B, H, 1, D]
    const float* k,          // [B, H_kv, seq_k, D]   (one rank's KV shard)
    const float* v,          // [B, H_kv, seq_k, D]
    float* out,              // [B, H, 1, D]          (accumulator)
    float* m, float* l,      // [B, H, 1]             (online-softmax state)
    const AttentionShape& shape,
    int q_pos_global,        // absolute position of the decoded token
    int k_offset,            // global position of k[..., 0, :]
    bool causal,
    cudaStream_t stream = 0);
```

Implementation: this is just `launch_attention_step` with `seq_q = 1`. The
existing flash kernel already handles `seq_q < BR` (the BR tile loop runs
exactly one iteration). For Phase 3 we will dispatch through the same
templated body and only special-case if profiling shows the one-row case wants
a tighter implementation (e.g., warp-shuffle reduction instead of shared-mem
softmax).

The fp16 path mirrors this with `__half` Q/K/V and the existing wmma matmuls.

### Decode ring loop

```cpp
struct DecodeStepResult {
  RingResult timings;
  // out is written back into the caller's buffer; no copy here.
};

DecodeStepResult run_ring_decode_step(
    const float* q_new,        // [B, H, 1, D] (replicated on every rank)
    const float* k_new,        // [B, H_kv, 1, D] (only meaningful on owner rank)
    const float* v_new,        // [B, H_kv, 1, D] (only meaningful on owner rank)
    int new_token_pos,         // global position of the new token
    DeviceKVCache<float>& cache,
    float* out,                // [B, H, 1, D] output for this rank's slice
    const RingConfig& cfg);
```

This will:

1. On rank `r_t = owner(new_token_pos)`: append `k_new`, `v_new` to local
   cache.
2. Init online-softmax state.
3. Run `cp_size` ring steps: rotate K/V around the ring (NCCL on
   `stream_copy`, kernel on `stream_compute`, same overlap pattern as
   prefill), call `launch_attention_decode_step` at each step.
4. Finalize.

Open question: does the decoded token attend only to its own rank's cache plus
rotated copies of the others, or does it need any special handling at the
seam? For coarse zigzag, the answer is "no special handling" — every cached
position has a globally consistent `(k_offset, j)` index and the causal mask
predicate works uniformly. We'll re-examine this when we move to fine-grained
zigzag (TODO #1 in the project root).

## Open design choices (locked in here)

1. **`RingMode` extension or orthogonal `RunMode`?**
   Decision: **orthogonal `RunMode { Prefill, Decode }`**. `RingMode` keeps
   meaning inside decode (we still want overlap for the cp_size KV-rotation
   steps, even though each step is small). The CLI gains a `--run` flag in
   addition to the existing `--mode`.

2. **KV cache ownership: per-call or session object?**
   Decision: **session object** (`RingSession`). The session owns the
   `DeviceKVCache`, the persistent online-softmax buffers, the NCCL
   communicator, and the streams. `run_ring_attention(session, …)` and
   `run_ring_decode_step(session, …)` both take a session reference. This
   is fewer lines than threading raw pointers through the API once we have
   multiple decode steps, and it makes the eventual "multi-prompt" extension
   cheap (one session per prompt).

3. **fp16 only for decode, or both?**
   Decision: **support both, prioritize fp16**. Decode is latency-bound; fp16
   matches what production inference uses. But the fp32 path is trivial to
   keep (the kernel already exists) and is useful as a numerical sanity
   check during development. We will not add separate fp32-only optimization
   work for decode.

## Phasing

- **Phase 0** — Mode cleanup. Make `ring-overlap` the CLI default, label the
  other modes as baselines in `--help` and README. Pure docs/UX change.
- **Phase 1** — Python reference (this doc's `inference.py` + pytest cases).
  Pinning correctness before we touch CUDA.
- **Phase 2** — `DeviceKVCache` + unit tests in C++. No kernel changes yet;
  exercises append / round-trip / bounds.
- **Phase 3** — `launch_attention_decode_step` and its fp16 variant.
  Standalone tests against the Python reference (single rank, no ring yet).
- **Phase 4** — `run_ring_decode_step`, the multi-rank decode loop. Multi-rank
  smoke tests under `mpirun -n 2`, comparing against the Python reference
  for prefill-then-decode.
- **Phase 5** — CLI + benchmarks. Extend `ring_attention_cli` (or add a new
  `apps/ring_inference_cli/`) with `--run decode --decode-tokens N`. Produce
  the two plots that justify the work: TTFT vs. `S_prompt × cp_size` and
  per-token decode latency vs. `S_prompt × cp_size`.

## Correctness gate

Single source of truth: the Python reference. For every CUDA-side change in
Phases 3–4, the test harness will:

1. Generate Q/K/V/new-token inputs from a fixed seed.
2. Run Python prefill + N decode steps to get the expected outputs.
3. Run the CUDA path on the same inputs.
4. Compare with `atol=1e-3, rtol=1e-3` (fp16) or `atol=1e-5, rtol=1e-5` (fp32).

This mirrors the existing `--verify` gate for prefill and reuses the same
`cpu_attention` machinery where possible.

## Benchmarks (Phase 5)

The decode mode justifies itself only if we can show:

- **TTFT** scales the way the prefill ring scales (it's the same code path,
  so this is just a regression check).
- **Per-token decode latency** is *flat or sublinear* in `cp_size` for fixed
  context length (the comm cost per step shrinks as the per-rank shard
  shrinks; if comm latency dominates over per-step compute, we see
  diminishing returns).
- **Per-token decode latency** grows linearly in context length (more cached
  KV → more dot products per step).

Both plotted across `cp_size ∈ {1, 2, 4}` and `S_prompt ∈ {1k, 4k, 16k, 64k}`
on `gpu-turing`.
