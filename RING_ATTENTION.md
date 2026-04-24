# Ring Attention — algorithmic reference

## Why context parallelism (CP)?

Tensor parallelism + sequence parallelism distributes weights and activations, but the
attention block still sees the full sequence per GPU. Even with full recomputation (~30%
compute overhead), per-layer boundary activations scale linearly with sequence length.

CP splits the input sequence across GPUs for the *entire* model (not just SP regions):
- MLPs and LayerNorms are unaffected (per-token independence).
- No expensive weight communication (inputs split, not weights).
- Gradients synchronized via all-reduce over the CP group after backprop.
- Attention is the exception — each token needs K/V from all sequence positions.

Memory impact example (8B model): no parallelism → 1k tokens; TP=2 → 16k; TP=2 CP=4 → 64k.

## Ring Attention

Each GPU holds a Q/K/V slice (1/cp_size of the sequence). One ring pass = `cp_size` steps:

1. **Non-blocking send** current K/V to the next rank.
2. **Compute** local attention with current K/V (overlaps with step 1).
3. **Receive** K/V from the previous rank → go to step 1.

Output accumulates via **online softmax**: per step, update running `(m, ℓ, O)` (max,
sum-of-exp, weighted output) — same numerics as FlashAttention but across ranks.

## Zig-Zag assignment (causal balance)

Naive sequential token assignment causes severe load imbalance under causal masking: the
first GPU sees only early tokens and has the most non-masked K/V pairs; later GPUs are mostly
masked out in early steps.

**Fix**: interleave first and last tokens on the same GPU.
For `cp_size` GPUs, GPU `i` holds tokens at indices `i` and `2*cp_size - 1 - i`
(and multiples thereof for longer sequences). This makes every GPU's portion of the causal
matrix equally dense.

## Communication strategies

| Strategy | Memory | Complexity |
|---|---|---|
| All-gather (ZeRO-3 style) | High — stores all K/V at once | Single large collective |
| Ring (all-to-all) | Low — one extra chunk at a time | Many small steps, overlap with compute |

Ring is preferred: memory scales with 1/cp_size; latency hides behind computation.
