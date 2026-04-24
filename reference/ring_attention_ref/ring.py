"""Simulated ring attention — returns per-rank output shards.

Since the per-rank output is a deterministic function of the full attention
result (just sliced by the rank's token ownership), the oracle computes full
attention once and partitions the result. The CUDA implementation accumulates
the same numbers via online softmax across ring steps; our tests pin down
*what* it must produce, not *how*.
"""

from __future__ import annotations

import torch

from ring_attention_ref.oracle import full_attention
from ring_attention_ref.zigzag import partition, zigzag_indices


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cp_size: int,
    causal: bool,
    zig_zag: bool,
) -> torch.Tensor:
    """Full attention output, partitioned into per-rank shards.

    Returns shape (cp_size, batch, heads, seq // cp_size, head_dim).
    With zig_zag=True the rank mapping follows `zigzag_indices(seq, cp_size)`;
    otherwise ranks own contiguous token blocks.
    """
    seq = q.shape[-2]
    if seq % cp_size != 0:
        raise ValueError(f"seq_len={seq} must be divisible by cp_size={cp_size}")

    out = full_attention(q, k, v, causal=causal)

    if zig_zag:
        return partition(out, zigzag_indices(seq, cp_size=cp_size))
    return out.reshape(*out.shape[:-2], cp_size, seq // cp_size, out.shape[-1]).movedim(-3, 0)
