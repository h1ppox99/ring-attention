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
    n_splits: int = 2,
) -> torch.Tensor:
    """Full attention output, partitioned into per-rank shards.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(batch, heads, seq, head_dim)``.
    cp_size : int
        Number of ranks in the context-parallel ring.
    causal : bool
        Apply causal masking when ``True``.
    zig_zag : bool
        Use zig-zag token assignment for load balance; otherwise contiguous blocks.
    n_splits : int
        Number of sub-groups per rank when ``zig_zag=True``. Default is 2.

    Returns
    -------
    torch.Tensor
        Shape ``(cp_size, batch, heads, seq // cp_size, head_dim)``.

    Raises
    ------
    ValueError
        If ``seq`` is not divisible by ``cp_size``.
    """
    seq = q.shape[-2]
    if seq % cp_size != 0:
        raise ValueError(f"seq_len={seq} must be divisible by cp_size={cp_size}")

    out = full_attention(q, k, v, causal=causal)

    if zig_zag:
        return partition(out, zigzag_indices(seq, cp_size=cp_size, n_splits=n_splits))
    return out.reshape(*out.shape[:-2], cp_size, seq // cp_size, out.shape[-1]).movedim(-3, 0)
