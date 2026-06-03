"""Zig-zag token partitioning for causal ring attention load balance.

Rank i owns n_splits positions within each macro-chunk of length n_splits*cp_size.
Sub-groups are paired inward: (0, n-1), (1, n-2), ... to balance early/late work.
"""

from __future__ import annotations

import torch


def zigzag_indices(seq_len: int, cp_size: int, n_splits: int = 2) -> torch.Tensor:
    """Token indices owned by each rank under zig-zag assignment.

    Parameters
    ----------
    seq_len : int
        Total sequence length; must be divisible by ``n_splits * cp_size``.
    cp_size : int
        Number of ranks in the ring.
    n_splits : int
        Number of sub-groups per rank (>= 2). Each rank owns ``n_splits``
        evenly-spaced chunks of the sequence, paired symmetrically so that
        early (cheap under causal) and late (expensive) chunks balance out.
        Default is 2 for backward compatibility.

    Returns
    -------
    torch.Tensor
        ``LongTensor`` of shape ``(cp_size, seq_len // cp_size)`` where row ``i``
        lists the original token positions assigned to rank ``i``.

    Raises
    ------
    ValueError
        If ``seq_len`` is not divisible by ``n_splits * cp_size``.
    """
    chunk = n_splits * cp_size
    if seq_len % chunk != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by n_splits*cp_size={chunk}")
    num_chunks = seq_len // chunk
    chunk_starts = torch.arange(num_chunks) * chunk  # (num_chunks,)
    ranks = torch.arange(cp_size).unsqueeze(1)  # (cp_size, 1)

    parts: list[torch.Tensor] = []

    for sg in range(n_splits):
        k = sg // 2
        if sg % 2 == 0:
            local_offset = k * cp_size + ranks
        else:
            local_offset = (n_splits - 1 - k) * cp_size + (cp_size - 1 - ranks)
        parts.append(chunk_starts.unsqueeze(0) + local_offset)

    return torch.stack(parts, dim=-1).reshape(cp_size, -1)


def striped_indices(seq_len: int, cp_size: int) -> torch.Tensor:
    """Token indices owned by each rank under striped assignment.

    Token ``i`` is assigned to rank ``i % cp_size``, so rank ``r`` owns the
    arithmetic progression ``r, r + cp_size, r + 2*cp_size, …``. Like zig-zag
    this balances per-rank work under a causal mask, but with a single
    contiguous-in-local-memory shard whose *global* positions are strided.

    Parameters
    ----------
    seq_len : int
        Total sequence length; must be divisible by ``cp_size``.
    cp_size : int
        Number of ranks in the ring.

    Returns
    -------
    torch.Tensor
        ``LongTensor`` of shape ``(cp_size, seq_len // cp_size)`` where row ``r``
        lists the original token positions assigned to rank ``r``, i.e.
        ``indices[r, j] = r + j * cp_size``.

    Raises
    ------
    ValueError
        If ``seq_len`` is not divisible by ``cp_size``.
    """
    if seq_len % cp_size != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by cp_size={cp_size}")
    shard_len = seq_len // cp_size
    ranks = torch.arange(cp_size).unsqueeze(1)  # (cp_size, 1)
    cols = torch.arange(shard_len).unsqueeze(0)  # (1, shard_len)
    return ranks + cols * cp_size  # (cp_size, shard_len)


def partition(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather tensor slices along the sequence dim per rank.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``(batch, heads, seq, head_dim)``.
    indices : torch.Tensor
        Shape ``(cp_size, seq // cp_size)`` from :func:`zigzag_indices`.

    Returns
    -------
    torch.Tensor
        Shape ``(cp_size, batch, heads, seq // cp_size, head_dim)``.
    """
    cp_size, shard_len = indices.shape
    gathered = x.index_select(-2, indices.flatten())
    new_shape = (*gathered.shape[:-2], cp_size, shard_len, gathered.shape[-1])
    return gathered.reshape(new_shape).movedim(-3, 0)


def unpartition(shards: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Scatter per-rank shards back to a sequence-ordered tensor.

    Parameters
    ----------
    shards : torch.Tensor
        Shape ``(cp_size, batch, heads, seq // cp_size, head_dim)``.
    indices : torch.Tensor
        Shape ``(cp_size, seq // cp_size)`` from :func:`zigzag_indices`.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, heads, seq, head_dim)``.
    """
    # Move cp_size back before the seq dim, then flatten → (batch, heads, cp_size*shard_len, ...).
    flat = shards.movedim(0, -3).flatten(-3, -2)
    out = torch.empty_like(flat)
    out.index_copy_(-2, indices.flatten(), flat)
    return out
