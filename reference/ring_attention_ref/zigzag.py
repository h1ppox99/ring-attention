"""Zig-zag token partitioning for causal ring attention load balance.

Rank i owns positions (i, 2*cp_size - 1 - i) within each chunk of length 2*cp_size.
"""

from __future__ import annotations

import torch


def zigzag_indices(seq_len: int, cp_size: int) -> torch.Tensor:
    """Token indices owned by each rank under zig-zag assignment.

    Parameters
    ----------
    seq_len : int
        Total sequence length; must be divisible by ``2 * cp_size``.
    cp_size : int
        Number of ranks in the ring.

    Returns
    -------
    torch.Tensor
        ``LongTensor`` of shape ``(cp_size, seq_len // cp_size)`` where row ``i``
        lists the original token positions assigned to rank ``i``.

    Raises
    ------
    ValueError
        If ``seq_len`` is not divisible by ``2 * cp_size``.
    """
    chunk = 2 * cp_size
    if seq_len % chunk != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by 2*cp_size={chunk}")
    num_chunks = seq_len // chunk
    chunk_starts = torch.arange(num_chunks) * chunk  # (num_chunks,)
    ranks = torch.arange(cp_size).unsqueeze(1)  # (cp_size, 1)
    # Per chunk, each rank owns the (i)-th and (2*cp_size - 1 - i)-th positions.
    lows = chunk_starts.unsqueeze(0) + ranks  # (cp_size, num_chunks)
    highs = chunk_starts.unsqueeze(0) + (chunk - 1 - ranks)  # (cp_size, num_chunks)
    return torch.stack([lows, highs], dim=-1).reshape(cp_size, -1)


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
