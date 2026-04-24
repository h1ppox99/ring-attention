"""Zig-zag token partitioning for causal ring attention load balance.

Rank i owns positions (i, 2*cp_size - 1 - i) within each chunk of length 2*cp_size.
"""

from __future__ import annotations

import torch


def zigzag_indices(seq_len: int, cp_size: int) -> torch.Tensor:
    """Indices of the original tokens owned by each rank.

    Returns a LongTensor of shape (cp_size, seq_len // cp_size) where row i lists
    the original positions assigned to rank i, in the order the rank stores them.
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
    """Gather x along the sequence dim (-2) per rank.

    x:        (batch, heads, seq, head_dim)
    indices:  (cp_size, seq // cp_size) from zigzag_indices
    returns:  (cp_size, batch, heads, seq // cp_size, head_dim)
    """
    cp_size, shard_len = indices.shape
    gathered = x.index_select(-2, indices.flatten())
    new_shape = (*gathered.shape[:-2], cp_size, shard_len, gathered.shape[-1])
    return gathered.reshape(new_shape).movedim(-3, 0)


def unpartition(shards: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Scatter per-rank shards back to a single sequence-ordered tensor.

    shards:   (cp_size, batch, heads, seq // cp_size, head_dim)
    indices:  (cp_size, seq // cp_size)
    returns:  (batch, heads, seq, head_dim)
    """
    # Move cp_size back before the seq dim, then flatten → (batch, heads, cp_size*shard_len, ...).
    flat = shards.movedim(0, -3).flatten(-3, -2)
    out = torch.empty_like(flat)
    out.index_copy_(-2, indices.flatten(), flat)
    return out
