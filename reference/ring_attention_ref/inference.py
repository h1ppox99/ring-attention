"""Reference for ring-attention inference (prefill + autoregressive decode).

The oracle does not simulate ring rotation. Its job is to pin down the numerics
the CUDA decode path must reproduce: a single query row attending to the full
distributed KV history.

Prefill populates per-rank K/V caches. Each decode step appends one K/V row to
its owner rank and computes attention of the new query against the concatenated
cache. Concatenation order is irrelevant because attention over keys is
permutation-invariant on the key dimension and no causal mask is needed once
the cache only holds positions <= the current decoded token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import scaled_dot_product_attention

from ring_attention_ref.oracle import full_attention
from ring_attention_ref.zigzag import partition, striped_indices, zigzag_indices


@dataclass
class KVCache:
    """Per-rank K/V cache for distributed inference.

    Parameters
    ----------
    k, v : torch.Tensor
        Shape ``(cp_size, batch, kv_heads, s_max, head_dim)``.
    current_len : torch.Tensor
        ``(cp_size,) int64`` — number of populated KV rows per rank.
    """

    k: torch.Tensor
    v: torch.Tensor
    current_len: torch.Tensor

    @property
    def cp_size(self) -> int:
        """Number of ranks in the ring."""
        return int(self.k.shape[0])

    @property
    def s_max(self) -> int:
        """Cache capacity per rank."""
        return int(self.k.shape[-2])

    def total_len(self) -> int:
        """Total cached tokens across all ranks."""
        return int(self.current_len.sum().item())

    def gather_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate the populated portion of every rank's cache.

        Returns
        -------
        k_full, v_full : torch.Tensor
            Shape ``(batch, kv_heads, total_len, head_dim)``. Rank order is
            preserved but ordering does not affect attention numerics.
        """
        cp_size = self.cp_size
        k_shards: list[torch.Tensor] = []
        v_shards: list[torch.Tensor] = []
        for r in range(cp_size):
            n = int(self.current_len[r].item())
            k_shards.append(self.k[r, :, :, :n, :])
            v_shards.append(self.v[r, :, :, :n, :])
        return torch.cat(k_shards, dim=-2), torch.cat(v_shards, dim=-2)


def prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cp_size: int,
    causal: bool,
    zig_zag: bool,
    striped: bool = False,
    s_max: int | None = None,
) -> tuple[torch.Tensor, KVCache]:
    """Run full prefill and return per-rank output shards plus populated cache.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(batch, heads, seq, head_dim)``. ``heads`` must equal
        ``kv_heads`` (MHA only in the Phase-1 reference).
    cp_size : int
        Number of ranks in the ring.
    causal : bool
        Apply causal masking when ``True``.
    zig_zag : bool
        Use zig-zag token assignment for load balance; otherwise contiguous
        blocks.
    striped : bool, optional
        Use striped token assignment (token ``i`` -> rank ``i % cp_size``).
        Mutually exclusive with ``zig_zag``. Defaults to ``False``.
    s_max : int, optional
        Cache capacity per rank. Defaults to ``seq // cp_size`` (no decode
        headroom); pass a larger value to leave room for decoded tokens.

    Returns
    -------
    out_shards : torch.Tensor
        ``(cp_size, batch, heads, seq // cp_size, head_dim)`` — per-rank
        attention output for the prefill, identical to
        :func:`ring_attention_ref.ring.ring_attention`.
    cache : KVCache
        Cache pre-populated with the prefill K/V partitioned per the chosen
        scheme; the rest of each rank's buffer is zero up to ``s_max``.

    Raises
    ------
    ValueError
        If ``seq`` is not divisible by ``cp_size`` or ``s_max`` is too small.
    """
    batch, heads, seq, head_dim = q.shape
    kv_heads = k.shape[1]
    if heads != kv_heads:
        raise ValueError(
            f"Phase-1 reference is MHA-only: heads={heads} must equal kv_heads={kv_heads}"
        )
    if seq % cp_size != 0:
        raise ValueError(f"seq_len={seq} must be divisible by cp_size={cp_size}")
    if zig_zag and striped:
        raise ValueError("zig_zag and striped are mutually exclusive (pick one scheme)")
    s_r = seq // cp_size
    if s_max is None:
        s_max = s_r
    if s_max < s_r:
        raise ValueError(f"s_max={s_max} must be >= seq/cp_size={s_r}")

    full_out = full_attention(q, k, v, causal=causal)
    if zig_zag or striped:
        if zig_zag:
            idx = zigzag_indices(seq, cp_size=cp_size)
        else:
            idx = striped_indices(seq, cp_size=cp_size)
        out_shards = partition(full_out, idx)
        k_shards = partition(k, idx)
        v_shards = partition(v, idx)
    else:
        out_shards = full_out.reshape(batch, heads, cp_size, s_r, head_dim).movedim(-3, 0)
        k_shards = k.reshape(batch, kv_heads, cp_size, s_r, head_dim).movedim(-3, 0)
        v_shards = v.reshape(batch, kv_heads, cp_size, s_r, head_dim).movedim(-3, 0)

    cache_k = q.new_zeros(cp_size, batch, kv_heads, s_max, head_dim)
    cache_v = q.new_zeros(cp_size, batch, kv_heads, s_max, head_dim)
    cache_k[:, :, :, :s_r, :] = k_shards
    cache_v[:, :, :, :s_r, :] = v_shards
    current_len = torch.full((cp_size,), s_r, dtype=torch.int64)

    return out_shards, KVCache(k=cache_k, v=cache_v, current_len=current_len)


def decode_step(
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache: KVCache,
    *,
    owner_rank: int,
) -> tuple[torch.Tensor, KVCache]:
    """One autoregressive decode step.

    Appends ``(k_new, v_new)`` to ``cache`` at ``owner_rank`` and computes
    attention of ``q_new`` against the full cached KV (including the new row).

    Parameters
    ----------
    q_new : torch.Tensor
        ``(batch, heads, 1, head_dim)``. Replicated on every rank in the CUDA
        path; here it's just the query for the new token.
    k_new, v_new : torch.Tensor
        ``(batch, kv_heads, 1, head_dim)``.
    cache : KVCache
        Pre-step cache. Not mutated; a new cache is returned.
    owner_rank : int
        Which rank receives the new K/V row.

    Returns
    -------
    out : torch.Tensor
        ``(batch, heads, 1, head_dim)`` — attention output for the new token.
        The same value is conceptually present on every rank in the CUDA path.
    new_cache : KVCache
        Cache with the new K/V row appended at ``owner_rank``.

    Raises
    ------
    ValueError
        If ``owner_rank`` is out of range or that rank's cache is full.
    """
    cp_size = cache.cp_size
    s_max = cache.s_max
    if not 0 <= owner_rank < cp_size:
        raise ValueError(f"owner_rank={owner_rank} out of range [0, {cp_size})")
    pos = int(cache.current_len[owner_rank].item())
    if pos >= s_max:
        raise ValueError(f"cache full on rank {owner_rank}: current_len={pos} == s_max={s_max}")

    new_k = cache.k.clone()
    new_v = cache.v.clone()
    new_k[owner_rank, :, :, pos : pos + 1, :] = k_new
    new_v[owner_rank, :, :, pos : pos + 1, :] = v_new
    new_current_len = cache.current_len.clone()
    new_current_len[owner_rank] = pos + 1
    new_cache = KVCache(k=new_k, v=new_v, current_len=new_current_len)

    full_k, full_v = new_cache.gather_kv()
    out = scaled_dot_product_attention(q_new, full_k, full_v, is_causal=False)
    return out, new_cache
