"""Tests for the prefill + decode reference.

The core invariants:

1. ``prefill`` produces the same per-rank output shards as ``ring_attention``
   (the existing oracle for the all-at-once forward pass).
2. ``prefill`` followed by ``N`` ``decode_step`` calls reproduces the rows of
   causal full attention on the concatenated ``[prompt | new tokens]``
   sequence.
3. Cache state evolves predictably: only the owner rank grows on each decode
   step.

These tests are the gate the CUDA decode path must pass once it exists.
"""

from __future__ import annotations

import pytest
import torch

from ring_attention_ref import KVCache, decode_step, full_attention, prefill, ring_attention


def _make_qkv(
    seq: int, *, batch: int = 1, heads: int = 2, head_dim: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (Q, K, V) of shape ``(batch, heads, seq, head_dim)``."""
    q = torch.randn(batch, heads, seq, head_dim)
    k = torch.randn(batch, heads, seq, head_dim)
    v = torch.randn(batch, heads, seq, head_dim)
    return q, k, v


# ---- Phase 1: prefill matches the existing ring_attention oracle -------------


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_prefill_output_matches_ring_attention_sequential(cp_size: int, causal: bool) -> None:
    """Sequential partitioning: prefill output shards must equal ring_attention."""
    q, k, v = _make_qkv(seq=32)

    out_shards, _ = prefill(q, k, v, cp_size=cp_size, causal=causal, zig_zag=False)
    expected = ring_attention(q, k, v, cp_size=cp_size, causal=causal, zig_zag=False)

    torch.testing.assert_close(out_shards, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("causal", [False, True])
def test_prefill_output_matches_ring_attention_zigzag(cp_size: int, causal: bool) -> None:
    """Zigzag partitioning: prefill output shards must equal ring_attention."""
    q, k, v = _make_qkv(seq=32)

    out_shards, _ = prefill(q, k, v, cp_size=cp_size, causal=causal, zig_zag=True)
    expected = ring_attention(q, k, v, cp_size=cp_size, causal=causal, zig_zag=True)

    torch.testing.assert_close(out_shards, expected, atol=1e-5, rtol=1e-5)


# ---- Phase 1: cache shape and state after prefill ----------------------------


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_cache_shape_and_fill_after_prefill(cp_size: int) -> None:
    seq, batch, heads, head_dim = 32, 1, 2, 16
    q, k, v = _make_qkv(seq=seq, batch=batch, heads=heads, head_dim=head_dim)

    _, cache = prefill(q, k, v, cp_size=cp_size, causal=False, zig_zag=False, s_max=seq)

    assert cache.cp_size == cp_size
    assert cache.s_max == seq
    assert cache.k.shape == (cp_size, batch, heads, seq, head_dim)
    assert cache.v.shape == (cp_size, batch, heads, seq, head_dim)
    expected_fill = torch.full((cp_size,), seq // cp_size, dtype=torch.int64)
    torch.testing.assert_close(cache.current_len, expected_fill)
    assert cache.total_len() == seq


def test_prefill_s_max_smaller_than_shard_raises() -> None:
    q, k, v = _make_qkv(seq=32)
    with pytest.raises(ValueError, match="s_max"):
        prefill(q, k, v, cp_size=4, causal=False, zig_zag=False, s_max=4)


def test_prefill_seq_not_divisible_raises() -> None:
    q, k, v = _make_qkv(seq=10)
    with pytest.raises(ValueError, match="divisible"):
        prefill(q, k, v, cp_size=4, causal=False, zig_zag=False)


def test_prefill_gqa_not_supported() -> None:
    """Phase-1 reference is MHA-only; GQA inputs must be rejected explicitly."""
    q = torch.randn(1, 4, 32, 16)
    k = torch.randn(1, 2, 32, 16)
    v = torch.randn(1, 2, 32, 16)
    with pytest.raises(ValueError, match="MHA-only"):
        prefill(q, k, v, cp_size=2, causal=False, zig_zag=False)


# ---- Phase 1: decode trajectory matches causal full attention ----------------


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize(("prompt_len", "n_decode"), [(8, 4), (16, 8), (32, 1)])
def test_decode_trajectory_matches_causal_full_sequential(
    cp_size: int, prompt_len: int, n_decode: int
) -> None:
    """Prefill + N decode steps must equal rows [prompt_len:prompt_len+N] of
    causal full attention over the concatenated sequence.

    Why this is the right oracle: a causal LM at position ``t`` attends to
    positions ``[0, t]``. Decode rolls forward one ``t`` at a time, and at
    each step the new query sees only its own history — which is exactly the
    same as slicing the causal full-attention output at row ``t``.
    """
    total = prompt_len + n_decode
    q_full, k_full, v_full = _make_qkv(seq=total, batch=1, heads=2, head_dim=16)

    causal_full_out = full_attention(q_full, k_full, v_full, causal=True)
    expected = causal_full_out[..., prompt_len:total, :]

    _, cache = prefill(
        q_full[..., :prompt_len, :],
        k_full[..., :prompt_len, :],
        v_full[..., :prompt_len, :],
        cp_size=cp_size,
        causal=True,
        zig_zag=False,
        s_max=total,
    )

    decode_outs: list[torch.Tensor] = []
    for d in range(n_decode):
        t = prompt_len + d
        out, cache = decode_step(
            q_full[..., t : t + 1, :],
            k_full[..., t : t + 1, :],
            v_full[..., t : t + 1, :],
            cache,
            owner_rank=d % cp_size,
        )
        decode_outs.append(out)
    got = torch.cat(decode_outs, dim=-2)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_decode_trajectory_zigzag_prefill(cp_size: int) -> None:
    """Zigzag prefill must produce the same decode trajectory as sequential.

    The cache layout differs (zigzag scatters K/V positions across ranks) but
    attention is permutation-invariant on the key dim, so the decoded outputs
    must match.
    """
    prompt_len, n_decode = 16, 4
    total = prompt_len + n_decode
    q_full, k_full, v_full = _make_qkv(seq=total)

    causal_full_out = full_attention(q_full, k_full, v_full, causal=True)
    expected = causal_full_out[..., prompt_len:total, :]

    _, cache = prefill(
        q_full[..., :prompt_len, :],
        k_full[..., :prompt_len, :],
        v_full[..., :prompt_len, :],
        cp_size=cp_size,
        causal=True,
        zig_zag=True,
        s_max=total,
    )

    decode_outs = []
    for d in range(n_decode):
        t = prompt_len + d
        out, cache = decode_step(
            q_full[..., t : t + 1, :],
            k_full[..., t : t + 1, :],
            v_full[..., t : t + 1, :],
            cache,
            owner_rank=d % cp_size,
        )
        decode_outs.append(out)
    got = torch.cat(decode_outs, dim=-2)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


# ---- Phase 1: decode cache mechanics -----------------------------------------


def test_decode_only_owner_rank_grows() -> None:
    """Each decode step bumps current_len on exactly one rank."""
    cp_size = 4
    q, k, v = _make_qkv(seq=16)
    _, cache = prefill(q, k, v, cp_size=cp_size, causal=False, zig_zag=False, s_max=32)

    initial = cache.current_len.clone()
    q_new = torch.randn(1, 2, 1, 16)
    k_new = torch.randn(1, 2, 1, 16)
    v_new = torch.randn(1, 2, 1, 16)

    _, new_cache = decode_step(q_new, k_new, v_new, cache, owner_rank=2)

    diff = new_cache.current_len - initial
    expected = torch.tensor([0, 0, 1, 0], dtype=torch.int64)
    torch.testing.assert_close(diff, expected)
    # Other ranks' cache contents must be byte-identical.
    for r in (0, 1, 3):
        torch.testing.assert_close(new_cache.k[r], cache.k[r])
        torch.testing.assert_close(new_cache.v[r], cache.v[r])


def test_decode_does_not_mutate_input_cache() -> None:
    """``decode_step`` is pure; the original cache must be unchanged."""
    q, k, v = _make_qkv(seq=16)
    _, cache = prefill(q, k, v, cp_size=2, causal=False, zig_zag=False, s_max=32)
    snapshot_lens = cache.current_len.clone()
    snapshot_k = cache.k.clone()

    q_new = torch.randn(1, 2, 1, 16)
    k_new = torch.randn(1, 2, 1, 16)
    v_new = torch.randn(1, 2, 1, 16)
    decode_step(q_new, k_new, v_new, cache, owner_rank=0)

    torch.testing.assert_close(cache.current_len, snapshot_lens)
    torch.testing.assert_close(cache.k, snapshot_k)


def test_decode_invalid_owner_rank_raises() -> None:
    q, k, v = _make_qkv(seq=16)
    _, cache = prefill(q, k, v, cp_size=2, causal=False, zig_zag=False)
    q_new = torch.randn(1, 2, 1, 16)
    k_new = torch.randn(1, 2, 1, 16)
    v_new = torch.randn(1, 2, 1, 16)
    with pytest.raises(ValueError, match="owner_rank"):
        decode_step(q_new, k_new, v_new, cache, owner_rank=5)


def test_decode_cache_full_raises() -> None:
    """Once a rank's cache hits ``s_max``, further appends to it must fail."""
    q, k, v = _make_qkv(seq=8)
    # s_max == s_r so the cache is already full on every rank.
    _, cache = prefill(q, k, v, cp_size=2, causal=False, zig_zag=False, s_max=4)
    q_new = torch.randn(1, 2, 1, 16)
    k_new = torch.randn(1, 2, 1, 16)
    v_new = torch.randn(1, 2, 1, 16)
    with pytest.raises(ValueError, match="cache full"):
        decode_step(q_new, k_new, v_new, cache, owner_rank=0)


def test_kvcache_gather_returns_total_len_rows() -> None:
    """gather_kv must concatenate exactly ``total_len`` populated rows."""
    cache = KVCache(
        k=torch.zeros(2, 1, 2, 8, 4),
        v=torch.zeros(2, 1, 2, 8, 4),
        current_len=torch.tensor([3, 5], dtype=torch.int64),
    )
    full_k, full_v = cache.gather_kv()
    assert full_k.shape == (1, 2, 8, 4)
    assert full_v.shape == (1, 2, 8, 4)
    assert cache.total_len() == 8
