"""Tests for simulated ring attention.

The oracle's ring output per rank must equal `full_attention(...)` sliced by the
partition indices (sequential or zig-zag). This pins down the numerics the CUDA
implementation is expected to match.
"""

from __future__ import annotations

import pytest
import torch

from ring_attention_ref import full_attention, partition, ring_attention, zigzag_indices


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_ring_noncausal_sequential_matches_full(cp_size: int) -> None:
    seq = 32
    q = torch.randn(2, 2, seq, 16)
    k = torch.randn(2, 2, seq, 16)
    v = torch.randn(2, 2, seq, 16)

    ring_out = ring_attention(q, k, v, cp_size=cp_size, causal=False, zig_zag=False)
    expected = full_attention(q, k, v, causal=False)
    expected_shards = expected.reshape(2, 2, cp_size, seq // cp_size, 16).movedim(2, 0)

    assert ring_out.shape == (cp_size, 2, 2, seq // cp_size, 16)
    torch.testing.assert_close(ring_out, expected_shards, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_ring_noncausal_zigzag_matches_full(cp_size: int) -> None:
    seq = 32
    q = torch.randn(1, 2, seq, 16)
    k = torch.randn(1, 2, seq, 16)
    v = torch.randn(1, 2, seq, 16)

    ring_out = ring_attention(q, k, v, cp_size=cp_size, causal=False, zig_zag=True)
    expected = full_attention(q, k, v, causal=False)
    idx = zigzag_indices(seq, cp_size=cp_size)
    expected_shards = partition(expected, idx)

    torch.testing.assert_close(ring_out, expected_shards, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_ring_causal_zigzag_matches_full(cp_size: int) -> None:
    seq = 32
    q = torch.randn(1, 2, seq, 16)
    k = torch.randn(1, 2, seq, 16)
    v = torch.randn(1, 2, seq, 16)

    ring_out = ring_attention(q, k, v, cp_size=cp_size, causal=True, zig_zag=True)
    expected = full_attention(q, k, v, causal=True)
    idx = zigzag_indices(seq, cp_size=cp_size)
    expected_shards = partition(expected, idx)

    torch.testing.assert_close(ring_out, expected_shards, atol=1e-5, rtol=1e-5)


def test_invalid_seq_raises() -> None:
    q = torch.randn(1, 2, 10, 16)
    k = torch.randn(1, 2, 10, 16)
    v = torch.randn(1, 2, 10, 16)
    with pytest.raises(ValueError):
        ring_attention(q, k, v, cp_size=4, causal=False, zig_zag=False)


def test_cp_size_1_equals_full_attention() -> None:
    q = torch.randn(1, 2, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)

    ring_out = ring_attention(q, k, v, cp_size=1, causal=True, zig_zag=False)
    expected = full_attention(q, k, v, causal=True).unsqueeze(0)
    torch.testing.assert_close(ring_out, expected, atol=1e-6, rtol=1e-6)
