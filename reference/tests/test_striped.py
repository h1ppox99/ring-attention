"""Tests for striped token partitioning.

Spec: token ``i`` is assigned to rank ``i % cp_size``, so rank ``r`` owns the
arithmetic progression ``r, r + cp_size, r + 2*cp_size, …``. Like zig-zag this
balances per-rank work under a causal mask; unlike zig-zag it is a single
strided shard rather than two contiguous sub-groups.
"""

from __future__ import annotations

import pytest
import torch

from ring_attention_ref import partition, striped_indices, unpartition


def test_cp_size_1_is_identity() -> None:
    idx = striped_indices(8, cp_size=1)
    assert idx.shape == (1, 8)
    torch.testing.assert_close(idx, torch.arange(8).unsqueeze(0))


def test_golden_indices_cp2_seq8() -> None:
    idx = striped_indices(8, cp_size=2)
    # rank 0 owns the evens {0, 2, 4, 6}; rank 1 owns the odds {1, 3, 5, 7}.
    expected = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]])
    torch.testing.assert_close(idx, expected)


def test_golden_indices_cp4_seq16() -> None:
    idx = striped_indices(16, cp_size=4)
    expected = torch.tensor(
        [
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
        ]
    )
    torch.testing.assert_close(idx, expected)


@pytest.mark.parametrize(("seq", "cp_size"), [(8, 2), (16, 4), (32, 4), (64, 8)])
def test_partition_covers_all_positions(seq: int, cp_size: int) -> None:
    idx = striped_indices(seq, cp_size=cp_size)
    assert idx.shape == (cp_size, seq // cp_size)
    flat = idx.flatten().sort().values
    torch.testing.assert_close(flat, torch.arange(seq))


@pytest.mark.parametrize(("seq", "cp_size"), [(8, 2), (16, 4), (32, 4)])
def test_partition_unpartition_roundtrip(seq: int, cp_size: int) -> None:
    x = torch.randn(2, 3, seq, 8)  # (batch, heads, seq, head_dim)
    idx = striped_indices(seq, cp_size=cp_size)
    shards = partition(x, idx)
    assert shards.shape == (cp_size, 2, 3, seq // cp_size, 8)
    restored = unpartition(shards, idx)
    torch.testing.assert_close(restored, x)


def test_invalid_seq_raises() -> None:
    with pytest.raises(ValueError):
        striped_indices(10, cp_size=4)  # 10 not divisible by cp_size
