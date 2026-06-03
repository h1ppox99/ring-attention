"""Tests for zig-zag token partitioning.

Spec (RING_ATTENTION.md): for cp_size GPUs, rank i owns positions i and
2*cp_size - 1 - i inside each chunk of length 2*cp_size, strided over the sequence.
"""

from __future__ import annotations

import pytest
import torch

from ring_attention_ref import partition, unpartition, zigzag_indices


def test_cp_size_1_is_identity() -> None:
    idx = zigzag_indices(8, cp_size=1)
    assert idx.shape == (1, 8)
    torch.testing.assert_close(idx, torch.arange(8).unsqueeze(0))


def test_golden_indices_cp2_seq8() -> None:
    idx = zigzag_indices(8, cp_size=2)
    # rank 0 owns {0, 3, 4, 7}; rank 1 owns {1, 2, 5, 6}
    expected = torch.tensor([[0, 3, 4, 7], [1, 2, 5, 6]])
    torch.testing.assert_close(idx, expected)


def test_golden_indices_cp4_seq16() -> None:
    idx = zigzag_indices(16, cp_size=4)
    expected = torch.tensor(
        [
            [0, 7, 8, 15],
            [1, 6, 9, 14],
            [2, 5, 10, 13],
            [3, 4, 11, 12],
        ]
    )
    torch.testing.assert_close(idx, expected)


@pytest.mark.parametrize(("seq", "cp_size"), [(8, 2), (16, 4), (32, 4), (64, 8)])
def test_partition_covers_all_positions(seq: int, cp_size: int) -> None:
    idx = zigzag_indices(seq, cp_size=cp_size)
    assert idx.shape == (cp_size, seq // cp_size)
    flat = idx.flatten().sort().values
    torch.testing.assert_close(flat, torch.arange(seq))


@pytest.mark.parametrize(("seq", "cp_size"), [(8, 2), (16, 4), (32, 4)])
def test_partition_unpartition_roundtrip(seq: int, cp_size: int) -> None:
    x = torch.randn(2, 3, seq, 8)  # (batch, heads, seq, head_dim)
    idx = zigzag_indices(seq, cp_size=cp_size)
    shards = partition(x, idx)
    assert shards.shape == (cp_size, 2, 3, seq // cp_size, 8)
    restored = unpartition(shards, idx)
    torch.testing.assert_close(restored, x)


def test_invalid_seq_raises() -> None:
    with pytest.raises(ValueError):
        zigzag_indices(10, cp_size=4)  # 10 not divisible by 2*cp_size


def test_n_splits_4_golden_cp2_seq16() -> None:
    idx = zigzag_indices(16, cp_size=2, n_splits=4)
    # n=4, P=2: macro-chunk size=8. rank 0 owns chunks [0,7,2,5]; rank 1 owns [1,6,3,4].
    # Flattened over the single macro-chunk:
    expected = torch.tensor([[0, 7, 2, 5, 8, 15, 10, 13], [1, 6, 3, 4, 9, 14, 11, 12]])
    torch.testing.assert_close(idx, expected)


@pytest.mark.parametrize(
    ("seq", "cp_size", "n_splits"), [(16, 2, 4), (24, 2, 4), (24, 3, 4), (32, 4, 4)]
)
def test_n_splits_covers_all_positions(seq: int, cp_size: int, n_splits: int) -> None:
    idx = zigzag_indices(seq, cp_size=cp_size, n_splits=n_splits)
    assert idx.shape == (cp_size, seq // cp_size)
    flat = idx.flatten().sort().values
    torch.testing.assert_close(flat, torch.arange(seq))


@pytest.mark.parametrize(("seq", "cp_size", "n_splits"), [(16, 2, 4), (24, 2, 4)])
def test_n_splits_roundtrip(seq: int, cp_size: int, n_splits: int) -> None:
    x = torch.randn(2, 3, seq, 8)
    idx = zigzag_indices(seq, cp_size=cp_size, n_splits=n_splits)
    shards = partition(x, idx)
    assert shards.shape == (cp_size, 2, 3, seq // cp_size, 8)
    restored = unpartition(shards, idx)
    torch.testing.assert_close(restored, x)


def test_n_splits_default_equals_2() -> None:
    idx2 = zigzag_indices(16, cp_size=2, n_splits=2)
    idx_default = zigzag_indices(16, cp_size=2)
    torch.testing.assert_close(idx2, idx_default)
