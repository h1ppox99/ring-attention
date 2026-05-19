"""Round-trip tests for the fixture generator.

Verifies that generate_fixtures:
1. Writes files with the expected naming scheme.
2. Produces outputs with correct shapes.
3. Matches the ring_attention oracle (values are exact since fixtures store
   the oracle's own output).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ring_attention_ref import ring_attention
from ring_attention_ref.fixtures import generate_fixtures


def test_generates_expected_file_count() -> None:
    """One file per (cp_size × seq × causal) combination."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = generate_fixtures(
            Path(tmp),
            cp_sizes=[1, 2],
            seq_lens=[128],
            heads=2,
            head_dim=32,
        )
        assert len(paths) == 4  # 2 cp_sizes × 1 seq × 2 causal values


def test_fixture_shapes() -> None:
    """Output tensor has shape (cp_size, batch, heads, seq // cp_size, head_dim)."""
    with tempfile.TemporaryDirectory() as tmp:
        batch, heads, head_dim = 1, 2, 32
        paths = generate_fixtures(
            Path(tmp),
            cp_sizes=[2, 4],
            seq_lens=[64],
            batch=batch,
            heads=heads,
            head_dim=head_dim,
        )
        for path in paths:
            data = torch.load(path, weights_only=True)
            cp = data["cp_size"]
            seq = data["seq"]
            expected_shape = (cp, batch, heads, seq // cp, head_dim)
            assert (
                tuple(data["out"].shape) == expected_shape
            ), f"{path.name}: expected {expected_shape}, got {tuple(data['out'].shape)}"


def test_fixture_values_match_oracle() -> None:
    """Stored output matches a fresh ring_attention call on the same Q/K/V."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = generate_fixtures(
            Path(tmp),
            cp_sizes=[1, 4],
            seq_lens=[64],
            heads=2,
            head_dim=32,
        )
        for path in paths:
            data = torch.load(path, weights_only=True)
            fresh = ring_attention(
                data["q"],
                data["k"],
                data["v"],
                cp_size=data["cp_size"],
                causal=data["causal"],
                zig_zag=data["zigzag"],
            )
            torch.testing.assert_close(data["out"], fresh, atol=1e-6, rtol=1e-6)


def test_default_args_produce_files() -> None:
    """Calling generate_fixtures() with no cp_sizes/seq_lens uses the defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = generate_fixtures(Path(tmp), heads=2, head_dim=32)
        # Defaults: cp_sizes=[1,2,4], seq_lens=[256,1024], causal x2 → 12 files
        assert len(paths) == 12


def test_fixture_files_named_correctly() -> None:
    """File names follow the p{P}_s{S}_c{C}.pt convention."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = generate_fixtures(
            Path(tmp),
            cp_sizes=[2],
            seq_lens=[256],
            heads=2,
            head_dim=32,
        )
        names = {p.name for p in paths}
        assert "p2_s256_c0.pt" in names
        assert "p2_s256_c1.pt" in names
