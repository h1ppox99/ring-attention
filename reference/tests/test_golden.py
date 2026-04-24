"""Golden fixtures the CUDA implementation will validate against.

Each fixture bundles inputs, config, and expected per-rank output for one canonical
scenario. On first run the fixture is written to disk; subsequent runs load and
compare — a regression guard against silent changes to the reference numerics.

The `.pt` files are what the CUDA test harness loads via `torch::load` later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import pytest
import torch

from ring_attention_ref import ring_attention

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class GoldenConfig:
    name: str
    batch: int
    heads: int
    seq: int
    head_dim: int
    cp_size: int
    causal: bool
    zig_zag: bool
    seed: int


CONFIGS: list[GoldenConfig] = [
    GoldenConfig("cp2_noncausal_seq", 2, 4, 64, 32, cp_size=2, causal=False, zig_zag=False, seed=0),
    GoldenConfig("cp2_causal_zigzag", 2, 4, 64, 32, cp_size=2, causal=True, zig_zag=True, seed=1),
    GoldenConfig("cp4_noncausal_seq", 2, 4, 64, 32, cp_size=4, causal=False, zig_zag=False, seed=2),
    GoldenConfig("cp4_causal_zigzag", 2, 4, 64, 32, cp_size=4, causal=True, zig_zag=True, seed=3),
]


class Fixture(TypedDict):
    config: dict[str, object]
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor


def _compute(cfg: GoldenConfig) -> Fixture:
    gen = torch.Generator().manual_seed(cfg.seed)
    shape = (cfg.batch, cfg.heads, cfg.seq, cfg.head_dim)
    q = torch.randn(shape, generator=gen)
    k = torch.randn(shape, generator=gen)
    v = torch.randn(shape, generator=gen)
    out = ring_attention(q, k, v, cp_size=cfg.cp_size, causal=cfg.causal, zig_zag=cfg.zig_zag)
    return {"config": asdict(cfg), "q": q, "k": k, "v": v, "out": out}


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_golden_fixture(cfg: GoldenConfig) -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{cfg.name}.pt"

    computed = _compute(cfg)
    assert computed["out"].shape == (
        cfg.cp_size,
        cfg.batch,
        cfg.heads,
        cfg.seq // cfg.cp_size,
        cfg.head_dim,
    )

    if not path.exists():
        torch.save(computed, path)
        pytest.skip(f"Seeded fixture {path.name} — re-run to validate.")

    saved = torch.load(path, weights_only=False)
    assert saved["config"] == computed["config"]
    # Inputs are seeded deterministically — require bit-exact match.
    for key in ("q", "k", "v"):
        torch.testing.assert_close(saved[key], computed[key], atol=0.0, rtol=0.0)
    # Output may differ by ~1e-6 across SDPA backends (e.g. CUDA vs CPU runner);
    # CLAUDE.md allows atol=1e-3 for fp16 — use tighter fp32 tolerance here.
    torch.testing.assert_close(saved["out"], computed["out"], atol=1e-5, rtol=1e-5)
