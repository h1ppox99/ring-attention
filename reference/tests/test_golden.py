"""Golden fixtures the CUDA implementation will validate against.

Each fixture bundles canonical inputs and expected per-rank output for one scenario.
The committed `.pt` files are the source of truth: tests load the inputs from the
fixture, recompute the output, and assert the result matches within tolerance. This
is the same protocol the CUDA test harness will use (`torch::load` the fixture,
run the kernel on the loaded inputs, compare outputs).

If a fixture file is missing (e.g. a new config was added) the test seeds it on
first run and skips; re-running then validates normally.
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


def _seed_fixture(cfg: GoldenConfig, path: Path) -> None:
    gen = torch.Generator().manual_seed(cfg.seed)
    shape = (cfg.batch, cfg.heads, cfg.seq, cfg.head_dim)
    q = torch.randn(shape, generator=gen)
    k = torch.randn(shape, generator=gen)
    v = torch.randn(shape, generator=gen)
    out = ring_attention(q, k, v, cp_size=cfg.cp_size, causal=cfg.causal, zig_zag=cfg.zig_zag)
    fixture: Fixture = {"config": asdict(cfg), "q": q, "k": k, "v": v, "out": out}
    torch.save(fixture, path)


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.name)
def test_golden_fixture(cfg: GoldenConfig) -> None:
    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / f"{cfg.name}.pt"

    if not path.exists():
        _seed_fixture(cfg, path)
        pytest.skip(f"Seeded fixture {path.name} — re-run to validate.")

    saved: Fixture = torch.load(path, weights_only=False)
    assert saved["config"] == asdict(cfg)

    out = ring_attention(
        saved["q"],
        saved["k"],
        saved["v"],
        cp_size=cfg.cp_size,
        causal=cfg.causal,
        zig_zag=cfg.zig_zag,
    )
    assert out.shape == saved["out"].shape
    # Different SDPA backends (CPU vs CUDA) reduce in different orders; ~1e-6 drift
    # is normal. The CUDA harness uses atol=1e-3 for fp16; here we run fp32 so we
    # can hold a tighter bound.
    torch.testing.assert_close(out, saved["out"], atol=1e-5, rtol=1e-5)
