"""Generate golden fixture files for the ring-attention correctness tests.

Each fixture contains Q, K, V tensors (generated from a fixed seed) and the
expected per-rank output shards from the Python oracle.  The C++ ``--verify``
mode uses its own ``cpu_attention`` reference rather than loading these files;
the fixtures are used for Python-side round-trip testing and offline inspection.

Usage
-----
    uv run python -m ring_attention_ref.fixtures --out reference/tests/fixtures/m4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ring_attention_ref import ring_attention


def generate_fixtures(
    out_dir: Path,
    *,
    seed: int = 0,
    batch: int = 1,
    heads: int = 4,
    head_dim: int = 64,
    cp_sizes: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> list[Path]:
    """Generate and save golden fixture files.

    Parameters
    ----------
    out_dir : Path
        Directory to write ``.pt`` files into (created if absent).
    seed : int
        Torch manual seed for reproducible Q/K/V generation.
    batch, heads, head_dim : int
        Tensor dimensions shared across all generated configs.
    cp_sizes : list[int], optional
        Ring sizes to sweep; defaults to ``[1, 2, 4]``.
    seq_lens : list[int], optional
        Sequence lengths to sweep; defaults to ``[256, 1024]``.

    Returns
    -------
    list[Path]
        Paths of all written fixture files.
    """
    if cp_sizes is None:
        cp_sizes = [1, 2, 4]
    if seq_lens is None:
        seq_lens = [256, 1024]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for cp_size in cp_sizes:
        for seq in seq_lens:
            for causal in (False, True):
                torch.manual_seed(seed)
                q = torch.randn(batch, heads, seq, head_dim)
                k = torch.randn(batch, heads, seq, head_dim)
                v = torch.randn(batch, heads, seq, head_dim)

                out = ring_attention(q, k, v, cp_size=cp_size, causal=causal, zig_zag=False)

                fname = f"p{cp_size}_s{seq}_c{int(causal)}.pt"
                path = out_dir / fname
                torch.save(
                    {
                        "q": q,
                        "k": k,
                        "v": v,
                        "out": out,
                        "cp_size": cp_size,
                        "seq": seq,
                        "causal": causal,
                        "zigzag": False,
                        "batch": batch,
                        "heads": heads,
                        "head_dim": head_dim,
                        "seed": seed,
                    },
                    path,
                )
                written.append(path)

    return written


def main() -> None:  # pragma: no cover
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate ring-attention golden fixtures")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reference/tests/fixtures/m4"),
        help="Output directory for .pt fixture files",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    args = parser.parse_args()

    paths = generate_fixtures(
        args.out,
        seed=args.seed,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
    )
    for p in paths:
        data = torch.load(p, weights_only=True)
        print(f"  {p.name}  out_shape={tuple(data['out'].shape)}")
    print(f"Generated {len(paths)} fixtures in {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
