"""Compare the flat ring (ring-overlap, "1D") against the hierarchical ring
(ring-2d) and check the measured speedup against the theoretical
``(P-1)/(P-G)`` inter-node round reduction.

Input: a single CSV (from ``scripts/slurm/ring2d_sweep.sbatch``) holding rows for
*both* ``ring-overlap`` and ``ring-2d`` at a fixed total sequence, swept over
cp_size on a 4-GPU/node cluster. The ``mode`` column distinguishes the two.

Two figures are written under NEW names so the existing 1D plots
(``strong_scaling.png`` / ``weak_scaling.png``) are left untouched:

  ring2d_vs_1d_strong.png        speedup t(1)/t(N) vs cp, one curve per
                                 (mode, head_dim), against the linear ideal.
  ring2d_speedup_vs_theory.png   measured ring-2d/flat speedup t_1d/t_2d vs cp,
                                 overlaid with theory (P-1)/(P-G), G=4.

The cluster is 4 GPUs/node, so for cp<=4 everything is single-node (N=1, no
inter-node link) and theory predicts ~1x; the win appears at cp in {8,12,16}.

Usage
-----
    uv run python scripts/plot_ring2d_compare.py results/bench_ring2d.csv \
        --out results/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

GPUS_PER_NODE = 4


def load_csv(path: Path) -> pd.DataFrame:
    """Load the sweep CSV, dropping any non-data banner lines."""
    df = pd.read_csv(path)
    numeric = ["cp_size", "seq", "head_dim", "total_ms"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=numeric).reset_index(drop=True)
    df["cp_size"] = df["cp_size"].astype(int)
    df["head_dim"] = df["head_dim"].astype(int)
    return df


def _topology(cp: int) -> tuple[int, int]:
    """(N, G) for cp GPUs on a GPUS_PER_NODE-per-node cluster, block layout."""
    if cp <= GPUS_PER_NODE:
        return 1, cp  # single node
    return cp // GPUS_PER_NODE, GPUS_PER_NODE


def theory_speedup(cp: int) -> float:
    """Flat->2D inter-node round reduction (P-1)/(P-G); 1x when single-node."""
    n, g = _topology(cp)
    if n == 1:
        return 1.0
    return (cp - 1) / (cp - g)


def _cp_axis(ax: plt.Axes, cp_vals: list[int]) -> None:
    ax.set_xticks(cp_vals)
    ax.set_xticklabels([str(c) for c in cp_vals])
    ax.set_xlim(min(cp_vals) - 0.5, max(cp_vals) + 0.5)
    ax.set_xlabel("Number of GPUs (cp_size)")


def plot_strong_overlay(df: pd.DataFrame, out_dir: Path) -> None:
    """Speedup t(1)/t(N) vs cp, one line per (mode, head_dim)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {
        "ring-overlap": dict(linestyle="--", marker="s"),
        "ring-2d": dict(linestyle="-", marker="o"),
    }
    label_mode = {"ring-overlap": "1D (flat)", "ring-2d": "2D (hierarchical)"}

    for (mode, hd), grp in df.groupby(["mode", "head_dim"]):
        grp = grp.sort_values("cp_size")
        base = grp.loc[grp["cp_size"] == 1, "total_ms"]
        if base.empty:
            continue
        t1 = base.iloc[0]
        ax.plot(
            grp["cp_size"],
            t1 / grp["total_ms"],
            label=f"{label_mode.get(mode, mode)}  D={hd}",
            **styles.get(mode, {}),
        )

    cp_vals = sorted(df["cp_size"].unique())
    ax.plot(cp_vals, cp_vals, "k:", alpha=0.4, label="ideal (linear)")
    _cp_axis(ax, cp_vals)
    ax.set_ylabel("Speedup  t(1) / t(N)")
    ax.set_title("Strong scaling — flat (1D) vs hierarchical (2D) ring, NCCL, causal+zigzag")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "ring2d_vs_1d_strong.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_speedup_vs_theory(df: pd.DataFrame, out_dir: Path) -> None:
    """Measured ring-2d/flat speedup vs cp, overlaid with theory (P-1)/(P-G)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    piv = df.pivot_table(
        index=["head_dim", "cp_size"], columns="mode", values="total_ms", aggfunc="min"
    )
    if "ring-overlap" not in piv or "ring-2d" not in piv:
        print("WARNING: need both ring-overlap and ring-2d rows; skipping theory plot")
        plt.close(fig)
        return

    for hd, grp in piv.groupby(level="head_dim"):
        grp = grp.dropna(subset=["ring-overlap", "ring-2d"])
        cps = [c for (_, c) in grp.index]
        speedup = (grp["ring-overlap"] / grp["ring-2d"]).to_numpy()
        ax.plot(cps, speedup, marker="o", label=f"measured  D={hd}")

    cp_vals = sorted(df["cp_size"].unique())
    ax.plot(
        cp_vals,
        [theory_speedup(c) for c in cp_vals],
        "k--",
        marker="x",
        alpha=0.7,
        label="theory (P-1)/(P-G)",
    )
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)

    _cp_axis(ax, cp_vals)
    ax.set_ylabel("Speedup  t(flat) / t(2D)")
    ax.set_title("Hierarchical-ring speedup vs theory (4 GPUs/node; win appears once nodes>1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "ring2d_speedup_vs_theory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

    # Console table for the goal's coherence check.
    print("\ncp   N  G   theory   measured(by head_dim)")
    for cp in cp_vals:
        n, g = _topology(cp)
        meas = []
        for hd, grp in piv.groupby(level="head_dim"):
            grp = grp.dropna(subset=["ring-overlap", "ring-2d"])
            row = grp[grp.index.get_level_values("cp_size") == cp]
            if not row.empty:
                meas.append(f"D{hd}={row['ring-overlap'].iloc[0] / row['ring-2d'].iloc[0]:.2f}x")
        print(f"{cp:<4} {n}  {g}   {theory_speedup(cp):.2f}x    {'  '.join(meas)}")


def main() -> None:
    """CLI entry point."""  # pragma: no cover
    parser = argparse.ArgumentParser(description="Compare flat vs hierarchical ring scaling")
    parser.add_argument("csv", type=Path, help="results/bench_ring2d.csv (both modes)")
    parser.add_argument(
        "--out", type=Path, default=Path("results/figures"), help="Output directory for PNG figures"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(df[["mode", "cp_size", "head_dim", "total_ms"]].to_string(index=False))

    plot_strong_overlay(df, args.out)
    plot_speedup_vs_theory(df, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
