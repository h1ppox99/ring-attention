"""Plot strong- and weak-scaling curves for ring attention.

The benchmark sweeps the production configuration only — ring-overlap, causal,
zigzag — so each plot is a small set of clean curves (one per seq / head_dim)
against the ideal-scaling reference. The comm/compute/wait split is *not*
plotted here: in overlap mode those phases run concurrently, so they are not
additive and a stacked bar misrepresents them. Use the Nsight Systems timeline
from ``scripts/slurm/profile.sbatch`` for the overlap story instead.

Strong scaling (default, from strong_scaling.sbatch -> bench_ring.csv):
  x = cp_size (number of GPUs), y = speedup = total_ms(1) / total_ms(N),
  with the linear ideal line.

Weak scaling (--weak, from weak_scaling.sbatch -> bench_weak.csv, where seq
grows with cp_size so work per rank is constant):
  x = cp_size, y = weak-scaling efficiency = total_ms(1) / total_ms(N),
  ideal = 1.0 (flat).

Usage
-----
    uv run python scripts/plot_scaling.py results/bench_ring.csv --out results/figures/
    uv run python scripts/plot_scaling.py results/bench_weak.csv --out results/figures/ --weak
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV, dropping non-data lines the CLI interleaves into stdout.

    The CLI prints per-rank banners (``rank 0/1 local_rank ...``) and may repeat
    the header; both land in the CSV as junk rows. We coerce the numeric columns
    and drop any row that failed to parse (NaN in the keys we plot on).
    """
    df = pd.read_csv(path)
    numeric = ["cp_size", "batch", "heads", "seq", "head_dim", "causal", "zigzag", "total_ms"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["cp_size", "seq", "head_dim", "total_ms"]).reset_index(drop=True)
    return df


def _gpu_axis(ax: plt.Axes, cp_vals: list[int]) -> None:
    """Linear GPU-count axis with one integer tick per measured cp_size.

    cp_size is a dense 1..16 sweep, so a linear axis keeps every tick legible
    (a log2 axis crowds 14/15/16 together). The ideal speedup line y=x is still
    straight on linear axes.
    """
    ax.set_xticks(cp_vals)
    ax.set_xticklabels([str(c) for c in cp_vals])
    ax.set_xlim(min(cp_vals) - 0.5, max(cp_vals) + 0.5)
    ax.set_xlabel("Number of GPUs (cp_size)")


def plot_strong(df: pd.DataFrame, out_dir: Path) -> None:
    """Speedup vs cp_size, one line per (seq, head_dim), against the ideal line."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for (seq, hd), grp in df.groupby(["seq", "head_dim"]):
        grp = grp.sort_values("cp_size")
        baseline = grp.loc[grp["cp_size"] == 1, "total_ms"]
        if baseline.empty:
            continue
        t1 = baseline.iloc[0]
        ax.plot(
            grp["cp_size"],
            t1 / grp["total_ms"],
            marker="o",
            label=f"seq={int(seq)}  head_dim={int(hd)}",
        )

    cp_vals = sorted(int(c) for c in df["cp_size"].unique())
    ax.plot(cp_vals, cp_vals, "k--", alpha=0.5, label="ideal (linear)")

    _gpu_axis(ax, cp_vals)
    ax.set_ylabel("Speedup  t(1) / t(N)")
    ax.set_title("Strong scaling — ring-overlap, causal, zigzag")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "strong_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_weak(df: pd.DataFrame, out_dir: Path) -> None:
    """Weak-scaling efficiency vs cp_size (seq grows with cp_size, fixed work/rank).

    One line per head_dim; ideal efficiency is a flat 1.0.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for hd, grp in df.groupby("head_dim"):
        grp = grp.sort_values("cp_size")
        baseline = grp.loc[grp["cp_size"] == 1, "total_ms"]
        if baseline.empty:
            continue
        t1 = baseline.iloc[0]
        ax.plot(grp["cp_size"], t1 / grp["total_ms"], marker="o", label=f"head_dim={int(hd)}")

    cp_vals = sorted(int(c) for c in df["cp_size"].unique())
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="ideal (flat)")

    _gpu_axis(ax, cp_vals)
    ax.set_ylabel("Weak-scaling efficiency  t(1) / t(N)")
    ax.set_ylim(0, 1.2)
    ax.set_title("Weak scaling — ring-overlap, causal, zigzag")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "weak_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    """CLI entry point."""  # pragma: no cover
    parser = argparse.ArgumentParser(description="Plot ring-attention scaling results")
    parser.add_argument(
        "csv",
        type=Path,
        help="bench_ring.csv (strong_scaling.sbatch) or bench_weak.csv (weak_scaling.sbatch)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for PNG figures",
    )
    parser.add_argument(
        "--weak",
        action="store_true",
        help="Treat input as weak-scaling data (seq grows with cp_size).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(df[["cp_size", "seq", "head_dim", "total_ms"]].to_string(index=False))

    if args.weak:
        plot_weak(df, args.out)
    else:
        plot_strong(df, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
