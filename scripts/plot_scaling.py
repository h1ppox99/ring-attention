"""Plot scaling and communication/compute breakdown from bench_ring.csv.

Generates three figures:
  1. Speedup vs cp_size  (total_ms(1) / total_ms(N))
  2. Parallel efficiency vs cp_size  (speedup / N)
  3. Stacked-bar comm vs comp vs wait per (mode, seq, cp_size)

Usage
-----
    uv run python scripts/plot_scaling.py results/bench_ring.csv --out results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV, dropping duplicate header rows printed by the CLI."""
    df = pd.read_csv(path)
    # Drop rows where 'mode' literally equals 'mode' (duplicate headers).
    df = df[df["mode"] != "mode"].reset_index(drop=True)
    numeric = [
        "cp_size",
        "batch",
        "heads",
        "seq",
        "head_dim",
        "causal",
        "zigzag",
        "iters",
        "comm_ms",
        "comp_ms",
        "wait_ms",
        "total_ms",
    ]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def plot_speedup(df: pd.DataFrame, out_dir: Path) -> None:
    """Speedup and parallel efficiency vs cp_size, one line per (mode, seq, head_dim, causal)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_sp, ax_eff = axes

    groups = df.groupby(["mode", "seq", "head_dim", "causal"])
    for (mode, seq, hd, causal), grp in groups:
        grp = grp.sort_values("cp_size")
        baseline = grp.loc[grp["cp_size"] == 1, "total_ms"]
        if baseline.empty:
            continue
        t1 = baseline.iloc[0]
        label = f"{mode} s={seq} d={hd} c={int(causal)}"
        speedup = t1 / grp["total_ms"]
        ax_sp.plot(grp["cp_size"], speedup, marker="o", label=label)
        ax_eff.plot(grp["cp_size"], speedup / grp["cp_size"], marker="o", label=label)

    cp_vals = sorted(df["cp_size"].unique())
    ax_sp.plot(cp_vals, cp_vals, "k--", alpha=0.4, label="ideal")
    ax_eff.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="ideal")

    ax_sp.set_xlabel("cp_size")
    ax_sp.set_ylabel("Speedup")
    ax_sp.set_title("Strong scaling speedup")
    ax_sp.legend(fontsize=6)
    ax_sp.grid(True)

    ax_eff.set_xlabel("cp_size")
    ax_eff.set_ylabel("Parallel efficiency")
    ax_eff.set_title("Parallel efficiency")
    ax_eff.set_ylim(0, 1.2)
    ax_eff.legend(fontsize=6)
    ax_eff.grid(True)

    fig.tight_layout()
    out = out_dir / "scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked-bar: comm / comp / wait per (mode, seq, cp_size).

    One figure per (seq, head_dim, causal) combination in the data.
    """
    for (seq, hd, causal), grp in df.groupby(["seq", "head_dim", "causal"]):
        grp = grp.sort_values(["mode", "cp_size"])
        labels = [f"{row.mode}\nn={int(row.cp_size)}" for _, row in grp.iterrows()]
        x = range(len(labels))

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
        ax.bar(x, grp["comp_ms"], label="comp_ms", color="steelblue")
        ax.bar(x, grp["comm_ms"], bottom=grp["comp_ms"], label="comm_ms", color="darkorange")
        ax.bar(
            x,
            grp["wait_ms"],
            bottom=grp["comp_ms"] + grp["comm_ms"],
            label="wait_ms",
            color="firebrick",
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("ms per iteration")
        ax.set_title(f"Comm/compute breakdown  seq={seq}  head_dim={hd}  causal={int(causal)}")
        ax.legend()
        ax.grid(axis="y", alpha=0.4)
        fig.tight_layout()

        fname = f"breakdown_s{seq}_d{hd}_c{int(causal)}.png"
        out = out_dir / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def main() -> None:
    """CLI entry point."""  # pragma: no cover
    parser = argparse.ArgumentParser(description="Plot ring-attention scaling results")
    parser.add_argument("csv", type=Path, help="bench_ring.csv produced by bench_ring.sbatch")
    parser.add_argument(
        "--out", type=Path, default=Path("results"), help="Output directory for PNG figures"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(df[["mode", "cp_size", "seq", "head_dim", "causal", "total_ms"]].to_string(index=False))

    plot_speedup(df, args.out)
    plot_breakdown(df, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
