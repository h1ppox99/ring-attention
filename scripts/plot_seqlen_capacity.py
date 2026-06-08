"""Plot the sequence-length CAPACITY of ring attention vs the number of GPUs.

This is the headline scalability figure: ring attention's per-GPU memory
footprint is O(S / cp_size), so the longest sequence a single attention layer
can process grows ~linearly with the GPU count. The measured curve rides the
ideal-linear line, and the per-GPU footprint is flat — a direct measurement of
the O(S/P) invariant that is the whole point of context parallelism.

Scope (stated honestly, not compared to LLM context windows): this is one
attention layer's forward WORKING SET — Q, the ring K/V buffers, the output,
scratch, and the softmax stats. It deliberately excludes model weights, the KV
cache across all layers, and any training state (activations, gradients,
optimizer). Those dominate a real model's memory and would shrink the absolute
numbers by 1-2 orders of magnitude; the *linear scaling law* is what transfers.

Two figures are produced from results/bench_capacity.csv
(scripts/slurm/seqlen_capacity.sbatch):

  seqlen_capacity.png    Max per-layer sequence length vs #GPUs (linear axes),
                         measured against the ideal-linear reference.

  seqlen_per_gpu.png     Max tokens held PER GPU vs #GPUs — flat, which is the
                         *reason* capacity scales: the O(S/P) memory invariant.

Usage
-----
    uv run python scripts/plot_seqlen_capacity.py results/bench_capacity.csv \
        --out results/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def _set_style() -> None:
    """Use a Computer Modern (LaTeX-like) serif font for all figures.

    Relies on matplotlib's bundled ``cmr10`` font and the ``cm`` mathtext
    fontset, so no system TeX installation is required.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    """Load the capacity CSV, coercing numerics and dropping unparsable rows."""
    df = pd.read_csv(path)
    numeric = ["cp_size", "gpus", "nodes_used", "max_seq", "max_seq_per_gpu", "gpu_mem_mb"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["cp_size", "max_seq"]).reset_index(drop=True)
    df = df[df["max_seq"] > 0].sort_values("cp_size").reset_index(drop=True)
    return df


def _human_tokens(n: float) -> str:
    """Compact token-count label: 1300000 -> '1.3M', 262144 -> '262K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:.0f}"


def plot_capacity(df: pd.DataFrame, out_dir: Path) -> None:
    """Max context length vs #GPUs (log y) with ideal-linear + model windows."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    cp = df["cp_size"].to_numpy()
    seq = df["max_seq"].to_numpy()

    # Measured capacity.
    ax.plot(
        cp,
        seq,
        "o-",
        color="#1f77b4",
        lw=2.2,
        ms=7,
        zorder=4,
        label="Measured maximum sequence length",
    )

    y_top = seq.max()

    ax.set_ylim(0, y_top * 1.12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _human_tokens(v)))
    ax.set_xticks(cp)
    ax.set_xlim(cp.min() - 0.4, cp.max() + 0.4)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Max sequence length per attention layer (tokens)")

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out = out_dir / "seqlen_capacity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_per_gpu(df: pd.DataFrame, out_dir: Path) -> None:
    """Max tokens held per GPU vs #GPUs — the flat O(S/P) memory invariant."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    cp = df["cp_size"].to_numpy()
    per_gpu = df["max_seq_per_gpu"].to_numpy()
    mean_pg = float(per_gpu.mean())

    ax.plot(cp, per_gpu, "s-", color="#2ca02c", lw=2.0, ms=6, label="max tokens per GPU (measured)")
    ax.axhline(
        mean_pg,
        color="0.5",
        ls="--",
        lw=1.2,
        label=rf"mean $\approx$ {_human_tokens(mean_pg)} tokens/GPU",
    )

    ax.set_xticks(cp)
    ax.set_ylim(0, per_gpu.max() * 1.25)
    ax.set_xlabel("Number of GPUs in the ring (cp_size)")
    ax.set_ylabel("Max tokens resident per GPU\n(attention-layer working set)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _human_tokens(v)))
    ax.set_title("Per-GPU footprint stays flat: the O(S / cp_size) memory invariant")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = out_dir / "seqlen_per_gpu.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    """CLI entry point."""  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Plot ring-attention sequence-length capacity vs GPU count"
    )
    parser.add_argument("csv", type=Path, help="bench_capacity.csv (seqlen_capacity.sbatch)")
    parser.add_argument(
        "--out", type=Path, default=Path("results/figures"), help="Output directory for PNG figures"
    )
    args = parser.parse_args()

    _set_style()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(df[["cp_size", "max_seq", "max_seq_per_gpu"]].to_string(index=False))

    plot_capacity(df, args.out)
    plot_per_gpu(df, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
