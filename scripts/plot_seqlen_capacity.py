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
    cp1 = df.loc[df["cp_size"] == cp.min()].iloc[0]
    base_cp, base_seq = float(cp1["cp_size"]), float(cp1["max_seq"])

    # Reachable region (filled triangle) under the measured capacity curve. A
    # LINEAR y-axis is deliberate: linear capacity scaling is a straight line
    # here, the clearest possible statement of "context grows with GPU count".
    ax.fill_between(cp, seq, 0, color="#1f77b4", alpha=0.10, zorder=1)

    # Ideal linear capacity, anchored at the smallest measured cp_size.
    ideal = base_seq * (cp / base_cp)
    ax.plot(
        cp,
        ideal,
        "--",
        color="0.45",
        lw=1.6,
        zorder=2,
        label=f"ideal linear  (∝ #GPUs, from cp={int(base_cp)})",
    )

    # Measured capacity.
    ax.plot(
        cp,
        seq,
        "o-",
        color="#1f77b4",
        lw=2.2,
        ms=7,
        zorder=4,
        label="measured max sequence (ring attention)",
    )

    y_top = max(seq.max(), ideal.max())

    # Annotate the headline endpoint: the full-ring capacity and its multiplier.
    top = df.iloc[-1]
    mult = float(top["max_seq"]) / base_seq
    ax.annotate(
        f"{int(top['cp_size'])} GPUs → {_human_tokens(float(top['max_seq']))} tokens\n"
        f"{mult:.1f}× a single GPU",
        xy=(float(top["cp_size"]), float(top["max_seq"])),
        xytext=(-10, -55),
        textcoords="offset points",
        ha="right",
        fontsize=9.5,
        fontweight="bold",
        color="#0d3b66",
        arrowprops=dict(arrowstyle="->", color="#0d3b66", lw=1.2),
    )
    # Annotate the single-GPU point.
    ax.annotate(
        f"1 GPU: {_human_tokens(base_seq)} tokens",
        xy=(base_cp, base_seq),
        xytext=(10, 18),
        textcoords="offset points",
        ha="left",
        fontsize=9,
        color="#7a3b00",
        arrowprops=dict(arrowstyle="->", color="#7a3b00", lw=1.0),
    )

    ax.set_ylim(0, y_top * 1.12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _human_tokens(v)))
    ax.set_xticks(cp)
    ax.set_xlim(cp.min() - 0.4, cp.max() + 0.4)
    ax.set_xlabel("Number of GPUs in the ring (cp_size)")
    ax.set_ylabel("Max sequence length per attention layer (tokens)")

    meta = df.iloc[0]
    gpu_mem = float(meta["gpu_mem_mb"])
    mem_str = f"{gpu_mem / 1024:.0f} GB/GPU" if gpu_mem > 0 else "Turing"
    ax.set_title(
        "Ring attention scales sequence length linearly with GPU count\n"
        f"single attention layer · {meta['dtype']}, heads={int(meta['heads'])}, "
        f"head_dim={int(meta['head_dim'])}, batch={int(meta['batch'])}, "
        f"causal+zigzag, {mem_str}",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.25)
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
        mean_pg, color="0.5", ls="--", lw=1.2, label=f"mean ≈ {_human_tokens(mean_pg)} tokens/GPU"
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

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(df[["cp_size", "max_seq", "max_seq_per_gpu"]].to_string(index=False))

    plot_capacity(df, args.out)
    plot_per_gpu(df, args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
