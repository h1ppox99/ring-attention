"""Plot bench_attention CSV output.

Three subplots, faceted by head_dim:
  - kernel time vs sequence length (log-log)
  - achieved GFLOPS vs sequence length
  - speedup of flash vs naive vs cpu

Usage
-----
    uv run python scripts/plot_bench.py results/bench_overnight.csv [out.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _filter_to_first_bh(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first (batch, heads) combo to avoid duplicate curves."""
    bh = df[["batch", "heads"]].drop_duplicates().iloc[0]
    return df[(df["batch"] == bh["batch"]) & (df["heads"] == bh["heads"])].copy()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    csv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.with_suffix(".png")

    df = pd.read_csv(csv_path)
    df = _filter_to_first_bh(df)
    head_dims = sorted(df["head_dim"].unique())

    fig, axes = plt.subplots(3, len(head_dims), figsize=(4.2 * len(head_dims), 10), squeeze=False)
    for col, d in enumerate(head_dims):
        sub = df[df["head_dim"] == d]
        ax_t, ax_g, ax_s = axes[0, col], axes[1, col], axes[2, col]

        for (kernel, causal), g in sub.groupby(["kernel", "causal"]):
            g = g.sort_values("seq")
            style = "--" if causal else "-"
            label = f"{kernel} {'causal' if causal else 'full'}"
            ax_t.plot(g["seq"], g["time_ms"], style, marker="o", label=label)
            ax_g.plot(g["seq"], g["gflops"], style, marker="o", label=label)

        # Speedup: flash / each baseline.
        flash = sub[sub["kernel"] == "flash"].set_index(["seq", "causal"])
        for kernel in ("cpu", "naive"):
            base = sub[sub["kernel"] == kernel].set_index(["seq", "causal"])
            if base.empty:
                continue
            joined = base.join(flash[["time_ms"]], rsuffix="_flash", how="inner")
            joined["speedup"] = joined["time_ms"] / joined["time_ms_flash"]
            for causal_flag in sorted(joined.index.get_level_values("causal").unique()):
                slc = joined.xs(causal_flag, level="causal").sort_index()
                style = "--" if causal_flag else "-"
                ax_s.plot(
                    slc.index,
                    slc["speedup"],
                    style,
                    marker="o",
                    label=f"vs {kernel} {'causal' if causal_flag else 'full'}",
                )

        for ax in (ax_t, ax_g, ax_s):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("sequence length")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        ax_t.set_yscale("log")
        ax_s.set_yscale("log")
        ax_t.set_ylabel("kernel time (ms)")
        ax_g.set_ylabel("achieved GFLOPS")
        ax_s.set_ylabel("flash speedup")
        ax_t.set_title(f"head_dim = {d}")

    bh = df[["batch", "heads"]].iloc[0]
    fig.suptitle(f"bench_attention — batch={bh['batch']} heads={bh['heads']} — {csv_path.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
