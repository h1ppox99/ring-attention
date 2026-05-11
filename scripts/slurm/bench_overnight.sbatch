#!/bin/bash
#SBATCH --job-name=ring-attn-bench
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=results/bench-%j.out
#SBATCH --error=results/bench-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results

source scripts/env/activate.sh

# ── Build (no-op if up to date) ────────────────────────────────────────────────
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j --target bench_attention >/dev/null

BENCH=./build/release/apps/bench_attention/bench_attention

# ── Sanity: tests still green ──────────────────────────────────────────────────
echo "=== ctest ==="
cmake --build build/release -j >/dev/null
ctest --test-dir build/release --output-on-failure
echo ""

# ── Full sweep, two (batch, heads) configurations ──────────────────────────────
echo "=== bench sweep (B=1, H=8) ==="
"$BENCH" --bh 1 8  > results/bench_b1_h8.csv
echo "wrote results/bench_b1_h8.csv"

echo "=== bench sweep (B=2, H=16) ==="
"$BENCH" --bh 2 16 > results/bench_b2_h16.csv
echo "wrote results/bench_b2_h16.csv"

# Combined CSV for plotting.
{
  head -1 results/bench_b1_h8.csv
  tail -n +2 results/bench_b1_h8.csv
  tail -n +2 results/bench_b2_h16.csv
} > results/bench_overnight.csv
echo "wrote results/bench_overnight.csv"

# ── Nsight Compute on representative configs ───────────────────────────────────
# Profile only the flash kernel; skip warmup launches via --launch-skip.
echo "=== Nsight Compute ==="
if command -v ncu >/dev/null 2>&1; then
  for cfg in "2048 64 0" "4096 64 1" "2048 128 0"; do
    read S D C <<< "$cfg"
    out="results/ncu_s${S}_d${D}_c${C}"
    echo "  profiling seq=$S head_dim=$D causal=$C → $out"
    ncu --set full \
        --kernel-name regex:flash_attention_kernel \
        --launch-skip 3 --launch-count 3 \
        -f -o "$out" \
        "$BENCH" --bh 1 8 --only "$S" "$D" "$C" --kernels flash \
        > "${out}.log" 2>&1 || echo "    ncu failed for $cfg (see ${out}.log)"
  done
else
  echo "  ncu not available, skipping profiling"
fi

echo ""
echo "=== Done. Artifacts in results/ ==="
ls -la results/ | tail -20
