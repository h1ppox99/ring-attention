#!/bin/bash
#SBATCH --job-name=validate-prefill
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=results/validate-prefill-%j.out
#SBATCH --error=results/validate-prefill-%j.err
#
# Correctness gate for FP16 prefill KV transit (Round 14). ring_smoke runs the
# fp32 --verify at tol=1e-3 (seq=128 — the worst case for fp16 quant error,
# weakest averaging). Larger-seq verify runs characterize the error trend.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh
export OMPI_MCA_rmaps_base_oversubscribe=1

cmake --build build/release -j

echo "============================================================"
echo "Prefill ring correctness (FP16 transit)"
echo "============================================================"
ctest --test-dir build/release -R "ring_smoke" --output-on-failure

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
echo ""
echo "=== fp32-compute / quantized-transit max_err vs seq (cp2, causal+zigzag, D=128, H=4) ==="
for seq in 128 256 1024 4096; do
  for mode in ring-blocking ring-overlap; do
    echo -n "seq=${seq} ${mode}: "
    mpirun -n 2 ${CLI} --seq ${seq} --heads 4 --head_dim 128 --iters 1 \
      --causal 1 --zigzag 1 --mode ${mode} --verify 2>/dev/null | grep -i verify || echo "(no verify line)"
  done
done
