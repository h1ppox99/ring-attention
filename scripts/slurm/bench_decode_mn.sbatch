#!/bin/bash
#SBATCH --job-name=bench-decode-mn
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:08:00
#SBATCH --output=results/bench-decode-mn-%j.out
#SBATCH --error=results/bench-decode-mn-%j.err
#
# Clean (no-profiler) end-to-end decode latency at cp4 (1 node) and cp8 (2
# nodes). Reports per-token comm_ms/comp_ms/total_ms via the CLI's CSV.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh
CLI=./build/release/apps/ring_attention_cli/ring_attention_cli

for cp in 4 8; do
  echo "===== cp_size=${cp}  prompt=16384 D=128 H=8 ====="
  mpirun -n ${cp} "${CLI}" --run decode --prompt-len 16384 --decode-tokens 16 \
    --batch 1 --heads 8 --head_dim 128 --csv --csv-header 2>/dev/null | tail -20
done
