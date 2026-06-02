#!/bin/bash
#SBATCH --job-name=round15-check
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=results/round15-check-%j.out
#SBATCH --error=results/round15-check-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh
export OMPI_MCA_rmaps_base_oversubscribe=1

cmake --build build/release -j

echo "=== decode correctness (no regression from barrier removal) ==="
ctest --test-dir build/release -R "decode" --output-on-failure | tail -8

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
echo ""
echo "=== clean decode latency (prompt=16384 D=128 H=8, 16 tokens) ==="
for cp in 4 8; do
  echo "--- cp_size=${cp} ---"
  mpirun -n ${cp} "${CLI}" --run decode --prompt-len 16384 --decode-tokens 16 \
    --batch 1 --heads 8 --head_dim 128 --csv 2>/dev/null | grep -i "summary"
done
