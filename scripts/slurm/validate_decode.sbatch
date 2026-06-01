#!/bin/bash
#SBATCH --job-name=validate-decode
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=results/validate-decode-%j.out
#SBATCH --error=results/validate-decode-%j.err
#
# Correctness gate for the FP16 inter-rank KV transfer (Round 12). Runs the
# decode + ring-decode CTests (cp_size=2 and cp_size=4 exercise the FP16 path)
# and the FP16 attention tests. The ring_decode test prints
# "ring_decode max_err=... tol=1e-3 PASS|FAIL".

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh

# OMPI needs spare slots when ctest launches mpirun -n 4 inside the allocation.
export OMPI_MCA_rmaps_base_oversubscribe=1

cmake --build build/release -j

echo "============================================================"
echo "Decode correctness"
echo "============================================================"
ctest --test-dir build/release -R "decode" --output-on-failure

echo ""
echo "=== max_err (direct test_ring_decode, production-ish shape) ==="
TRD=build/release/tests/test_ring_decode
echo "-- cp2 --"; mpirun -n 2 ${TRD} --prompt_len 2048 --n_decode 4 --heads 8 --head_dim 128 2>/dev/null | grep -i max_err
echo "-- cp4 --"; mpirun -n 4 ${TRD} --prompt_len 2048 --n_decode 4 --heads 8 --head_dim 128 2>/dev/null | grep -i max_err
