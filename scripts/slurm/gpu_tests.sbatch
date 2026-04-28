#!/bin/bash
#SBATCH --job-name=ring-attn-tests
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/env/activate.sh

# ── Build ──────────────────────────────────────────────────────────────────────
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/release -j

# ── Node health check ──────────────────────────────────────────────────────────
echo "=== hello_mpi_cuda ==="
mpirun -n 2 ./build/release/apps/hello_mpi_cuda/hello_mpi_cuda
echo ""

# ── C++ / CUDA unit tests ──────────────────────────────────────────────────────
echo "=== ctest ==="
ctest --test-dir build/release --output-on-failure

echo ""
echo "All GPU tests passed."
