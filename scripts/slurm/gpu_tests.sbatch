#!/bin/bash
#SBATCH --job-name=ring-attn-tests
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/env/activate.sh

CLI=./build/coverage/apps/ring_attention_cli/ring_attention_cli

# ── Build (coverage-instrumented) ─────────────────────────────────────────────
cmake --preset=coverage -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/coverage -j

# ── Node health check ──────────────────────────────────────────────────────────
echo "=== hello_mpi_cuda ==="
mpirun -n 4 ./build/coverage/apps/hello_mpi_cuda/hello_mpi_cuda
echo ""

# ── C++ / CUDA unit tests (ctest) ─────────────────────────────────────────────
echo "=== ctest ==="
ctest --test-dir build/coverage --output-on-failure -j1
echo ""

# ── C++ coverage report ───────────────────────────────────────────────────────
echo "=== coverage ==="
gcovr -r . --filter 'src/' --filter 'tests/' \
      --gcov-ignore-errors=source_not_found \
      --txt "results/cov-${SLURM_JOB_ID}.txt" \
      --sort uncovered-number \
      --print-summary
echo ""

# ── allgather --verify ────────────────────────────────────────────────────────
# Pass criterion: max_err < 1e-3 for cp_size ∈ {1,2,4}, causal ∈ {0,1}.
echo "=== allgather --verify ==="
PASS=1
for n in 1 2 4; do
  for causal in 0 1; do
    echo -n "  cp_size=${n} causal=${causal} ... "
    output=$(mpirun -n "${n}" "${CLI}" \
      --seq 1024 --head_dim 64 --causal "${causal}" \
      --mode allgather --verify 2>&1)
    echo "${output}" | grep -E "rank 0"
    if echo "${output}" | grep -q "PASS"; then
      echo "    → PASS"
    else
      echo "    → FAIL"
      echo "${output}"
      PASS=0
    fi
  done
done

if [ "${PASS}" -eq 1 ]; then
  echo ""
  echo "All allgather --verify tests PASSED."
else
  echo ""
  echo "allgather --verify FAILED — see output above."
  exit 1
fi

echo ""
echo "All GPU tests passed."
