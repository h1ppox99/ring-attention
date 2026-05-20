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

# Helper: run one --verify invocation and print PASS/FAIL.
# Usage: check_verify <n> <mode> <causal> [--zigzag]
check_verify() {
  local n=$1 mode=$2 causal=$3
  shift 3
  local extra="$*"
  local tag="cp_size=${n} mode=${mode} causal=${causal}${extra:+ ${extra}}"
  echo -n "  ${tag} ... "
  local output
  output=$(mpirun -n "${n}" "${CLI}" \
    --seq 1024 --head_dim 64 --causal "${causal}" \
    --mode "${mode}" --verify ${extra} 2>&1)
  echo "${output}" | grep -E "rank 0" || true
  if echo "${output}" | grep -q "PASS"; then
    echo "    → PASS"
  else
    echo "    → FAIL"
    echo "${output}"
    PASS=0
  fi
}

PASS=1

# ── allgather --verify ────────────────────────────────────────────────────────
echo "=== allgather --verify ==="
for n in 1 2 4; do
  for causal in 0 1; do
    check_verify "${n}" allgather "${causal}"
  done
done

# ── ring-blocking --verify ────────────────────────────────────────────────────
echo ""
echo "=== ring-blocking --verify ==="
for n in 1 2 4; do
  for causal in 0 1; do
    check_verify "${n}" ring-blocking "${causal}"
  done
done

# ── ring-overlap --verify ─────────────────────────────────────────────────────
echo ""
echo "=== ring-overlap --verify ==="
for n in 1 2 4; do
  for causal in 0 1; do
    check_verify "${n}" ring-overlap "${causal}"
  done
done

# ── zigzag ring-blocking --verify ─────────────────────────────────────────────
# seq=1024 must be divisible by 2*cp_size; cp_size ∈ {1,2,4} all satisfy this.
echo ""
echo "=== ring-blocking --zigzag --verify ==="
for n in 1 2 4; do
  for causal in 0 1; do
    check_verify "${n}" ring-blocking "${causal}" --zigzag
  done
done

# ── zigzag ring-overlap --verify ──────────────────────────────────────────────
echo ""
echo "=== ring-overlap --zigzag --verify ==="
for n in 1 2 4; do
  for causal in 0 1; do
    check_verify "${n}" ring-overlap "${causal}" --zigzag
  done
done

if [ "${PASS}" -eq 1 ]; then
  echo ""
  echo "All GPU tests PASSED."
else
  echo ""
  echo "One or more GPU tests FAILED — see output above."
  exit 1
fi
