#!/bin/bash
#SBATCH --job-name=ring-attn-tests
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --error=results/slurm-%j.err
#
# Default allocation is single-node (4 GPUs). To also exercise the multi-node
# binding path and cross-node ring rotation, resubmit with:
#   sbatch --nodes=2 --ntasks-per-node=4 --gres=gpu:4 scripts/slurm/gpu_tests.sbatch
# The multi-node block at the bottom auto-activates when SLURM_NNODES >= 2.

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
# Aggregate failure into CTEST_FAIL instead of aborting under `set -e`, so a
# single broken CTest entry does not skip the rest of the matrix below.
echo "=== ctest ==="
CTEST_FAIL=0
ctest --test-dir build/coverage --output-on-failure -j1 || CTEST_FAIL=1
if [ "${CTEST_FAIL}" -eq 1 ]; then
  echo "  → ctest reported failures (continuing; will fail the job at the end)"
fi
echo ""

# ── C++ coverage report ───────────────────────────────────────────────────────
# Optional: only runs if gcovr is on PATH. Install with:
#   uv tool install --python /usr/bin/python3.11 gcovr
# (or `pip install --user gcovr`). Missing-tool is a warning, not a failure.
echo "=== coverage ==="
if command -v gcovr >/dev/null 2>&1; then
  # --gcov-ignore-parse-errors: cpu_attention.cpp's innermost loops execute
  # billions of times under the verify matrix, which trips a long-standing
  # gcov bug (gcc bugzilla #68080) where very large hit counts get encoded as
  # "suspicious". Warn-once-per-file lets the report continue while still
  # flagging the issue in the log.
  # --exclude-throw-branches / --exclude-unreachable-branches: gcov counts
  # implicit exception-propagation edges out of every std::vector / cudaMalloc /
  # std::string call, which inflates branch denominators by ~2x. These edges are
  # untestable in passing tests, so excluding them gives a meaningful branch %.
  gcovr -r . --filter 'src/' --filter 'tests/' \
        --gcov-ignore-errors=source_not_found \
        --gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file \
        --exclude-throw-branches \
        --exclude-unreachable-branches \
        --txt "results/cov-${SLURM_JOB_ID}.txt" \
        --sort uncovered-number \
        --print-summary
else
  echo "  gcovr not found on PATH — skipping coverage report."
  echo "  Install with: uv tool install --python /usr/bin/python3.11 gcovr"
fi
echo ""

# Helper: run one --verify invocation and print PASS/FAIL.
# Usage: check_verify <n> <mode> <causal> [--zigzag]
# Shape can be overridden via env vars: SEQ (default 1024), HEAD_DIM (default 64).
check_verify() {
  local n=$1 mode=$2 causal=$3
  shift 3
  local extra="$*"
  local seq="${SEQ:-1024}"
  local head_dim="${HEAD_DIM:-64}"
  local tag="cp_size=${n} mode=${mode} causal=${causal} seq=${seq} D=${head_dim}${extra:+ ${extra}}"
  echo -n "  ${tag} ... "
  local output
  output=$(mpirun -n "${n}" "${CLI}" \
    --seq "${seq}" --head_dim "${head_dim}" --causal "${causal}" \
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

# ── Shape sweep on ring-overlap+zigzag+causal ────────────────────────────────
# The hot path for the final report. Catches shape-specific bugs (e.g. D=128
# register spilling, large-seq kernel tiling) that the fixed 1024×64 matrix
# above would miss. cp_size=2 is enough: shape bugs are kernel-local, not
# rotation-local.
echo ""
echo "=== ring-overlap --zigzag --verify (shape sweep, cp_size=2, causal=1) ==="
for HEAD_DIM in 32 64 128; do
  for SEQ in 512 4096; do
    export SEQ HEAD_DIM
    check_verify 2 ring-overlap 1 --zigzag
  done
done
unset SEQ HEAD_DIM

# ── Multi-node block (auto-skipped on single-node allocations) ───────────────
# Only meaningful when SLURM gave us >1 node. Exercises the local-rank GPU
# binding path (MPI_Comm_split_type) and the cross-node MPI ring rotation,
# both of which are invisible to the single-node matrix above.
if [ "${SLURM_NNODES:-1}" -ge 2 ]; then
  N_TOTAL=$((SLURM_NNODES * 4))
  echo ""
  echo "=== multi-node verify (nodes=${SLURM_NNODES}, n=${N_TOTAL}) ==="
  # seq must be divisible by 2*cp_size for zigzag → pick a comfortable multiple.
  export SEQ=$((512 * N_TOTAL))
  export HEAD_DIM=64
  for mode in ring-blocking ring-overlap; do
    for causal in 0 1; do
      check_verify "${N_TOTAL}" "${mode}" "${causal}" --zigzag
    done
  done
  unset SEQ HEAD_DIM
else
  echo ""
  echo "=== multi-node block SKIPPED (SLURM_NNODES=${SLURM_NNODES:-1}) ==="
  echo "    Resubmit with --nodes=2 --ntasks-per-node=4 to exercise this path."
fi

if [ "${PASS}" -eq 1 ] && [ "${CTEST_FAIL}" -eq 0 ]; then
  echo ""
  echo "All GPU tests PASSED."
else
  echo ""
  echo "One or more GPU tests FAILED — see output above."
  echo "  ctest:        $([ "${CTEST_FAIL}" -eq 0 ] && echo PASS || echo FAIL)"
  echo "  verify matrix: $([ "${PASS}" -eq 1 ] && echo PASS || echo FAIL)"
  exit 1
fi
