#!/bin/bash
#SBATCH --job-name=ring-validate
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=results/validate-%j.out
#SBATCH --error=results/validate-%j.err
#
# Correctness gate for the final report. Three parts:
#   1. Prefill multi-rank --verify matrix: every ring mode × cp_size {1,2,4}
#      × causal × zigzag, plus a head_dim/seq shape sweep and (when the
#      allocation spans >1 node) a cross-node ring-rotation check. This is the
#      "multi-rank consistency (1,2,4 ranks)" evidence for the writeup.
#   2. Decode max_err: test_ring_decode at a production-ish shape (cp2/cp4)
#      exercises the FP16 inter-rank KV transfer path.
#   3. Prefill FP16-transit max_err vs seq: characterizes the quantization
#      error trend (seq=128 is the worst case — weakest softmax averaging).
#
# ctest unit tests and gcov coverage live in scripts/coverage.sh; this script
# is the multi-rank / multi-node behavioural gate that ctest cannot express.
#
# To also exercise the cross-node path, resubmit with:
#   sbatch --nodes=2 --ntasks-per-node=4 --gres=gpu:4 scripts/slurm/validate.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh

# OMPI needs spare slots when launching mpirun inside the allocation.
export OMPI_MCA_rmaps_base_oversubscribe=1

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j

PASS=1

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

echo "============================================================"
echo "Part 1 — Prefill multi-rank --verify matrix"
echo "============================================================"

for mode in allgather ring-blocking ring-overlap; do
  echo ""
  echo "=== ${mode} --verify ==="
  for n in 1 2 4; do
    for causal in 0 1; do
      check_verify "${n}" "${mode}" "${causal}"
    done
  done
done

# Zigzag variants (seq=1024 divisible by 2*cp_size for cp ∈ {1,2,4}).
for mode in ring-blocking ring-overlap; do
  echo ""
  echo "=== ${mode} --zigzag --verify ==="
  for n in 1 2 4; do
    for causal in 0 1; do
      check_verify "${n}" "${mode}" "${causal}" --zigzag
    done
  done
done

# Shape sweep on the report hot path (ring-overlap+zigzag+causal). Catches
# shape-specific bugs (D=128 register spilling, large-seq tiling) the fixed
# 1024×64 matrix misses. cp_size=2 suffices: shape bugs are kernel-local.
echo ""
echo "=== ring-overlap --zigzag --verify (shape sweep, cp_size=2, causal=1) ==="
for HEAD_DIM in 32 64 128; do
  for SEQ in 512 4096; do
    export SEQ HEAD_DIM
    check_verify 2 ring-overlap 1 --zigzag
  done
done
unset SEQ HEAD_DIM

# Multi-node block (auto-skipped on single-node allocations). Exercises the
# local-rank GPU binding (MPI_Comm_split_type) and cross-node ring rotation.
if [ "${SLURM_NNODES:-1}" -ge 2 ]; then
  N_TOTAL=$((SLURM_NNODES * 4))
  echo ""
  echo "=== multi-node verify (nodes=${SLURM_NNODES}, n=${N_TOTAL}) ==="
  export SEQ=$((512 * N_TOTAL))  # divisible by 2*cp_size for zigzag
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

echo ""
echo "============================================================"
echo "Part 2 — Decode max_err (FP16 inter-rank KV transfer)"
echo "============================================================"
TRD=build/release/tests/test_ring_decode
echo "-- cp2 --"; mpirun -n 2 ${TRD} --prompt_len 2048 --n_decode 4 --heads 8 --head_dim 128 2>/dev/null | grep -i max_err
echo "-- cp4 --"; mpirun -n 4 ${TRD} --prompt_len 2048 --n_decode 4 --heads 8 --head_dim 128 2>/dev/null | grep -i max_err

echo ""
echo "============================================================"
echo "Part 3 — Prefill FP16-transit max_err vs seq (cp2, causal+zigzag, D=128, H=4)"
echo "============================================================"
for seq in 128 256 1024 4096; do
  for mode in ring-blocking ring-overlap; do
    echo -n "seq=${seq} ${mode}: "
    mpirun -n 2 ${CLI} --seq ${seq} --heads 4 --head_dim 128 --iters 1 \
      --causal 1 --zigzag 1 --mode ${mode} --verify 2>/dev/null | grep -i verify || echo "(no verify line)"
  done
done

echo ""
if [ "${PASS}" -eq 1 ]; then
  echo "Validation PASSED (Part 1 verify matrix green; inspect Parts 2-3 max_err vs tol=1e-3)."
else
  echo "Validation FAILED — see Part 1 output above."
  exit 1
fi
