#!/bin/bash
#SBATCH --job-name=ring2d-validate
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:25:00
#SBATCH --output=results/ring2d-validate-%j.out
#SBATCH --error=results/ring2d-validate-%j.err
#
# Correctness gate + speedup A/B for the hierarchical (2D) ring (--mode ring-2d,
# src/ring_loop.cu::run_ring_2d). See docs/hierarchical_ring.md.
#
# The pure schedule (Ring2DSchedule) is unit-tested CPU-only by
# tests/test_ring2d_schedule.cpp; the single-node orchestrator (N=1) is covered
# by the ring_smoke_ring-2d* ctests. This script is the only place the
# *inter-node* tier (N>=2) is exercised end-to-end, so it needs >=2 nodes:
#   sbatch                scripts/slurm/validate_ring2d.sbatch   # 2 nodes -> cp8
#   sbatch --nodes=4 ...  scripts/slurm/validate_ring2d.sbatch   # 4 nodes -> cp16
#
# Part 1: ring-2d --verify against the CPU oracle across cp_size, causal,
#         contiguous + zigzag. Bit-identical accumulation ⇒ same tol as the
#         flat ring (max_err < 1e-3 for fp32).
# Part 2: total_ms A/B vs ring-overlap at the same shape — confirms the
#         P-1 -> P-G round reduction shows up as wall-clock speedup once the
#         allocation spans >1 node (the inter-node link is the bottleneck).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh

# OMPI needs spare slots when launching mpirun inside the allocation.
export OMPI_MCA_rmaps_base_oversubscribe=1

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j

NNODES="${SLURM_NNODES:-1}"
MAX_CP=$((NNODES * 4))
echo "nodes=${NNODES}  max cp_size=${MAX_CP}  nodelist=${SLURM_JOB_NODELIST:-?}"
if [ "${NNODES}" -lt 2 ]; then
  echo "WARNING: single-node allocation — the inter-node tier (N>=2) is NOT exercised."
  echo "         Resubmit with --nodes=2 (or 4) to validate cross-node ring-2d."
fi

PASS=1

# Run one ring-2d --verify and report PASS/FAIL.
# Usage: verify_one <cp_size> <causal> <zigzag_n>  (SEQ/HEAD_DIM via env)
verify_one() {
  local n=$1 causal=$2 zigzag_n=$3
  if [ "${n}" -gt "${MAX_CP}" ]; then return 0; fi
  local seq="${SEQ:-2048}" head_dim="${HEAD_DIM:-64}"
  echo -n "  cp=${n} causal=${causal} zigzag_n=${zigzag_n} seq=${seq} D=${head_dim} ... "
  local out
  out=$(mpirun -n "${n}" "${CLI}" \
    --seq "${seq}" --head_dim "${head_dim}" --heads 4 --iters 1 \
    --causal "${causal}" --zigzag-n "${zigzag_n}" --mode ring-2d --verify 2>&1)
  if echo "${out}" | grep -q "PASS"; then
    echo "PASS  ($(echo "${out}" | grep -o 'max_err=[^ ]*' | head -1))"
  else
    echo "FAIL"; echo "${out}"; PASS=0
  fi
}

echo "============================================================"
echo "Part 1 — ring-2d --verify (vs CPU oracle)"
echo "============================================================"
# seq=2048 is divisible by 2*cp for cp ∈ {1,2,4,8,16}.
for n in 2 4 8 16; do
  for causal in 0 1; do
    verify_one "${n}" "${causal}" 1   # zigzag (production)
  done
  verify_one "${n}" 1 0               # contiguous, causal
done

echo ""
echo "============================================================"
echo "Part 2 — total_ms A/B: ring-2d vs ring-overlap (zigzag, causal)"
echo "============================================================"
# Both modes now use NCCL (ring-2d host-stages NOTHING — GPU-direct ncclSend/Recv
# with FP16 transit on both the intra- and inter-node ring), so the default build
# is already an apples-to-apples transport comparison: the only difference is the
# algorithm (flat P-1 inter-node rounds vs hierarchical P-G). cp values must be
# uniform block layouts (multiples of 4, plus the single-node cases).
SEQ=16384; HEAD_DIM=128
echo "  shape: seq=${SEQ} heads=4 D=${HEAD_DIM} iters=20 causal=1 zigzag_n=1  (both NCCL, FP16 transit)"
printf "  %-6s %-14s %-14s %-10s %-10s\n" "cp" "ring-overlap" "ring-2d" "speedup" "theory"
for n in 4 8 12 16; do
  if [ "${n}" -gt "${MAX_CP}" ]; then continue; fi
  if (( SEQ % (2 * n) != 0 )); then continue; fi
  t_ov=$(mpirun -n "${n}" "${CLI}" --seq ${SEQ} --heads 4 --head_dim ${HEAD_DIM} \
    --iters 20 --causal 1 --zigzag-n 1 --mode ring-overlap --csv 2>/dev/null \
    | grep "^ring-overlap," | awk -F, '{print $13}')
  t_2d=$(mpirun -n "${n}" "${CLI}" --seq ${SEQ} --heads 4 --head_dim ${HEAD_DIM} \
    --iters 20 --causal 1 --zigzag-n 1 --mode ring-2d --csv 2>/dev/null \
    | grep "^ring-2d," | awk -F, '{print $13}')
  spd=$(awk -v a="${t_ov}" -v b="${t_2d}" 'BEGIN{ if (b>0) printf "%.2fx", a/b; else print "-" }')
  # theory = (P-1)/(P-G), G=4 for the multi-node points (=1x on a single node).
  th=$(awk -v p="${n}" 'BEGIN{ g=(p<=4?p:4); n2=p/g; if (n2<=1) print "1.00x"; else printf "%.2fx", (p-1)/(p-g) }')
  printf "  %-6s %-14s %-14s %-10s %-10s\n" "${n}" "${t_ov:-?}" "${t_2d:-?}" "${spd}" "${th}"
done

echo ""
if [ "${PASS}" -eq 1 ]; then
  echo "ring-2d validation PASSED (Part 1 verify matrix green)."
else
  echo "ring-2d validation FAILED — see Part 1 output above."
  exit 1
fi
