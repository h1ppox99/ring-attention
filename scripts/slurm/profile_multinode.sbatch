#!/bin/bash
#SBATCH --job-name=ring-mn-profile
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=results/profile-mn-%j.out
#SBATCH --error=results/profile-mn-%j.err
#
# Multi-node profile: 2 nodes x 4 GPUs = 8 ranks. Exercises the ring kernels
# across the inter-node Ethernet fabric (cp_size=8) in addition to the
# single-node configs (cp<=4). Writes timestamped archives matching
# profile_decode.sbatch / profile.sbatch so the optimize loop can diff them.
#
# Outputs:
#   results/nsys/<timestamp>/decode_<cfg>_p<N>.nsys-rep   (decode ring timeline)
#   results/nsys/<timestamp>/prefill_<mode>_p8.nsys-rep   (prefill, 2 nodes)
#   results/ncu/<timestamp>/decode_<cfg>.ncu-rep          (decode kernel deep dive)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
TIMESTAMP="${PROFILE_TIMESTAMP:-$(date +%Y-%m-%d_%H-%M)}"
NSYS_DIR=results/nsys/${TIMESTAMP}
NCU_DIR=results/ncu/${TIMESTAMP}

mkdir -p "${NSYS_DIR}" "${NCU_DIR}" "/tmp/${USER}"
echo "${TIMESTAMP}" > results/.last_profile_timestamp

cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j

# Production decode shape (matches profile_decode.sbatch).
PROMPT=16384
HEAD_DIM=128
HEADS=8
TOKENS_NSYS=30
TOKENS_NCU=2

# Prefill shape for the multi-node kernel timelines.
SEQ=16384
ITERS_NSYS=5

echo "============================================================"
echo "Multi-node profile  nodes=2 ranks=8"
echo "  decode: prompt=${PROMPT} head_dim=${HEAD_DIM} heads=${HEADS}"
echo "  prefill: seq=${SEQ} head_dim=${HEAD_DIM} heads=${HEADS}"
echo "  nodelist: ${SLURM_JOB_NODELIST:-?}"
echo "============================================================"

# ----------------------------------------------------------------------------
# Phase 1 — Nsight Systems: decode ring timelines (cp1/2/4 single node, cp8 = 2 nodes)
# ----------------------------------------------------------------------------
echo ""
echo "[1/3] Nsight Systems — decode timelines"
echo "------------------------------------------------------------"

run_nsys_decode() {
  local cp_size=$1
  local tag=$2
  echo ""
  echo ">>> nsys decode: cp_size=${cp_size}"
  mpirun -n "${cp_size}" nsys profile \
    --trace=cuda,nvtx,mpi,osrt \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="${NSYS_DIR}/decode_${tag}_p${cp_size}_%q{OMPI_COMM_WORLD_RANK}" \
    --force-overwrite=true \
    --stats=true \
    "${CLI}" \
      --run decode \
      --prompt-len "${PROMPT}" \
      --decode-tokens "${TOKENS_NSYS}" \
      --batch 1 --heads "${HEADS}" --head_dim "${HEAD_DIM}" \
    2>&1 | tail -25
}

run_nsys_decode 4 "cp4"
run_nsys_decode 8 "cp8"

# ----------------------------------------------------------------------------
# Phase 2 — Nsight Systems: prefill kernels across 2 nodes (cp8)
# ----------------------------------------------------------------------------
echo ""
echo "[2/3] Nsight Systems — prefill timelines (2 nodes, cp8)"
echo "------------------------------------------------------------"

run_nsys_prefill() {
  local mode=$1
  local extra=$2
  local tag=$3
  echo ""
  echo ">>> nsys prefill: mode=${mode} ${extra}"
  mpirun -n 8 nsys profile \
    --trace=cuda,nvtx,mpi,osrt \
    --output="${NSYS_DIR}/prefill_${tag}_p8_%q{OMPI_COMM_WORLD_RANK}" \
    --force-overwrite=true \
    --stats=true \
    "${CLI}" \
      --mode "${mode}" \
      --seq "${SEQ}" --head_dim "${HEAD_DIM}" \
      --batch 1 --heads "${HEADS}" --causal 1 \
      ${extra} \
      --iters "${ITERS_NSYS}" \
    2>&1 | tail -25
}

run_nsys_prefill "ring-blocking" ""           "ring-blocking"
run_nsys_prefill "ring-overlap"  "--zigzag-n 2" "ring-overlap-zigzag"

# ----------------------------------------------------------------------------
# Phase 3 — Nsight Compute: decode kernel deep dive (single rank, node-independent)
# ----------------------------------------------------------------------------
echo ""
echo "[3/3] Nsight Compute — decode kernel (single rank)"
echo "------------------------------------------------------------"

run_ncu_decode() {
  local head_dim=$1
  local tag=$2
  echo ""
  echo ">>> ncu decode: head_dim=${head_dim}"
  TMPDIR="/tmp/${USER}" mpirun -n 1 -x TMPDIR ncu \
    --set full \
    --kernel-name 'regex:attention_decode_(step|split)_kernel|attention_step_kernel' \
    --launch-skip 2 \
    --launch-count 1 \
    --export "${NCU_DIR}/decode_${tag}.ncu-rep" \
    --force-overwrite \
    "${CLI}" \
      --run decode \
      --prompt-len "${PROMPT}" \
      --decode-tokens "${TOKENS_NCU}" \
      --batch 1 --heads "${HEADS}" --head_dim "${head_dim}" \
    2>&1 | tail -20
}

run_ncu_decode 64  "p${PROMPT}_d64"
run_ncu_decode 128 "p${PROMPT}_d128"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Multi-node profiling complete  (timestamp: ${TIMESTAMP})"
echo "============================================================"
echo ""
echo "Nsight Systems files:"
ls -lh "${NSYS_DIR}"/*.nsys-rep 2>/dev/null | head -30 || echo "  (none)"
echo ""
echo "Nsight Compute files:"
ls -lh "${NCU_DIR}"/*.ncu-rep 2>/dev/null || echo "  (none)"
