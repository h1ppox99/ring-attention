#!/bin/bash
#SBATCH --job-name=ring-profile
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/profile-%j.out
#SBATCH --error=results/profile-%j.err
#
# Nsight Systems (multi-GPU timeline) + Nsight Compute (kernel analysis).
# Usage:
#   sbatch scripts/slurm/profile.sbatch
# Or interactive:
#   salloc --partition=gpu-turing --gres=gpu:4 --ntasks=4 --time=00:30:00
#   bash scripts/slurm/profile.sbatch
#
# Outputs:
#   results/nsys/<mode>_p<N>.nsys-rep   <- open with nsys-ui on your laptop
#   results/ncu/flash_*.ncu-rep         <- open with ncu-ui on your laptop

set -euo pipefail

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
NSYS_DIR=results/nsys
NCU_DIR=results/ncu

mkdir -p "${NSYS_DIR}" "${NCU_DIR}" "/tmp/${USER}"

# Build (no-op if already up to date)
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j

# Common shape: large enough to see overlap, small enough to stay under 1 min
SEQ=16384
HEAD_DIM=128
BATCH=1
HEADS=4
ITERS_NSYS=3   # few iters: warmup is enough to see the pipeline pattern
ITERS_NCU=3    # a few iters to ensure kernels are present to profile

echo "============================================================"
echo "Ring Attention Profiling"
echo "  seq=${SEQ}  head_dim=${HEAD_DIM}  batch=${BATCH}  heads=${HEADS}"
echo "============================================================"

# ----------------------------------------------------------------------------
# Phase 1 — Nsight Systems: timeline for each mode
# ----------------------------------------------------------------------------
echo ""
echo "[1/2] Nsight Systems — multi-GPU timelines"
echo "------------------------------------------------------------"

run_nsys() {
  local mode=$1
  local extra=$2  # "" or "--zigzag 1"
  local tag=$3    # short name for the output file

  echo ""
  echo ">>> nsys: mode=${mode} ${extra}"

  # %q{OMPI_COMM_WORLD_RANK}: one .nsys-rep file per MPI rank
  mpirun -n 4 nsys profile \
    --trace=cuda,nvtx,mpi,osrt \
    --output="${NSYS_DIR}/${tag}_p4_%q{OMPI_COMM_WORLD_RANK}" \
    --force-overwrite=true \
    --stats=true \
    "${CLI}" \
      --mode "${mode}" \
      --seq "${SEQ}" \
      --head_dim "${HEAD_DIM}" \
      --batch "${BATCH}" --heads "${HEADS}" \
      --causal 1 \
      ${extra} \
      --iters "${ITERS_NSYS}" \
    2>&1 | tail -40
}

# All 3 modes with causal masking
run_nsys "allgather"      ""             "allgather"
run_nsys "ring-blocking"  ""             "ring-blocking"
run_nsys "ring-overlap"   ""             "ring-overlap"

# ring-overlap + zigzag: the "production" configuration
run_nsys "ring-overlap"   "--zigzag 1"   "ring-overlap-zigzag"

# ----------------------------------------------------------------------------
# Phase 2 — Nsight Compute: kernel analysis
# ----------------------------------------------------------------------------
echo ""
echo "[2/2] Nsight Compute — attention_step_kernel analysis"
echo "------------------------------------------------------------"

run_ncu() {
  local head_dim=$1
  local causal=$2
  local tag=$3

  echo ""
  echo ">>> ncu: head_dim=${head_dim} causal=${causal}"

  # ncu via mpirun: MPI_Init requires mpirun even for 1 rank.
  # TMPDIR: avoids conflicts on the shared /tmp/nsight-compute-lock
  # (ncu has no --tmpdir flag; it honors the env var instead).
  TMPDIR="/tmp/${USER}" mpirun -n 1 -x TMPDIR ncu \
    --set full \
    --kernel-name attention_step_kernel \
    --launch-skip 0 \
    --launch-count 1 \
    --export "${NCU_DIR}/flash_${tag}.ncu-rep" \
    --force-overwrite \
    "${CLI}" \
      --mode ring-blocking \
      --seq "${SEQ}" \
      --head_dim "${head_dim}" \
      --batch "${BATCH}" --heads "${HEADS}" \
      --causal "${causal}" \
      --iters "${ITERS_NCU}" \
    2>&1 | tail -20
}

# Three variants for the report:
#   - head_dim=64,  non-causal: simple baseline
#   - head_dim=128, non-causal: higher register + smem pressure
#   - head_dim=128, causal:     impact of causal masking
run_ncu 64  0 "s${SEQ}_d64_c0"
run_ncu 128 0 "s${SEQ}_d128_c0"
run_ncu 128 1 "s${SEQ}_d128_c1"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Profiling complete"
echo "============================================================"
echo ""
echo "Nsight Systems files (open with nsys-ui):"
ls -lh "${NSYS_DIR}"/*.nsys-rep 2>/dev/null || echo "  (none)"
echo ""
echo "Nsight Compute files (open with ncu-ui):"
ls -lh "${NCU_DIR}"/*.ncu-rep 2>/dev/null || echo "  (none)"
echo ""
echo "To retrieve on your laptop:"
echo "  scp -r cluster:${PWD}/${NSYS_DIR} ./"
echo "  scp -r cluster:${PWD}/${NCU_DIR}  ./"
