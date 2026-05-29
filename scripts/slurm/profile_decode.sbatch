#!/bin/bash
#SBATCH --job-name=ring-decode-profile
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/profile-decode-%j.out
#SBATCH --error=results/profile-decode-%j.err
#
# Decode-mode profile: nsys timelines for the ring rotation, ncu deep dive on
# the attention_step_kernel invocation that runs with seq_q=1 (the decode
# regime). Writes timestamped archives matching profile.sbatch.
#
# Outputs:
#   results/nsys/<timestamp>/decode_<cfg>_p<N>.nsys-rep
#   results/ncu/<timestamp>/decode_<cfg>.ncu-rep

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

# Single decode config representative of the benchmark sweep:
#   - prompt_len=16384 — long enough that comp dominates
#   - head_dim=128 — worst case from bench_decode (49ms/token at cp_size=1)
#   - heads=8 — production-shaped
PROMPT=16384
HEAD_DIM=128
HEADS=8
TOKENS_NSYS=4    # enough to see one ring cycle plus a stable repeat
TOKENS_NCU=2     # ncu needs at least one decoded token after warmup

echo "============================================================"
echo "Decode Profile  prompt=${PROMPT}  head_dim=${HEAD_DIM}  heads=${HEADS}"
echo "============================================================"

# ----------------------------------------------------------------------------
# Phase 1 — Nsight Systems: ring decode timeline
# ----------------------------------------------------------------------------
echo ""
echo "[1/2] Nsight Systems — decode timelines"
echo "------------------------------------------------------------"

run_nsys_decode() {
  local cp_size=$1
  local tag=$2

  echo ""
  echo ">>> nsys decode: cp_size=${cp_size}"

  mpirun -n "${cp_size}" nsys profile \
    --trace=cuda,nvtx,mpi,osrt \
    --output="${NSYS_DIR}/decode_${tag}_p${cp_size}_%q{OMPI_COMM_WORLD_RANK}" \
    --force-overwrite=true \
    --stats=true \
    "${CLI}" \
      --run decode \
      --prompt-len "${PROMPT}" \
      --decode-tokens "${TOKENS_NSYS}" \
      --batch 1 --heads "${HEADS}" --head_dim "${HEAD_DIM}" \
    2>&1 | tail -30
}

run_nsys_decode 1 "cp1"
run_nsys_decode 2 "cp2"
run_nsys_decode 4 "cp4"

# ----------------------------------------------------------------------------
# Phase 2 — Nsight Compute: attention_step_kernel under seq_q=1
# ----------------------------------------------------------------------------
echo ""
echo "[2/2] Nsight Compute — attention_step_kernel (decode regime)"
echo "------------------------------------------------------------"

# cp_size=1: single rank, isolates the kernel from comm noise. The kernel
# instance we want is the FIRST seq_q=1 invocation after the warmup decode
# step the CLI does internally. With prompt=16384 + warmup the kernel fires
# twice before we'd want to capture, so skip the warmup launch:
#   warmup pass: 1 attention_step_kernel call
#   timed loop : 1 per token × TOKENS_NCU tokens
# launch-skip=1 captures the first timed-loop kernel.
run_ncu_decode() {
  local head_dim=$1
  local tag=$2

  echo ""
  echo ">>> ncu decode: head_dim=${head_dim}"

  # Round 5 onwards: the seq_q=1 path dispatches to attention_decode_step_kernel,
  # so filter on that. (attention_step_kernel kept as a fallback alternation
  # in case the dispatch in launch_attention_step is ever disabled.)
  TMPDIR="/tmp/${USER}" mpirun -n 1 -x TMPDIR ncu \
    --set full \
    --kernel-name 'regex:attention_(decode_step|step)_kernel' \
    --launch-skip 1 \
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

# Two head_dim values matching the bench_decode sweep — D=128 is the
# worst case (~49ms/token at cp_size=1); D=64 is the well-behaved one.
run_ncu_decode 64  "p${PROMPT}_d64"
run_ncu_decode 128 "p${PROMPT}_d128"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Decode profiling complete  (timestamp: ${TIMESTAMP})"
echo "============================================================"
echo ""
echo "Nsight Systems files:"
ls -lh "${NSYS_DIR}"/*.nsys-rep 2>/dev/null | head -20 || echo "  (none)"
echo ""
echo "Nsight Compute files:"
ls -lh "${NCU_DIR}"/*.ncu-rep 2>/dev/null || echo "  (none)"
