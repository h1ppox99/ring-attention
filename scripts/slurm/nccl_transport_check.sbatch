#!/bin/bash
#SBATCH --job-name=ring-nccl-chk
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:12:00
#SBATCH --output=results/nccl-chk-%j.out
#SBATCH --error=results/nccl-chk-%j.err
#
# A/B of prefill comm_ms at the real shape (seq=16384, cp8) with default NCCL
# vs. multi-socket tuning (NSOCKS/NTHREADS) and a larger socket buffer. If
# comm_ms is flat across these, single-stream TCP is already at line rate and
# there is no inter-node bandwidth left to recover.
# (Interface speed/MTU already confirmed in job 88571: eth0=1000Mb/s mtu=1500,
#  eth1 down, on both nodes.)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j >/dev/null
echo "nodelist: ${SLURM_JOB_NODELIST:-?}"

run_prefill() {
  local tag=$1; shift
  echo
  echo "=== prefill seq=16384 cp8 :: ${tag} ==="
  mpirun -n 8 "$@" \
    "${CLI}" \
      --mode ring-overlap \
      --seq 16384 --head_dim 128 \
      --batch 1 --heads 8 --causal 1 \
      --zigzag 1 \
      --iters 5 \
      --csv-header --csv \
    2>&1 | grep -E "^mode,|^ring-overlap," || true
}

run_prefill "default"            -x NCCL_DEBUG=WARN
run_prefill "nsocks4_nthreads4"  -x NCCL_DEBUG=WARN -x NCCL_NSOCKS_PERTHREAD=4 -x NCCL_SOCKET_NTHREADS=4
run_prefill "nsocks2_buffsize8M" -x NCCL_DEBUG=WARN -x NCCL_NSOCKS_PERTHREAD=2 -x NCCL_SOCKET_NTHREADS=2 -x NCCL_BUFFSIZE=8388608
echo
echo "DONE"
