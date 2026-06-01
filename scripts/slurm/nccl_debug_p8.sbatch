#!/bin/bash
#SBATCH --job-name=ring-nccl-dbg
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:08:00
#SBATCH --output=results/nccl-dbg-%j.out
#SBATCH --error=results/nccl-dbg-%j.err
#
# Minimal 2-node x 4-GPU (cp8) run with NCCL_DEBUG=INFO to dump the chosen
# inter-node transport, NIC selection, channel/ring topology and tuned
# thresholds. Small seq + 1 iter: we only need the ncclCommInitRank banner,
# not a perf run. Confirms whether the inter-node link is NET/Socket (TCP/
# Ethernet) vs NET/IB and how many channels cross the node boundary.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j >/dev/null

echo "nodelist: ${SLURM_JOB_NODELIST:-?}"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV,TUNING

mpirun -n 8 \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  "${CLI}" \
    --mode ring-overlap \
    --seq 4096 --head_dim 128 \
    --batch 1 --heads 8 --causal 1 \
    --zigzag 1 \
    --iters 1 \
  2>&1
