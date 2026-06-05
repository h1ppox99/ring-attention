#!/bin/bash
#SBATCH --job-name=comm-probe
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=results/comm-probe-%j.out
#SBATCH --error=results/comm-probe-%j.err
#
# Measure the (alpha + beta*n) point-to-point communication constants used by
# the performance model (docs/TODO.md / writeup.tex section "Performance model"):
#
#   inter-node (alpha', beta')  : host<->host MPI ping-pong over eth0 (1 GbE).
#   intra-node (alpha,  beta )  : host<->host MPI ping-pong within one node
#                                 (OpenMPI shared-memory / vader BTL).
#
# Host buffers are the right probe here: GPUDirect RDMA is disabled on this
# cluster (NIC gdr=0) and the NCCL-off ring path host-stages, so host<->host
# bandwidth is what bounds the inter-node transfer. Both measurements run in a
# single 2-node allocation.
#
# Run:  sbatch scripts/slurm/comm_probe.sbatch
# Output: results/comm-probe-<jobid>.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh

SRC=scripts/comm_probe/comm_probe.cpp
BIN=build/comm_probe
mkdir -p build
mpic++ -O2 -std=c++17 "$SRC" -o "$BIN"

# OpenMPI may need oversubscribe headroom inside a SLURM allocation.
export OMPI_MCA_rmaps_base_oversubscribe=1

echo "============================================================"
echo "INTER-NODE link  (alpha', beta')  — 1 rank per node, over eth0"
echo "============================================================"
# One rank on each of the two allocated nodes.
mpirun -n 2 --map-by ppr:1:node "$BIN"

echo
echo "============================================================"
echo "INTRA-NODE link  (alpha, beta)    — 2 ranks pinned to one node"
echo "============================================================"
# Both ranks on the first allocated node (shared-memory BTL path).
FIRST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
mpirun -n 2 --host "${FIRST}:2" "$BIN"

echo
echo "comm_probe complete."
