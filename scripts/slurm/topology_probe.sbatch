#!/bin/bash
#SBATCH --job-name=ring-topo
#SBATCH --partition=gpu-turing
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=results/topology-%j.out
#SBATCH --error=results/topology-%j.err
#
# Characterize the cluster topology to motivate a topology-aware (hierarchical)
# ring. The strong-scaling sweep shows a cliff at cp=5 — the first config that
# leaves a single 4-GPU node and crosses the inter-node fabric — so this probe
# captures the two relevant levels:
#   1. Intra-node: GPU<->GPU and GPU<->NIC links (NVLink / PCIe / NUMA).
#   2. Inter-node: NIC link speed and whether any InfiniBand / GPUDirect RDMA
#      exists (decides the bandwidth ceiling).
# Plus NCCL's own detected topology and the per-connection transport it picks
# (P2P / SHM intra-node vs NET/Socket vs NET/IB inter-node).
#
# gpu-turing nodes are homogeneous, so the hardware probes are captured once on
# the first allocated node and are representative of all nodes. The NCCL graph
# run spans 2 nodes (cp=8) to exercise the inter-node fabric. Run with:
#   sbatch scripts/slurm/topology_probe.sbatch
#
# Outputs (the writeup's topology evidence):
#   results/topology/<timestamp>/gpu_topo.txt    nvidia-smi topo -m / nvlink / p2p
#   results/topology/<timestamp>/network.txt     NICs, link speed, InfiniBand
#   results/topology/<timestamp>/hwloc.txt       lstopo node map / NUMA / CPU
#   results/topology/<timestamp>/nccl_graph.txt  NCCL INIT/GRAPH/NET log
#   results/topology/<timestamp>/nccl_topo.xml   NCCL's detected topology (XML)

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
TIMESTAMP="${TOPO_TIMESTAMP:-$(date +%Y-%m-%d_%H-%M)}"
OUT=results/topology/${TIMESTAMP}
mkdir -p "${OUT}"

echo "============================================================"
echo "Topology probe"
echo "  nodes=${SLURM_NNODES:-?}  nodelist=${SLURM_JOB_NODELIST:-?}"
echo "  host(node0)=$(hostname)  output=${OUT}"
echo "============================================================"

# Best-effort probe: label it, run it (pipelines allowed via bash -c), and
# append to a file. Never abort the script if a tool is missing or fails — these
# are diagnostics, and not every cluster ships every tool (ibstat, lstopo, ...).
probe() {
  local out=$1 label=$2 cmd=$3
  {
    echo "### ${label}"
    echo "\$ ${cmd}"
    bash -c "${cmd}" 2>&1 || echo "(command unavailable or failed)"
    echo ""
  } >> "${out}"
}

# ----------------------------------------------------------------------------
# Phase 1 — GPU interconnect (intra-node): NVLink vs PCIe vs SYS
# ----------------------------------------------------------------------------
echo "[1/4] GPU interconnect -> ${OUT}/gpu_topo.txt"
GPU="${OUT}/gpu_topo.txt"
: > "${GPU}"
probe "${GPU}" "GPUs on this node"            "nvidia-smi -L"
probe "${GPU}" "GPU<->GPU / GPU<->NIC matrix" "nvidia-smi topo -m"
probe "${GPU}" "P2P read capability matrix"   "nvidia-smi topo -p2p r"
probe "${GPU}" "NVLink status"                "nvidia-smi nvlink --status"
probe "${GPU}" "NVLink capabilities"          "nvidia-smi nvlink --capabilities"

# ----------------------------------------------------------------------------
# Phase 2 — Network fabric (inter-node): link speed + InfiniBand presence
# ----------------------------------------------------------------------------
echo "[2/4] Network fabric  -> ${OUT}/network.txt"
NET="${OUT}/network.txt"
: > "${NET}"
probe "${NET}" "Network interfaces (state)"   "ip -br link"
probe "${NET}" "Network interfaces (addrs)"   "ip -br addr"
probe "${NET}" "Per-interface link speed"     "for i in /sys/class/net/*; do n=\$(basename \$i); [ \"\$n\" = lo ] && continue; echo \"\$n: \$(cat \$i/speed 2>/dev/null || echo ?) Mb/s\"; done"
probe "${NET}" "ethtool link details"         "for i in /sys/class/net/*; do n=\$(basename \$i); [ \"\$n\" = lo ] && continue; echo \"== \$n ==\"; ethtool \$n 2>/dev/null | grep -iE 'speed|duplex|link detected'; done"
probe "${NET}" "InfiniBand devices (ibstat)"  "ibstat"
probe "${NET}" "InfiniBand verbs (ibv_devinfo)" "ibv_devinfo"
probe "${NET}" "RDMA link"                     "rdma link"

# ----------------------------------------------------------------------------
# Phase 3 — Node topology (NUMA / PCIe tree / CPU)
# ----------------------------------------------------------------------------
echo "[3/4] Node topology   -> ${OUT}/hwloc.txt"
HW="${OUT}/hwloc.txt"
: > "${HW}"
probe "${HW}" "hwloc node map"     "lstopo-no-graphics --of console || lstopo --of txt"
probe "${HW}" "NUMA hardware"      "numactl --hardware"
probe "${HW}" "CPU topology"       "lscpu"

# ----------------------------------------------------------------------------
# Phase 4 — NCCL's detected topology + chosen transport (across 2 nodes)
# ----------------------------------------------------------------------------
echo "[4/4] NCCL graph (cp=8, 2 nodes) -> ${OUT}/nccl_graph.txt"
NCCL="${OUT}/nccl_graph.txt"

# Build (no-op if up to date) so the CLI exists.
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j

# Record whether the binary is actually linked against NCCL — if not, the ring
# uses the host-staged MPI path and the NCCL_DEBUG log below will be empty.
{
  echo "### NCCL linkage of ${CLI}"
  ldd "${CLI}" 2>/dev/null | grep -i nccl || echo "(CLI not linked against NCCL: host-staged MPI path)"
  echo ""
} > "${NCCL}"

# NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET dumps the rings/trees NCCL builds and the
# transport per connection. NCCL_TOPO_DUMP_FILE writes the detected topology as
# XML. A tiny run (seq divisible by 2*8=16) is enough to trigger init + a step.
# -x exports the env vars to the remote ranks under Open MPI.
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,ENV \
NCCL_TOPO_DUMP_FILE="${OUT}/nccl_topo.xml" \
  mpirun -n 8 \
    -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS -x NCCL_TOPO_DUMP_FILE \
    "${CLI}" \
      --mode ring-overlap --zigzag 1 \
      --seq 16384 --head_dim 128 \
      --batch 1 --heads 4 --causal 1 \
      --iters 3 \
    >> "${NCCL}" 2>&1 || echo "(NCCL graph run failed — see ${NCCL})"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Topology probe complete  (timestamp: ${TIMESTAMP})"
echo "============================================================"
ls -lh "${OUT}"
echo ""
echo "Key things to read off these files:"
echo "  network.txt : is the inter-node link ~1000 Mb/s Ethernet? any IB/RDMA?"
echo "  gpu_topo.txt: are the 4 GPUs NVLink (NV#) or PCIe (PIX/PHB/SYS)?"
echo "  nccl_graph.txt: does inter-node use NET/Socket (TCP) or NET/IB (RDMA)?"
echo ""
echo "Retrieve with:  scp -r cluster:${PWD}/${OUT} ./"
