#!/bin/bash
#SBATCH --job-name=ring-weak
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/weak-%j.out
#SBATCH --error=results/weak-%j.err
#
# Weak scaling: fixed work PER RANK, increasing rank count. The total
# sequence grows with the ring: seq = PER_RANK * cp_size, so each GPU always
# owns PER_RANK tokens. In ideal weak scaling total_ms stays flat as cp_size
# grows; the rise above flat is communication/overlap overhead.
#
# Sweeps every cp_size ∈ 1..16 (up to 4 nodes × 4 GPUs) — seq = PER_RANK*cp is
# always divisible by 2*ZIGZAG_N*cp (PER_RANK divisible by 2*ZIGZAG_N), so all 16
# values are valid for zigzag. head_dim is swept over {32, 64, 128, 256}.
# ZIGZAG_N: number of zig-zag passes (default 1; each pass = 2 sub-groups, set 2 to override).
#   sbatch scripts/slurm/weak_scaling.sbatch              # full 4-node sweep
#   sbatch --nodes=2 scripts/slurm/weak_scaling.sbatch    # cap at cp8 (2 nodes)
#   sbatch --nodes=1 scripts/slurm/weak_scaling.sbatch    # single node, cp<=4
#
# cp_size values larger than the allocation (nodes*4) are skipped.
#
# Output: results/bench_weak.csv  (same schema as bench_ring.csv)
#   columns: mode,cp_size,batch,heads,seq,head_dim,causal,zigzag_n,striped,iters,
#            comm_ms,comp_ms,wait_ms,total_ms,max_err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
CSV=results/bench_weak.csv

cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/release -j

mkdir -p results

NNODES="${SLURM_NNODES:-1}"
MAX_CP=$((NNODES * 4))
echo "nodes=${NNODES}  max cp_size=${MAX_CP}  nodelist=${SLURM_JOB_NODELIST:-?}"

# Tokens owned by each rank, held constant across the sweep. Must be divisible
# by 2*ZIGZAG_N so that seq = PER_RANK*cp_size is divisible by 2*ZIGZAG_N*cp_size.
PER_RANK=4096
ITERS=20
ZIGZAG_N="${ZIGZAG_N:-1}"  # number of zig-zag passes; override with: ZIGZAG_N=2 sbatch ...

echo "mode,cp_size,batch,heads,seq,head_dim,causal,zigzag_n,striped,iters,comm_ms,comp_ms,wait_ms,total_ms,max_err" \
  > "${CSV}"

# Production config: ring-overlap, causal, zigzag.
bench_one() {
  local cp_size=$1 head_dim=$2
  if [ "${cp_size}" -gt "${MAX_CP}" ]; then return 0; fi
  local seq=$((PER_RANK * cp_size))  # fixed work per rank
  echo "cp_size=${cp_size}  seq=${seq} (per_rank=${PER_RANK})  head_dim=${head_dim}  (ring-overlap causal zigzag_n=${ZIGZAG_N})"
  mpirun -n "${cp_size}" "${CLI}" \
    --batch 1 --heads 4 \
    --seq "${seq}" \
    --head_dim "${head_dim}" \
    --causal 1 \
    --mode ring-overlap \
    --zigzag-n "${ZIGZAG_N}" \
    --iters "${ITERS}" \
    --csv 2>/dev/null \
    | grep "^ring-overlap," \
    >> "${CSV}"
}

for cp_size in $(seq 1 16); do
  for head_dim in 32 64 128 256; do
    bench_one "${cp_size}" "${head_dim}"
  done
done

echo ""
echo "Weak-scaling benchmark complete. Results in ${CSV}"
echo "Rows: $(wc -l < "${CSV}")"
echo "Plot with: uv run python scripts/plot_scaling.py ${CSV} --out results/figures/ --weak"
