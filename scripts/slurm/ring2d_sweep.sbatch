#!/bin/bash
#SBATCH --job-name=ring2d-sweep
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/ring2d-sweep-%j.out
#SBATCH --error=results/ring2d-sweep-%j.err
#
# Strong-scaling sweep comparing the flat ring (ring-overlap, "1D") against the
# hierarchical ring (ring-2d) at fixed total sequence, both on NCCL. Emits one
# CSV that plot_ring2d_compare.py turns into the 1D-vs-2D comparison figures
# (saved under NEW names in results/figures/, leaving the existing 1D plots).
#
#   sbatch                scripts/slurm/ring2d_sweep.sbatch   # 4 nodes -> cp16
#   sbatch --nodes=2 ...                                       # 2 nodes -> cp8
#
# ring-2d needs a uniform block layout (equal GPUs/node), so cp_size is swept
# over the values that tile cleanly with 4 GPUs/node: {1,2,3,4} on one node and
# {8,12,16} across nodes. (cp=5,6,7,... are uneven and ring-2d rejects them; the
# flat ring's behaviour there is already covered by strong_scaling.sbatch.)
#
# Output: results/bench_ring2d.csv  (same schema as bench_ring.csv; the `mode`
# column distinguishes ring-overlap from ring-2d).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/env/activate.sh
export OMPI_MCA_rmaps_base_oversubscribe=1

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
CSV=results/bench_ring2d.csv
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 >/dev/null
cmake --build build/release -j

mkdir -p results

NNODES="${SLURM_NNODES:-1}"
MAX_CP=$((NNODES * 4))
echo "nodes=${NNODES}  max cp_size=${MAX_CP}  nodelist=${SLURM_JOB_NODELIST:-?}"

# Fixed total sequence (strong scaling). 20160 = 2^6·3^2·5·7 is divisible by
# 2*cp for every cp in the sweep set, so zigzag is valid throughout.
SEQ=20160
ITERS=20

echo "mode,cp_size,batch,heads,seq,head_dim,causal,zigzag,iters,comm_ms,comp_ms,wait_ms,total_ms,max_err" \
  > "${CSV}"

# Production config: causal + zigzag. ring-2d only accepts uniform layouts.
bench_one() {
  local mode=$1 cp_size=$2 head_dim=$3
  if [ "${cp_size}" -gt "${MAX_CP}" ]; then return 0; fi
  if (( SEQ % (2 * cp_size) != 0 )); then return 0; fi
  echo "  mode=${mode} cp=${cp_size} head_dim=${head_dim} seq=${SEQ}"
  mpirun -n "${cp_size}" "${CLI}" \
    --batch 1 --heads 4 --seq "${SEQ}" --head_dim "${head_dim}" \
    --causal 1 --zigzag 1 --mode "${mode}" --iters "${ITERS}" --csv 2>/dev/null \
    | grep "^${mode}," >> "${CSV}" || echo "    (skipped: ${mode} cp=${cp_size} returned no row)"
}

# Uniform-layout cp_size values for a 4-GPU/node cluster.
for cp_size in 1 2 3 4 8 12 16; do
  for head_dim in 64 128; do
    bench_one ring-overlap "${cp_size}" "${head_dim}"
    bench_one ring-2d      "${cp_size}" "${head_dim}"
  done
done

echo ""
echo "Sweep complete -> ${CSV}  ($(wc -l < "${CSV}") rows)"
echo "Plot with: uv run python scripts/plot_ring2d_compare.py ${CSV} --out results/figures/"
