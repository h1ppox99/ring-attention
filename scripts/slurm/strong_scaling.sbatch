#!/bin/bash
#SBATCH --job-name=ring-strong
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/strong-%j.out
#SBATCH --error=results/strong-%j.err
#
# Strong scaling: fixed total problem size (seq=20160), increasing rank count.
# Production configuration only: ring-overlap + causal + zigzag. Sweeps every
# cp_size ∈ 1..16 (up to 4 nodes × 4 GPUs) except 11 and 13, which zigzag's
# seq % (2*ZIGZAG_N*cp) == 0 requirement cannot satisfy at a feasible seq. head_dim
# is swept over {32, 64, 128, 256}. Speedup is total_ms(cp=1) / total_ms(cp=N).
# ZIGZAG_N: number of zig-zag passes (default 1; each pass = 2 sub-groups, set 2 for 4-way).
#
#   sbatch scripts/slurm/strong_scaling.sbatch              # full 4-node sweep
#   sbatch --nodes=2 scripts/slurm/strong_scaling.sbatch    # cap at cp8 (2 nodes)
#   sbatch --nodes=1 scripts/slurm/strong_scaling.sbatch    # single node, cp<=4
#
# cp_size values larger than the allocation (nodes*4) are skipped, so the
# same script works at 1, 2, or 4 nodes without edits.
#
# Output: results/bench_ring.csv  (consumed by scripts/plot_scaling.py)
#   columns: mode,cp_size,batch,heads,seq,head_dim,causal,zigzag_n,striped,iters,
#            comm_ms,comp_ms,wait_ms,total_ms,max_err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
CSV=results/bench_ring.csv

cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/release -j

mkdir -p results

# Total ranks available in this allocation (defaults to single node = 4).
NNODES="${SLURM_NNODES:-1}"
MAX_CP=$((NNODES * 4))
echo "nodes=${NNODES}  max cp_size=${MAX_CP}  nodelist=${SLURM_JOB_NODELIST:-?}"

# Write CSV header once (the CLI also prints it per run; plot_scaling.py deduplicates).
echo "mode,cp_size,batch,heads,seq,head_dim,causal,zigzag_n,striped,iters,comm_ms,comp_ms,wait_ms,total_ms,max_err" \
  > "${CSV}"

ITERS=20
ZIGZAG_N="${ZIGZAG_N:-1}"  # number of zig-zag passes; override with: ZIGZAG_N=2 sbatch ...

# Fixed total problem size. seq=20160 = 2^6 * 3^2 * 5 * 7 is divisible by
# 2*cp_size for every cp_size in 1..16 EXCEPT 11 and 13 (those would need a
# factor of 11/13, forcing seq >= 1.4M tokens — infeasible on Turing). The
# divisibility guard in bench_one skips those two so the sweep never aborts.
SEQ=20160

# Production config: ring-overlap, causal, zigzag.
bench_one() {
  local cp_size=$1 head_dim=$2
  # Skip cp_size values that exceed the allocation.
  if [ "${cp_size}" -gt "${MAX_CP}" ]; then return 0; fi
  # Zigzag requires seq divisible by 2*ZIGZAG_N*cp_size; skip cp values it can't satisfy.
  if (( SEQ % (2 * ZIGZAG_N * cp_size) != 0 )); then
    echo "skip cp_size=${cp_size}  (seq=${SEQ} not divisible by 2*${ZIGZAG_N}*cp_size=$((2 * ZIGZAG_N * cp_size)))"
    return 0
  fi
  echo "cp_size=${cp_size}  seq=${SEQ}  head_dim=${head_dim}  (ring-overlap causal zigzag_n=${ZIGZAG_N})"
  mpirun -n "${cp_size}" "${CLI}" \
    --batch 1 --heads 4 \
    --seq "${SEQ}" \
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
echo "Strong-scaling benchmark complete. Results in ${CSV}"
echo "Rows: $(wc -l < "${CSV}")"
echo "Plot with: uv run python scripts/plot_scaling.py ${CSV} --out results/figures/"
