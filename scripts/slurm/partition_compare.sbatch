#!/bin/bash
#SBATCH --job-name=ring-partition-cmp
#SBATCH --partition=gpu-turing
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/partition-cmp-%j.out
#SBATCH --error=results/partition-cmp-%j.err
#
# Partition-scheme comparison: contiguous vs striped vs zig-zag, side by side,
# on the SAME shape. Unlike strong_scaling/weak_scaling (which hardcode zig-zag),
# this sweeps the four schemes together so their curves can be compared directly.
#
# Two axes (pick with AXIS):
#   AXIS=seq : fixed cp_size (CP), sweep SEQS  -> compute-vs-comm regime crossover.
#   AXIS=cp  : fixed SEQ, sweep CPS            -> load-imbalance vs rank-count.
#
# Schemes per point (distinguished in the CSV by the zigzag_n/striped columns):
#   contiguous : no flag            (zigzag_n=0, striped=0)
#   striped    : --striped 1        (zigzag_n=0, striped=1)
#   zigzag-n1  : --zigzag-n 1        (2 sub-groups)
#   zigzag-n2  : --zigzag-n 2        (4 sub-groups)
#
# Divisibility: striped/contiguous need seq % cp == 0; --zigzag-n N needs
# seq % (2*N*cp) == 0. Points that don't satisfy it are skipped, not aborted.
#
# Examples:
#   sbatch scripts/slurm/partition_compare.sbatch                         # seq sweep, cp=4
#   AXIS=cp sbatch scripts/slurm/partition_compare.sbatch                 # cp sweep, seq=65536
#   AXIS=seq CP=4 SEQS="16384 32768 65536" sbatch scripts/slurm/partition_compare.sbatch
#   AXIS=cp SEQ=65536 CPS="4 8 16" sbatch scripts/slurm/partition_compare.sbatch
#
# Output: results/partition_compare_<axis>_<YYYYmmdd-HHMMSS>.csv  (timestamped per
# launch; override with CSV=<path>). Columns:
#   mode,cp_size,batch,heads,seq,head_dim,causal,zigzag_n,striped,iters,
#   comm_ms,comp_ms,wait_ms,total_ms,max_err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli

cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/release -j

mkdir -p results

NNODES="${SLURM_NNODES:-1}"
MAX_CP=$((NNODES * 4))
echo "nodes=${NNODES}  max cp_size=${MAX_CP}  nodelist=${SLURM_JOB_NODELIST:-?}"

# --- knobs (all overridable from the sbatch command line via env) ---
AXIS="${AXIS:-seq}"                       # seq | cp
CP="${CP:-4}"                             # fixed cp_size when AXIS=seq
SEQ="${SEQ:-65536}"                       # fixed seq when AXIS=cp
SEQS="${SEQS:-16384 32768 65536 131072}"  # seq list when AXIS=seq
CPS="${CPS:-4 8 16}"                       # cp list when AXIS=cp
HEAD_DIM="${HEAD_DIM:-128}"
HEADS="${HEADS:-4}"
ITERS="${ITERS:-20}"

# CSV name is timestamped (launch date-time) so repeated runs never clobber each
# other. Override the whole path with CSV=... if you want a fixed name.
CSV="${CSV:-results/partition_compare_${AXIS}_$(date +%Y%m%d-%H%M%S).csv}"
echo "writing -> ${CSV}"

echo "mode,cp_size,batch,heads,seq,head_dim,causal,zigzag_n,striped,iters,comm_ms,comp_ms,wait_ms,total_ms,max_err" \
  > "${CSV}"

# Run one (scheme, cp, seq) point. $5 = required seq divisor for that scheme.
run_scheme() {
  local cp=$1 seq=$2 hd=$3 label=$4 div=$5
  shift 5
  if [ "${cp}" -gt "${MAX_CP}" ]; then return 0; fi
  if (( seq % div != 0 )); then
    echo "skip ${label}  cp=${cp} seq=${seq}  (seq not divisible by ${div})"
    return 0
  fi
  echo "scheme=${label}  cp=${cp}  seq=${seq}  head_dim=${hd}"
  mpirun -n "${cp}" "${CLI}" \
    --batch 1 --heads "${HEADS}" \
    --seq "${seq}" \
    --head_dim "${hd}" \
    --causal 1 \
    --mode ring-overlap \
    "$@" \
    --iters "${ITERS}" \
    --csv 2>/dev/null \
    | grep "^ring-overlap," \
    >> "${CSV}"
}

# All four schemes at one (cp, seq, head_dim) point.
bench_point() {
  local cp=$1 seq=$2 hd=$3
  run_scheme "${cp}" "${seq}" "${hd}" "contiguous" "${cp}"
  run_scheme "${cp}" "${seq}" "${hd}" "striped"    "${cp}"            --striped 1
  run_scheme "${cp}" "${seq}" "${hd}" "zigzag-n1"  $((2 * cp))        --zigzag-n 1
  run_scheme "${cp}" "${seq}" "${hd}" "zigzag-n2"  $((4 * cp))        --zigzag-n 2
}

case "${AXIS}" in
  seq)
    echo "AXIS=seq  cp=${CP}  head_dim=${HEAD_DIM}  seqs=[${SEQS}]"
    for seq in ${SEQS}; do bench_point "${CP}" "${seq}" "${HEAD_DIM}"; done
    ;;
  cp)
    echo "AXIS=cp  seq=${SEQ}  head_dim=${HEAD_DIM}  cps=[${CPS}]"
    for cp in ${CPS}; do bench_point "${cp}" "${SEQ}" "${HEAD_DIM}"; done
    ;;
  *)
    echo "ERROR: AXIS must be 'seq' or 'cp' (got '${AXIS}')" >&2
    exit 1
    ;;
esac

echo ""
echo "Partition comparison complete. Results in ${CSV}"
echo "Rows: $(wc -l < "${CSV}")"
