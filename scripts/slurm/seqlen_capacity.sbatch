#!/bin/bash
#SBATCH --job-name=ring-capacity
#SBATCH --partition=gpu-turing
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/capacity-%j.out
#SBATCH --error=results/capacity-%j.err
#
# Sequence-length CAPACITY sweep: how long a context can we fit as we add GPUs?
#
# This is the headline scalability experiment for the report. Ring attention's
# per-GPU memory footprint is O(S / cp_size): each rank only ever materializes
# its own shard plus a constant number of in-flight K/V buffers. So the longest
# sequence that fits on the cluster should grow ~LINEARLY with the number of
# GPUs. A single Turing GPU OOMs at some S_max; 20 GPUs should reach ~20*S_max.
#
# For each cp_size we BINARY-SEARCH the largest total sequence length S that
# runs without an out-of-memory abort (one real forward pass, --iters 1, in the
# production config: ring-overlap + causal + zigzag). We seed each search from
# the previous cp_size's result, because capacity is expected to scale linearly
# — so the doubling phase is short and the whole 20-GPU sweep fits in 30 min.
#
# Detection: on cudaMalloc failure the kernel calls std::abort(), so mpirun
# returns non-zero. Exit 0 == "fits"; anything else == "does not fit".
#
#   sbatch scripts/slurm/seqlen_capacity.sbatch                 # full 5-node sweep
#   sbatch --nodes=2 scripts/slurm/seqlen_capacity.sbatch      # cap at cp8
#   DTYPE=fp32 sbatch scripts/slurm/seqlen_capacity.sbatch     # fp32 instead of fp16
#   CP_LIST="1 4 8 16 20" sbatch scripts/slurm/seqlen_capacity.sbatch
#
# Results are written incrementally, so a partial sweep (e.g. if the 30-min
# wall is hit) still yields a usable CSV.
#
# Output: results/bench_capacity.csv
#   columns: dtype,heads,head_dim,batch,causal,zigzag,cp_size,gpus,nodes_used,
#            max_seq,max_seq_per_gpu,gpu_mem_mb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
CSV=results/bench_capacity.csv

cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/release -j

mkdir -p results

# ---------------------------------------------------------------------------
# Configuration (override from the environment / sbatch --export).
# ---------------------------------------------------------------------------
DTYPE="${DTYPE:-fp16}"     # fp16 is the realistic long-context inference dtype
HEADS="${HEADS:-4}"
HEAD_DIM="${HEAD_DIM:-128}"
BATCH="${BATCH:-1}"
SEED_SEQ="${SEED_SEQ:-16384}"   # first doubling seed for cp_size=1
PROBE_TIMEOUT="${PROBE_TIMEOUT:-120}"  # seconds per probe; guards against hangs
# Converge the binary search to within TOL_FRAC of the boundary (relative), not
# to the zigzag granularity. The plot only needs ~couple-% accuracy, and each
# extra probe is a full mpirun (NCCL re-init across up to 5 nodes), so chasing
# 2-token precision would balloon the sweep past the 30-min wall.
TOL_FRAC="${TOL_FRAC:-0.02}"

NNODES="${SLURM_NNODES:-1}"
MAX_CP=$((NNODES * 4))

# Default sweep: a representative set that draws a clean line and includes the
# cp4->cp5 node-crossing. Restrict to what the allocation can host.
CP_LIST="${CP_LIST:-1 2 3 4 5 6 8 10 12 14 16 18 20}"

echo "nodes=${NNODES}  max cp_size=${MAX_CP}  dtype=${DTYPE}  heads=${HEADS}  head_dim=${HEAD_DIM}"
echo "nodelist=${SLURM_JOB_NODELIST:-?}"

# Per-GPU total memory (homogeneous nodes -> probe device 0 once).
# `sed -n 1p` (not `head -1`) consumes all input, so nvidia-smi's later lines
# (one per GPU) don't trigger SIGPIPE -> pipefail -> a spurious extra value.
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n 1p)"
GPU_MEM_MB="${GPU_MEM_MB:-0}"
echo "gpu_mem_mb=${GPU_MEM_MB}"

echo "dtype,heads,head_dim,batch,causal,zigzag,cp_size,gpus,nodes_used,max_seq,max_seq_per_gpu,gpu_mem_mb" \
  > "${CSV}"

# round_down VALUE STEP -> largest multiple of STEP that is <= VALUE
round_down() { echo $(( ($1 / $2) * $2 )); }

# fits CP SEQ -> return 0 if every device buffer for (cp_size=CP, seq=SEQ) is
# allocatable, non-zero otherwise. --mem-probe allocates all buffers (and the
# NCCL transport) then exits WITHOUT running attention: a clean exit means the
# config fits in device memory; an OOM aborts via cudaCheck (non-zero exit).
# This measures the MEMORY capacity in ~2s/probe — running the real O(S^2)
# forward pass at the boundary costs minutes and measures compute, not memory.
# Output is suppressed; only the exit code matters. The timeout guards hangs.
fits() {
  local cp=$1 seq=$2
  timeout "${PROBE_TIMEOUT}" mpirun -n "${cp}" "${CLI}" \
    --batch "${BATCH}" --heads "${HEADS}" \
    --seq "${seq}" \
    --head_dim "${HEAD_DIM}" \
    --dtype "${DTYPE}" \
    --causal 1 \
    --mode ring-overlap \
    --zigzag 1 \
    --iters 1 \
    --mem-probe \
    > /dev/null 2>&1
}

# Binary-search the largest seq (multiple of 2*cp) that fits for this cp_size.
# Seeds the doubling phase from $SEED (the previous cp's result) so the search
# is short under the expected linear capacity scaling.
SEED="${SEED_SEQ}"
for cp in ${CP_LIST}; do
  if [ "${cp}" -gt "${MAX_CP}" ]; then
    echo "skip cp_size=${cp} (exceeds allocation of ${MAX_CP} GPUs)"
    continue
  fi

  step=$((2 * cp))                       # zigzag granularity
  lo=0                                   # largest seq known to fit (0 = none yet)
  hi=$(round_down "${SEED}" "${step}")
  [ "${hi}" -lt "${step}" ] && hi=${step}

  echo "=== cp_size=${cp}  (step=${step}, seed=${hi}) ==="

  # Grow hi until it fails (establishes a fitting lo and a failing hi).
  while fits "${cp}" "${hi}"; do
    lo=${hi}
    hi=$((hi * 2))
    echo "  grow: lo=${lo} fits; try hi=${hi}"
  done
  echo "  bound: lo=${lo} (fits), hi=${hi} (fails)"

  # Binary search in (lo, hi]. Stop once the bracket is within TOL_FRAC of lo
  # (or one zigzag step, whichever is larger) — a couple-% boundary is all the
  # plot needs and keeps the probe count (and NCCL re-inits) low.
  # tol = max(step, lo*TOL_FRAC), recomputed as lo rises.
  tol() { awk -v lo="$1" -v f="${TOL_FRAC}" -v s="${step}" \
            'BEGIN{t=int(lo*f); if(t<s)t=s; print t}'; }
  while [ $((hi - lo)) -gt "$(tol "${lo}")" ]; do
    mid=$(round_down $(((lo + hi) / 2)) "${step}")
    if [ "${mid}" -le "${lo}" ]; then break; fi
    if fits "${cp}" "${mid}"; then
      lo=${mid}
      echo "  probe ${mid}: fits   -> lo=${lo}"
    else
      hi=${mid}
      echo "  probe ${mid}: OOM    -> hi=${hi}"
    fi
  done

  max_seq=${lo}
  per_gpu=$((max_seq / cp))
  nodes_used=$(( (cp + 3) / 4 ))
  echo "  >>> cp_size=${cp}: max_seq=${max_seq}  (per_gpu=${per_gpu}, nodes=${nodes_used})"

  echo "${DTYPE},${HEADS},${HEAD_DIM},${BATCH},1,1,${cp},${cp},${nodes_used},${max_seq},${per_gpu},${GPU_MEM_MB}" \
    >> "${CSV}"

  # Seed the next (larger) cp from this result: capacity ~ linear in cp, so
  # next cp's max is ~ (next_cp/cp)*max_seq. Use max_seq as a safe lower seed.
  [ "${max_seq}" -gt 0 ] && SEED=${max_seq}
done

echo ""
echo "Capacity sweep complete. Results in ${CSV}"
cat "${CSV}"
echo ""
echo "Plot with: uv run python scripts/plot_seqlen_capacity.py ${CSV} --out results/figures/"
