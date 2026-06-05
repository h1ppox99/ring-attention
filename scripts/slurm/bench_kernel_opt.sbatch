#!/bin/bash
#SBATCH --job-name=ring-kernel-opt
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=results/bench_kernel_opt-%j.out
#SBATCH --error=results/bench_kernel_opt-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/env/activate.sh

mkdir -p results

CSV_OPT=results/bench_ring_opt.csv
CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
ITERS=20

# --- Build baseline (current main) ---
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 -DUSE_NCCL=OFF
cmake --build build/release -j

echo "mode,cp_size,batch,heads,seq,head_dim,causal,zigzag,striped,iters,comm_ms,comp_ms,wait_ms,total_ms,max_err" \
  > results/bench_ring_baseline_cmp.csv

for cp_size in 1 2 4; do
  for seq in 4096 16384; do
    for head_dim in 64 128; do
      echo "baseline  cp=${cp_size}  seq=${seq}  D=${head_dim}"
      mpirun -n "${cp_size}" "${CLI}" \
        --batch 1 --heads 4 --seq "${seq}" --head_dim "${head_dim}" \
        --causal 1 --mode ring-overlap --iters "${ITERS}" --csv 2>/dev/null \
        | grep -v "^mode," >> results/bench_ring_baseline_cmp.csv
    done
  done
done

# --- Swap in the optimized kernel files only ---
git checkout origin/improve-kernel -- src/attention_step.cu src/flash_attention.cu

cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75 -DUSE_NCCL=OFF
cmake --build build/release -j --target ring_attention_cli

echo "mode,cp_size,batch,heads,seq,head_dim,causal,zigzag,striped,iters,comm_ms,comp_ms,wait_ms,total_ms,max_err" \
  > "${CSV_OPT}"

for cp_size in 1 2 4; do
  for seq in 4096 16384; do
    for head_dim in 64 128; do
      echo "opt  cp=${cp_size}  seq=${seq}  D=${head_dim}"
      mpirun -n "${cp_size}" "${CLI}" \
        --batch 1 --heads 4 --seq "${seq}" --head_dim "${head_dim}" \
        --causal 1 --mode ring-overlap --iters "${ITERS}" --csv 2>/dev/null \
        | grep -v "^mode," >> "${CSV_OPT}"
    done
  done
done

# --- Restore kernel files to main ---
git checkout HEAD -- src/attention_step.cu src/flash_attention.cu

echo ""
echo "Baseline: results/bench_ring_baseline_cmp.csv"
echo "Optimized: ${CSV_OPT}"
echo "Rows each: $(wc -l < "${CSV_OPT}")"
