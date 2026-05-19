#!/bin/bash
#SBATCH --job-name=ring-bench
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=results/bench-%j.out
#SBATCH --error=results/bench-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
CSV=results/bench_ring.csv

cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/release -j

mkdir -p results

# Write CSV header once (the CLI also prints it per run; plot_scaling.py deduplicates).
echo "mode,cp_size,batch,heads,seq,head_dim,causal,zigzag,iters,comm_ms,comp_ms,wait_ms,total_ms,max_err" \
  > "${CSV}"

ITERS=20

for mode in allgather; do
  for cp_size in 1 2 4; do
    for seq in 1024 4096 16384; do
      for head_dim in 64 128; do
        for causal in 0 1; do
          echo "mode=${mode}  cp_size=${cp_size}  seq=${seq}  head_dim=${head_dim}  causal=${causal}"
          mpirun -n "${cp_size}" "${CLI}" \
            --batch 1 --heads 4 \
            --seq "${seq}" \
            --head_dim "${head_dim}" \
            --causal "${causal}" \
            --mode "${mode}" \
            --iters "${ITERS}" \
            --csv 2>/dev/null \
            | grep -v "^mode," \
            >> "${CSV}"
        done
      done
    done
  done
done

echo ""
echo "Benchmark complete. Results in ${CSV}"
echo "Rows: $(wc -l < "${CSV}")"
