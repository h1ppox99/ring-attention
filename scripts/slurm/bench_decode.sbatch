#!/bin/bash
#SBATCH --job-name=ring-decode-bench
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/bench-decode-%j.out
#SBATCH --error=results/bench-decode-%j.err

# Per-token decode latency sweep. For each (cp_size, prompt_len), measures
# mean total/comp/comm ms over `--decode-tokens` autoregressive steps.
#
# Output CSV columns (per token):
#   run,cp_size,prompt_len,decode_token_idx,context_len,comm_ms,comp_ms,total_ms

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source scripts/env/activate.sh

CLI=./build/release/apps/ring_attention_cli/ring_attention_cli
CSV=results/bench_decode.csv

cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/release -j

mkdir -p results

# Write the CSV header once; subsequent runs append only data rows.
echo "run,cp_size,prompt_len,decode_token_idx,context_len,comm_ms,comp_ms,total_ms" > "${CSV}"

DECODE_TOKENS=16

bench_one() {
  local cp_size=$1 prompt=$2 heads=$3 head_dim=$4
  echo "cp_size=${cp_size}  prompt=${prompt}  heads=${heads}  head_dim=${head_dim}"
  mpirun -n "${cp_size}" "${CLI}" \
    --run decode \
    --prompt-len "${prompt}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --batch 1 --heads "${heads}" --head_dim "${head_dim}" \
    --csv 2>/dev/null \
    | grep -E "^decode,[0-9]" \
    >> "${CSV}"
}

# Sweep cp_size × prompt_len for the typical inference-shaped config (Llama-ish).
for cp_size in 1 2 4; do
  for prompt in 1024 4096 16384; do
    for head_dim in 64 128; do
      bench_one "${cp_size}" "${prompt}" 8 "${head_dim}"
    done
  done
done

echo ""
echo "Decode benchmark complete. Results in ${CSV}"
echo "Rows: $(wc -l < "${CSV}")"
