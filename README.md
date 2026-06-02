# Ring Attention

**A from-scratch C++/CUDA implementation, tuned for Stanford's Turing (sm_75) GPU cluster.**

[![CI](https://github.com/h1ppox99/ring-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/h1ppox99/ring-attention/actions/workflows/ci.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![CUDA 12.3](https://img.shields.io/badge/CUDA-12.3-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://www.python.org/)

---

## Content

**TODO** : rewrite this content more concisely

Attention is the one transformer block that cannot be trivially sequence-parallelised:
every query token needs the keys and values of *every* position, so the cost grows with the
full sequence length even after the rest of the model is split across GPUs. **Ring Attention**
removes that wall. Each GPU holds only a `1/cp_size` slice of Q/K/V and the K/V shards are
rotated around a ring of GPUs; partial attention is computed against each shard as it arrives
and merged with a FlashAttention-style **online softmax**. Memory per GPU scales with
`1/cp_size`, so the maximum context length grows linearly as you add GPUs — without ever
materialising the full sequence on any single device.

See Liu, Zaharia, and Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context* (2023),
[arXiv:2310.01889](https://arxiv.org/abs/2310.01889) for more details.

This repository is a from-scratch **CUDA/C++ implementation** with a matching
**Python/PyTorch reference** for correctness testing. It includes:

- **Zig-Zag token assignment** — interleaved first/last token layout that balances
  per-GPU work under a causal mask (otherwise early ranks do most of the work).
- **Communication/compute overlap** — K/V rotation (MPI or NCCL) hidden behind the
  attention kernel using two CUDA streams.
- **A hierarchical 2D ring**  that minimises slow inter-node hops on
  clusters with fast intra-node links and a thin (1 GbE) uplink per node.
- **Distributed-KV-cache decode**  autoregressive single-token decoding
  against a KV history sharded around the ring.
- **FP16 Tensor-Core path**, GQA/MQA support, and a benchmark/verification CLI.

### Repository layout

```
src/          CUDA/C++ library — ring loop, attention kernels, decode, KV cache
apps/         Executables: ring_attention_cli (driver), bench_attention, hello_mpi_cuda
tests/        CTest C++/CUDA unit tests (single-GPU)
reference/    Python/PyTorch reference implementation + pytest suite
scripts/      env/activate.sh (session setup), slurm/ (SBATCH jobs)
docs/         Design docs, hierarchical-ring analysis, write-up
```

---

## Getting Started

### Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/h1ppox99/ring-attention.git
cd ring-attention

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install command-line tools (pre-commit, clang-format)
uv tool install --python /usr/bin/python3.11 pre-commit
uv tool install --python /usr/bin/python3.11 clang-format

# 4. Set up the Python environment for the reference implementation
uv sync

# 5. Install the git hooks
pre-commit install --install-hooks

# 6. Load cluster modules and activate the environment
source scripts/env/activate.sh
```

> [!WARNING]
> **Cluster note.** Development targets the Stanford HPCC `gpu-turing` partition
> (Turing / `sm_75`, NVHPC 24.1 → nvcc 12.3, Open MPI 4.1.7, NCCL). The login node has no
> GPU — always run GPU work under `salloc`/`sbatch`.

### Verify your setup

```bash
# Python side
uv run pytest                        # all reference tests should pass
uv run ruff check .                  # no lint errors

# C++/CUDA side (requires cluster modules loaded)
cmake --preset=release               # configures the build; prints summary
cmake --build build/release -j       # compiles
```

To verify CUDA + MPI actually work on a compute node:

```bash
salloc --partition=gpu-turing --gres=gpu:2 --ntasks=2 --time=00:10:00
source scripts/env/activate.sh
mpirun -n 2 ./build/release/apps/hello_mpi_cuda/hello_mpi_cuda
```

You should see two lines, one per rank, each reporting a different GPU.

---

## Usage

The distributed driver is `apps/ring_attention_cli`. **One MPI rank per GPU.**

```bash
salloc --partition=gpu-turing --gres=gpu:2 --ntasks=2 --time=00:15:00
source scripts/env/activate.sh
cmake --build build/release -j

# Causal, zig-zag, overlap mode, fp16, with correctness check
mpirun -n 2 ./build/release/apps/ring_attention_cli/ring_attention_cli \
    --seq 4096 --heads 8 --head_dim 64 \
    --causal 1 --zigzag 1 \
    --mode ring-overlap --dtype fp16 --iters 10 --verify
```

`--verify` re-runs a CPU reference pass and reports the max absolute error; expect it within
`atol=1e-3, rtol=1e-3` (fp16 arithmetic).

### Decode mode

Autoregressive single-token decoding against a KV cache sharded around the ring:

```bash
mpirun -n 2 ./build/release/apps/ring_attention_cli/ring_attention_cli \
    --run decode --prompt-len 4096 --decode-tokens 8 \
    --heads 8 --head_dim 64 --dtype fp16
```

### Key flags

| Flag | Meaning |
|---|---|
| `--seq N` | total sequence length (must be divisible by `cp_size`, i.e. the number of ranks). |
| `--heads`, `--head_dim`, `--batch` | attention shape. |
| `--kv_heads K` | `0` = MHA; set `< heads` for GQA/MQA. |
| `--causal 0/1` | apply the lower-triangular mask. |
| `--zigzag 0/1` | interleaved token assignment for causal load balance (needs `seq` divisible by `2 * cp_size`). |
| `--striped 0/1` | striped token assignment: token `i` → rank `i % cp_size`. Balances causal-mask work as a single strided shard with a near-uniform per-ring-step mask (needs `seq` divisible by `cp_size`). Mutually exclusive with `--zigzag`; not supported with `--mode allgather`. |
| `--dtype fp32\|fp16` | precision. `fp16` exercises the Tensor-Core path (half-precision K/V on the wire, fp32 softmax accumulators). Default `fp32`. |
| `--iters N` | timed iterations. |
| `--verify` | re-run a CPU reference and report max absolute error. |
| `--csv` / `--csv-header` | emit one CSV result line; pair `--csv-header` with the first run when appending many runs to a file. |
| `--run prefill\|decode` | benchmark regime (default `prefill`). `decode` uses `--prompt-len` / `--decode-tokens`. |
| `--mem-probe` | allocate device buffers for the config and exit (capacity sweep; clean exit = fits, OOM = aborts). fp16 `ring-overlap` only. |

---

## Documentation

**TODO** : point to final report once added in `docs/`.

---

## How to cite

If you use this code in academic work, please cite it as:

```bibtex
@software{ring_attention,
  author  = {Wallaert, Hippolyte and Rabasse, Edouard},
  title   = {Ring Attention: Zig-Zag Context-Parallel Attention in CUDA/C++},
  year    = {2026},
  url     = {https://github.com/h1ppox99/ring-attention}
}
```

The underlying algorithm is from Liu, Zaharia, and Abbeel,
*Ring Attention with Blockwise Transformers for Near-Infinite Context* (2023),
[arXiv:2310.01889](https://arxiv.org/abs/2310.01889).

## Contributors

- **Hippolyte Wallaert** — [@h1ppox99](https://github.com/h1ppox99)
- **Edouard Rabasse** — [@edouard-rabasse](https://github.com/edouard-rabasse)

Developed for Stanford **CME 213** (Introduction to Parallel Computing using MPI, OpenMP,
and CUDA).
