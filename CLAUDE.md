# CLAUDE.md — Ring Attention project conventions

## Project overview

CUDA/C++ implementation of Zig-Zag Ring Attention for context parallelism, with a Python
reference implementation for correctness testing. Uses MPI for inter-GPU communication in a
ring pattern. See `RING_ATTENTION.md` for the algorithmic background.

## Environment (cluster)

**Cluster**: Stanford HPCC. Login node has no GPU — always use `salloc`/`sbatch` for GPU work.

**GPU partition**: `gpu-turing` — 5 nodes (`hpcc-gpu-5-[1-5]`), 4× Turing GPUs each (sm_75), 16 CPUs, 128 GB RAM.

**Every new session**:
```bash
source scripts/env/activate.sh   # loads course/cme213/nvhpc/24.1 + activates .venv
```

**Interactive GPU session**:
```bash
salloc --partition=gpu-turing --gres=gpu:2 --ntasks=2 --time=00:30:00
source scripts/env/activate.sh
```

The NVHPC 24.1 module provides: nvcc 12.3, Open MPI 4.1.7, NCCL, NVSHMEM, cuBLAS/cuFFT.
It sets `CC=nvc` and `CXX=nvc++` — these are used as the CMake CXX compiler.

## Build system

**Generator**: Unix Makefiles (Ninja not installed on this cluster).

```bash
cmake --preset=release          # configure (creates build/release/)
cmake --build build/release -j  # build all targets
ctest --test-dir build/release  # run C++ tests
```

For debug builds replace `release` with `debug`. Override CUDA arch with:
```bash
cmake --preset=release -DCMAKE_CUDA_ARCHITECTURES=75
```

**CUDA architecture**: `75` (Turing) — default in CMakeLists.txt. Change to `80` (A100) or `90` (H100) if running on a different cluster.

## Directory layout

```
src/          CUDA/C++ library code (ring attention kernels)
apps/         Executable targets (e.g., hello_mpi_cuda)
tests/        CTest-based C++/CUDA unit tests
reference/    Python reference implementation (ring_attention_ref package)
  ring_attention_ref/   Pure-Python/PyTorch reference
  tests/                pytest test suite
scripts/
  env/activate.sh       Module + venv setup — source every session
  slurm/                SBATCH job scripts
docs/design/            Per-milestone design documents
```

## Python reference

Managed by `uv`. Python 3.11 required.

```bash
uv sync               # install deps into .venv
uv run pytest         # run reference tests
uv run ruff check .   # lint
uv run mypy           # type-check
```

All Python code lives under `reference/`. The package is `ring_attention_ref`.
Style: ruff (line length 100, py311 target). Type annotations required (mypy strict).

## C++/CUDA conventions

- C++17, CUDA 17. Formatting: Google style, 2-space indent, 100-col limit (`.clang-format`).
- Host-device split: `.cu` for files containing kernels, `.cpp` for host-only code, `.cuh`/`.hpp` for headers.
- MPI ranks == GPU IDs (one rank per GPU). `MPI_Init` at entry, `MPI_Finalize` at exit.
- Use `mpirun -n N` (not `srun`) on this cluster. SLURM's PMIx server is incompatible with NVHPC 24.1's OpenMPI, so `srun` fails at `MPI_Init`. `mpirun` works fine inside a SLURM allocation.
- Prefer `cudaCheck(...)` macro around CUDA calls; `MPI_CHECK(...)` around MPI calls.

## Pre-commit hooks

Hooks run automatically on `git commit`. To run manually:
```bash
pre-commit run --all-files
```
Hooks: trailing whitespace, YAML check, large-file guard, ruff (Python), clang-format (C++/CUDA).

## Testing strategy

- **Python**: pytest in `reference/tests/` — correctness of the reference implementation.
- **C++/CUDA**: CTest in `tests/` — unit tests for kernels, can run on a single GPU.
- **Integration**: SBATCH scripts in `scripts/slurm/` — multi-GPU correctness and performance runs.
- Numerical tolerance: compare CUDA output against Python reference with `atol=1e-3, rtol=1e-3` (fp16 arithmetic).

## Key algorithmic notes

Ring Attention: each GPU holds a slice of Q, K, V. In a ring of `cp_size` steps:
1. Non-blocking send current K/V to the next rank.
2. Compute local attention with current K/V (overlap with communication).
3. Receive K/V from previous rank; repeat.

Causal masking requires **Zig-Zag token assignment** to balance compute:
- GPU `i` holds tokens at positions `[i, cp_size*2-1-i, cp_size*2+i, ...]` (interleaved first/last).
- This ensures each GPU has roughly equal work under a causal mask.

Online softmax (flash-attention style) accumulates the output incrementally:
`(m, ℓ, O)` — running max, sum of exp, and weighted output — updated each ring step.
