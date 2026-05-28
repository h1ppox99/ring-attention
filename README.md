## Getting Started

### Setup

```bash
# 1. Clone and enter the repo
git clone git@github.com:<your-username>/ring-attention.git
cd ring-attention

# 2. Install uv (if you don't already have it) — user-space, no admin needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install command-line tools (pre-commit, clang-format)
#    The --python flag is critical: without it, uv falls back to system
#    Python 3.6 and silently installs ancient versions.
uv tool install --python /usr/bin/python3.11 pre-commit
uv tool install --python /usr/bin/python3.11 clang-format

# 4. Set up the Python environment for the reference implementation
uv sync                              # installs deps from uv.lock into .venv

# 5. Install the git hooks
pre-commit install --install-hooks   # downloads and prepares hook environments

# 6. Load cluster modules and activate the environment
source scripts/env/activate.sh
```

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

### Running ring attention

The distributed driver is `apps/ring_attention_cli`. One MPI rank per GPU.

```bash
salloc --partition=gpu-turing --gres=gpu:2 --ntasks=2 --time=00:15:00
source scripts/env/activate.sh
cmake --build build/release -j

mpirun -n 2 ./build/release/apps/ring_attention_cli/ring_attention_cli \
    --seq 4096 --heads 8 --head_dim 64 \
    --causal 1 --zigzag 1 \
    --mode ring-overlap --iters 10 --verify
```

**Modes** (`--mode`, default `ring-overlap`):
- `ring-overlap` — **production path.** Ring rotation with two CUDA streams hiding MPI/NCCL behind the kernel. This is the only mode optimized; use it for any real benchmark.
- `ring-blocking` — *baseline.* Ring rotation with blocking comm between steps. Kept as the experimental control for the overlap numbers in `KERNEL_OPTIMIZATIONS.md`; do not use for production timings.
- `allgather` — *baseline.* `MPI_Allgather` the full K/V, one local pass. Reference for the "no ring" upper bound on memory and the lower bound on comm hiding.

**Flags**:
- `--causal 1` — apply lower-triangular mask.
- `--zigzag 1` — interleave token assignment so each rank gets balanced work under a causal mask (requires `seq` divisible by `2 * cp_size`).
- `--dtype {fp32,fp16}` — select compute/input precision. Use `fp16` to exercise the Tensor-Core path (half-precision K/V on the wire and in the kernel, fp32 online-softmax accumulators); `fp32` for the full-precision baseline. Default is `fp32`.
- `--verify` — re-run a CPU reference pass and report max absolute error.
- `--csv` — emit one CSV result line; pair with `--csv-header` on the first invocation when appending many runs to a file.

### Daily workflow

Every new shell session, before working:

```bash
source scripts/env/activate.sh       # loads modules + activates Python venv
```

Then:
- **Build:** `cmake --build build/release -j`
- **Run Python tests:** `uv run pytest`
- **Run C++ tests:** `ctest --test-dir build/release`
- **Submit a cluster job:** `sbatch scripts/slurm/<job>.sbatch`

Pre-commit hooks run automatically on `git commit`. To check everything manually:

```bash
pre-commit run --all-files
```

### Troubleshooting

- **CMake can't find MPI or CUDA:** modules aren't loaded. Run `source scripts/env/activate.sh`.
- **`cudaErrorNoDevice` when running:** you're on the login node (no GPU). Use `salloc` or `sbatch` to get a compute node.
- **`srun` fails at `MPI_Init` with `orte_ess_init failed`:** SLURM's PMIx is incompatible with NVHPC 24.1's OpenMPI. Use `mpirun -n N`, not `srun`, inside the allocation.

For more detail, see `docs/design/` (per-milestone design docs) and `CLAUDE.md` (project conventions).
