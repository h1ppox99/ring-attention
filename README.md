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
