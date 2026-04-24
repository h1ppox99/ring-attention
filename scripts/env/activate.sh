#!/usr/bin/env bash
# Source this script at the start of every session:
#   source scripts/env/activate.sh
#
# It loads the NVHPC module (nvcc, mpirun, NCCL, NVSHMEM) and activates
# the Python virtual environment managed by uv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load NVHPC 24.1 (provides nvcc 12.3, Open MPI, NCCL, NVSHMEM, math libs).
# Skip if already loaded to avoid duplicate PATH entries.
if ! module is-loaded course/cme213/nvhpc/24.1 2>/dev/null; then
    module load course/cme213/nvhpc/24.1
fi

# The NVHPC HPC-X MPI was built with OPAL_PREFIX pointing to /proj/nv/...,
# which doesn't exist on compute nodes.  Override it to the local install.
NVHPC_ROOT="/home/cme213/software/nvidia-hpc-sdk/2024_24.1/Linux_x86_64/24.1"
export OPAL_PREFIX="${NVHPC_ROOT}/comm_libs/12.3/hpcx/hpcx-2.17.1/ompi"

# Make sure uv's bin directory is on PATH.
export PATH="${HOME}/.local/bin:${PATH}"

# Activate Python venv (created by `uv sync`).
VENV="${REPO_ROOT}/.venv"
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
else
    echo "[activate.sh] WARNING: .venv not found — run 'uv sync' first." >&2
fi

echo "[activate.sh] Environment ready."
echo "  nvcc  : $(nvcc --version 2>/dev/null | grep 'release' | awk '{print $5, $6}' || echo 'not found')"
echo "  mpirun: $(mpirun --version 2>/dev/null | head -1 || echo 'not found')"
echo "  python: $(python --version 2>/dev/null || echo 'not found')"
