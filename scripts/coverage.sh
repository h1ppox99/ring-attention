#!/usr/bin/env bash
# Build the project with coverage instrumentation, run ctest, and emit a
# gcovr text report at results/coverage/cov.txt.
#
# Run from inside a GPU allocation (login node has no GPU, so ctest would
# fail — that's expected, see CLAUDE.md). Example:
#     salloc --partition=gpu-turing --gres=gpu:2 --ntasks=2 --time=00:30:00
#     source scripts/env/activate.sh
#     bash scripts/coverage.sh
#
# Extra arguments are forwarded to ctest, e.g.:
#     bash scripts/coverage.sh -L gpu
#     bash scripts/coverage.sh -R flash_attention
#
# gcov only instruments host code. CUDA device kernels are not measured —
# the report covers C++ host code (including the host portions of .cu files).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="build/coverage"
OUT_DIR="results/coverage"
TXT_REPORT="${OUT_DIR}/cov.txt"

mkdir -p "${OUT_DIR}"

# ── Configure + build (coverage preset forces g++ so --coverage works) ────────
cmake --preset=coverage -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build "${BUILD_DIR}" -j

# ── Wipe stale counter data so this run is not merged with the previous one.
# .gcno (notes) are kept — they are produced at compile time and are stable
# until the next rebuild.
find "${BUILD_DIR}" -name '*.gcda' -delete

# ── Run the tests. Tolerate failures so a single broken test still produces
# a partial report; the script's exit code propagates ctest's result.
ctest --test-dir "${BUILD_DIR}" --output-on-failure "$@"
CTEST_STATUS=$?

# ── Locate gcovr. Prefer the project venv so the script works in a fresh
# salloc shell where the user has not sourced scripts/env/activate.sh.
GCOVR="${REPO_ROOT}/.venv/bin/gcovr"
if [ ! -x "${GCOVR}" ]; then
  GCOVR="$(command -v gcovr || true)"
fi
if [ -z "${GCOVR}" ]; then
  echo "ERROR: gcovr not found. Run 'uv sync' from the repo root." >&2
  exit 1
fi

# ── Coverage report. Flags mirror scripts/slurm/gpu_tests.sbatch:
#   suspicious_hits.warn_once_per_file: gcc bug #68080 trips on the >1e9
#       iteration counts inside cpu_attention.cpp's reference loops.
#   --exclude-{throw,unreachable}-branches: drop implicit exception edges
#       from std::vector / cudaMalloc / std::string calls that gcov counts
#       but no passing test can cover, so branch % stays meaningful.
"${GCOVR}" -r . --filter 'src/' --filter 'tests/' \
      --gcov-ignore-errors=source_not_found \
      --gcov-ignore-parse-errors=suspicious_hits.warn_once_per_file \
      --exclude-throw-branches \
      --exclude-unreachable-branches \
      --txt "${TXT_REPORT}" \
      --sort uncovered-number \
      --print-summary

# ── Stamp the report with when it was generated. gcovr overwrites the file on
# each run, so we prepend the timestamp after gcovr has written it.
STAMP="Coverage report generated: $(date '+%Y-%m-%d %H:%M:%S %Z')"
printf '%s\n\n%s' "${STAMP}" "$(cat "${TXT_REPORT}")" > "${TXT_REPORT}"

echo ""
echo "Coverage report written to ${TXT_REPORT}"

exit "${CTEST_STATUS}"
