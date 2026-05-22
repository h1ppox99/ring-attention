#!/usr/bin/env bash
# Extract text summaries from Nsight Systems (.nsys-rep) and
# Nsight Compute (.ncu-rep) profile reports.
#
# Usage:
#   bash scripts/analyze_profiles.sh             # default dirs
#   NSYS_DIR=results/nsys NCU_DIR=results/ncu \
#       OUT_DIR=results/stats bash scripts/analyze_profiles.sh
#
# Outputs (under ${OUT_DIR}, default results/stats/):
#   nsys/<tag>__kern.csv      per-kernel time summary
#   nsys/<tag>__mem.csv       H2D/D2H/DtoD memcpy summary
#   nsys/<tag>__mpi.csv       MPI call summary (Isend, Irecv, Waitall, Allgather, ...)
#   ncu/<tag>__sol.csv        compute / mem / occupancy SoL metrics
#   summary.txt               human-readable digest (rank-0 kernel + MPI per mode)

set -euo pipefail

NSYS_DIR="${NSYS_DIR:-results/nsys}"
NCU_DIR="${NCU_DIR:-results/ncu}"
OUT_DIR="${OUT_DIR:-results/stats}"

mkdir -p "${OUT_DIR}/nsys" "${OUT_DIR}/ncu"

# ----------------------------------------------------------------------------
# Nsight Systems: kernel / memcpy / MPI summaries per .nsys-rep
# ----------------------------------------------------------------------------
echo "=== Nsight Systems ==="
shopt -s nullglob
for rep in "${NSYS_DIR}"/*.nsys-rep; do
  tag="$(basename "${rep}" .nsys-rep)"
  echo "  ${tag}"

  # --format=csv -> machine readable; --force-export=true regenerates if stale
  nsys stats --format=csv --force-export=true --quiet \
    --report cuda_gpu_kern_sum \
    --output "${OUT_DIR}/nsys/${tag}__kern" \
    "${rep}" >/dev/null

  nsys stats --format=csv --force-export=true --quiet \
    --report cuda_gpu_mem_time_sum \
    --output "${OUT_DIR}/nsys/${tag}__mem" \
    "${rep}" >/dev/null

  # mpi_event_sum only exists when the run was traced with --trace=...,mpi
  nsys stats --format=csv --force-export=true --quiet \
    --report mpi_event_sum \
    --output "${OUT_DIR}/nsys/${tag}__mpi" \
    "${rep}" >/dev/null 2>&1 || echo "    (no MPI events found)"
done

# ----------------------------------------------------------------------------
# Nsight Compute:
# ----------------------------------------------------------------------------
echo ""
echo "=== Nsight Compute ==="
for rep in "${NCU_DIR}"/*.ncu-rep; do
  tag="$(basename "${rep}" .ncu-rep)"
  echo "  ${tag}"

  # --csv  : machine readable; --page details : every collected metric
  # The grep filters to SoL throughput rows + occupancy — feel free to widen.
  ncu --import "${rep}" --csv --page details \
    > "${OUT_DIR}/ncu/${tag}__all.csv" 2>/dev/null

  {
    head -n 1 "${OUT_DIR}/ncu/${tag}__all.csv"
    grep -E "sm__throughput|dram__throughput|achieved_occupancy|sm__sass_thread_inst_executed_op_(f|h)" \
      "${OUT_DIR}/ncu/${tag}__all.csv" || true
  } > "${OUT_DIR}/ncu/${tag}__sol.csv"
done

# ----------------------------------------------------------------------------
#  top kernel + MPI breakdown for rank 0 of each mode
# ----------------------------------------------------------------------------
echo ""
echo "=== Building summary.txt ==="
SUMMARY="${OUT_DIR}/summary.txt"
{
  echo "Ring Attention profile digest"
  echo "Generated: $(date)"
  echo "=============================================================="
  for rep in "${NSYS_DIR}"/*_p4_0.nsys-rep; do
    tag="$(basename "${rep}" .nsys-rep)"
    mode="${tag%_p4_0}"
    echo ""
    echo "----- ${mode} (rank 0) -----"
    echo "[top kernels]"
    head -n 6 "${OUT_DIR}/nsys/${tag}__kern.csv" 2>/dev/null || echo "  (missing)"
    echo "[MPI events]"
    head -n 6 "${OUT_DIR}/nsys/${tag}__mpi.csv" 2>/dev/null || echo "  (missing)"
  done
} > "${SUMMARY}"

echo "Done. See ${SUMMARY} and CSVs under ${OUT_DIR}/."
