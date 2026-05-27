---
name: profile-optimize
description: Profile the ring-attention CUDA kernels, identify bottlenecks from Nsight Compute / Nsight Systems reports, apply optimizations, and validate the next profile against the previous one. Use when the user asks to profile-and-optimize the ring-attention kernels, run a round of optimization, or continue an optimization loop driven by profiling.
---

# profile-optimize

End-to-end loop: profile → analyze → edit → re-profile → validate. Designed for the ring-attention CUDA project at `/home/cme213/hippowal/ring-attention`.

**Argument:** `$1` = number of optimization rounds to run (default `1`). After each round, ask the user whether to continue before starting the next round, unless `$1` > 1 was explicitly passed.

---

## 0. Preconditions (verify, don't assume)

Before doing anything else:

1. Confirm working dir is `/home/cme213/hippowal/ring-attention`. If not, `cd` there.
2. Confirm `KERNEL_OPTIMIZATIONS.md` exists at the repo root — this is where bullets and changes get written. If absent, ask the user before creating.
3. Confirm `scripts/slurm/profile.sbatch` exists and writes to `results/ncu/<timestamp>/` and `results/nsys/<timestamp>/` (grep for `TIMESTAMP=` in the script — if the timestamped layout is missing, stop and tell the user the script needs the archive patch first).
4. Confirm the working tree is clean *or* clearly belongs to the optimization branch (`git status`, `git branch --show-current`). If there are uncommitted edits unrelated to optimization, ask before proceeding — a profile run on a dirty tree is not reproducible.

---

## 1. Run the profile (compute node, 4 GPUs × 4 tasks)

The SBATCH script has all directives baked in (`--gres=gpu:4 --ntasks=4 --partition=gpu-turing`). Submit it via `sbatch` and wait for it to finish — do **not** try to run it on the login node (no GPU there) and do **not** use `srun` for the inner `mpirun` (NVHPC's OpenMPI is PMIx-incompatible with SLURM's `srun`).

```bash
# Submit
JOBID=$(sbatch --parsable scripts/slurm/profile.sbatch)
echo "Submitted job ${JOBID}"

# Wait for completion — poll squeue. Profile runs ~5–15 min depending on shape.
until ! squeue -h -j "${JOBID}" 2>/dev/null | grep -q .; do sleep 30; done

# Check exit status + tail the log
sacct -j "${JOBID}" --format=JobID,State,ExitCode --noheader
tail -60 results/profile-${JOBID}.out
```

Use `Bash` with `run_in_background: false` and `timeout: 600000` for the `until …` poll, OR submit and use `Monitor`-style polling with longer sleeps. The profile usually takes 5–15 min; do not waste sleep cycles polling more often than every 30 s.

After completion, read the timestamp the script just wrote:

```bash
TS=$(cat results/.last_profile_timestamp)
echo "Latest profile: results/{ncu,nsys}/${TS}/"
ls -lh results/ncu/${TS}/ results/nsys/${TS}/
```

If the SBATCH job failed (non-zero exit or no reports produced), surface the error, stop the skill, and do **not** edit anything.

---

## 2. Analyze the reports (CLI only)

Use `ncu --import` and `nsys stats` to extract metrics. Never try to open the GUI tools — they're not available on the cluster CLI.

### Nsight Compute — kernel bottlenecks

Run on each `.ncu-rep` in the latest timestamp dir:

```bash
NCU_DIR=results/ncu/${TS}
for rep in ${NCU_DIR}/*.ncu-rep; do
  tag=$(basename "${rep}" .ncu-rep)
  echo "===== ${tag} ====="
  # details page (Speed-of-Light, Compute/Mem Workload Analysis, Occupancy, Warp State, Source Counters)
  ncu --import "${rep}" --page details --print-units base 2>/dev/null \
    > results/ncu/${TS}/${tag}__details.txt
  # quick scan: SoL + warp stall reasons + occupancy
  grep -E "Speed Of Light|Achieved Occupancy|Theoretical Occupancy|Stall|Bank Conflict|Memory Throughput|Compute (SM)|SM Frequency|Block Size|Registers Per Thread|Shared Memory" \
    results/ncu/${TS}/${tag}__details.txt | head -40
done
```

What to extract per kernel (write these down — they're going into KERNEL_OPTIMIZATIONS.md):
- **SoL %**: Compute SoL and Memory SoL. The bigger of the two is the headline limiter.
- **Achieved vs theoretical occupancy**: gap → launch config / resource pressure issue.
- **Top 2 warp stall reasons** + cycles. Common ones: Short Scoreboard (smem load→FMA), Long Scoreboard (global load), MIO Throttle, Wait, Barrier.
- **Bank conflicts**: shared load/store wavefront ratios > 1.0 → conflicts.
- **Spills**: `Stack Frame` / Local memory bytes > 0 → register spilling.
- **L1/L2/DRAM throughput** and hit rates.

### Nsight Systems — overlap / MPI

```bash
NSYS_DIR=results/nsys/${TS}
for rep in ${NSYS_DIR}/*.nsys-rep; do
  tag=$(basename "${rep}" .nsys-rep)
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,mpi_event_sum \
    --format csv --force-export=true --quiet \
    --output "${NSYS_DIR}/${tag}__" "${rep}" >/dev/null 2>&1 || true
done
# Compare modes (rank 0 only — that's enough to see structural patterns)
for tag in allgather ring-blocking ring-overlap ring-overlap-zigzag; do
  echo "----- ${tag} -----"
  head -5 ${NSYS_DIR}/${tag}_p4_0__cuda_gpu_kern_sum.csv 2>/dev/null
  head -5 ${NSYS_DIR}/${tag}_p4_0__mpi_event_sum.csv     2>/dev/null
done
```

What to extract:
- Total kernel time per mode (overlap vs blocking — is overlap actually winning?).
- MPI time breakdown: `MPI_Wait*`, `MPI_Isend/Irecv`, `MPI_Allgather`. Excess `MPI_Wait` time means comm isn't actually overlapped.
- Per-kernel time distribution: which kernel dominates?

### Output of step 2

A short mental model: **at most 3 bottlenecks** ranked by impact. Don't pad. If only one thing actually matters, list one.

---

## 3. Write bottlenecks into `KERNEL_OPTIMIZATIONS.md`

Append a new section to the existing file. Don't rewrite prior rounds.

Append a header for the new round with the timestamp:

```
## Round N — <one-line theme>

Profile: `results/ncu/<TS>/`, `results/nsys/<TS>/`. Baseline for comparison: `<previous TS>`.

### Bottlenecks identified

- **<kernel/area>**: <metric snapshot, e.g. "Short Scoreboard 6.8 cycles, 69% of inter-issue">. Hypothesis: <one line>.
- **<...>**: <...>.
```

Rules:
- Bullets are 1–2 lines each. No paragraphs. Cite the metric, not a generic claim.
- Match the existing style in `KERNEL_OPTIMIZATIONS.md` (see Round 1–3 for tone).
- Do **not** propose fixes here — bottlenecks only. Fixes come in step 5.

Round numbering: read the current file, find the highest existing `## Round N`, use `N+1`.

---

## 4. Pause for direction (if N=1 and not in a loop)

If `$1 == 1` (or unspecified), and bullets have been written, ask the user:
"Bottlenecks identified above. Want me to apply fixes for all of them, a subset, or stop here for you to review?"

If `$1 > 1`, skip this — proceed to step 5 with all bottlenecks.

---

## 5. Apply the changes

Edit the relevant `.cu` / `.cuh` / `.cpp` files. Follow project conventions (clang-format, Doxygen, 2-space indent, 100-col limit, host/device split).

**Critical:**
- Make surgical edits — touch only what the bottleneck demands. Don't restructure adjacent code.
- After every change, rebuild and run the C++ unit tests to confirm correctness:
  ```bash
  cmake --build build/release -j && ctest --test-dir build/release --output-on-failure
  ```
  If tests fail, the change is wrong — fix or revert before going further. Numerical tolerance is `atol=1e-3, rtol=1e-3` against the Python reference.
- If a fix needs a non-obvious explanation (subtle invariant, hidden constraint), one-line comment max. Otherwise no comment.

---

## 6. Document changes in `KERNEL_OPTIMIZATIONS.md`

Append a `### Changes` block under the same Round N header. Same bullet style as Round 1–3:

```
### Changes

- **<Title of change>** (`<file>:<lines>`). Theory: <1–2 lines>. Practice: <observed effect — leave blank if not yet measured; will fill in step 7>.
```

---

## 7. Re-profile and validate

Repeat step 1 to produce a new timestamp `TS_new`. Then compare against `TS_old` (the previous round's archive) using the same CLI extractions:

```bash
TS_OLD=<previous timestamp>
TS_NEW=$(cat results/.last_profile_timestamp)

# For each kernel report, diff the key metrics
for rep_new in results/ncu/${TS_NEW}/*.ncu-rep; do
  tag=$(basename "${rep_new}" .ncu-rep)
  rep_old=results/ncu/${TS_OLD}/${tag}.ncu-rep
  echo "===== ${tag}: ${TS_OLD} -> ${TS_NEW} ====="
  diff \
    <(ncu --import "${rep_old}" --page details --print-units base 2>/dev/null | grep -E "Speed Of Light|Occupancy|Stall|Bank Conflict") \
    <(ncu --import "${rep_new}" --page details --print-units base 2>/dev/null | grep -E "Speed Of Light|Occupancy|Stall|Bank Conflict") \
    | head -40
done
```

Validate each bullet from step 3:
- For each bottleneck claim, confirm the metric moved in the expected direction.
- If a metric did **not** improve, say so explicitly. Don't paper over it.

Fill in the **Practice** field of each change bullet in step 6 with the measured before→after numbers.

---

## 8. Loop or stop

- If `$1 > 1`, decrement and return to step 1 (the prior round's "TS_new" is the next round's "TS_old").
- Otherwise, stop. Summarize for the user:
  - Round N done. Bottlenecks addressed: N1, N2.
  - Metrics that improved: …
  - Metrics that did not improve (and why, if known): …
  - Where the reports live: `results/{ncu,nsys}/${TS_new}/`.

Do **not** commit changes or push — the user does that.

---

## Failure modes to surface immediately (don't paper over)

- SBATCH job hangs > 30 min → cancel (`scancel ${JOBID}`), check logs, ask user.
- ncu/nsys report missing or empty (0-byte) → the run didn't actually profile; check `results/profile-${JOBID}.err`.
- Tests fail after an edit → revert or fix before proceeding to re-profile.
- A metric *regresses* after a change → flag in the writeup, do not silently move on.
- "Bottleneck" is actually noise (single-digit % change run-to-run) → say so; don't fabricate a fix.
