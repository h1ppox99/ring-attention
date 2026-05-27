# Kernel optimization notes

Branch: `improve-kernel`. Driven by Nsight Compute feedback on
`flash_attention_kernel` (initially) and `attention_step_kernel` (the ring
driver's hot path, profiled by `scripts/slurm/profile.sbatch`). Each change
below lists *why it should help in theory* and *why it actually helped in
practice (observed metrics / ptxas output / test results)*.

Hardware: Turing sm_75, 32 banks (4 B each), 64 KB combined L1/shared per SM,
65,536 registers per SM, 255 registers per thread, 32 warps per SM.

Files touched: `src/flash_attention.cu`, `src/attention_step.cu`.

---

## Round 1 — bank conflicts and MIO instruction pressure

Originally applied to `flash_attention.cu` because the Nsight report file was
named `flash_*.ncu-rep`. Found out later the report's `--kernel-name` filter
actually targets `attention_step_kernel`, so the same changes were ported to
`attention_step.cu`.

### 1. Pad `Q_tile` rows from stride `D` to `D + 1`

- **Theory.** With stride `D` (a multiple of 32 for all supported head dims),
  the bank index for `Q_tile[tid * D + d]` collapses to `d % 32` across the
  warp — all 32 lanes hit the *same* bank, giving a 32-way conflict. Padding
  to `D + 1` shifts the bank index to `(tid + d) % 32`, so the 32 lanes hit 32
  distinct banks.
- **Practice.** Nsight had flagged "8.7-way average bank conflict, 88% of
  shared load wavefronts excessive." This is the load pattern that was
  dominating that number. Conflict-free after the pad.

### 2. Transpose the inner matmul loop nesting (`d` outer, `j` inner)

- **Theory.** The original `for j { for d { dot += Q[d] * K[j][d] } }` loads
  `Q_tile[tid*stride+d]` once per `(j, d)` pair — `D · BC` loads per
  K/V tile. Hoisting Q to the outer loop (`for d { q = Q[d]; for j { s[j] +=
  q * K[j][d] } }`) loads each Q element once per `d` and reuses it across
  all `BC` `j`s — `D` loads per tile, a factor of `BC` reduction.
  Per-`s[j]` accumulation order over `d` is unchanged, so numerics are
  identical.
- **Practice.** Drops shared-load instruction count on the Q side by `BC` ×
  (8–64 depending on shape), feeding directly into the MIO-queue stall
  reduction. Tests still pass with max diff ≤ 1.79e-7.

### 3. Switch the cooperative K/V tile load to `float4` (`LDS.128` / `STS.128`)

- **Theory.** A `float4` store packs four floats into one issued instruction.
  Same bytes moved, **4× fewer instructions in the MIO queue** —
  directly attacks the "11.3 cycles waiting for MIO queue not full" feedback.
- **Practice.** All `(BR, BC, D)` shapes satisfy `(BC * D) % (BR * 4) == 0`,
  so the loop count divides cleanly across threads. K/V/`K_tile`/`V_tile`
  addresses are 16 B-aligned by construction (D is a multiple of 4 and
  `head_k + j_global * D` is a multiple of D).

---

## Round 2 — kill `Q_tile` to break the 12.5 % occupancy ceiling

After Round 1, occupancy on the profiled config (D=128) was still
**12.5 % (1 warp / scheduler)** — Nsight said *"limited by required amount of
shared memory."* All in `attention_step.cu`.

### 4. Move Q from shared memory to per-thread registers (`Q_reg[D]`)

- **Theory.** Q is per-query-row — a thread only ever reads its own row, no
  cross-thread sharing — so shared memory was the wrong tier. The
  `Q_tile[BR][D+1]` was costing ~16.5 KB of smem at D=128 BR=32 purely as a
  staging area. `#pragma unroll` on the d loop keeps `Q_reg[d]`
  compile-time-indexed so nvcc does not lower the array to local memory.
- **Practice.** ptxas output:

  | (BR, BC, D)    | regs | spill stores | smem    |
  |---             |---   |---           |---      |
  | (16, 8, 256)   | 255  | 3.3 KB       | 16 KB   |
  | (32, 16, 128)  | 255  | 372 B        | 16 KB   |
  | (32, 32, 64)   | 231  | 0            | 16 KB   |
  | (64, 64, 32)   | 168  | 0            | 16 KB   |

  Smem halved across all configs (was 24–32 KB). D=32/64 have *zero spill* —
  Q lives entirely in registers. D=128 spills 372 B/thread (well below L1
  budget). D=256 spills heavily because `Q_reg[256]` exceeds Turing's
  255-reg cap — left as the corner case.

---

## Round 3 — more warps per block + float4 broadcasts

After Round 2 Nsight reported **25 % occupancy still limited by smem** at
D=128 (Wait — limited by smem at 16 KB?). Math: 64 KB SM smem / 16 KB block
= 4 blocks, × 1 warp/block (BR=32) = 4 warps = 12.5 %. Wrong arithmetic on my
part the first time around; the actual ceiling was *blocks-per-SM*, not
smem-bytes. Fix below addresses *warps per block*.

### 5. Double `BR` for D ≤ 128 (more warps per block, same smem)

- **Theory.** `K_tile` and `V_tile` sizes depend on `BC` and `D` but **not**
  on `BR`. So bumping `BR` packs more warps into the same per-block smem
  footprint — extra warps "for free" from a smem perspective.
- **Practice.** New tile shapes (`attention_step.cu` only):

  | D    | BR (old → new) | warps/block | theoretical occupancy |
  |---   |---             |---          |---                    |
  | 32   | 64 → 128       | 4           | 25 % → 37.5 %         |
  | 64   | 32 → 64        | 2           | 12.5 % → 25 %         |
  | 128  | 32 → 64        | 2           | 12.5 % → 25 %         |
  | 256  | 16 (unchanged) | 1           | 12.5 % (spill-bound)  |

  D=256 left at BR=16 because `Q_reg[256]` already spills 3.3 KB/thread;
  doubling BR would multiply local-memory traffic across more threads and
  push it past the L1 budget.

### 6. `float4` broadcast loads on `K_tile` / `V_tile` in the inner matmul and output accumulation

- **Theory.** `K_tile[j*D + d]` is a warp-wide broadcast (all 32 lanes
  read the same address) — already 1 wavefront per access, no bank conflict
  available to fix. But the *issued instruction count* is `D · BC` per K/V
  tile. Switching to `LDS.128` cuts that by 4× while keeping the same single
  broadcast wavefront. This is the Nsight-recommended *"use fewer, wider
  loads"* — the Short Scoreboard stall (cycles a warp spends waiting on a
  pending shared load before it can issue the dependent FMA) is *per
  instruction*, so 4× fewer load instructions directly relieves it.
- **Practice.** Short Scoreboard stall dropped from 8.2 cycles → **6.8
  cycles** (71.7 % → 69.3 % of inter-issue cycles). Inter-issue cycle count
  itself dropped from 11.4 → 9.8.

---

## Where things landed (and where they didn't)

After Round 3, Nsight reports:
- **Theoretical occupancy 25 %** at D=128, limited by **both registers
  *and* shared memory simultaneously** — the configuration is balanced
  against both ceilings, not one.
- **Short Scoreboard 6.8 cycles** at 69.3 % of inter-issue time — still the
  dominant stall reason.

This is the Turing architectural ceiling for the current per-thread design.
Each thread holds its full query row's state: `Q_reg[D] + O_i[D] = 2D`
floats. At D=128 that's **256 registers minimum per thread**, just under
Turing's 255-reg cap; with 64 threads/block, every block reserves
~16,320 / 65,536 SM-regs, hard-capping resident blocks at 4. The arithmetic
is symmetric:

```
regs:  65536 / (64 * 255) ≈ 4.01 blocks/SM
smem:  65536 / 16384       = 4.00 blocks/SM
```

4 blocks × 2 warps = 8 warps/SM = 25 %. Pushing past this needs a
**structural change** — either (a) FlashAttention-v2-style cooperation where
one warp owns one query row and the 32 lanes split D (so each thread holds
`D/32` floats of Q and O — ~8 floats at D=128), with a `__shfl_xor` reduction
for the dot product; or (b) newer hardware (Ampere/Hopper have larger
register files and larger smem per SM).

Lowering BC or applying `__launch_bounds__` to "force" more blocks were
considered and rejected: BC reduction is blocked by the register limit, and
`__launch_bounds__`-driven spills would push the per-block local-memory
working set past L1's capacity at the higher block counts (≥ 5 blocks/SM ⇒
local-mem thrashing).
