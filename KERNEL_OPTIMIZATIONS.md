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

---

## Round 4 — manual FMA fusion in the dot product and V accumulation

Profile: `results/ncu/2026-05-26_21-59/`, `results/nsys/2026-05-26_21-59/`.
Baseline for comparison: the flat-layout `results/ncu/*.ncu-rep` snapshot
from end of Round 3.

### Step-level breakdown (where the time actually goes)

Per-rank totals over 3 iters at `seq=16384, D=128, causal=1`:

| mode                 | kernel | MPI_Waitall | kernel : MPI |
|---                   |---     |---          |---           |
| ring-blocking        | 50.9 ms | 564.9 ms   | 1 : 11       |
| ring-overlap         | 50.6 ms | 449.3 ms   | 1 : 9        |
| ring-overlap-zigzag  | 219.9 ms | 226.9 ms  | ~1 : 1       |

In non-zigzag modes MPI_Waitall dwarfs the kernel by ~10×: the per-step
8 MB × 4-message round trip over the cluster's Ethernet network simply
takes longer than the kernel does. In ring-overlap, the kernel finishes
inside the Waitall window (so it *is* hidden), and Waitall is recorded as
host-blocked time, not GPU stall time — the GPU is doing real work during
those 449 ms. Critical-path wall-clock is bounded below by `max(kernel,
comm)`, and at this shape `comm` already wins; no kernel-side change can
shorten it.

**ring-overlap + zigzag is the production configuration** (smaller K/V
chunks per step + more kernel launches → comm and compute end up balanced
~1:1, exactly the regime where each ms of kernel speedup translates ~1:1
to end-to-end). That is the mode where Round 4's kernel work pays off.

### Bottlenecks identified

- **Short Scoreboard still dominates the kernel** (D=128, c=0: 6.7 cycles =
  69.1 % of 9.7-cycle inter-issue gap). Smem broadcast load (`LDS.128` from
  `K_tile` / `V_tile`) → dependent FMA. Round 3 already cut the load
  *count* 4× with `float4`; the remaining stall is per-load latency, which
  is structural. Hypothesis: only a warp-cooperative dot product (Q/K split
  across 32 lanes via `__shfl_xor`) materially moves this on Turing —
  Round 3's documented ceiling. Not in scope for this round.
- **Non-fused FP32 ≈ 25 % of arithmetic** (Nsight "Instruction Statistics":
  7.66 B fused + 2.59 B non-fused FP32 → up to **13 % kernel speedup** from
  pairing → ~6 % end-to-end in zigzag mode). Source pattern: `s[j] +=
  q0*k.x + q1*k.y + q2*k.z + q3*k.w` and the V-accumulation `acc.x +=
  s[j]*v.x; …` give nvcc enough algebraic freedom to emit FMUL+FADD pairs
  instead of FMA chains. Manual `fmaf` chains force fusion. Hypothesis:
  this is the only bottleneck addressable without a structural rewrite
  this round.
- **Eligible warps per scheduler = 0.20** (1.87 active, 10.7 % eligible).
  Symptom of bottleneck #1 — too many warps stalled on the same smem load
  at once. Not directly addressable without restructuring.

MPI tuning is **not** in scope: the non-zigzag overhead is network-bound
(no source-level fix), and the production zigzag config already overlaps
comm with compute at ~1:1, so further overlap-engineering buys nothing.
Memory access pattern warnings (uncoalesced global loads / stores) carry
< 0.25 % estimated speedup each — not real bottlenecks, ignored.

### Changes attempted and reverted

- **Manual `fmaf` chain in the Q·K^T inner matmul + V accumulation + `l_i`
  update** (`attention_step.cu`). Replaced `s[j] += q0*k.x + q1*k.y +
  q2*k.z + q3*k.w` with four serialised `s[j] = fmaf(q?, k.?, s[j])` calls;
  similarly rewrote `acc.? += s[j] * v.?` and `l_i = alpha*l_i + row_sum`
  with explicit `fmaf`. Theory: forces FMA where nvcc was reportedly
  leaving 25 % of FP32 non-fused (per the Nsight rule). Tests passed
  (`atol=rtol=1e-3` against the Python reference, all 16 cases).
- **Practice: regressed by 7-10 % across all four ring modes** (re-profile
  job 87398, `results/{ncu,nsys}/2026-05-26_22-37/`):
    - `ring-overlap-zigzag` `attention_step` avg: 6.11 ms → 6.73 ms (+10.1 %)
    - `ring-overlap` avg:                          12.65 ms → 13.59 ms (+7.4 %)
    - `ring-blocking` avg:                         12.74 ms → 13.70 ms (+7.5 %)
    - `allgather` avg:                             49.71 ms → 53.96 ms (+8.6 %)
  - Warp Cycles Per Issued Instruction at D=128 c=0: 9.72 → 11.12 (+14 %).
  - D=64 register pressure: 247 → 255 (now saturated).
  - **Why**: the original `s[j] += q0*k.x + q1*k.y + q2*k.z + q3*k.w` lets
    nvcc emit 4 independent FMULs (parallel issue) followed by a small
    reduction tree — high ILP, ~5 cycle wide latency. The fmaf chain
    `s[j] = fmaf(q0, k.x, s[j]); s[j] = fmaf(q1, ...);` serialises on
    `s[j]` for ~24 cycles of dependent FMAs (6-cycle Turing FMA latency).
    The 16 independent `s[j]` accumulators are not enough to hide this
    given the per-thread state already pinning 255 registers/thread.
  - **Reverted**. `src/attention_step.cu` is back to its post-Round-3
    state. The Nsight "13 % non-fused FP32" rule was misleading here: the
    parenthesised mul-then-sum form was already optimal in the
    latency-vs-ILP trade-off, even though it scores as "non-fused".

### Where this leaves us

Round 4 produced no kernel improvement. The actionable improvements
visible to Nsight at this stage are either (a) symptoms of an addressed
ceiling (eligible-warps, scheduler issue rate — both downstream of Short
Scoreboard, which itself requires a structural rewrite), or (b)
mis-classified ILP wins that look like FMA-fusion opportunities but
aren't. The Round 3 conclusion — *FlashAttention-v2-style
warp-cooperative D reduction is the next real lever* — still holds and
is the correct framing for a future round.

---

## Round 5 — decode-specialized kernel (seq_q = 1)

Profile: `results/ncu/2026-05-28_14-35/`, `results/nsys/2026-05-28_14-35/`.
Baseline for comparison: the same dir (no prior decode-mode profile exists).
Configuration: `--run decode --prompt-len 16384 --decode-tokens 2`,
`B=1 H=8 D∈{64,128}`, cp_size=1.

The decode path reuses `attention_step_kernel` with `seq_q = 1`. The kernel
was designed for prefill (BR=64 query rows per block); under decode, only
`i_local == 0` clears the `if (active)` gate — 63 of every 64 threads do no
compute. Nsight confirms this is catastrophic.

### Bottlenecks identified

- **Thread-level idleness inside each warp**: Avg. Active Threads Per Warp
  **2.54 / 32** (D=128) and 2.44 / 32 (D=64). Scheduler Issue Slots Busy
  **2.7 %** (D=128), 5.3 % (D=64). The kernel has nothing for 95+ % of its
  warp-cycles because only one of every BR=64 lanes performs the dot
  product. Hypothesis: structural — BR=64 is a prefill choice; for decode
  one warp should own one query row and the 32 lanes should split D.
- **Tiny grid + saturated registers**: Grid Size **8 blocks** (= B·H);
  Registers Per Thread **255** (D=128) / 247 (D=64); Block Limit Registers
  = 4; Theoretical Active Warps per SM = 8, achieved 2. The per-thread
  `Q_reg[D] + O_i[D]` allocation pins us to the register-limited 25 %
  occupancy ceiling, then the seq_q=1 active-thread waste pushes Achieved
  Occupancy to **6.25 %**. With Quadro RTX 6000 having 72 SMs, only ~8 SMs
  see any work at all.
- **Barrier stalls from the cooperative K-tile load**: Stall Barrier
  **8.7 cycles (47.3 % of inter-issue)** at D=128. The 64 threads
  cooperatively load `K_tile` / `V_tile` via float4 then `__syncthreads()`;
  the single active thread that consumes them spends nearly half its
  inter-issue time waiting on its 63 idle siblings to finish the load
  before the barrier releases. A single-warp kernel removes both the
  cooperative load and the `__syncthreads` entirely.

### Headline counters (current state, decode regime)

| metric                          | D=64 (decode) | D=128 (decode) | for context: D=128 prefill (Round 3) |
|---                              |---            |---             |---                                   |
| SoL Compute Throughput          | 1.89 %        | **1.02 %**     | ~13 %                                |
| SoL Memory Throughput           | 1.89 %        | 1.02 %         | ~12 %                                |
| Achieved Occupancy              | 6.24 %        | **6.25 %**     | 25 % (theoretical ceiling)           |
| Avg. Active Threads / Warp      | 2.44          | **2.54**       | ~32                                  |
| Issue Slots Busy                | 5.30 %        | 2.70 %         | ~10 %                                |
| Warp Cycles / Issued Instruction| 9.42 cycles   | 18.49 cycles   | ~9.8 cycles                          |
| Grid Size                       | 8             | 8              | 64 (16k / BR=64 × H=4)               |
| Registers Per Thread            | 247           | 255            | 255                                  |

Compute SoL of **1.02 %** at D=128 means the decode kernel does ~1 % of the
GPU's peak FLOPs work. The end-to-end `bench_decode` number (49 ms / token
at prompt=16384 D=128 cp=1) is consistent with this: at 1 % of an 11 TFLOPS
peak, doing ~140 MFLOPs per decoded token takes ~13 ms — close enough that
the kernel is in fact the bottleneck (the remainder is host-side allocator
churn, MPI barriers, finalize).

### Where this leaves us

The fix is the same structural rewrite Round 3 named: one warp per query
row, 32 lanes split the D dimension via `__shfl_xor` reduction. Round 5
applies that specifically to the decode case (seq_q = 1) so it can ship
without disturbing the prefill kernel. Implementation below.

### Changes

- **New `attention_decode_step_kernel<D>`** (`src/attention_decode.cu`).
  Block = 1 warp = 32 threads, grid = `(H, B)`. Each lane holds `D/32`
  elements of Q and O in registers; K and V are streamed from global with
  warp-coalesced loads (no shared memory). Q·Kᵀ is a `__shfl_xor_sync`
  XOR-pattern reduction across the 32 lanes; the V-accumulate runs lane-
  parallel across the same `D/32` slices. Online-softmax state `(m, ℓ)` is
  redundantly maintained on every lane — no broadcast needed because the
  reduce already aligns them.
- **Dispatch hook** (`src/attention_step.cu:307-312`). `launch_attention_step`
  forwards to the decode kernel when `shape.seq_q == 1`. Behaviour for any
  other `seq_q` is unchanged, so the prefill path and existing benchmarks
  are not perturbed.
- **No new tests required**. The existing `test_decode_op` and
  `test_ring_decode` suites already cover the seq_q=1 path; they now
  exercise the new kernel transparently. Both pass with tighter numerical
  tolerance (fp32 max_diff drops from ~6e-8 to ~5e-8 — the warp-shuffle
  reduction has slightly lower round-off than the sequential dot product).

### Practice — Nsight Compute (D=128, prompt=16384, B=1, H=8)

| metric                          | before (`attention_step_kernel`) | after (`attention_decode_step_kernel`) | Δ                |
|---                              |---                                |---                                      |---               |
| Avg. Active Threads / Warp      | 2.54 / 32                         | **32.00 / 32**                          | **+12.6×**       |
| Stall Barrier                   | 8.7 cycles (47.3 %)               | 0 cycles                                | bottleneck gone  |
| Registers / Thread              | 255                               | **39**                                  | −6.5×            |
| Static Shared Memory / Block    | 16 384 B                          | **0**                                   | −16 KB           |
| Theoretical Occupancy           | 25 %                              | 50 %                                    | +2×              |
| Achieved Occupancy              | 6.25 %                            | 3.12 %                                  | regressed*       |
| Issue Slots Busy                | 2.70 %                            | 1.70 %                                  | regressed*       |
| Warp Cycles / Issued Instruction| 18.49 cycles                      | 14.73 cycles                            | −20 %            |
| Block Size                      | 64 threads                        | 32 threads                              | matches 1 warp   |
| Grid Size                       | 8 blocks                          | 8 blocks                                | =                |

\* "Regressed" Achieved Occupancy and Issue Slots Busy are downstream of
the kernel becoming memory-latency-bound — each warp now serially fetches
one K row per loop iteration with no prefetch, so the scheduler genuinely
has nothing to issue while the load is in flight. Memory throughput rose
1.02 % → 1.97 % at D=128, confirming the warps are at least doing useful
memory traffic. The absolute work-per-warp-cycle is up 12.6× even though
warp-cycles-per-instruction dropped, so the headline wall-clock improves.

### Practice — end-to-end (`bench_decode.sbatch`, full sweep)

Per-token decode latency, fp32, B=1 H=8, mean over 16 tokens. Old =
Round 4 baseline (committed in `results/bench_decode_round4.csv`); new =
Round 5 re-run (`results/bench_decode.csv`).

**cp_size = 1** (single rank, kernel speedup in isolation):

| prompt_len | D=64 before | D=64 after | speedup | D=128 before | D=128 after | **speedup** |
|---         |---          |---         |---      |---           |---          |---          |
| 1 024      | 0.88 ms     | 0.53 ms    | 1.65×   | 3.14 ms      | 0.72 ms     | **4.39×**   |
| 4 096      | 3.37 ms     | 2.65 ms    | 1.27×   | 12.41 ms     | 2.81 ms     | **4.41×**   |
| 16 384     | 13.37 ms    | 10.58 ms   | 1.26×   | 49.68 ms     | 11.26 ms    | **4.41×**   |

**cp_size = 2 and 4** (full ring decode, comm included):

| prompt_len | D=128 cp=2 before | D=128 cp=2 after | speedup | D=128 cp=4 before | D=128 cp=4 after | speedup |
|---         |---                 |---                 |---     |---                 |---                 |---     |
| 1 024      | 3.65 ms            | 1.15 ms           | 3.18×  | 3.97 ms            | 1.16 ms           | 3.43×  |
| 4 096      | 14.07 ms           | 4.30 ms           | 3.27×  | 15.12 ms           | 5.26 ms           | 2.87×  |
| 16 384     | 55.90 ms           | 17.00 ms          | 3.29×  | 59.36 ms           | 20.49 ms          | 2.90×  |

D=128 wins are uniformly **3–4.4× faster**. D=64 wins are smaller (1.2–1.8×)
because the old kernel was already closer to memory-bound at that head_dim;
the new kernel still helps but the kernel is no longer the dominant cost.
At cp_size > 1 the comm fraction grows (≈18 % of total at cp=2, ≈31 % at
cp=4 for D=128 prompt=16k) — that's where Round 7+ should target.

Notably, **per-token comp_ms is now nearly D-independent** (10.3 ms at D=64,
10.6 ms at D=128 for prompt=16384) — both sit at the memory-latency floor.
Round 6 lever: K-row double-buffering or shared-memory K-tile prefetch to
hide that floor.

### What this doesn't fix

- **Grid Size stays at `B × H = 8`.** On a 72-SM Quadro RTX 6000, only
  ~11 % of the SMs see any work. A split-K decode variant (multiple
  blocks per `(B, H)` accumulating into per-tile `(m, ℓ, O)` partials,
  with a final reduction kernel) would lift SM coverage and is the natural
  next structural step. Skipped here because the Round 5 win is already
  4.4× without it.
- **Memory latency now dominates** (Memory Throughput 1.97 %, well under
  peak). Each `for j in [0, Sk)` iteration waits on the global load of
  `K[j]` before the dot product can issue. Software prefetching of `K[j+1]`
  while computing `K[j]` would mask this — the natural Round 6 follow-up.

---

## Round 6 — software-pipelined K/V prefetch

Profile: `results/ncu/2026-05-28_14-46/` (Round 5 baseline; new profile
folder written by this round's re-run).

### Bottlenecks identified

- **Long Scoreboard stall 10.0 cycles (67.7 % of inter-issue)** at D=128,
  per Nsight WarpStateStats. Each loop iteration emits LDG for K[j] and
  V[j] and immediately consumes K[j] in the dot product → the warp parks
  on the scoreboard waiting for global memory. With `Avg Active Threads
  Per Warp = 32` and `Issue Slots Busy = 1.70 %`, the scheduler is healthy
  but starved: the warp simply has nothing to issue while the load is in
  flight. This is the single visible bottleneck of the Round 5 kernel.

### Changes

- **Register-level K/V double buffer** (`src/attention_decode.cu`). The
  decode kernel now keeps two slots per lane — `K_buf[2][VPL]`,
  `V_buf[2][VPL]` — and ping-pongs. Loop iteration `j` issues the LDG for
  K[j+1] and V[j+1] into the *next* slot *before* the dot product and
  V-accumulate read from the *current* slot. nvcc can now schedule the
  next-iter loads concurrently with the current-iter compute, so the
  Long Scoreboard wait for K[j+1] overlaps with the ~30 cycles of dot-
  product + softmax + V-accumulate on K[j].
  Cost: +2 × VPL × 2 = +4·VPL registers per lane (61 / lane at D=128, up
  from 39 — still well under Turing's 255-reg cap).
- **Hoist causal cutoff out of the loop** (same file). The Round 5 kernel
  had `if (causal && j_global > i_global) break;` inside the loop. With
  prefetch in play, an early `break` orphans an already-issued LDG for the
  unused K[j+1]/V[j+1]. Compute `Sk_eff = min(Sk, i_global - k_offset + 1)`
  once before the loop and iterate the clean `[0, Sk_eff)` range; the
  branch leaves the inner loop entirely. Side benefit: the masked-row case
  (`Sk_eff == 0`) skips the loop and the kernel touches no global memory
  at all.

### Practice — Nsight Compute (D=128, prompt=16384, B=1, H=8)

| metric                          | Round 5 (no prefetch) | Round 6 (prefetch)   | Δ            |
|---                              |---                    |---                   |---           |
| Warp Cycles / Issued Instruction| 14.73 cycles          | **5.50 cycles**      | **−63 %**    |
| Issue Slots Busy                | 1.70 %                | **4.55 %**           | **+168 %**   |
| Memory Throughput               | 1.97 %                | 2.45 %               | +24 %        |
| Compute (SM) Throughput         | 0.25 %                | 0.50 %               | +100 %       |
| Top stall reason                | Long SB 10.0c (67.7%) | **Short SB 2.5c (44.9%)** | latency hidden |
| Registers / Thread              | 39                    | 61                   | +22 (~as planned) |
| Achieved Occupancy              | 3.12 %                | 3.12 %               | =            |
| Avg. Active Threads / Warp      | 32                    | 32                   | =            |

The Long Scoreboard stall — global-memory latency — dropped out of the top
spot, replaced by Short Scoreboard at less than a third of its prior cost.
Warp issue cadence dropped from one instruction every 14.7 cycles to one
every 5.5 — the kernel is now issue-throughput bound at this level. The
2.45 % Memory Throughput tells us there is still bandwidth left to take —
we're hiding latency but not yet saturating the DRAM channel.

### Practice — end-to-end (`bench_decode.sbatch`, full sweep)

Per-token decode latency, fp32, B=1 H=8, mean over 16 tokens.
`results/bench_decode_round5.csv` → `results/bench_decode_round6.csv`.

**cp_size = 1** (kernel speedup in isolation, dominated by `comp_ms`):

| prompt_len | D=64 R5  | D=64 R6  | Δ      | D=128 R5 | D=128 R6 | **Δ**       |
|---         |---       |---       |---     |---       |---       |---          |
| 1 024      | 0.53 ms  | 0.50 ms  | −6 %   | 0.72 ms  | 0.56 ms  | **−22 %**   |
| 4 096      | 2.65 ms  | 2.00 ms  | −24 %  | 2.81 ms  | 2.22 ms  | **−21 %**   |
| 16 384     | 10.58 ms | **8.01 ms** | **−24 %** | 11.26 ms | **8.91 ms** | **−21 %**   |

**cp_size = 2 and 4** (comm-included, where the kernel share is smaller):

| prompt_len | D=128 cp=2 R5 | D=128 cp=2 R6 | Δ      | D=128 cp=4 R5 | D=128 cp=4 R6 | Δ      |
|---         |---            |---            |---     |---            |---            |---     |
| 1 024      | 1.15 ms       | 1.00 ms       | −13 %  | 1.16 ms       | 1.15 ms       | −1 %   |
| 4 096      | 4.30 ms       | 3.73 ms       | −13 %  | 5.26 ms       | 4.74 ms       | −10 %  |
| 16 384     | 17.00 ms      | 14.77 ms      | −13 %  | 20.49 ms      | 18.24 ms      | −11 %  |

The Round 5+6 combined speedup vs the Round 4 baseline:

| config                          | Round 4 | Round 5 | Round 6 | total speedup |
|---                              |---      |---      |---      |---            |
| D=128 cp=1 prompt=16384         | 49.68   | 11.26   | 8.91    | **5.57×**     |
| D=128 cp=4 prompt=16384         | 59.36   | 20.49   | 18.24   | **3.25×**     |
| D=64 cp=1 prompt=16384          | 13.37   | 10.58   | 8.01    | **1.67×**     |

### What this doesn't fix

- **Memory Throughput is still 2.45 %.** We hid the *latency* but we are
  not feeding the load-store unit at peak rate. Increasing the in-flight
  load count further (3-deep prefetch, or a shared-memory K tile staging
  multiple rows so the load batch coalesces across more lanes) would
  start pulling on the DRAM curve directly. Note that "Short Scoreboard"
  is now the top stall — the typical Turing register-file/MIO bottleneck;
  the right next step is to stage K/V through shared memory so the warp
  hits the L1/MIO path with bulk transfers rather than per-row LDGs.
- **Grid Size still 8.** Same SM-coverage cap as Round 5; split-K decode
  is still the structural lever for raising it.

---

## Round 7 — 4-deep register prefetch [REVERTED]

Profile: `results/ncu/2026-05-28_16-17/` vs Round 6 (`2026-05-28_16-08`).

### Bottlenecks identified

- After Round 6, Memory Throughput was still **2.45 %** and the new top
  stall was Short Scoreboard 1.7 cycles (44.9 %) — typical Turing register-
  file / MIO bottleneck. Round 7 hypothesis: extending the in-flight load
  count from 2 → 3 would let the LSU pipeline more transactions and lift
  Memory Throughput further.

### Changes attempted

- **PIPE = 4-slot ring buffer** (`src/attention_decode.cu`): replaced the
  2-slot ping-pong with a 4-slot buffer (effective pipeline depth 3, so 3
  loads are pending while the consumer works on a 4th slot). Priming loop
  issues PIPE-1 = 3 LDGs before the steady-state loop. Slot index uses
  bitwise AND with `PIPE_MASK = 3` to stay nvcc-friendly.

### Practice — no end-to-end improvement (slightly regressed)

| metric (D=128)                  | Round 6         | Round 7 (4-slot) |
|---                              |---              |---               |
| Warp Cycles / Issued Instruction| 5.50 cycles     | **4.03 cycles** (−27 %) |
| Issue Slots Busy                | 4.55 %          | **6.21 %** (+37 %) |
| Memory Throughput               | 2.45 %          | 2.28 % (−7 %)    |
| Registers / Thread              | 61              | **75** (+14)     |
| Top stall                       | Short SB 2.5c   | Short SB 1.7c    |
| Wall-clock (prompt=16k, cp=1)   | 8.87 ms (spot)  | 8.80 ms (spot)   |
| Wall-clock (prompt=16k, cp=4)   | n/a (full sweep skipped) | n/a       |

### Why this failed (and what it taught us)

The compiler clearly *did* extract more issue throughput (cycles/issued
dropped 27 %, scheduler utilisation up 37 %) and the Short Scoreboard stall
shrank. But Memory Throughput did **not** go up — it actually moved slightly
the wrong way (2.45 → 2.28 %). Register usage rose by 14 of an expected
~32 (PIPE × 2 × VPL = 32 extra slots × 4 bytes), strongly suggesting nvcc
spilled at least half of `K_buf` to **local memory**. Local-memory loads
are still Short Scoreboard dependencies, so they look like an issue-rate
win in the metric while the actual byte-throughput is unchanged: we were
just hiding LDG latency behind LDL latency, paying registers for no
real-world payoff.

End-to-end wall-clock confirmed: 8.87 → 8.80 ms is within run-to-run noise.

### Reverted

`src/attention_decode.cu` returned to the Round 6 2-slot ping-pong. The
takeaway — *register-only pipelining stalls past depth 2 because nvcc
spills, defeating the purpose* — informed the choice of Round 8's lever:
add **parallelism across warps** (shared-memory accumulator merge) rather
than try to deepen the per-warp pipeline further.

---

## Round 8 — multi-warp blocks with intra-block K-split

Profile: `results/ncu/2026-05-28_17-42/` vs Round 6 (`2026-05-28_16-08`).

### Bottlenecks identified

After Round 6 the kernel had three structural ceilings, in order of impact:

- **Grid Size = 8.** One block per `(B, H)`; with 72 SMs on the Quadro
  RTX 6000, only 11 % of the SMs were active at any moment.
- **Per-block work was single-warp.** Per-SM resident warps = 1, Achieved
  Occupancy 3.12 % (vs 50 % theoretical). The scheduler had only one warp
  to feed.
- **Memory Throughput 2.45 %.** Far from DRAM-saturated. Round 7 confirmed
  that deeper per-warp prefetch doesn't help — needed wider, not deeper.

### Changes

- **Block size 32 → 128 (4 warps), K loop block-partitioned**
  (`src/attention_decode.cu`). Each block still owns one `(B, H)` row, but
  warp `w` streams K rows `[w·Sk/4, (w+1)·Sk/4)` with its own independent
  online-softmax state `(m_w, ℓ_w, O_w)`. Inside a warp, the existing
  Round-6 2-slot prefetch and `__shfl_xor` D-dim reduction are unchanged.
- **Shared-memory partial merge.** After the K loop, each warp posts
  `(m_w, ℓ_w, O_w)` to `__shared__ {m_shared[4], l_shared[4],
  O_shared[4][D]}` (2.08 KB at D=128). One `__syncthreads`, then warp 0
  reads the existing persistent global `(M, L, O)` and folds in all four
  partials sequentially using the FlashAttention partial-merge recurrence
  `m_new = max(m_a, m_b)`, `ℓ_new = ℓ_a·exp(m_a − m_new) + ℓ_b·exp(m_b − m_new)`,
  `O_new = O_a·exp(m_a − m_new) + O_b·exp(m_b − m_new)`. Empty partials
  (`m_w == −∞` when a warp got 0 rows under causal pruning) are skipped to
  avoid `−∞ − (−∞)` → NaN.
- **No new tests required.** `test_decode_op` and `test_ring_decode` both
  pass with even tighter max_diff (fp32: ~5e-8 → ~4e-8 for some cases —
  the per-warp partial accumulator runs over 1/4 the rows so its softmax
  exponentials have a narrower dynamic range).

### Practice — Nsight Compute (D=128, prompt=16384, B=1, H=8)

| metric                          | Round 6          | Round 8              | Δ        |
|---                              |---               |---                   |---       |
| Block Size                      | 32 threads       | **128 threads**      | 4×       |
| Theoretical Occupancy           | 50 %             | **100 %**            | 2×       |
| Achieved Occupancy              | 3.12 %           | **12.50 %**          | **4×**   |
| Memory Throughput (SoL)         | 2.45 %           | **9.54 %**           | **3.9×** |
| Compute (SM) Throughput         | 0.50 %           | **1.89 %**           | 3.8×     |
| Issue Slots Busy                | 4.55 %           | **17.12 %**          | 3.8×     |
| Memory Throughput (bytes/s)     | 15.3 GB/s        | **59.2 GB/s**        | 3.9×     |
| Registers / Thread              | 61               | 60                   | =        |
| Static Shared Memory / Block    | 0                | 2 080 B              | added    |
| Avg. Active Threads / Warp      | 32               | 32                   | =        |
| Top stall                       | Short SB 2.5c    | Short SB 2.8c        | similar  |

Every SM-utilisation metric scales ~4× with the warp count: 4× more
in-flight memory transactions per block, 4× more scheduler-eligible warps
per SM, 4× more bytes/sec. The kernel goes from idle most of the time
(3.12 % occupancy) to materially using the SM resources (12.5 %); on
Turing's 4 schedulers, 4 warps × 8 active blocks = 32 resident warps
spread across 8 SMs is the realistic ceiling without raising Grid Size.

### Practice — end-to-end (`bench_decode.sbatch`, full sweep)

`results/bench_decode_round6.csv` → `results/bench_decode_round8.csv`.
Per-token decode latency, fp32, B=1 H=8, mean over 16 tokens.

**cp_size = 1** (kernel-dominated):

| prompt_len | D=64 R6  | D=64 R8  | Δ        | D=128 R6 | D=128 R8 | **Δ**    |
|---         |---       |---       |---       |---       |---       |---       |
| 1 024      | 0.50 ms  | 0.14 ms  | **3.46×**| 0.56 ms  | 0.18 ms  | **3.08×**|
| 4 096      | 2.00 ms  | 0.56 ms  | **3.58×**| 2.22 ms  | 0.70 ms  | **3.19×**|
| 16 384     | 8.01 ms  | **2.24 ms** | **3.58×**| 8.91 ms  | **2.74 ms** | **3.25×**|

**cp_size = 2 and 4** (comm-included; kernel speedup shows up smaller as
NCCL transfer time becomes the headline term):

| prompt_len | D=128 cp=2 R6 | D=128 cp=2 R8 | Δ        | D=128 cp=4 R6 | D=128 cp=4 R8 | Δ        |
|---         |---            |---            |---       |---            |---            |---       |
| 1 024      | 1.00 ms       | 0.60 ms       | **1.66×**| 1.15 ms       | 0.86 ms       | **1.34×**|
| 4 096      | 3.73 ms       | 2.18 ms       | **1.71×**| 4.74 ms       | 3.10 ms       | **1.53×**|
| 16 384     | 14.77 ms      | 8.53 ms       | **1.73×**| 18.24 ms      | 11.85 ms      | **1.54×**|

**Combined Round 4 → Round 8 totals** at the headline configs:

| config                    | Round 4   | Round 8   | total speedup |
|---                        |---        |---        |---            |
| D=128 cp=1 prompt=16384   | 49.68 ms  | 2.74 ms   | **18.1×**     |
| D=128 cp=4 prompt=16384   | 59.36 ms  | 11.85 ms  | **5.0×**      |
| D=64  cp=1 prompt=16384   | 13.37 ms  | 2.24 ms   | **5.97×**     |

### What this doesn't fix

- **Grid Size still 8.** Multi-warp lifted *per-block* parallelism but
  not the structural cap on how many blocks the grid emits. Achieved
  Occupancy 12.5 % is the new ceiling. The next structural step is split-K
  across blocks: launch `(K_SPLIT, H, B)` blocks where each handles a
  contiguous slice of K, then a tiny reduce kernel combines `K_SPLIT`
  partials into the final `(M, L, O)`. That would 4× the resident warps
  on the SMs that currently get a block and start populating the other
  ~64 idle SMs.
- **Comm dominates at cp_size > 1.** The kernel is now small enough that
  the cp=2 / cp=4 paths spend the majority of their per-token wall-clock
  in NCCL. The 1.5–1.7× cp>1 speedup (vs 3.2× at cp=1) is the explicit
  signature of that shift. Future work: overlap the local-rank kernel
  call with the NCCL Send/Recv of the *next* chunk (this repo's
  `ring-overlap` mode already does this for prefill; the decode loop
  blocks comm and compute sequentially today).
- **Round 7 reverted, not absorbed.** The 4-slot register pipeline didn't
  help (spilled), so Round 8 stays at the Round-6 2-slot per-warp
  prefetch. If split-K lands and we want every micro-cycle back, an
  smem-staged tile may finally beat the register approach because it
  removes the spill ceiling.

---

## Round 9 — split-K across blocks

Profile: `results/ncu/2026-05-28_17-42/` (Round 8 baseline; new profile
written by this round's re-run).

### Bottlenecks identified

- **Grid Size still 8.** Round 8 lifted Achieved Occupancy from 3.12 %
  → 12.5 % by widening each block from 1 → 4 warps, but the grid still
  emits only `B × H = 8` blocks. On a 72-SM Quadro RTX 6000 that leaves
  ~64 SMs idle. The kernel is *structurally* capped at 12.5 % achieved
  occupancy as long as the grid stays at 8.
- **Memory Throughput 9.54 %.** Same root cause: only 8 SMs are issuing
  loads against DRAM. With 8× more concurrent blocks we should be able
  to hit ~30–40 % SoL.

### Changes

- **Two-kernel split-K design** (`src/attention_decode.cu`).
  1. `attention_decode_split_kernel<D, K_SPLIT>` — grid
     `(K_SPLIT, H, B)` = 64 blocks at the production shape with
     `K_SPLIT = 8`. Each block owns a `Sk / K_SPLIT` slice of K and runs
     the Round-8 4-warp pipeline on it, with the intra-block partial
     merge writing to a per-block `(m, ℓ, O)` partial buffer in global
     memory rather than to the persistent `(M, L, O)`.
  2. `attention_decode_reduce_kernel<D>` — grid `(H, B)`, 32 threads /
     block. Each warp reads the existing persistent `(M, L, O)` and folds
     in all `K_SPLIT` partials sequentially using the same FlashAttention
     partial-merge recurrence as Round 8's intra-block merge. Cost: a
     single-warp sweep with `K_SPLIT` iterations — a few μs.
- **Per-call temp workspace** sized
  `K_SPLIT × B × H × (D + 2)` floats. For the production shape (B=1, H=8,
  D=128, K_SPLIT=8): 8 × 1 × 8 × 130 = 8 320 floats = 32.5 KB. Allocated
  inside `launch_decode_typed` with a `DeviceTensor` and freed at end of
  call. The cudaMalloc/cudaFree overhead (~50 μs each) is dwarfed by the
  ~2 ms kernel time at prompt=16k; can be hoisted into the ring-decode
  session later if that 50 μs becomes a fraction of a smaller kernel.
- **API unchanged.** `launch_attention_decode_step` keeps the same
  signature; `K_SPLIT` is hidden inside the launch function. The reduce
  kernel runs after the compute kernel on the same stream, so the
  persistent `(M, L, O)` semantics observed by the ring loop are
  identical to Round 8.

### Practice — Nsight Compute (D=128, prompt=16384, B=1, H=8)

Comparing the Round 8 single-kernel against the Round 9 split kernel
(the reduce is single-warp and runs in a few μs — not the headline):

| metric                          | Round 8         | Round 9 (split)     | Δ            |
|---                              |---              |---                   |---           |
| Grid Size                       | 8               | **64**               | **8×**       |
| Memory Throughput (SoL)         | 9.54 %          | **69.72 %**          | **7.3×**     |
| Memory Throughput (bytes/s)     | 59.2 GB/s       | **432 GB/s**         | 7.3×         |
| Compute (SM) Throughput         | 1.89 %          | 14.03 %              | 7.4×         |
| Issue Slots Busy                | 17.12 %         | 16.15 %              | ≈            |
| Achieved Occupancy              | 12.50 %         | 12.49 %              | ≈            |
| Theoretical Occupancy           | 100 %           | 100 %                | =            |
| Avg. Active Threads / Warp      | 32              | 32                   | =            |
| Top stall                       | Short SB 2.8c   | Short SB 3.0c        | similar      |

The kernel is now **DRAM-bandwidth-bound**: 432 GB/s out of ~600 GB/s peak
on the Quadro RTX 6000. Compute throughput tracks memory the same way it
did in Round 8 because per-block compute hasn't changed — we just have 8×
more blocks issuing the same load pattern in parallel. Per-block Achieved
Occupancy stays at 12.5 % (4 warps/SM) but the *grid* now covers 64 of the
72 SMs instead of 8, which is what unlocks the memory-bandwidth scaling.

### Practice — end-to-end (`bench_decode.sbatch`, full sweep)

`results/bench_decode_round8.csv` → `results/bench_decode_round9.csv`.
Per-token decode latency, fp32, B=1 H=8, mean over 16 tokens.

**cp_size = 1** (kernel-dominated):

| prompt_len | D=64 R8  | D=64 R9  | Δ        | D=128 R8 | D=128 R9 | **Δ**    |
|---         |---       |---       |---       |---       |---       |---       |
| 1 024      | 0.14 ms  | 0.06 ms  | **2.20×**| 0.18 ms  | 0.07 ms  | **2.49×**|
| 4 096      | 0.56 ms  | 0.15 ms  | **3.62×**| 0.70 ms  | 0.24 ms  | **2.88×**|
| 16 384     | 2.24 ms  | **0.56 ms** | **4.00×**| 2.74 ms  | **1.02 ms** | **2.70×**|

Note: at cp=1 prompt=16384 D=128 the kernel itself is 0.34 ms; the
remaining 0.68 ms of total is the cudaMalloc/cudaFree pair of the
per-call workspace (~50 µs × 3 buffers × 2 sync points). This is
mechanical overhead, not algorithmic — moving the workspace into a
session-level cache would shave ~0.5 ms more off the cp=1 numbers but
won't matter at cp>1 where comm dominates anyway.

**cp_size = 2 and 4** (comm-dominated):

| prompt_len | D=128 cp=2 R8 | D=128 cp=2 R9 | Δ        | D=128 cp=4 R8 | D=128 cp=4 R9 | Δ        |
|---         |---            |---            |---       |---            |---            |---       |
| 1 024      | 0.60 ms       | 0.51 ms       | 1.17×    | 0.86 ms       | 0.84 ms       | 1.03×    |
| 4 096      | 2.18 ms       | 1.73 ms       | 1.26×    | 3.10 ms       | 2.68 ms       | 1.16×    |
| 16 384     | 8.53 ms       | 6.59 ms       | **1.29×**| 11.85 ms      | 10.08 ms      | **1.18×**|

**Combined Round 4 → Round 9 totals** at the headline configs:

| config                    | Round 4   | Round 9   | total speedup |
|---                        |---        |---        |---            |
| D=128 cp=1 prompt=16384   | 49.68 ms  | 1.02 ms   | **48.7×**     |
| D=128 cp=4 prompt=16384   | 59.36 ms  | 10.08 ms  | **5.89×**     |
| D=64  cp=1 prompt=16384   | 13.37 ms  | 0.56 ms   | **23.9×**     |
| D=64  cp=1 prompt=1024    | 0.88 ms   | 0.06 ms   | **14.7×**     |

### What this doesn't fix

- **Workspace cudaMalloc/Free per call.** At cp=1 prompt=16k D=128 the
  raw kernel is 0.34 ms but total is 1.02 ms — most of the gap is the
  per-call workspace allocation. A `RingDecodeSession`-level workspace
  would close that gap (the design doc already named the session as the
  right home for streams/comm/cache).
- **Comm dominates at cp_size > 1.** Round 9's kernel speedup is buried
  behind NCCL Send/Recv time at cp_size > 1; D=128 cp=4 prompt=16k spends
  ~80 % of the per-token wall-clock in comm now. Overlapping the local
  kernel with the next-step NCCL transfer (the same trick `ring-overlap`
  uses for prefill, but adapted to the single-shot decode call) is the
  next lever for the cp > 1 numbers.
- **Memory-bandwidth ceiling reached at ~70 %.** The kernel sits at the
  Turing PCIe-DDR ceiling for this workload. Going further requires
  reducing the actual byte traffic — fp16 K/V cache would halve the bytes
  and probably double the throughput in absolute terms.

---

## Round 10 — cached decode workspace (cudaMalloc out of the hot path)

Profile: `results/ncu/2026-05-28_18-32/` (Round 9 baseline; new profile
written by this round's re-run).

### Bottlenecks identified

- **Per-call workspace allocation dominates the cp=1 wall-clock.**
  Round 9 left a 0.68 ms gap between the kernel (`comp_ms = 0.34 ms`) and
  the total per-token decode time (`total_ms = 1.02 ms`) at D=128
  prompt=16k cp=1. The decode launch path calls
  `DeviceTensor<float> M_partial(...), L_partial(...), O_partial(...)`
  — 3 cudaMalloc + 3 cudaFree per decode kernel call — and inside the
  ring loop `run_ring_decode_step` adds 4 more transit buffers. The
  cudaMalloc cost is small per call (~30–50 µs) but the *count* adds up:
  3 + 4 = 7 alloc/free pairs per cp=1 decode step, ~cp_size× that at
  cp_size > 1. At the kernel's new 0.34 ms time the alloc cost is the
  *single largest* contributor to total wall-clock.
- **Memory throughput is already at ~70 % of peak** (Round 9). Further
  kernel wins require either (a) cutting the byte traffic (fp16 KV) or
  (b) the comm/comp overlap for cp>1 — neither of which the per-call
  allocator overhead is on the critical path for. Round 10 closes the
  cheap allocator gap first so the next round sees a clean baseline.

### Changes

- **Process-static decode workspace** (`src/attention_decode.cu`). A
  single static buffer `g_workspace` holds the K_SPLIT × B × H × (D + 2)
  floats of partial state. `ensure_workspace(needed)` grows it on demand
  and never frees — the buffer persists for the life of the process.
  `launch_decode_typed` slices `M_partial`, `L_partial`, `O_partial`
  pointers out of `g_workspace` instead of allocating fresh
  `DeviceTensor`s. Eliminates 3 cudaMalloc + 3 cudaFree per decode kernel
  call.
- **Thread-safety caveat.** The static workspace is not stream- or
  thread-safe; concurrent decode launches on different streams would
  race on the partial buffers. This is fine for the current
  `ring_attention_cli` and the test suite (single host thread,
  default stream); a `RingDecodeSession`-owned workspace is the right
  long-term home but adds API surface area not yet motivated by a real
  caller.

### Practice — Nsight Compute (D=128, prompt=16384, B=1, H=8)

The split kernel itself is **unchanged** at the device level, so the
device metrics are identical to Round 9 within run-to-run noise:

| metric                          | Round 9         | Round 10        |
|---                              |---              |---              |
| Memory Throughput (SoL)         | 69.72 %         | 70.27 %         |
| Compute (SM) Throughput         | 14.03 %         | 14.13 %         |
| Issue Slots Busy                | 16.15 %         | 16.17 %         |
| Memory Throughput (bytes/s)     | 432 GB/s        | 433 GB/s        |
| Warp Cycles / Issued Instruction| 6.20 cycles     | 6.18 cycles     |
| Achieved Occupancy              | 12.49 %         | 12.49 %         |

This is the expected signature: Round 10's win is host-side (allocator
overhead removed), not device-side. The kernel itself was already
DRAM-bandwidth-bound at ~70 % of peak.

### Practice — end-to-end (`bench_decode.sbatch`, full sweep)

`results/bench_decode_round9.csv` → `results/bench_decode_round10.csv`.
Per-token decode latency, fp32, B=1 H=8, mean over 16 tokens.

**cp_size = 1** (the configuration most exposed to fixed per-call cost):

| prompt_len | D=64 R9  | D=64 R10 | Δ        | D=128 R9 | D=128 R10 | **Δ**    |
|---         |---       |---       |---       |---       |---        |---       |
| 1 024      | 0.06 ms  | 0.04 ms  | **1.44×**| 0.07 ms  | 0.06 ms   | 1.19×    |
| 4 096      | 0.15 ms  | 0.14 ms  | 1.06×    | 0.24 ms  | 0.23 ms   | 1.06×    |
| 16 384     | 0.56 ms  | 0.55 ms  | 1.02×    | 1.02 ms  | **0.88 ms**   | **1.16×**|

The relative gain is biggest at small `prompt_len` where the kernel is
fastest and the fixed allocator cost is a larger fraction of the total.
At prompt=16k D=128 the kernel itself is 0.32 ms (down from 0.34 ms in
R9, equal within noise); the remaining 0.56 ms of total comes from the
`run_ring_decode_step` transit-buffer allocs (4 × ~67 MB at this shape)
plus kernel-launch overhead — Round 11's natural target if cp=1 still
matters.

**cp_size = 2 and 4** (comm-dominated; very little for Round 10 to give
because the kernel is no longer a meaningful slice of the per-token wall):

| prompt_len | D=128 cp=2 R9 | D=128 cp=2 R10 | Δ      | D=128 cp=4 R9 | D=128 cp=4 R10 | Δ      |
|---         |---            |---             |---     |---            |---             |---     |
| 1 024      | 0.51 ms       | 0.48 ms        | 1.05×  | 0.84 ms       | 0.80 ms        | 1.05×  |
| 4 096      | 1.73 ms       | 1.69 ms        | 1.02×  | 2.68 ms       | 2.65 ms        | 1.01×  |
| 16 384     | 6.59 ms       | 6.61 ms        | ≈ 1.0× | 10.08 ms      | 10.01 ms       | 1.01×  |

**Combined Round 4 → Round 10 totals** at the headline configs:

| config                    | Round 4   | Round 10  | total speedup |
|---                        |---        |---        |---            |
| D=128 cp=1 prompt=16384   | 49.68 ms  | 0.88 ms   | **56.5×**     |
| D=128 cp=1 prompt=1024    | 3.14 ms   | 0.06 ms   | **52.3×**     |
| D=64  cp=1 prompt=16384   | 13.37 ms  | 0.55 ms   | **24.3×**     |
| D=128 cp=4 prompt=16384   | 59.36 ms  | 10.01 ms  | **5.93×**     |

### What this doesn't fix

- **Transit-buffer allocations inside `run_ring_decode_step`** are now
  the visible host overhead at cp=1: 4 × `DeviceTensor<float>(transit_elem)`
  per call, where `transit_elem = B × kv_H × max_chunk × D` reaches
  ~67 MB / buffer at the production shape. Folding these into the same
  static-workspace pattern is the cheapest available next win and would
  likely close most of the remaining cp=1 gap.
- **Comm dominates at cp_size > 1.** Unchanged from Round 9 — the
  per-call allocator removal has nothing to do with NCCL latency.
  Decode-mode compute/comm overlap is still the next architectural
  lever for the cp>1 numbers.

---

## Round 11 — cached ring-decode transit buffers [REVERTED]

Profile: same `results/ncu/2026-05-28_18-53/` (kernel is unchanged;
this round targeted host-side allocator overhead in the orchestrator).
Bench comparison: `bench-decode-88132.out` (Round 10 baseline) vs
`bench-decode-88139.out` (Round 11 candidate).

### Bottlenecks identified

- **Transit-buffer allocations in `run_ring_decode_step`** were
  hypothesised to dominate the cp=1 host overhead after Round 10. Per
  call the function constructs six `DeviceTensor<float>` instances —
  `K_a`, `K_b`, `V_a`, `V_b` of `B × kv_H × max_chunk × D` floats each,
  plus tiny `m_d`, `l_d` of `B × H` floats — and destructs them at
  scope exit. At the production shape (B=1, kv_H=8, prompt=16384,
  D=128) each of the four transit buffers is ~67 MB.
- At cp>1 the kernel calls (cp_size of them) all share the same six
  buffers — they're allocated once at the top of the function — so the
  per-token overhead is independent of cp_size.

### Changes attempted

- **Process-static ring workspace** (`src/ring_decode.cu`). One single
  static buffer holds `[K_a | K_b | V_a | V_b | m | ℓ]` contiguously.
  `ensure_ring_workspace(needed)` grows on demand (Round 10 pattern),
  never frees. The six former `DeviceTensor` constructions become
  pointer slices out of the workspace — no allocation in the hot path.

### Practice — regression at cp=1, no change at cp>1

| metric (D=128, prompt=16k) | Round 10 (88132) | Round 11 (88139) | Δ |
|---                          |---               |---               |---|
| cp=1  total_ms              | 0.881            | **1.020**        | +15.7 % (slower) |
| cp=1  comp_ms               | 0.326            | **0.382**        | +17.2 % (slower) |
| cp=2  total_ms              | 6.611            | 6.614            | within noise |
| cp=2  comp_ms               | 0.338            | 0.338            | within noise |
| cp=4  total_ms              | 10.135           | 10.138           | within noise |
| cp=4  comp_ms               | 0.371            | 0.370            | within noise |

`comp_ms` is the sum of cuda-event-timed kernel intervals — so the
regression is **inside the GPU kernel time**, not the host alloc path.
Repeated the bench (job 88139) to rule out noise; the regression
reproduced. Correctness preserved end-to-end (`ring_decode max_err=7.45e-08`,
all 10 GPU tests pass).

### Why this failed (hypothesis)

The premise was wrong. `cudaMalloc/cudaFree` of 67 MB inside a hot
loop are *not* free, but CUDA's caching allocator (and our
`DeviceTensor` wrapper which sits on top of it) already pools repeat
same-size allocations — the per-call alloc cost was nowhere near the
0.5 ms / 6 alloc-pairs the original bullet estimated. Swapping for a
namespace-static workspace removed the (small) alloc cost but
**changed the resident memory footprint**: the 268 MB workspace now
lives permanently in the address space across the whole run, whereas
the `DeviceTensor` version reused the same address bands per call.
The cp=1 kernel is bandwidth-bound (Round 10's 70 % memory SoL), and a
permanently-resident large workspace appears to perturb either the L2
victim set or the page-table residency enough to add ~50 µs to a
kernel that totals ~330 µs. The cp>1 cases don't show it because
comm dominates by 10×, swallowing any sub-100 µs kernel jitter.

### Reverted

`src/ring_decode.cu` restored to the original `DeviceTensor` allocations
(`git checkout -- src/ring_decode.cu`). The takeaway — *resident
workspace caching at this size can perturb the kernel's effective
memory hierarchy and is not free* — argues for *per-shape* caches
rather than one big monolithic workspace if we ever revisit this.
