Ring Attention v1 — Profiling-Driven Optimization Plan

  0. Decisions baked in (push back on any of these)

  - Anchor architecture: Llama-3 8B shape (or a more recent Gemma model - see which one to choose) — B=1, H=32, D=128. Single layer per callfor detailed analysis + benchmarking on one full inference pass.
  - Workload matrix: S ∈ {4k, 8k, 16k, 32k, 64k}, cp_size ∈ {1, 2, 4}, dtype ∈ {fp32, fp16}, causal ∈ {off, on}, mode ∈ {ring-blocking, ring-overlap}. ≈ 240 cells; not all need every metric.
  - Hardware: 4×Quadro RTX 6000 (sm_75), single node ?, PCIe (no NVLink/NVSwitch). Same node throughout — pin the assumption.
  - Response variables: comp_ms, comm_ms, wait_ms, total_ms (already emitted by the benchmark), plus per-rank breakdown and per-kernel Nsight metrics.

  1. What v1 is (the system we are profiling)

  The codebase as of c++coverage:

  - Kernels: attention_step.cu (fp32 scalar) and attention_step_fp16.cu (fp16 + WMMA tensor cores). Both support D ∈ {32, 64, 128, 256}. Online softmax with running (m, ℓ, O).
  - Distribution: ring of cp_size ranks, NCCL Send/Recv on a dedicated stream_copy, compute on stream_compute. ring-blocking and ring-overlap modes both in.
  - Load balance: coarse zigzag (RingPartition::Mode::Zigzag, 2 contiguous sub-groups per rank).
  - Numerics: validated against the Python reference at atol=rtol=1e-3.

  Out-of-scope for v1 profiling (and acknowledged as such in the report): no backward, no decode mode, no attn_bias, no fine-grained zigzag.

  2. The analytical model (built alongside v1)

  A single function predict(S, D, H, B, P, dtype, mode, causal) → (t_comp, t_comm, t_wait, t_total):

  FLOPs_per_step    ≈ 4 · B · H · (S/2P)² · D · zigzag_factor(P, causal)
  KV_bytes_per_step = 2 · B · (S/P) · H · D · sizeof(dtype)
  t_comp_step       = FLOPs_per_step / (P_eff · peak_tflops · η_kernel)
  t_comm_step       = KV_bytes_per_step / BW_pcie_eff + α_nccl
  t_total           = Σ_steps  ( blocking ? t_comp+t_comm : max(t_comp, t_comm) ) + α_launch · P

  Calibrated constants (measured once on this cluster):
  - peak_tflops per dtype from a microbenchmark (cublasGemmEx).
  - BW_pcie_eff from a NCCL pair-wise ncclSend microbench.
  - α_nccl, α_launch from idle-loop measurements at small S.
  - η_kernel per dtype from the v1 roofline (this is the term whose drift across versions tells the story).

  The model is a first-order alpha-beta-gamma model. It is expected to be wrong; the wrongness is informative.

  3. Per-version profiling protocol

  Each version repeats this loop. Apply it first to v1 to define the baseline; then to v2, v3, v4.

  Step A — Coarse sweep (always)

  Run the full workload matrix under ring-overlap and ring-blocking. Output: results/v{N}/sweep.csv with comp_ms, comm_ms, wait_ms, total_ms per cell,
  mean ± stdev over 20 iters after 5 warmup. Visualization: stacked bars across S for each (cp_size, dtype, causal) panel.

  Step B — Model fit (always)

  Calibrate η_kernel and BW_pcie_eff on the v1 data. Plot predicted vs measured total_ms (scatter, log-log) and residual heatmap over (S, cp_size). The
  cells with the largest residuals are where the model is broken — those are the bottlenecks.

  Step C — Targeted kernel profile (Nsight Compute)

  Pick 3–4 representative cells (one compute-bound, one comm-bound, one large-D, one short-S). For each:

  - Roofline: arithmetic intensity vs achieved FLOPs/s.
  - SM metrics: smsp__cycles_active.avg.pct_of_peak_sustained_elapsed, achieved occupancy, registers/thread, smem/block.
  - Tensor pipe: smsp__inst_executed_pipe_tensor.sum.per_cycle_active (fp16 only — should be near 1 if WMMA is saturated).
  - Memory: dram__bytes.sum.per_second vs peak HBM; l1tex__data_bank_conflicts_pipe_lsu.sum.
  - Stalls: top warp stall reasons (stall_long_sb, stall_mio_throttle, …).

  Step D — Timeline profile (Nsight Systems)

  Two representative cells (short-S and long-S). Capture both ranks' streams. Extract programmatically:
  - Gap between stream_copy NCCL end and stream_compute kernel start (the real wait, separate from the benchmark's wait_ms).
  - Per-rank kernel time variance (load imbalance signal).
  - Idle gaps between ring steps (launch overhead).

  Step E — Diagnose & fix

  The bottleneck identified in B+C+D defines v2. Document the hypothesis explicitly:

  ▎ "v1 residual is largest at (S=64k, P=4, fp16, causal). Nsight shows 38% register-spill traffic at D=128 because float O_i[D] lives in local memory.
  ▎ Hypothesis: tiling the D dimension into chunks of 32 will eliminate spill and gain ≈X% on comp_ms. Predicted total_ms after fix: Y. Implementing as
  ▎ v2."

  Step F — Validate

  Re-run Step A on v2. Plot v1 vs v2 deltas. Re-fit the model; report whether η_kernel moved as predicted. If not, the model was wrong about the
  mechanism — that, too, gets written up.

  4. Anticipated v1 bottlenecks (the hypotheses going in)

  Listed by regime so you know what to look for. The actual ranking comes out of Step B.

  1. Long-S compute regime (S ≥ 32k, cp_size=1 or large local Sl). Likely SM-bound. Look at: tensor pipe utilization in fp16 (if < 70% there's room),
  register pressure at D=128/256 (spills to local mem are the prime suspect), shared-memory bank conflicts in the QK^T / softmax stages.
  2. Short-S launch-overhead regime (S ≤ 8k). comp_ms and comm_ms both small; total_ms dominated by α_launch · P (per-step CPU dispatch + cudaEvent
  setup). Look at: per-step idle gaps in Nsight Systems, NCCL kernel time floor.
  3. Communication regime (fp16, P=4, S moderate). PCIe at ≈12 GB/s effective per direction. Look at: BW_pcie_eff extracted from comm_ms matches the
  microbench; if it doesn't, NCCL is doing extra copies or contention. Also check whether ring-overlap actually achieves wait_ms → 0 at all sizes or
  only some.
  4. Load imbalance under causal masking (causal=on, low cp_size). Coarse zigzag (2 contiguous sub-groups) is known suboptimal at small P. Look at:
  per-rank comp_ms variance from the timeline; if rank-skew is ≥10%, fine-grained zigzag (TODO #1) is a v2 candidate.
  5. Memory regime (S = 128k at cp_size=1 or 2). May simply OOM. The boundary itself is data — it goes on the scaling-frontier plot.
  6. fp16 wmma underutilization. The fp16 kernel uses 16×16×16 WMMA tiles. With BR=BC=16, D=128, the K-dimension is 128/16 = 8 wmma steps. Tail effects
  in S (when local Sl isn't a multiple of 16) and the softmax-rescale stage are not on tensor cores. Look at: cycles in softmax/rescale vs cycles in
  WMMA.

  5. Deliverables out of v1 specifically

  - results/v1/sweep.csv — the full matrix.
  - results/v1/roofline.png — kernel position vs sm_75 roof, both dtypes.
  - results/v1/timeline_*.qdrep — Nsight Systems traces at 2 cells.
  - results/v1/ncu_*.ncu-rep — Nsight Compute reports at 4 cells.
  - models/predictor.py — fitted v1 model.
  - docs/design/v1_baseline.md — narrative: what we ran, what we saw, residual heatmap, the ranked bottleneck list with evidence, and the v2 hypothesis.

  The v1 doc commits to v2 before any v2 code is written. That ordering is the point of the methodology — it keeps you honest about whether profiling
  actually drove the next change.

  6. Practical application: long-context prefill of a 7B-class model

  This section describes a *demo track* that runs in parallel with the v1→v4 optimization track. Its purpose is twofold: (a) anchor the synthetic
  benchmarks in a real workload so the report can claim "we processed a real model end-to-end," and (b) validate the analytical model from §2 against
  a 32-layer stacked workload rather than a single-layer call. It is *not* an attempt to do ML research — no fine-tuning, no quality metrics, no
  perplexity. The application is purely a latency/throughput demonstration.

  6.1 Scope: prefill only, no decode

  The application runs a *single forward pass* of a 7B-class transformer over a long input — the "prefill" phase of LLM inference. It emits the
  logits of the next token, then stops. Full autoregressive generation requires TODO #8 (decode-mode ring attention with a distributed KV cache),
  which is out of scope for v1–v4. Restricting to prefill keeps the workload entirely within what the current kernel and ring loop support:
  - Q, K, V are all of length S (no length-1 query against a cache).
  - Causal masking is on.
  - Single batch (B=1), single forward pass per measurement.

  Concretely: feed in a ~64k-token document, do the forward pass, print "processed N tokens in T seconds on P GPUs at L ms/layer." That is the demo.

  6.2 Model choice: Llama-3 8B

  Anchor on Llama-3 8B (Meta's openly distributed weights) because:
  - Its attention shape (H=32, D=128, MHA — ignoring GQA for now) matches the §0 anchor exactly, so all v1 profiling data transfers directly.
  - 32 transformer layers → 32 stacked ring-attention calls per forward pass → exercises the launch-overhead regime at scale.
  - 8B params × fp16 ≈ 16 GB → fits comfortably across 4× RTX 6000 (24 GB each) even with weights *replicated* per rank.
  - Tokenizer and weights are HuggingFace-compatible; integration is a known-quantity problem.

  Llama-3 8B technically has GQA (H_q=32, H_kv=8). For v1's demo we treat it as MHA by replicating the K/V heads to 32 — slight memory waste but
  avoids depending on TODO #7. When GQA is added (likely v3 or v4) the demo automatically benefits.

  6.3 Parallelism layout: weights replicated, sequence partitioned

  This is *context parallelism only* — no tensor parallelism, no pipeline parallelism. Each of the P GPUs holds:
  - The full Llama-3 8B weights (~16 GB fp16, replicated). Cheap on RTX 6000.
  - 1/P of the input sequence (the local Q/K/V slice).

  Per layer: MLPs, embedding lookup, RMSNorm, RoPE, output projection — all run *locally* on each rank's slice (no cross-rank communication needed,
  because they are token-wise operations). The attention op is the *only* place where ranks talk to each other — and that is exactly where the ring
  runs. This is the cleanest possible setup for isolating the parallel-computing contribution: every millisecond of cross-GPU communication is
  attention communication.

  6.4 Integration path: HuggingFace + monkeypatch

  The minimum-viable integration, in dependency order:

  1. pybind11 binding `ring_attention_torch.so` exposing `ring_attention(q, k, v, causal=True) → o` that accepts and returns `torch.Tensor` on CUDA.
     The binding wraps `run_ring_attention` from `src/`. PyTorch tensors expose `.data_ptr()` — passing those into the existing C++ entrypoint
     requires no buffer copies. ~150 lines of glue.
  2. A `RingAttention(nn.Module)` wrapper in Python that takes `(q, k, v)` of shape `(B, S_local, H, D)` and calls the binding.
  3. Monkey-patch `transformers.models.llama.modeling_llama.LlamaAttention.forward` to replace its scaled-dot-product-attention call with our module.
     RoPE stays in PyTorch (token-wise, no cross-rank); only the attention dot-product/softmax is replaced.
  4. A launcher script `apps/inference/long_context_prefill.py` that:
     - Initializes MPI/NCCL via `torch.distributed` (NCCL backend, same communicator the C++ side uses — share via the existing `nccl_utils.hpp`).
     - Loads Llama-3 8B weights on each rank.
     - Tokenizes a long input document (provided as a CLI arg — e.g., a public-domain novel concatenated to 64k tokens).
     - Partitions the input sequence by zigzag across ranks (same partition logic as the C++ side; expose `ring_partition` to Python).
     - Runs the forward pass under `torch.no_grad()`.
     - Emits per-layer timing + total prefill latency to `results/v{N}/inference_prefill.json`.

  Risks and de-scope ladder, worst-case first:
  - If pybind11 + PyTorch interop is painful: fall back to running the existing CLI binary with pre-saved Q/K/V tensors per layer, dumped from a
    PyTorch run with the stock attention. Less elegant but decouples the integration problem entirely.
  - If Llama-3 8B weights are gated/unavailable: substitute with Mistral 7B (same shape family) or with a *randomly-initialized* model of the same
    architecture. For a latency demo, real weights aren't strictly required — they just make the demo sound real.
  - If 32-layer end-to-end has correctness drift from fp16 accumulation across layers: the demo doesn't need bit-exact outputs against a reference
    implementation. Show that the logits are *plausible* (top-k token IDs match a single-GPU PyTorch run on a shorter prefix) and move on.

  6.5 What to measure

  The demo's measurements feed back into the §2 model:
  - End-to-end prefill latency at S ∈ {8k, 16k, 32k, 64k} on P ∈ {1, 2, 4}.
  - Per-layer breakdown: time in attention (our ring kernel) vs time in MLP/norm/RoPE (stock PyTorch). Validates that the ring is the bottleneck — or
    isn't — and pins the fraction of total prefill time that v1→v4 optimizations can affect (Amdahl ceiling for this application).
  - Predicted vs measured: sum the §2 predictor's `total_ms` across 32 layers and compare to the measured attention-only time. If they disagree, the
    delta is launch overhead, host gaps between layers, or NCCL communicator-reuse effects — all *new* phenomena that single-layer benchmarks miss.
  - Comparison baseline: single-GPU PyTorch SDPA on the longest sequence that fits (probably S ≤ 16k on one RTX 6000). Plot "tokens prefilled per
    second" for both. The crossover point is the demo's headline number.

  6.6 What this section is NOT

  - Not an ML quality study. Perplexity, accuracy, summarization quality are not measured.
  - Not autoregressive generation. No decode loop, no KV cache management, no sampling.
  - Not a training demo. No backward pass.
  - Not a fair comparison against vLLM / TensorRT-LLM. Those are highly engineered inference servers; we are demonstrating a parallel-attention
    kernel, not a serving stack.

  The framing in the report is: *"we built a context-parallel attention system; here it is plugged into a real 8B model doing real prefill; here is
  the end-to-end latency; here is which fraction of that latency our kernel is responsible for; here is how our predictor extrapolates to 128k
  context on hardware we don't have."*

  6.7 Deliverables for the application track

  - `bindings/ring_attention_torch/` — pybind11 + setup.py for the PyTorch binding.
  - `apps/inference/long_context_prefill.py` — launcher script.
  - `results/v{N}/inference_prefill.json` — per-layer + total timings, one per version of the kernel.
  - `docs/design/inference_application.md` — narrative: integration architecture, measurement protocol, the v1 result + Amdahl analysis (attention
    fraction of total prefill), comparison to single-GPU SDPA baseline, predicted scaling to longer context.
  - A short reproducibility note: exact Llama-3 checkpoint hash, tokenizer version, input document, SLURM command.

  Sequencing: the binding + launcher are built once against v1. Every subsequent kernel version (v2, v3, v4) is automatically picked up by rebuilding
  the shared library — no Python-side changes. This means the application track costs ~1 week of integration work, then ~1 hour per kernel version to
  re-measure.
