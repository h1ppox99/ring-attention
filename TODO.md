# TODO

## TODO 7 — Grouped-Query Attention (GQA / MQA) [COMPLETED]

Added `kv_heads` to `AttentionShape` (default 0 = MHA) and `RingConfig`, then propagated a
`kv_H` parameter through all four kernels (`attention_step`, `attention_step_fp16`,
`flash_attention`, `flash_attention_fp16`) so each head `h` indexes K/V at `h % kv_H` instead
of `h`. The ring-loop orchestrators (`ring_loop.cu`, `ring_loop_fp16.cu`) were updated to size
K/V buffers and MPI transfers by `kv_H` rather than `H`, and correctness tests covering
GQA (8Q/2KV) and MQA (4Q/1KV) were added to all four kernel test suites.
