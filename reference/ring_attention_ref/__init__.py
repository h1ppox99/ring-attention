"""PyTorch reference for Ring Attention — correctness oracle for the CUDA implementation."""

from ring_attention_ref.inference import KVCache, decode_step, prefill
from ring_attention_ref.oracle import full_attention
from ring_attention_ref.ring import ring_attention
from ring_attention_ref.zigzag import partition, striped_indices, unpartition, zigzag_indices

__all__ = [
    "KVCache",
    "decode_step",
    "full_attention",
    "partition",
    "prefill",
    "ring_attention",
    "striped_indices",
    "unpartition",
    "zigzag_indices",
]
