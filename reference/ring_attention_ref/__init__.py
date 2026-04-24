"""PyTorch reference for Ring Attention — correctness oracle for the CUDA implementation."""

from ring_attention_ref.oracle import full_attention
from ring_attention_ref.ring import ring_attention
from ring_attention_ref.zigzag import partition, unpartition, zigzag_indices

__all__ = ["full_attention", "partition", "ring_attention", "unpartition", "zigzag_indices"]
