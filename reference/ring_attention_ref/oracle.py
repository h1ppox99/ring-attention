"""Full-attention oracle: ground truth every other reference compares to."""

from __future__ import annotations

import torch
from torch.nn.functional import scaled_dot_product_attention


def full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    """Scaled dot-product attention over the full sequence.

    q, k, v: (batch, heads, seq, head_dim). Returns a tensor of the same shape.
    """
    return scaled_dot_product_attention(q, k, v, is_causal=causal)
