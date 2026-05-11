"""Full-attention oracle: ground truth every other reference compares to."""

from __future__ import annotations

import torch
from torch.nn.functional import scaled_dot_product_attention


def full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    """Scaled dot-product attention over the full sequence.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(batch, heads, seq, head_dim)``.
    causal : bool
        Apply causal (lower-triangular) masking when ``True``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, heads, seq, head_dim)``.
    """
    return scaled_dot_product_attention(q, k, v, is_causal=causal)
