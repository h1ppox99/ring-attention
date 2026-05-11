"""Tests for the full-attention oracle — the ground truth all other refs compare to."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from ring_attention_ref import full_attention


def _manual_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    """Textbook softmax(Q Kᵀ / √d) V, done in fp32 for numerical stability."""
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    scale = 1.0 / math.sqrt(qf.shape[-1])
    scores = qf @ kf.transpose(-2, -1) * scale
    if causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        mask = torch.ones(seq_q, seq_k, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return (scores.softmax(dim=-1) @ vf).to(q.dtype)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("head_dim", [16, 32, 64])
@pytest.mark.parametrize("seq", [16, 128])
def test_matches_manual_softmax(causal: bool, heads: int, head_dim: int, seq: int) -> None:
    batch = 2
    q = torch.randn(batch, heads, seq, head_dim)
    k = torch.randn(batch, heads, seq, head_dim)
    v = torch.randn(batch, heads, seq, head_dim)

    out = full_attention(q, k, v, causal=causal)
    expected = _manual_attention(q, k, v, causal=causal)

    assert out.shape == expected.shape
    assert out.dtype == q.dtype
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
def test_matches_sdpa(causal: bool) -> None:
    q = torch.randn(1, 2, 32, 16)
    k = torch.randn(1, 2, 32, 16)
    v = torch.randn(1, 2, 32, 16)

    out = full_attention(q, k, v, causal=causal)
    expected = scaled_dot_product_attention(q, k, v, is_causal=causal)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_deterministic() -> None:
    q = torch.randn(1, 2, 32, 16)
    k = torch.randn(1, 2, 32, 16)
    v = torch.randn(1, 2, 32, 16)

    out1 = full_attention(q, k, v, causal=False)
    out2 = full_attention(q, k, v, causal=False)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)


def test_fp16_dtype_preserved() -> None:
    q = torch.randn(1, 2, 32, 16, dtype=torch.float16)
    k = torch.randn(1, 2, 32, 16, dtype=torch.float16)
    v = torch.randn(1, 2, 32, 16, dtype=torch.float16)

    out = full_attention(q, k, v, causal=True)
    assert out.dtype == torch.float16
    assert out.shape == (1, 2, 32, 16)


def test_causal_future_tokens_masked() -> None:
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)

    out_causal = full_attention(q, k, v, causal=True)
    out_first_only = full_attention(q[..., :1, :], k[..., :1, :], v[..., :1, :], causal=True)

    torch.testing.assert_close(out_causal[..., :1, :], out_first_only, atol=1e-6, rtol=1e-6)
