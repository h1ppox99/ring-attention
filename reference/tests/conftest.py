"""Shared pytest fixtures for ring-attention reference tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    torch.manual_seed(0)
