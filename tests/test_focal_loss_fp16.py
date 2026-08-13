"""Regression guard: focal loss stays finite on fp16-saturated logits.

The 2026-07-23 cnnin1k-f10-s0 divergence: under fp16 autocast
the loss's [eps, 1 - eps] clamp is a no-op on the upper side because
``1 - 1e-6`` is not representable in half precision (it rounds to 1.0), so a
penalty-reduced negative pixel whose sigmoid saturates to fp16 1.0 produces
``log(0) = -inf``. The loss now casts its inputs to fp32 (study precision
rule: logit/log math in fp32), which restores the clamp and
bounds the worst-case per-pixel penalty at -log(1e-6).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.train.losses import penalty_reduced_focal_loss  # noqa: E402


def _saturating_batch(dtype: torch.dtype):
    """One positive center plus a saturated Gaussian-shoulder negative."""

    logits = torch.full((1, 8, 8), -8.0, dtype=dtype)
    target = torch.zeros((1, 8, 8), dtype=dtype)
    target[0, 4, 4] = 1.0  # positive center
    logits[0, 4, 4] = 12.0
    # Shoulder pixel: target < 1 makes it a penalty-reduced NEGATIVE, but a
    # confident detector legitimately drives its logit into saturation
    # (fp16 sigmoid(x) == 1.0 exactly for x >= ~9).
    target[0, 4, 5] = 0.8
    logits[0, 4, 5] = 12.0
    return logits, target


def test_fp16_saturated_shoulder_is_finite():
    loss = penalty_reduced_focal_loss(*_saturating_batch(torch.float16))
    assert torch.isfinite(loss), "fp16-saturated shoulder pixel must not yield inf"


def test_fp16_matches_fp32_away_from_saturation():
    torch.manual_seed(0)
    logits32 = torch.randn(2, 16, 16) * 3.0
    target = torch.rand(2, 16, 16)
    target[0, 3, 3] = 1.0
    loss16 = penalty_reduced_focal_loss(logits32.half(), target.half())
    loss32 = penalty_reduced_focal_loss(logits32, target)
    # Half-precision INPUTS may still be used by callers; the fp32 math keeps
    # the result within input-rounding distance of the fp32 reference.
    assert torch.allclose(loss16, loss32, rtol=2e-3, atol=1e-4)
