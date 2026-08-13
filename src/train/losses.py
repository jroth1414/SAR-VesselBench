"""Penalty-reduced focal loss + Gaussian heatmap targets.

CenterNet (*Objects as Points*) recipe: each vessel stamps a Gaussian blob
(sigma = 2 output px) on the stride-4 target map; the loss is the
penalty-reduced pixelwise focal loss with alpha=2, beta=4. LOW-confidence
labels stamp an ignore disk (radius 3 output px) where the loss is zeroed.
"""

from __future__ import annotations

from typing import Sequence

import torch

GAUSSIAN_SIGMA_PX = 2.0
IGNORE_RADIUS_PX = 3.0
FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0


def render_target(
    size: int,
    centers: Sequence[tuple[float, float]],
    ignore_centers: Sequence[tuple[float, float]] = (),
    *,
    sigma: float = GAUSSIAN_SIGMA_PX,
    ignore_radius: float = IGNORE_RADIUS_PX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render (heatmap, loss_mask) at output resolution.

    ``centers``/``ignore_centers`` are (row, col) in OUTPUT-map pixels.
    Centers are quantized to INTEGER pixels before stamping — the CenterNet
    reference (`ct_int = ct.astype(int)`) — so every object's peak pixel is
    EXACTLY 1.0 and the focal loss's positive mask (`target >= 1`) is
    non-empty. Stamping at fractional centers silently yields max(target) < 1
    and zero positive-loss terms. The heatmap takes the max over per-vessel
    Gaussians; the loss mask zeroes an ignore disk around every
    LOW-confidence label.
    """

    heatmap = torch.zeros(size, size)
    mask = torch.ones(size, size)

    if centers:
        ys = torch.arange(size).view(-1, 1)
        xs = torch.arange(size).view(1, -1)
        for row, col in centers:
            row_int = int(min(max(row, 0), size - 1))
            col_int = int(min(max(col, 0), size - 1))
            gaussian = torch.exp(
                -((ys - row_int) ** 2 + (xs - col_int) ** 2) / (2.0 * sigma**2)
            )
            heatmap = torch.maximum(heatmap, gaussian)

    if ignore_centers:
        ys = torch.arange(size).view(-1, 1)
        xs = torch.arange(size).view(1, -1)
        for row, col in ignore_centers:
            disk = ((ys - row) ** 2 + (xs - col) ** 2) <= ignore_radius**2
            mask[disk] = 0.0

    return heatmap, mask


def penalty_reduced_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    alpha: float = FOCAL_ALPHA,
    beta: float = FOCAL_BETA,
) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss over sigmoid logits.

    ``target`` is the rendered Gaussian heatmap in [0, 1]; pixels equal to 1
    are positives, everything else is a penalty-reduced negative. Normalized
    by the number of positives (min 1). ``mask`` zeroes ignored pixels.
    """

    # Logit/log math must stay fp32 (study precision rule): in fp16
    # the upper clamp bound 1 - 1e-6 rounds to exactly 1.0 (largest half below
    # 1.0 is 1 - 2^-11), so a saturated negative-pixel sigmoid (logit >~ 9)
    # slips through as 1.0 and log(1 - prob) = -inf. Measured: killed
    # cnnin1k-f10-s0 deterministically at epoch 18 (2026-07-23).
    logits = logits.float()
    target = target.float()
    if mask is not None:
        mask = mask.float()

    prob = torch.sigmoid(logits)
    eps = 1e-6
    prob = prob.clamp(eps, 1.0 - eps)

    positive = (target >= 1.0 - 1e-6).float()
    negative_weight = (1.0 - target).pow(beta)

    pos_loss = -torch.log(prob) * (1.0 - prob).pow(alpha) * positive
    neg_loss = -torch.log(1.0 - prob) * prob.pow(alpha) * negative_weight * (1.0 - positive)
    loss = pos_loss + neg_loss
    if mask is not None:
        loss = loss * mask
        positive = positive * mask

    n_positives = positive.sum().clamp(min=1.0)
    return loss.sum() / n_positives


def gaussian_radius_centers(
    labels: Sequence[dict],
    *,
    input_stride: int = 4,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split chip-space labels into (positive, ignore) output-space centers.

    Positives: ``is_vessel`` truthy with confidence HIGH/MEDIUM. LOW
    confidence -> ignore. Non-vessel HIGH labels (fixed infrastructure) are
    background — neither positive nor ignored (Appendix B protocol).
    Chip-pixel coords are divided by ``input_stride`` to land on the
    stride-4 map.
    """

    positives: list[tuple[float, float]] = []
    ignores: list[tuple[float, float]] = []
    for label in labels:
        row = float(label["chip_row"]) / input_stride
        col = float(label["chip_col"]) / input_stride
        confidence = str(label.get("confidence") or "").upper()
        if confidence == "LOW":
            ignores.append((row, col))
        elif bool(label.get("is_vessel")) and confidence in ("HIGH", "MEDIUM"):
            positives.append((row, col))
    return positives, ignores
