"""Foreground-balanced chip sampling.

Epoch-level weighted sampler so ~50% of sampled chips contain at least one
HIGH/MEDIUM vessel — most open-ocean chips are empty, and an unweighted
epoch would starve the loss of positives.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import WeightedRandomSampler

FOREGROUND_FRAC = 0.5


def build_foreground_balanced_sampler(
    foreground: Sequence[bool],
    *,
    fg_frac: float = FOREGROUND_FRAC,
    num_samples: int | None = None,
    seed: int = 0,
) -> WeightedRandomSampler:
    flags = torch.as_tensor(list(foreground), dtype=torch.bool)
    n_fg = int(flags.sum())
    n_bg = int((~flags).sum())
    if n_fg == 0 or n_bg == 0:
        weights = torch.ones(len(flags), dtype=torch.double)
    else:
        weights = torch.where(
            flags,
            torch.tensor(fg_frac / n_fg, dtype=torch.double),
            torch.tensor((1.0 - fg_frac) / n_bg, dtype=torch.double),
        )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=num_samples or len(flags),
        replacement=True,
        generator=generator,
    )
