"""Shared heatmap head + per-family adapter.

One head class attaches to either backbone through a small adapter that
upsamples the backbone feature map to the common stride-4 response map:
ViT stride-16 needs two upsample blocks, ConvNeXt stride-32 needs three.
Each block is ``[ConvTranspose2d, GroupNorm(32), GELU]``; the head then
applies a 3x3 conv -> 1 channel (logits at train, sigmoid at inference).

The adapter block-count is the ONLY architecture-conditional difference in
the whole detector; the guard ``test_backbone_parity`` asserts both adapters
emit the same (B, C, 128, 128) stride-4 geometry for a 512 input, so an
adapter wired to the wrong stride fails CI (parity guard).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

HEAD_CHANNELS = 256
OUTPUT_STRIDE = 4


def _upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        ),
        nn.GroupNorm(32, out_channels),
        nn.GELU(),
    )


class HeatmapHead(nn.Module):
    """Adapter (backbone stride -> 4) + 3x3 conv -> 1-channel heatmap."""

    def __init__(
        self,
        in_channels: int,
        in_stride: int,
        *,
        head_channels: int = HEAD_CHANNELS,
        output_stride: int = OUTPUT_STRIDE,
    ) -> None:
        super().__init__()
        ratio = in_stride / output_stride
        n_blocks = math.log2(ratio)
        if n_blocks != int(n_blocks) or n_blocks < 1:
            raise ValueError(
                f"in_stride {in_stride} must be output_stride {output_stride} "
                f"times a power of two"
            )

        blocks: list[nn.Module] = []
        channels = in_channels
        for _ in range(int(n_blocks)):
            blocks.append(_upsample_block(channels, head_channels))
            channels = head_channels
        self.adapter = nn.Sequential(*blocks)
        self.conv = nn.Conv2d(head_channels, 1, kernel_size=3, padding=1)
        self.out_channels = head_channels
        self.output_stride = output_stride

        # CenterNet-style focal-friendly bias init: start the heatmap near
        # a low foreground prior instead of 0.5.
        nn.init.constant_(self.conv.bias, -2.19)  # -log((1 - 0.1) / 0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Backbone feature map -> heatmap LOGITS (B, 1, H4, W4)."""

        return self.conv(self.adapter(features))

    def adapter_features(self, features: torch.Tensor) -> torch.Tensor:
        """Expose the stride-4 adapter output for the parity guard."""

        return self.adapter(features)
