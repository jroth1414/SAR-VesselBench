"""The two study backbones behind one feature interface (DEVPLAN P3.1).

- ViT track (arms 1-4): timm ``vit_base_patch16_224``, ``in_chans=3``,
  learnable pos-embed resampled to the input grid (``dynamic_img_size``), so a
  512x512 input yields a stride-16 (B, 768, 32, 32) feature map.
- CNN track (arms 5-8): timm ``convnextv2_base``, ``in_chans=3``,
  ``features_only`` stage-3 output: stride-32 (B, 1024, 16, 16) at 512 input.

Both are wrapped to expose a uniform ``forward(x) -> (B, C, H, W)`` plus
``out_channels`` / ``out_stride`` attributes consumed by the head adapter
(P3.3). Never hand-roll patch-embed surgery: any channel-count adaptation
happens exclusively through timm's ``in_chans`` (Repeat-with-rescaling,
ground rule 12).
"""

from __future__ import annotations

import torch
import torch.nn as nn

VIT_TIMM_ID = "vit_base_patch16_224"
CNN_TIMM_ID = "convnextv2_base"
IN_CHANS = 3  # fixed [VH, VV, VH-VV] study-wide


class ViTBackbone(nn.Module):
    """ViT-B/16 exposing the stride-16 patch-token grid as (B, 768, H16, W16)."""

    def __init__(self) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(
            VIT_TIMM_ID,
            pretrained=False,
            num_classes=0,
            in_chans=IN_CHANS,
            dynamic_img_size=True,  # pos-embed resampled to the input grid
        )
        self.out_channels = self.model.embed_dim  # 768
        self.out_stride = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        tokens = self.model.forward_features(x)  # (B, prefix + N, C)
        patches = tokens[:, self.model.num_prefix_tokens :, :]
        grid_h = height // self.out_stride
        grid_w = width // self.out_stride
        if patches.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"token count {patches.shape[1]} != {grid_h}x{grid_w} grid — "
                "input size must be a multiple of the 16px patch size"
            )
        return (
            patches.transpose(1, 2)
            .reshape(batch, self.out_channels, grid_h, grid_w)
            .contiguous()
        )


class ConvNeXtBackbone(nn.Module):
    """ConvNeXt-V2-Base exposing the stride-32 stage-3 map (B, 1024, H32, W32).

    Deliberately built as the PLAIN timm model (``forward_features``), NOT
    ``features_only``: the feature-extraction wrapper flattens module names
    (``stages.0`` -> ``stages_0``), which silently defeats timm's
    ``group_matcher`` so layer-wise lr decay degenerates to full lr on every
    layer — the P3.6 sweep caught the CNN track training without the decay
    the ViT track had (a cross-track fairness violation, and the likely
    bigearthnet_s1 divergence trigger).
    """

    def __init__(self) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(
            CNN_TIMM_ID,
            pretrained=False,
            num_classes=0,
            in_chans=IN_CHANS,
        )
        self.out_channels = self.model.num_features  # 1024
        self.out_stride = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.forward_features(x)
        if features.ndim != 4:
            raise RuntimeError(
                f"expected a NCHW stage-3 map, got shape {tuple(features.shape)}"
            )
        return features


def build_backbone(family: str) -> ViTBackbone | ConvNeXtBackbone:
    """'vit' -> ViT-B/16, 'cnn' -> ConvNeXt-V2-B (the only two families)."""

    if family == "vit":
        return ViTBackbone()
    if family == "cnn":
        return ConvNeXtBackbone()
    raise ValueError(f"unknown backbone family: {family!r} (use 'vit' or 'cnn')")
