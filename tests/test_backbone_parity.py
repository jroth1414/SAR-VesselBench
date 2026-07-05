"""Guard test (DEVPLAN §1b guard 2): within-track param parity + cross-track
adapter output-geometry parity.

Within each track the four arms must instantiate the byte-identical
architecture (only weights differ); across tracks both head adapters must
emit the same stride-4 (B, C, 128, 128) map for a 512 input, so a mis-wired
adapter (wrong stride or wrong C) fails CI. Runs CPU-only with random inits
(``load_weights=False``) — no downloads.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from src.models.backbones import build_backbone  # noqa: E402
from src.models.heatmap_head import HeatmapHead  # noqa: E402
from src.models.init_loaders import INIT_FAMILY, build_init  # noqa: E402


def _param_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_within_track_param_parity():
    vit_counts = {
        name: _param_count(build_init(name, load_weights=False))
        for name, family in INIT_FAMILY.items()
        if family == "vit"
    }
    cnn_counts = {
        name: _param_count(build_init(name, load_weights=False))
        for name, family in INIT_FAMILY.items()
        if family == "cnn"
    }

    assert len(vit_counts) == 4 and len(cnn_counts) == 4
    assert len(set(vit_counts.values())) == 1, f"ViT arms diverge: {vit_counts}"
    assert len(set(cnn_counts.values())) == 1, f"CNN arms diverge: {cnn_counts}"

    # Cross-track sizes are close but not identical (~86M vs ~89M) —
    # reported, never asserted equal (DEVPLAN §1b guard 2).
    vit_params = next(iter(vit_counts.values()))
    cnn_params = next(iter(cnn_counts.values()))
    assert vit_params != cnn_params
    assert abs(vit_params - cnn_params) / max(vit_params, cnn_params) < 0.10


def test_cross_track_adapter_geometry_parity():
    x = torch.randn(1, 3, 512, 512)
    shapes = {}
    for family in ("vit", "cnn"):
        backbone = build_backbone(family).eval()
        head = HeatmapHead(backbone.out_channels, backbone.out_stride).eval()
        with torch.no_grad():
            features = backbone(x)
            adapted = head.adapter_features(features)
            logits = head(features)
        shapes[family] = tuple(adapted.shape)
        assert logits.shape == (1, 1, 128, 128), f"{family} logits {logits.shape}"

    assert shapes["vit"] == shapes["cnn"], (
        f"adapter geometry drift: vit {shapes['vit']} vs cnn {shapes['cnn']}"
    )
    _, channels, height, width = shapes["vit"]
    assert (height, width) == (128, 128), "stride-4 map for a 512 input"
    assert channels == shapes["cnn"][1]
