"""Regression test for the P3.6 layer-decay degeneration (fairness bug).

timm's ``param_groups_layer_decay`` fails SILENTLY when a model's module
names don't match its group matcher: every parameter lands at lr_scale 1.0.
The P3.6 sweep caught the CNN track training without the layer-wise decay
the ViT track had (the features_only wrapper had flattened module names) —
a violation of the identical-optimizer contract (ground rule 2) and the
likely bigearthnet_s1 fp16 divergence trigger. Both tracks must now produce
a real multi-scale decay ladder, and a degenerate grouping must raise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("lightning")

from src.train.lit_modules import HeatmapLitModule  # noqa: E402


@pytest.mark.parametrize("init_name", ["vit_random", "cnn_random"])
def test_layer_decay_ladder_is_real_and_applied(init_name):
    module = HeatmapLitModule(
        init_name=init_name, load_weights=False, lr=1.0e-4, layer_decay=0.65
    )
    config = module.configure_optimizers()
    optimizer = config["optimizer"]

    # The ladder must exist in the OPTIMIZER's actual per-group lrs — timm's
    # lr_scale is only applied by timm schedulers, so an un-baked ladder
    # sits inert and every layer trains at base lr (second half of the P3.6
    # optimizer bug).
    lrs = sorted({round(g["lr"], 14) for g in optimizer.param_groups})
    assert len(lrs) >= 8, (
        f"{init_name}: only {len(lrs)} distinct group lrs — the decay "
        f"ladder is not applied (lrs: {lrs})"
    )
    # Ratios are invariant to the warmup/cosine schedule state; both tracks
    # share the 13-rung ViT-B-depth convention, so max/min == 0.65^-12.
    assert lrs[0] / lrs[-1] == pytest.approx(0.65**12, rel=0.01), (
        f"{init_name}: ladder span {lrs[0] / lrs[-1]:.6f}, expected "
        f"0.65^12 — the tracks' ladders are not depth-equalized"
    )

    n_grouped = sum(len(g["params"]) for g in optimizer.param_groups)
    n_params = sum(1 for _ in module.backbone.parameters()) + sum(
        1 for _ in module.head.parameters()
    )
    assert n_grouped == n_params, "some parameters missing from the optimizer"
