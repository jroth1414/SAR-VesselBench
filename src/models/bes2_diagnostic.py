"""Diagnostic-only BigEarthNet-S2 initialization variants.

The eight-arm registry is unchanged. Replay restores the historical converted
weight after the current production load. The reset variant recreates the same
seeded three-channel target and transfers every downloaded backbone tensor
except ``stem.0.weight``, matching timm's unsupported input-convolution
fallback.  Both paths leave the RNG at the same post-backbone state, so the
detector head and all later stochastic streams remain byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

import torch

from src.models.init_loaders import (
    build_init,
    map_bigearthnet_keys,
    repeat_with_rescaling,
)

from src.models.bes2_transfer import (
    BES2TransferError as BES2DiagnosticError,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_STEM_KEY,
    STEM_KEY,
    load_post_stem,
    sha256_file,
    source_state as _source_state,
    state_identity,
    state_tensor_hashes,
    tensor_sha256,
)


DIAGNOSTIC_VARIANTS: Final = ("current_replay", "first_conv_reset")


def _load_reset_target(
    *,
    source_state: Mapping[str, torch.Tensor],
    weights_root: str | Path,
    rng_before_target: torch.Tensor,
    rng_after_target: torch.Tensor,
) -> tuple[torch.nn.Module, str, list[str]]:
    """Load every mapped S2 tensor except the first convolution."""

    try:
        torch.random.set_rng_state(rng_before_target)
        backbone = build_init(
            "cnn_random",
            load_weights=False,
            weights_root=weights_root,
        )
        rng_after_reset_target = torch.random.get_rng_state().clone()
        if not torch.equal(rng_after_reset_target, rng_after_target):
            raise BES2DiagnosticError(
                "seeded reset target consumed a different RNG stream than "
                "the production S2 target"
            )

        fresh_stem_hash, loaded_keys = load_post_stem(
            backbone.model, source_state, map_bigearthnet_keys(source_state)
        )
        return backbone, fresh_stem_hash, loaded_keys
    finally:
        # Production replay and reset must initialize the shared head from the
        # identical RNG state even if a diagnostic assertion fails.
        torch.random.set_rng_state(rng_after_target)


def build_bes2_diagnostic_backbone(
    variant: str,
    *,
    weights_root: str | Path = "data/weights",
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Build one exact S2 diagnostic variant and its initialization manifest."""

    if variant not in DIAGNOSTIC_VARIANTS:
        raise ValueError(
            f"unknown BigEarthNet-S2 diagnostic variant {variant!r}; "
            f"use one of {DIAGNOSTIC_VARIANTS}"
        )

    rng_before_target = torch.random.get_rng_state().clone()
    production = build_init(
        "bigearthnet_s2",
        load_weights=True,
        weights_root=weights_root,
    )
    rng_after_target = torch.random.get_rng_state().clone()

    checkpoint, source_state = _source_state(weights_root)
    source_stem = source_state[SOURCE_STEM_KEY]
    production_state = production.model.state_dict()
    expected_current = repeat_with_rescaling(source_stem, 3)
    # Preserve the historical replay after Arm 6 adopts the approved fallback.
    # Only the diagnostic replay restores the superseded converted weight.
    with torch.no_grad():
        production_state[STEM_KEY].copy_(expected_current)

    production_hashes = state_tensor_hashes(production.model)
    fresh_stem_hash: str | None = None
    loaded_target_keys = sorted(production_hashes)
    backbone = production
    if variant == "first_conv_reset":
        backbone, fresh_stem_hash, loaded_target_keys = _load_reset_target(
            source_state=source_state,
            weights_root=weights_root,
            rng_before_target=rng_before_target,
            rng_after_target=rng_after_target,
        )

    active_hashes = state_tensor_hashes(backbone.model)
    changed = sorted(
        name
        for name in production_hashes
        if production_hashes[name] != active_hashes[name]
    )
    expected_changed = [] if variant == "current_replay" else [STEM_KEY]
    if changed != expected_changed:
        raise BES2DiagnosticError(
            f"{variant}: expected changed keys {expected_changed}, got {changed}"
        )
    if variant == "first_conv_reset" and active_hashes[STEM_KEY] != fresh_stem_hash:
        raise BES2DiagnosticError("reset stem does not match seeded target initialization")

    mapping = map_bigearthnet_keys(source_state.keys())
    dropped_source_keys = sorted(set(source_state) - set(mapping))
    manifest: dict[str, object] = {
        "schema": 1,
        "variant": variant,
        "source_init": "bigearthnet_s2",
        "source_checkpoint": {
            "relative_path": "bigearthnet_s2/model.safetensors",
            "sha256": sha256_file(checkpoint),
        },
        "source_stem": {
            "key": SOURCE_STEM_KEY,
            "shape": list(source_stem.shape),
            "sha256": tensor_sha256(source_stem),
            "channels": [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B11",
                "B12",
            ],
        },
        "target_input_channels": ["VH", "VV", "VH_minus_VV"],
        "stem_policy": (
            "production-repeat-with-rescaling-10-to-3"
            if variant == "current_replay"
            else "timm-unsupported-input-conv-fallback-seeded-target-weight"
        ),
        "mapped_source_key_count": len(mapping),
        "dropped_source_keys": dropped_source_keys,
        "loaded_target_keys": loaded_target_keys,
        "excluded_target_keys": (
            [] if variant == "current_replay" else [STEM_KEY]
        ),
        "retained_seeded_target_keys": (
            [] if variant == "current_replay" else [STEM_KEY]
        ),
        "changed_from_production_replay": changed,
        "production_stem_sha256": production_hashes[STEM_KEY],
        "seeded_target_stem_sha256": fresh_stem_hash,
        "active_stem_sha256": active_hashes[STEM_KEY],
        "backbone_tensor_sha256": active_hashes,
        "backbone_state_sha256": state_identity(active_hashes),
        "parameter_count": sum(
            parameter.numel() for parameter in backbone.parameters()
        ),
    }
    return backbone, manifest


def add_head_identity(
    manifest: Mapping[str, object],
    head: torch.nn.Module,
) -> dict[str, object]:
    """Bind the detector-head initialization without changing it."""

    hashes = state_tensor_hashes(head)
    return {
        **dict(manifest),
        "head_state_sha256": state_identity(hashes),
        "head_tensor_sha256": hashes,
        "head_parameter_count": sum(
            parameter.numel() for parameter in head.parameters()
        ),
    }
