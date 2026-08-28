"""Diagnostic-only BigEarthNet-S2 initialization variants.

The core arm registry and production loader remain unchanged.  The replay
variant calls that loader directly.  The reset variant recreates the same
seeded three-channel target and transfers every downloaded backbone tensor
except ``stem.0.weight``, matching timm's unsupported input-convolution
fallback.  Both paths leave the RNG at the same post-backbone state, so the
detector head and all later stochastic streams remain byte-identical.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Mapping

import torch

from src.models.init_loaders import (
    build_init,
    map_bigearthnet_keys,
    repeat_with_rescaling,
)

DIAGNOSTIC_VARIANTS: Final = ("current_replay", "first_conv_reset")
STEM_KEY: Final = "stem.0.weight"
SOURCE_STEM_KEY: Final = "model.vision_encoder.stem.0.weight"
SOURCE_CHECKPOINT_SHA256: Final = (
    "b09d0e41cc683878243a9128a6f4724d6a71d562318beeae716f0dce9cbbf454"
)


class BES2DiagnosticError(RuntimeError):
    """A diagnostic initialization violated its exact single-variable contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor metadata and exact contiguous CPU bytes."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(map(str, value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_tensor_hashes(module: torch.nn.Module) -> dict[str, str]:
    return {
        name: tensor_sha256(value)
        for name, value in sorted(module.state_dict().items())
    }


def state_identity(tensor_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensor_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _source_state(
    weights_root: str | Path,
) -> tuple[Path, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file

    checkpoint = Path(weights_root) / "bigearthnet_s2" / "model.safetensors"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise BES2DiagnosticError(
            f"BigEarthNet-S2 checkpoint must be a regular non-symlink file: {checkpoint}"
        )
    observed = sha256_file(checkpoint)
    if observed != SOURCE_CHECKPOINT_SHA256:
        raise BES2DiagnosticError(
            "BigEarthNet-S2 checkpoint SHA-256 mismatch: "
            f"expected {SOURCE_CHECKPOINT_SHA256}, got {observed}"
        )
    state = dict(load_file(checkpoint, device="cpu"))
    if SOURCE_STEM_KEY not in state:
        raise BES2DiagnosticError(
            f"BigEarthNet-S2 checkpoint lacks {SOURCE_STEM_KEY}"
        )
    stem = state[SOURCE_STEM_KEY]
    if tuple(stem.shape) != (128, 10, 4, 4):
        raise BES2DiagnosticError(
            f"unexpected BigEarthNet-S2 source stem shape: {tuple(stem.shape)}"
        )
    return checkpoint, state


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

        target_state = backbone.model.state_dict()
        mapping = map_bigearthnet_keys(source_state.keys())
        mapped_targets = list(mapping.values())
        if len(mapped_targets) != len(set(mapped_targets)):
            raise BES2DiagnosticError("BigEarthNet-S2 key mapping is not one-to-one")
        missing_before_reset = sorted(set(target_state) - set(mapped_targets))
        unexpected = sorted(set(mapped_targets) - set(target_state))
        if missing_before_reset or unexpected:
            raise BES2DiagnosticError(
                "downloaded S2 mapping no longer covers the target exactly: "
                f"missing={missing_before_reset}, unexpected={unexpected}"
            )

        fresh_stem_hash = tensor_sha256(target_state[STEM_KEY])
        retained: dict[str, torch.Tensor] = {}
        for source_key, target_key in mapping.items():
            if target_key == STEM_KEY:
                continue
            source_tensor = source_state[source_key]
            target_tensor = target_state[target_key]
            if (
                source_tensor.shape != target_tensor.shape
                or source_tensor.dtype != target_tensor.dtype
            ):
                raise BES2DiagnosticError(
                    f"downloaded tensor {source_key} cannot load as {target_key}: "
                    f"{tuple(source_tensor.shape)}/{source_tensor.dtype} != "
                    f"{tuple(target_tensor.shape)}/{target_tensor.dtype}"
                )
            retained[target_key] = source_tensor

        result = backbone.model.load_state_dict(retained, strict=False)
        missing = sorted(result.missing_keys)
        if missing != [STEM_KEY] or result.unexpected_keys:
            raise BES2DiagnosticError(
                "reset transfer must omit exactly stem.0.weight: "
                f"missing={missing}, unexpected={sorted(result.unexpected_keys)}"
            )
        if tensor_sha256(backbone.model.state_dict()[STEM_KEY]) != fresh_stem_hash:
            raise BES2DiagnosticError(
                "reset stem changed while loading post-stem S2 tensors"
            )
        return backbone, fresh_stem_hash, sorted(retained)
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
    if not torch.equal(production_state[STEM_KEY].cpu(), expected_current.cpu()):
        raise BES2DiagnosticError(
            "production BigEarthNet-S2 stem is not the exact current "
            "repeat-with-rescaling conversion"
        )

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
