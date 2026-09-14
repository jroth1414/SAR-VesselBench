"""Owner-approved S2 post-stem transfer (DEVPLAN amendment 2026-09-14).

Retain the seeded target weight, as in timm 1.0.27 _builder.load_pretrained's
NotImplementedError fallback. timm does not automatically take that branch
for in_chans=3; this explicit S2-only policy adopts its weight-drop semantics.
The source identity/checks and tensor hashing originate in the accepted
Sprint-10 diagnostic. No channel conversion is performed here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Mapping

import torch

STEM_KEY: Final = "stem.0.weight"
SOURCE_STEM_KEY: Final = "model.vision_encoder.stem.0.weight"
SOURCE_CHECKPOINT_SHA256: Final = (
    "b09d0e41cc683878243a9128a6f4724d6a71d562318beeae716f0dce9cbbf454"
)


class BES2TransferError(RuntimeError):
    """An S2 transfer violated the approved exact single-weight contract."""


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


def source_state(
    weights_root: str | Path,
) -> tuple[Path, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file

    checkpoint = Path(weights_root) / "bigearthnet_s2" / "model.safetensors"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise BES2TransferError(
            f"BigEarthNet-S2 checkpoint must be a regular non-symlink file: {checkpoint}"
        )
    observed = sha256_file(checkpoint)
    if observed != SOURCE_CHECKPOINT_SHA256:
        raise BES2TransferError(
            "BigEarthNet-S2 checkpoint SHA-256 mismatch: "
            f"expected {SOURCE_CHECKPOINT_SHA256}, got {observed}"
        )
    state = dict(load_file(checkpoint, device="cpu"))
    if SOURCE_STEM_KEY not in state:
        raise BES2TransferError(
            f"BigEarthNet-S2 checkpoint lacks {SOURCE_STEM_KEY}"
        )
    stem = state[SOURCE_STEM_KEY]
    if tuple(stem.shape) != (128, 10, 4, 4):
        raise BES2TransferError(
            f"unexpected BigEarthNet-S2 source stem shape: {tuple(stem.shape)}"
        )
    return checkpoint, state


STEM_POLICY: Final = "timm-unsupported-input-conv-fallback-seeded-target-weight"


def load_post_stem(
    model: torch.nn.Module,
    source: Mapping[str, torch.Tensor],
    mapping: Mapping[str, str],
) -> tuple[str, list[str]]:
    """Keep exactly the fresh target stem weight; load and verify everything else."""

    target = model.state_dict()
    mapped = list(mapping.values())
    if len(mapped) != len(set(mapped)):
        raise BES2TransferError("BigEarthNet-S2 key mapping is not one-to-one")
    missing = sorted(set(target) - set(mapped))
    unexpected = sorted(set(mapped) - set(target))
    if missing or unexpected:
        raise BES2TransferError(
            f"downloaded S2 mapping no longer covers the target exactly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if (
        mapping.get(SOURCE_STEM_KEY) != STEM_KEY
        or tuple(source[SOURCE_STEM_KEY].shape) != (128, 10, 4, 4)
        or tuple(target[STEM_KEY].shape) != (128, 3, 4, 4)
    ):
        raise BES2TransferError("S2 fallback requires the official 10-band/3-band stem pair")
    fresh_stem_hash = tensor_sha256(target[STEM_KEY])
    retained = {}
    for source_key, target_key in mapping.items():
        if target_key == STEM_KEY:
            continue
        value = source[source_key]
        if (
            value.shape != target[target_key].shape
            or value.dtype != target[target_key].dtype
        ):
            raise BES2TransferError(f"downloaded tensor {source_key} cannot load as {target_key}")
        retained[target_key] = value
    result = model.load_state_dict(retained, strict=False)
    if sorted(result.missing_keys) != [STEM_KEY] or result.unexpected_keys:
        raise BES2TransferError(
            f"reset transfer must omit exactly {STEM_KEY}: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    loaded = model.state_dict()
    if tensor_sha256(loaded[STEM_KEY]) != fresh_stem_hash:
        raise BES2TransferError("reset stem changed while loading post-stem S2 tensors")
    for key, value in retained.items():
        if not torch.equal(loaded[key], value):
            raise BES2TransferError(f"post-stem S2 tensor was not loaded exactly: {key}")
    return fresh_stem_hash, sorted(retained)
