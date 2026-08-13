"""Eight backbone initializations behind one enum.

Within a track all four loaders produce the IDENTICAL architecture; only the
weights differ. Each loader prints matched/missing/unexpected key counts and
refuses to load a downloaded checkpoint whose ``LICENSE.note`` is missing
(the P3.2 license gate).

Key-mapping functions are pure (they map source key names to timm names)
so the CPU-offline half of ``test_fm_checkpoints_load`` can verify coverage
against committed key manifests without downloading anything.

Checkpoint facts (verified at download, revisions in data/weights/*/SOURCE.note):
- SatDINO ``satdino-vit_base-16.pth``: ``ckpt['teacher']`` in timm ViT naming
  plus one extra ``gsd_register`` (SatDINO's GSD embedding — unused here).
- SARMAE ``SARMAE_vitb_checkpoint-last``: ``ckpt['model']`` with
  ``sar_encoder.*`` (MAE encoder + decoder), ``optical_encoder.*`` and
  ``sar_alignment_ffn.*`` branches; we keep the sar_encoder ENCODER only.
- BigEarthNet ``model.safetensors``: timm ConvNeXt-V2-B under a
  ``model.vision_encoder.`` prefix with a 19-class head (dropped). Stems are
  2-band (S1: VV,VH) / 10-band (S2), adapted to the fixed 3-channel input by
  Repeat-with-rescaling.
- ImageNet ViT ``timm/vit_base_patch16_224.augreg_in1k`` and ImageNet CNN
  ``timm/convnextv2_base.fcmae_ft_in1k``: canonical timm safetensors; only
  the classification heads are dropped. The former is supervised AugReg;
  the latter is FCMAE followed by supervised ImageNet-1K fine-tuning.

Channel adaptation note (flagged for review): timm's
``adapt_input_conv`` implements Repeat-with-rescaling only FROM 3-channel
sources; the BigEarthNet stems need 2->3 and 10->3, so
``repeat_with_rescaling`` below generalizes the identical tile-and-rescale
math (cyclic tile to the target count, scale by src/dst to preserve
activation magnitude). No other patch-embed/stem surgery anywhere
(ground rule 12).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Iterable, Mapping

import torch

from src.models.backbones import ConvNeXtBackbone, ViTBackbone, build_backbone

INIT_FAMILY: dict[str, str] = {
    "vit_random": "vit",
    "satdino_b": "vit",
    "sarmae_b": "vit",
    "vit_imagenet": "vit",
    "cnn_random": "cnn",
    "bigearthnet_s2": "cnn",
    "bigearthnet_s1": "cnn",
    "cnn_imagenet": "cnn",
}
INIT_NAMES = tuple(INIT_FAMILY)


@dataclass(frozen=True)
class PinnedImageNetCheckpoint:
    """Local artifact identity for a core ImageNet initialization."""

    weights_subdir: str
    filename: str
    source: str


IMAGENET_CHECKPOINTS: Final[dict[str, PinnedImageNetCheckpoint]] = {
    "vit_imagenet": PinnedImageNetCheckpoint(
        weights_subdir="imagenet_vit_augreg_in1k",
        filename="model.safetensors",
        source=(
            "timm/vit_base_patch16_224.augreg_in1k"
            "@458542882691a06a8b667c6fb5fe5c9573093a81"
        ),
    ),
    "cnn_imagenet": PinnedImageNetCheckpoint(
        weights_subdir="imagenet_cnn_fcmae_ft_in1k",
        filename="model.safetensors",
        source=(
            "timm/convnextv2_base.fcmae_ft_in1k"
            "@7b29800e499fdc06de5b612970f3384dc8d29ca5"
        ),
    ),
}

DOWNLOADED_WEIGHTS_SUBDIR: dict[str, str] = {
    "satdino_b": "satdino",
    "sarmae_b": "sarmae",
    "bigearthnet_s2": "bigearthnet_s2",
    "bigearthnet_s1": "bigearthnet_s1",
    "vit_imagenet": IMAGENET_CHECKPOINTS["vit_imagenet"].weights_subdir,
    "cnn_imagenet": IMAGENET_CHECKPOINTS["cnn_imagenet"].weights_subdir,
}


def map_satdino_keys(keys: Iterable[str]) -> dict[str, str]:
    """SatDINO teacher keys -> timm ViT keys (drop the GSD register)."""

    return {
        key: key
        for key in keys
        if key != "gsd_register" and not key.startswith("head.")
    }


def map_sarmae_keys(keys: Iterable[str]) -> dict[str, str]:
    """SARMAE model keys -> timm ViT keys: sar_encoder ENCODER only."""

    mapping: dict[str, str] = {}
    for key in keys:
        if not key.startswith("sar_encoder."):
            continue  # optical_encoder.*, sar_alignment_ffn.* — other branches
        stripped = key.removeprefix("sar_encoder.")
        if stripped.startswith("decoder_") or stripped == "mask_token":
            continue  # MAE decoder — not part of the transferred encoder
        mapping[key] = stripped
    return mapping


def map_bigearthnet_keys(keys: Iterable[str]) -> dict[str, str]:
    """BigEarthNet classifier keys -> plain timm ConvNeXt-V2 keys.

    Strip the ``model.vision_encoder.`` prefix; names then map 1:1 onto the
    plain timm model (the backbone is deliberately NOT the features_only
    wrapper — see ConvNeXtBackbone). Only the 19-class classifier
    (``head.fc``) is dropped; ``head.norm`` maps through (present in the
    target's state dict, unused by ``forward_features``).
    """

    mapping: dict[str, str] = {}
    for key in keys:
        if not key.startswith("model.vision_encoder."):
            continue
        stripped = key.removeprefix("model.vision_encoder.")
        if stripped.startswith("head.fc."):
            continue  # 19-class BigEarthNet classifier — dropped
        mapping[key] = stripped
    return mapping


def map_imagenet_vit_keys(keys: Iterable[str]) -> dict[str, str]:
    """AugReg ImageNet-1K timm ViT keys -> headless timm ViT keys."""

    return {key: key for key in keys if not key.startswith("head.")}


def map_imagenet_cnn_keys(keys: Iterable[str]) -> dict[str, str]:
    """FCMAE->IN1K ConvNeXt-V2 keys -> canonical `num_classes=0` target.

    The class-logit projection (`head.fc`) is dropped. Canonical timm
    ConvNeXt with `num_classes=0` retains `head.norm` in its state dict,
    so those two normalization tensors map through for full target coverage;
    they are unused because :class:`ConvNeXtBackbone` calls
    `forward_features` rather than `forward_head`.
    """

    return {key: key for key in keys if not key.startswith("head.fc.")}


def repeat_with_rescaling(weight: torch.Tensor, target_chans: int) -> torch.Tensor:
    """Adapt a conv stem weight (O, I, kH, kW) to ``target_chans`` inputs.

    The same tile-and-rescale rule timm's ``adapt_input_conv`` applies from
    3-channel sources, generalized to any source count: tile the source
    channels cyclically to the target count, then scale by src/dst so the
    expected activation magnitude is preserved.
    """

    out_ch, in_ch, k_h, k_w = weight.shape
    if in_ch == target_chans:
        return weight
    repeats = -(-target_chans // in_ch)  # ceil
    tiled = weight.repeat(1, repeats, 1, 1)[:, :target_chans]
    return tiled * (in_ch / float(target_chans))


def _load_mapped(
    backbone_module: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
    key_map: Mapping[str, str],
    *,
    name: str,
    stem_key: str | None = None,
) -> None:
    """Load mapped weights, adapt the stem if asked, and report coverage."""

    target = backbone_module.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for src_key, dst_key in key_map.items():
        tensor = source_state[src_key]
        if dst_key == stem_key and tensor.shape[1] != target[dst_key].shape[1]:
            tensor = repeat_with_rescaling(tensor, target[dst_key].shape[1])
        mapped[dst_key] = tensor

    missing = [key for key in target if key not in mapped]
    unexpected = [key for key in mapped if key not in target]
    result = backbone_module.load_state_dict(
        {key: value for key, value in mapped.items() if key in target},
        strict=False,
    )
    print(
        f"[{name}] matched={len(mapped) - len(unexpected)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        raise RuntimeError(
            f"{name}: {len(missing)} backbone keys not covered by the "
            f"checkpoint (first: {missing[:5]}) — refusing a partial load "
            "(loading guard: no silent random weights)"
        )
    if result.unexpected_keys:
        raise RuntimeError(f"{name}: unexpected keys slipped through: {result.unexpected_keys[:5]}")


def _require_license(weights_dir: Path, name: str) -> None:
    note = weights_dir / "LICENSE.note"
    if not note.exists():
        raise FileNotFoundError(
            f"{name}: {note} is missing — the P3.2 license gate refuses to "
            "load a downloaded backbone without its recorded license"
        )


def _require_pinned_imagenet_checkpoint(name: str, weights_dir: Path) -> Path:
    """Return the expected ImageNet artifact path or fail before deserialization.

    Identity relies on the revision-pinned download source recorded in
    ``IMAGENET_CHECKPOINTS`` and the value-sensitive structural checks in
    ``tests/test_fm_checkpoints_load.py``; the loader does not hash the file.
    """

    metadata = IMAGENET_CHECKPOINTS[name]
    checkpoint_path = weights_dir / metadata.filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"{name}: pinned checkpoint is missing: {checkpoint_path} "
            f"(expected {metadata.source})"
        )
    return checkpoint_path


def build_init(
    name: str,
    *,
    load_weights: bool = True,
    weights_root: str | Path = "data/weights",
) -> ViTBackbone | ConvNeXtBackbone:
    """Build the backbone for one arm's initialization.

    ``load_weights=False`` returns the bare architecture (used by the parity
    guard and CI, where no checkpoints exist).
    """

    if name not in INIT_FAMILY:
        raise ValueError(f"unknown init {name!r}; use one of {INIT_NAMES}")
    backbone = build_backbone(INIT_FAMILY[name])
    if not load_weights or name in ("vit_random", "cnn_random"):
        return backbone

    weights_root = Path(weights_root)
    if name in DOWNLOADED_WEIGHTS_SUBDIR:
        weights_dir = weights_root / DOWNLOADED_WEIGHTS_SUBDIR[name]
        _require_license(weights_dir, name)

    if name == "satdino_b":
        ckpt = dict(
            torch.load(
                weights_dir / "satdino-vit_base-16.pth",
                map_location="cpu",
                weights_only=False,
            )["teacher"]
        )
        # SatDINO pos_embed layout is [cls, patches..., gsd_register] (see
        # modeling_satdino.py interpolate_pos_encoding); timm expects
        # [cls, patches...], so drop the trailing register position.
        ckpt["pos_embed"] = ckpt["pos_embed"][:, :-1]
        _load_mapped(backbone.model, ckpt, map_satdino_keys(ckpt.keys()), name=name)
    elif name == "sarmae_b":
        ckpt = torch.load(
            weights_dir / "SARMAE_vitb_checkpoint-last",
            map_location="cpu",
            weights_only=False,
        )["model"]
        _load_mapped(backbone.model, ckpt, map_sarmae_keys(ckpt.keys()), name=name)
    elif name in ("bigearthnet_s1", "bigearthnet_s2"):
        from safetensors.torch import load_file

        state = load_file(weights_dir / "model.safetensors")
        _load_mapped(
            backbone.model,
            state,
            map_bigearthnet_keys(state.keys()),
            name=name,
            stem_key="stem.0.weight",
        )
    elif name in ("vit_imagenet", "cnn_imagenet"):
        from safetensors.torch import load_file

        checkpoint_path = _require_pinned_imagenet_checkpoint(name, weights_dir)
        state = load_file(checkpoint_path)
        mapper = (
            map_imagenet_vit_keys if name == "vit_imagenet" else map_imagenet_cnn_keys
        )
        _load_mapped(backbone.model, state, mapper(state.keys()), name=name)
    return backbone
