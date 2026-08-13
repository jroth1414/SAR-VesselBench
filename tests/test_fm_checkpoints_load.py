"""Guard test: downloaded backbones load value-sensitively.

Two halves:

(a) CPU-offline STRUCTURAL half (runs in CI, no downloads): committed key
    manifests (``tests/manifests/*.json``, recorded at download from the
    pinned revisions) are pushed through the loader key-mapping functions and
    must cover the timm target architecture's state dict exactly — missing==0
    proves the mapping yields a full, non-partial load.

(b) GPU-box VALUE-SENSITIVE half (skips when ``data/weights`` is absent):
    actually loads each checkpoint and asserts a fixed probe of encoder
    tensors differs from a fresh trunc-normal init by a minimum L2 — the
    anti-"arm quietly running on random weights" check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from src.models.init_loaders import (  # noqa: E402
    IMAGENET_CHECKPOINTS,
    _require_pinned_imagenet_checkpoint,
    build_init,
    map_bigearthnet_keys,
    map_sarmae_keys,
    map_imagenet_cnn_keys,
    map_imagenet_vit_keys,
    map_satdino_keys,
)

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "data" / "weights"

STRUCTURAL_CASES = (
    # (manifest, mapper, timm target kwargs, family)
    ("satdino.json", map_satdino_keys, "vit"),
    ("sarmae.json", map_sarmae_keys, "vit"),
    ("bigearthnet_s1.json", map_bigearthnet_keys, "cnn"),
    ("bigearthnet_s2.json", map_bigearthnet_keys, "cnn"),
    ("imagenet_vit.json", map_imagenet_vit_keys, "vit"),
    ("imagenet_cnn.json", map_imagenet_cnn_keys, "cnn"),
)

IMAGENET_CASES = (
    (
        "vit_imagenet",
        "imagenet_vit.json",
        map_imagenet_vit_keys,
        {"head.weight", "head.bias"},
    ),
    (
        "cnn_imagenet",
        "imagenet_cnn.json",
        map_imagenet_cnn_keys,
        {"head.fc.weight", "head.fc.bias"},
    ),
)

VALUE_PROBE_SEED = 0x5EED
VALUE_PROBE_MIN_L2 = 1.0


@pytest.mark.parametrize("manifest_name,mapper,family", STRUCTURAL_CASES)
def test_structural_key_coverage(manifest_name, mapper, family):
    """(a) manifest keys, mapped, must cover the target architecture exactly."""

    manifest = json.loads((MANIFEST_DIR / manifest_name).read_text())
    mapped_targets = set(mapper(manifest["keys"]).values())

    from src.models.backbones import build_backbone

    target_keys = set(build_backbone(family).model.state_dict().keys())

    missing = target_keys - mapped_targets
    assert not missing, (
        f"{manifest_name}: {len(missing)} target keys not covered "
        f"(first: {sorted(missing)[:5]}) — a partial load would leave these "
        "silently random"
    )
    extra = mapped_targets - target_keys
    assert not extra, f"{manifest_name}: mapped keys not in target: {sorted(extra)[:5]}"


@pytest.mark.parametrize("name,manifest_name,mapper,expected_excluded", IMAGENET_CASES)
def test_imagenet_manifest_metadata_and_head_exclusions(
    name, manifest_name, mapper, expected_excluded
):
    """Manifest identity and exclusions must match the loader's pinned metadata."""

    manifest = json.loads((MANIFEST_DIR / manifest_name).read_text())
    pinned = IMAGENET_CHECKPOINTS[name]
    assert manifest["source"] == pinned.source
    assert manifest["file"] == pinned.filename
    assert manifest["sha256"] == pinned.sha256

    mapping = mapper(manifest["keys"])
    excluded = set(manifest["keys"]) - set(mapping)
    assert excluded == expected_excluded
    mapped_targets = set(mapping.values())
    assert not ({"head.weight", "head.bias"} & mapped_targets)
    assert not any(key.startswith("head.fc.") for key in mapped_targets)
    if name == "cnn_imagenet":
        # Canonical timm num_classes=0 retains this norm, but our backbone
        # calls forward_features and never consumes the classifier head.
        assert {"head.norm.weight", "head.norm.bias"} <= mapped_targets


@pytest.mark.parametrize("name", tuple(IMAGENET_CHECKPOINTS))
def test_imagenet_hash_guard_rejects_substitution(name, tmp_path):
    """A same-named but non-pinned local artifact must fail before loading."""

    pinned = IMAGENET_CHECKPOINTS[name]
    weights_dir = tmp_path / pinned.weights_subdir
    weights_dir.mkdir()
    (weights_dir / pinned.filename).write_bytes(b"not the pinned checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _require_pinned_imagenet_checkpoint(name, weights_dir)


DOWNLOADED = (
    "satdino_b",
    "sarmae_b",
    "bigearthnet_s1",
    "bigearthnet_s2",
    "vit_imagenet",
    "cnn_imagenet",
)


@pytest.mark.parametrize("name", DOWNLOADED)
@pytest.mark.skipif(
    not WEIGHTS_DIR.exists(),
    reason="data/weights absent — value-sensitive half runs on the GPU boxes",
)
def test_value_sensitive_load(name):
    """(b) loaded encoder tensors must differ from a fresh random init."""

    # Reset to the identical initialization before both constructions. Without
    # this, random-vs-random easily clears the L2 threshold and makes the guard
    # a false positive even if no checkpoint tensor was loaded.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(VALUE_PROBE_SEED)
        loaded = build_init(name, weights_root=WEIGHTS_DIR).state_dict()
        torch.manual_seed(VALUE_PROBE_SEED)
        fresh = build_init(name, load_weights=False).state_dict()

    probe = [
        key
        for key in loaded
        if "blocks.0." in key or "stages_0.blocks.0" in key or "stem_0" in key
    ][:6]
    assert probe, f"{name}: no probe keys found"
    l2 = sum(float((loaded[k].float() - fresh[k].float()).norm()) for k in probe)
    assert l2 > VALUE_PROBE_MIN_L2, (
        f"{name}: probe L2 vs fresh init is {l2:.3f} — loaded weights look "
        "random (partial/failed load)"
    )
