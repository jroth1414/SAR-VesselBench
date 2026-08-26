"""Isolation and exactness guards for the BigEarthNet-S2 diagnostic variants."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src.models import bes2_diagnostic as diagnostic  # noqa: E402
from src.models.heatmap_head import HeatmapHead  # noqa: E402
from src.models.init_loaders import INIT_FAMILY, repeat_with_rescaling  # noqa: E402


class _TinyCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, kernel_size=4, stride=4),
            torch.nn.GroupNorm(32, 128),
        )
        self.stages = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
        )


class _TinyBackbone(torch.nn.Module):
    out_channels = 128
    out_stride = 8

    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyCore()

    def forward(self, value):
        return self.model.stages(self.model.stem(value))


def _source_for(target: _TinyBackbone) -> dict[str, torch.Tensor]:
    source: dict[str, torch.Tensor] = {}
    for index, (name, value) in enumerate(target.model.state_dict().items(), start=1):
        source_name = f"model.vision_encoder.{name}"
        if name == diagnostic.STEM_KEY:
            source[source_name] = torch.full(
                (128, 10, 4, 4),
                index / 100.0,
                dtype=value.dtype,
            )
        else:
            source[source_name] = torch.full_like(value, index / 100.0)
    source["model.vision_encoder.head.fc.weight"] = torch.ones(2, 128)
    source["model.vision_encoder.head.fc.bias"] = torch.ones(2)
    return source


@pytest.fixture
def tiny_loader(monkeypatch, tmp_path):
    checkpoint = tmp_path / "bigearthnet_s2" / "model.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture")
    source_holder: dict[str, dict[str, torch.Tensor]] = {}

    def fake_build_init(name, *, load_weights=True, weights_root=None):
        backbone = _TinyBackbone()
        if name == "bigearthnet_s2":
            assert load_weights is True
            source = _source_for(backbone)
            source_holder["state"] = source
            mapped = {}
            target = backbone.model.state_dict()
            for source_name, value in source.items():
                if source_name.startswith("model.vision_encoder.head.fc."):
                    continue
                target_name = source_name.removeprefix("model.vision_encoder.")
                if target_name == diagnostic.STEM_KEY:
                    value = repeat_with_rescaling(value, 3)
                mapped[target_name] = value
            assert set(mapped) == set(target)
            backbone.model.load_state_dict(mapped, strict=True)
            return backbone
        assert name == "cnn_random"
        assert load_weights is False
        return backbone

    def fake_source_state(weights_root):
        return checkpoint, source_holder["state"]

    monkeypatch.setattr(diagnostic, "build_init", fake_build_init)
    monkeypatch.setattr(diagnostic, "_source_state", fake_source_state)
    monkeypatch.setattr(
        diagnostic,
        "sha256_file",
        lambda path: diagnostic.SOURCE_CHECKPOINT_SHA256,
    )
    return tmp_path


def _build_pair(weights_root: Path):
    torch.manual_seed(0)
    replay, replay_manifest = diagnostic.build_bes2_diagnostic_backbone(
        "current_replay", weights_root=weights_root
    )
    replay_head = HeatmapHead(
        replay.out_channels, replay.out_stride, head_channels=32
    )
    replay_head_hash = diagnostic.state_identity(
        diagnostic.state_tensor_hashes(replay_head)
    )

    torch.manual_seed(0)
    reset, reset_manifest = diagnostic.build_bes2_diagnostic_backbone(
        "first_conv_reset", weights_root=weights_root
    )
    reset_head = HeatmapHead(
        reset.out_channels, reset.out_stride, head_channels=32
    )
    reset_head_hash = diagnostic.state_identity(
        diagnostic.state_tensor_hashes(reset_head)
    )
    return (
        replay,
        replay_manifest,
        replay_head,
        replay_head_hash,
        reset,
        reset_manifest,
        reset_head,
        reset_head_hash,
    )


def test_replay_is_production_loader_identity(tiny_loader):
    torch.manual_seed(0)
    expected = diagnostic.build_init(
        "bigearthnet_s2", load_weights=True, weights_root=tiny_loader
    )
    expected_hashes = diagnostic.state_tensor_hashes(expected.model)

    torch.manual_seed(0)
    replay, manifest = diagnostic.build_bes2_diagnostic_backbone(
        "current_replay", weights_root=tiny_loader
    )

    assert diagnostic.state_tensor_hashes(replay.model) == expected_hashes
    assert manifest["changed_from_production_replay"] == []
    assert manifest["excluded_target_keys"] == []
    assert manifest["loaded_target_keys"] == sorted(expected_hashes)


def test_reset_omits_exactly_first_convolution_and_preserves_head(tiny_loader):
    (
        replay,
        replay_manifest,
        replay_head,
        replay_head_hash,
        reset,
        reset_manifest,
        reset_head,
        reset_head_hash,
    ) = _build_pair(tiny_loader)

    replay_hashes = diagnostic.state_tensor_hashes(replay.model)
    reset_hashes = diagnostic.state_tensor_hashes(reset.model)
    changed = sorted(
        key for key in replay_hashes if replay_hashes[key] != reset_hashes[key]
    )
    assert changed == [diagnostic.STEM_KEY]
    assert reset_manifest["changed_from_production_replay"] == [
        diagnostic.STEM_KEY
    ]
    assert reset_manifest["excluded_target_keys"] == [diagnostic.STEM_KEY]
    assert reset_manifest["retained_seeded_target_keys"] == [
        diagnostic.STEM_KEY
    ]
    assert diagnostic.STEM_KEY not in reset_manifest["loaded_target_keys"]
    assert set(reset_manifest["loaded_target_keys"]) == (
        set(reset_hashes) - {diagnostic.STEM_KEY}
    )
    assert reset_manifest["seeded_target_stem_sha256"] == reset_hashes[
        diagnostic.STEM_KEY
    ]
    assert replay_manifest["production_stem_sha256"] == replay_hashes[
        diagnostic.STEM_KEY
    ]
    assert replay_head_hash == reset_head_hash
    assert diagnostic.state_tensor_hashes(replay_head) == (
        diagnostic.state_tensor_hashes(reset_head)
    )


def test_reset_stem_matches_fresh_seed_zero_target_byte_for_byte(tiny_loader):
    torch.manual_seed(0)
    fresh = diagnostic.build_init(
        "cnn_random", load_weights=False, weights_root=tiny_loader
    )
    fresh_stem = fresh.model.state_dict()[diagnostic.STEM_KEY].detach().clone()

    torch.manual_seed(0)
    reset, manifest = diagnostic.build_bes2_diagnostic_backbone(
        "first_conv_reset", weights_root=tiny_loader
    )
    reset_stem = reset.model.state_dict()[diagnostic.STEM_KEY]

    assert torch.equal(reset_stem, fresh_stem)
    assert manifest["active_stem_sha256"] == diagnostic.tensor_sha256(fresh_stem)


def test_variants_preserve_parameter_count_and_feature_geometry(tiny_loader):
    (
        replay,
        replay_manifest,
        replay_head,
        _,
        reset,
        reset_manifest,
        reset_head,
        _,
    ) = _build_pair(tiny_loader)

    assert replay_manifest["parameter_count"] == reset_manifest["parameter_count"]
    assert replay.out_channels == reset.out_channels
    assert replay.out_stride == reset.out_stride

    value = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        replay_logits = replay_head(replay(value))
        reset_logits = reset_head(reset(value))
    assert replay_logits.shape == reset_logits.shape == (1, 1, 16, 16)


def test_diagnostic_names_do_not_enter_core_initialization_registry():
    assert tuple(INIT_FAMILY) == (
        "vit_random",
        "satdino_b",
        "sarmae_b",
        "vit_imagenet",
        "cnn_random",
        "bigearthnet_s2",
        "bigearthnet_s1",
        "cnn_imagenet",
    )
    assert not set(diagnostic.DIAGNOSTIC_VARIANTS) & set(INIT_FAMILY)
