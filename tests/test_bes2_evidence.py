"""CPU tests for BES2 covariance, activation, drift, and data isolation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.analysis.bes2_contract import BES2ContractError  # noqa: E402
from src.analysis.bes2_evidence import (  # noqa: E402
    _covariance_evidence,
    activation_statistics,
    layer_drift,
    manifest_train_indices,
    validate_data_scope,
)


def test_covariance_records_exact_third_channel_redundancy():
    rng = np.random.default_rng(0)
    vh = rng.normal(size=512)
    vv = rng.normal(size=512)
    values = np.stack([vh, vv, vh - vv], axis=1)

    evidence = _covariance_evidence(values)

    assert evidence["effective_matrix_rank"] == 2
    assert evidence["raw_vh_minus_vv_formula"]["max_abs_residual"] == 0.0
    assert evidence["raw_vh_minus_vv_formula"]["rmse"] == 0.0
    assert evidence["centered_linear_redundancy"][
        "third_from_first_two_coefficients"
    ] == pytest.approx([1.0, -1.0])


def test_covariance_records_scale_aware_normalized_redundancy():
    rng = np.random.default_rng(1)
    vh = rng.normal(size=512)
    vv = rng.normal(size=512)
    values = np.stack(
        [
            (vh - 2.0) / 3.0,
            (vv + 1.0) / 4.0,
            ((vh - vv) - 3.0) / 5.0,
        ],
        axis=1,
    )

    evidence = _covariance_evidence(values)

    assert evidence["effective_matrix_rank"] == 2
    assert evidence["centered_linear_redundancy"][
        "third_from_first_two_coefficients"
    ] == pytest.approx([3.0 / 5.0, -4.0 / 5.0])
    assert evidence["centered_linear_redundancy"]["rmse"] < 1.0e-12
    assert evidence["raw_vh_minus_vv_formula"]["rmse"] > 0.0


class _TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
        )
        self.model.stages = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(4, 4, kernel_size=3, padding=1),
                    torch.nn.GELU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
                    torch.nn.GELU(),
                ),
            ]
        )

    def forward(self, value):
        value = self.model.stem(value)
        for stage in self.model.stages:
            value = stage(value)
        return value


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _TinyBackbone()
        self.head = torch.nn.Conv2d(8, 1, 1)

    def forward(self, value):
        return self.head(self.backbone(value))

    def _layer_decay_param_groups(self):
        stem = list(self.backbone.model.stem.parameters())
        stages = list(self.backbone.model.stages.parameters())
        head = list(self.head.parameters())
        return [
            {"params": stem, "lr_scale": 0.65},
            {"params": stages, "lr_scale": 0.8},
            {"params": head, "lr_scale": 1.0},
        ]


def test_activation_statistics_report_finite_stage_ranks():
    torch.manual_seed(0)
    model = _TinyModel()
    images = [
        np.random.default_rng(index).normal(size=(3, 32, 32)).astype(
            np.float32
        )
        for index in range(6)
    ]

    evidence = activation_statistics(
        model,
        images,
        device="cpu",
        batch_size=2,
    )

    assert set(evidence) == {
        "pre_stem",
        "post_first_convolution",
        "post_stem",
        "post_stage_0",
        "post_stage_1",
    }
    assert all(item["nonfinite_count"] == 0 for item in evidence.values())
    assert evidence["post_stage_1"]["channel_count"] == 8


def test_layer_drift_binds_first_convolution_and_lr_scale(tmp_path):
    torch.manual_seed(0)
    initial = _TinyModel()
    best = {
        name: value.detach().clone()
        for name, value in initial.state_dict().items()
    }
    best["backbone.model.stem.0.weight"] += 0.25
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"epoch": 4, "state_dict": best}, checkpoint)

    evidence = layer_drift(initial, checkpoint)

    first = evidence["first_convolution"]
    assert first["update_norm"] > 0.0
    assert first["lr_scale"] == 0.65
    assert evidence["groups"]["stem"]["update_norm"] > 0.0
    assert evidence["groups"]["stage_0"]["update_norm"] == 0.0


def _data_view(tmp_path: Path):
    repo = Path(__file__).resolve().parents[1]
    splits = json.loads((repo / "data/splits.json").read_text())["splits"]
    train = sorted(splits["train"])
    dev8 = sorted(splits["dev"])[:8]

    root = tmp_path / "view"
    chips = root / "chips"
    grd = root / "raw" / "GRD"
    labels = root / "raw" / "labels"
    chips.mkdir(parents=True)
    grd.mkdir(parents=True)
    labels.mkdir(parents=True)
    for scene in train:
        (chips / scene).mkdir()
    for scene in dev8:
        (grd / scene).mkdir()
    with (labels / "train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_id"])
        writer.writeheader()
        for scene in [*train, *dev8]:
            writer.writerow({"scene_id": scene})
    config = {
        "paths": {
            "chips": str(chips),
            "raw_xview3": str(root / "raw"),
            "splits": str(repo / "data/splits.json"),
            "stats": str(repo / "data/stats.json"),
        }
    }
    return repo, root, config, splits


def test_data_scope_accepts_train_dev8_and_rejects_test(tmp_path):
    repo, root, config, splits = _data_view(tmp_path)
    scope = validate_data_scope(
        repo=repo,
        data_view_root=root,
        data_config=config,
    )
    assert len(scope["train_scene_ids"]) == 111
    assert len(scope["dev_scene_ids"]) == 8

    (root / "raw" / "GRD" / sorted(splits["test"])[0]).mkdir()
    with pytest.raises(BES2ContractError, match="unexpected"):
        validate_data_scope(
            repo=repo,
            data_view_root=root,
            data_config=config,
        )


def test_manifest_batch_binding_skips_samples_outside_fraction(tmp_path):
    retained = []
    entries = []
    for index in range(16):
        path = tmp_path / "chips" / f"scene-{index}" / f"chip-{index}.npy"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fixture")
        retained.append(path)
        entries.append(
            {
                "kind": "chip",
                "path": f"chips/scene-{index}/chip-{index}.npy",
            }
        )
    entries.append(
        {
            "kind": "chip",
            "path": "chips/scene-outside-f50/missing.npy",
        }
    )
    dataset = type("Dataset", (), {"chip_paths": retained})()

    assert manifest_train_indices({"entries": entries}, dataset) == list(
        range(16)
    )
