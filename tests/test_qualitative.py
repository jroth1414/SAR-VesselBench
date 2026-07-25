from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.qualitative import _checkpoint_state, select_gallery_chip_paths


def test_checkpoint_state_accepts_weights_only_export():
    payload = {"backbone.weight": object(), "head.bias": object()}

    state, hparams = _checkpoint_state(payload)

    assert state is payload
    assert hparams == {}


def test_checkpoint_state_accepts_lightning_payload():
    state_dict = {"backbone.weight": object()}
    payload = {
        "state_dict": state_dict,
        "hyper_parameters": {"init_name": "vit_random"},
    }

    state, hparams = _checkpoint_state(payload)

    assert state is state_dict
    assert hparams["init_name"] == "vit_random"


def test_checkpoint_state_rejects_unrecognized_mapping():
    with pytest.raises(ValueError, match="neither a Lightning checkpoint"):
        _checkpoint_state({"optimizer_states": []})


def _write_chip(
    root: Path,
    scene: str,
    stem: str,
    *,
    confidence: str = "MEDIUM",
    row: float = 400.0,
    col: float = 400.0,
) -> Path:
    scene_dir = root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    chip_path = scene_dir / f"{stem}.npy"
    chip_path.write_bytes(b"fixture")
    sidecar = {
        "scene_id": scene,
        "chip_size": 800,
        "labels": [
            {
                "chip_row": row,
                "chip_col": col,
                "is_vessel": True,
                "confidence": confidence,
            }
        ],
    }
    chip_path.with_suffix(".json").write_text(
        json.dumps(sidecar),
        encoding="utf-8",
    )
    return chip_path


def test_gallery_selection_uses_available_split_scenes_and_center_positives(
    tmp_path: Path,
):
    chips = tmp_path / "chips"
    included = _write_chip(chips, "dev-scene", "included")
    _write_chip(chips, "dev-scene", "low", confidence="LOW")
    _write_chip(chips, "dev-scene", "outside", row=20.0, col=20.0)
    _write_chip(chips, "test-scene", "wrong-split")
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "splits": {
                    "train": [],
                    "dev": ["dev-scene", "missing-dev-scene"],
                    "test": ["test-scene"],
                    "eval_final": [],
                }
            }
        ),
        encoding="utf-8",
    )

    selected = select_gallery_chip_paths(
        chips,
        splits_path=splits,
        split="dev",
        count=16,
        seed=0,
    )

    assert selected == [included]


def test_gallery_selection_refuses_train_split(tmp_path: Path):
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"splits": {"train": [], "dev": [], "test": []}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="restricted to dev or test"):
        select_gallery_chip_paths(
            tmp_path,
            splits_path=splits,
            split="train",
            count=1,
            seed=0,
        )
