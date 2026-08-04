from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts import diagnose_eval_contract as diagnostic
from src.eval.scorer import PredictionPoint
from src.references import locateanything_zs, yolo26_ref


def _write_diagnostic_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "data/raw/xview3/labels").mkdir(parents=True)
    (repo / "src/eval").mkdir(parents=True)
    runs = repo / "runs"

    dev = [f"dev-{index}" for index in range(8)]
    splits = {
        "splits": {
            "train": ["train-0"],
            "dev": list(reversed(dev)),
            "test": ["test-0"],
            "eval_final": ["final-0"],
        }
    }
    (repo / "data/splits.json").write_text(json.dumps(splits), newline="\n")
    (repo / "data/stats.json").write_text("{}\n")
    (repo / "configs/detector.yaml").write_text("frozen: true\n")
    (repo / "src/eval/scorer.py").write_text("# frozen fixture\n")
    arms = {
        "arms": {"vit_imagenet": {"short": "vitin1k"}},
        "label_fracs": [0.1],
        "seeds": {"core": [0]},
    }
    (repo / "configs/arms.yaml").write_text(yaml.safe_dump(arms), newline="\n")

    rows = []
    for index in range(517):
        rows.append(
            {
                "scene_id": dev[index % 8],
                "detect_scene_column": 10.0,
                "detect_scene_row": 20.0,
                "confidence": "MEDIUM" if index % 2 else "HIGH",
                "is_vessel": True,
                "source": "ais",
                "distance_from_shore_km": 9999.99,
            }
        )
    for index in range(107):
        rows.append(
            {
                "scene_id": dev[index % 8],
                "detect_scene_column": 1000.0,
                "detect_scene_row": 2000.0,
                "confidence": "HIGH",
                "is_vessel": False,
                "source": "ais",
                "distance_from_shore_km": 9999.99,
            }
        )
    for index in range(118):
        rows.append(
            {
                "scene_id": dev[index % 8],
                "detect_scene_column": 3000.0,
                "detect_scene_row": 4000.0,
                "confidence": "LOW",
                "is_vessel": None,
                "source": "ais",
                "distance_from_shore_km": 9999.99,
            }
        )
    pd.DataFrame(rows).to_csv(repo / "data/raw/xview3/labels/train.csv", index=False)

    run = runs / "vitin1k-f10-s0"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/best.ckpt").write_bytes(b"best-checkpoint")
    (run / "checkpoints/last.ckpt").write_bytes(b"last-checkpoint")
    (run / "final_metrics.json").write_bytes(b'{"legacy": true}\n')
    return repo, runs, sorted(dev)


def test_diagnostic_is_dev8_only_independent_and_canonical_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, runs, expected_scenes = _write_diagnostic_fixture(tmp_path)
    final_path = runs / "vitin1k-f10-s0/final_metrics.json"
    final_before = final_path.read_bytes()
    checkpoint_before = {
        name: (runs / f"vitin1k-f10-s0/checkpoints/{name}.ckpt").read_bytes()
        for name in diagnostic.CHECKPOINT_KINDS
    }

    read_paths = []
    real_read_csv = pd.read_csv

    def tracked_read_csv(path, *args, **kwargs):
        read_paths.append(Path(path))
        return real_read_csv(path, *args, **kwargs)

    infer_scenes = []

    def fake_loader(checkpoint: Path, device: str):
        assert device == "cpu"
        return checkpoint.stem

    def fake_infer(model, scene_dir: Path, **kwargs):
        assert kwargs["precision"] == "32-true"
        infer_scenes.append(scene_dir.name)
        score = 0.8 if model == "best" else 0.3
        return [
            PredictionPoint(
                x_m=100.0,
                y_m=200.0,
                score=score,
                distance_from_shore_km=9999.99,
            )
        ]

    monkeypatch.setattr(pd, "read_csv", tracked_read_csv)
    monkeypatch.setattr(diagnostic, "git_sha", lambda _repo: "a" * 40)
    output = diagnostic.run_diagnostic(
        repo=repo,
        runs_root=runs,
        data_cfg={
            "paths": {
                "raw_xview3": "data/raw/xview3",
                "splits": "data/splits.json",
                "stats": "data/stats.json",
            }
        },
        det_cfg={
            "decode": {"candidate_floor": 0.05, "d_nms_m": 120},
            "eval": {"tile_px": 512, "tile_stride_px": 384, "infer_batch": 8},
            "schedule": {"precision": "32-true"},
        },
        run_ids=["vitin1k-f10-s0"],
        checkpoint_kinds=["best", "last"],
        timestamp="20260804T120000Z",
        device="cpu",
        model_loader=fake_loader,
        epoch_reader=lambda checkpoint: 2 if checkpoint.stem == "best" else 4,
        infer_fn=fake_infer,
    )

    assert output == runs / "diagnostics/eval-contract-20260804T120000Z"
    assert read_paths == [repo / "data/raw/xview3/labels/train.csv"]
    assert infer_scenes == expected_scenes * 2
    assert not any(scene.startswith(("test-", "final-")) for scene in infer_scenes)
    payload = json.loads((output / "diagnostic.json").read_text())
    assert payload["diagnostic_schema"] == 1
    assert payload["reportable"] is False
    assert payload["scene_ids"] == expected_scenes
    assert payload["ground_truth_contract"]["counts"] == {
        "positive": 517,
        "background": 107,
        "ignore": 118,
    }
    assert [item["threshold"] for item in payload["checkpoints"]] == [0.8, 0.3]
    assert [
        item["checkpoint"]["epoch_zero_based"] for item in payload["checkpoints"]
    ] == [2, 4]
    assert all("per_scene" in item["metrics"] for item in payload["checkpoints"])
    assert all("near_shore" in item["metrics"]["slices"] for item in payload["checkpoints"])
    assert final_path.read_bytes() == final_before
    for name, content in checkpoint_before.items():
        assert (runs / f"vitin1k-f10-s0/checkpoints/{name}.ckpt").read_bytes() == content
    assert not list((runs / "diagnostics").glob(".eval-contract-*"))


def test_diagnostic_refuses_overlap_before_reading_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("scene_id\n")
    splits = tmp_path / "splits.json"
    dev = [f"dev-{index}" for index in range(8)]
    splits.write_text(
        json.dumps(
            {
                "splits": {
                    "train": ["train"],
                    "dev": dev,
                    "test": [dev[0]],
                    "eval_final": ["final"],
                }
            }
        )
    )
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *_args, **_kwargs: pytest.fail("overlap must fail before CSV access"),
    )
    with pytest.raises(RuntimeError, match="split overlap"):
        diagnostic.load_dev_contract(
            splits_path=splits, train_labels_csv=train_csv
        )


def test_diagnostic_rejects_non_core_ids_and_checkpoint_symlinks(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="unsafe or non-core"):
        diagnostic.validate_run_ids(["../victim"], {"vitin1k-f10-s0"})
    runs = tmp_path / "runs"
    checkpoint_dir = runs / "vitin1k-f10-s0/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"outside")
    (checkpoint_dir / "best.ckpt").symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink"):
        diagnostic.checkpoint_path(runs, "vitin1k-f10-s0", "best")


def test_r3_uses_shared_contract_and_retains_low_ignores() -> None:
    labels = [
        {
            "chip_col": 2.0,
            "chip_row": 3.0,
            "confidence": "HIGH",
            "is_vessel": True,
            "source": "manual",
            "distance_from_shore_km": 1.5,
        },
        {
            "chip_col": 4.0,
            "chip_row": 5.0,
            "confidence": "HIGH",
            "is_vessel": False,
            "source": "ais",
            "distance_from_shore_km": 8.0,
        },
        {
            "chip_col": 6.0,
            "chip_row": 7.0,
            "confidence": "LOW",
            "is_vessel": None,
            "source": "ais",
            "distance_from_shore_km": 9.0,
        },
    ]
    assert locateanything_zs.chip_label_counts(labels) == {
        "positive": 1,
        "background": 1,
        "ignore": 1,
    }
    points = locateanything_zs.chip_ground_truth_from_labels(labels)
    assert len(points) == 2
    assert (points[0].x_m, points[0].y_m, points[0].source) == (20.0, 30.0, "Manual")
    assert points[1].confidence == "LOW"
    assert points[1].is_low_confidence
    with pytest.raises(ValueError, match="is_vessel is missing"):
        locateanything_zs.chip_label_counts(
            [{"confidence": "HIGH", "is_vessel": None}]
        )


def test_r2_export_uses_shared_classifier_and_preserved_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = [
        {
            "chip_col": 10.0,
            "chip_row": 20.0,
            "confidence": "medium",
            "is_vessel": "true",
            "vessel_length_m": 80.0,
        },
        {
            "chip_col": 30.0,
            "chip_row": 40.0,
            "confidence": "HIGH",
            "is_vessel": "false",
        },
        {
            "chip_col": 50.0,
            "chip_row": 60.0,
            "confidence": "LOW",
            "is_vessel": None,
        },
    ]
    assert len(yolo26_ref.yolo_label_lines(labels, 800)) == 1
    assert yolo26_ref.EXPECTED_GT_COUNTS == {
        "dev": {"positive": 1479, "background": 804, "ignore": 441},
        "test": {"positive": 1165, "background": 420, "ignore": 325},
    }

    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "runs/yolo26-f100/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preserved-r2")
    monkeypatch.setattr(
        yolo26_ref,
        "EXPECTED_R2_BEST_SHA256",
        yolo26_ref._sha256_file(checkpoint),
    )
    selected, digest = yolo26_ref._known_r2_checkpoint(checkpoint)
    assert selected == checkpoint
    assert digest == yolo26_ref._sha256_file(checkpoint)
    alternate = tmp_path / "runs/yolo26-f100/weights/last.pt"
    alternate.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="preserved"):
        yolo26_ref._known_r2_checkpoint(alternate)


def test_reference_result_sources_are_schema2_and_atomic() -> None:
    assert yolo26_ref.RESULT_SCHEMA == 2
    assert locateanything_zs.RESULT_SCHEMA == 2
    r2_source = Path(yolo26_ref.__file__).read_text()
    r3_source = Path(locateanything_zs.__file__).read_text()
    assert '"training_disposition": "preserved-best-pt-rescore-only"' in r2_source
    assert '"eval_contract_disposition": "full-rerun-under-corrected-contract"' in (
        r3_source
    )
    assert '"legacy_result_reused": False' in r3_source
    assert "atomic_write_json(out, payload)" in r2_source
    assert "atomic_write_json(out, payload)" in r3_source
