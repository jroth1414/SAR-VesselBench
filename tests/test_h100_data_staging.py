from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.h100 import campaign, data_staging, source_validation
from scripts.handoff import runtime_amendment
from scripts import score_test_cohort
from src.eval.heldout_contract import HeldoutContractError
from src.eval.result_contract import sha256_file


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
        "sha1": _digest(path, "sha1"),
    }


def _write_splits(path: Path) -> dict[str, tuple[str, ...]]:
    scope = {
        "train": ("train-a",),
        "dev8": tuple(f"dev-{index}" for index in range(8)),
        "test": ("test-a",),
        "eval_final": ("final-a",),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": list(scope["train"]),
                    "dev": list(scope["dev8"]),
                    "test": list(scope["test"]),
                    "eval_final": list(scope["eval_final"]),
                }
            }
        ),
        encoding="utf-8",
    )
    return scope


def test_filtered_label_views_physically_exclude_forbidden_rows(tmp_path: Path) -> None:
    splits = tmp_path / "splits.json"
    scope = _write_splits(splits)
    source = tmp_path / "source.csv"
    source.write_text(
        "scene_id,value\n"
        + "".join(
            f"{scene_id},1\n"
            for scene_id in (
                *scope["train"],
                *scope["dev8"],
                *scope["test"],
                *scope["eval_final"],
            )
        ),
        encoding="utf-8",
    )

    training = tmp_path / "training.csv"
    data_staging.filter_training_labels(
        source, training, splits_path=splits, production=False
    )
    training_bytes = training.read_bytes()
    assert b"test-a" not in training_bytes
    assert b"final-a" not in training_bytes
    assert {row.split(",", 1)[0] for row in training.read_text().splitlines()[1:]} == (
        set(scope["train"]) | set(scope["dev8"])
    )

    scoring = tmp_path / "scoring.csv"
    data_staging.filter_score_labels(
        source, scoring, splits_path=splits, production=False
    )
    scoring_bytes = scoring.read_bytes()
    assert all(scene.encode() not in scoring_bytes for scene in scope["dev8"])
    assert b"final-a" not in scoring_bytes
    assert {row.split(",", 1)[0] for row in scoring.read_text().splitlines()[1:]} == (
        set(scope["train"]) | set(scope["test"])
    )


def test_training_campaign_stager_never_selects_test_labels_or_wheelhouse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    scope = _write_splits(repo / "data/splits.json")
    runs = tmp_path / "runs"
    runs.mkdir()
    base = tmp_path / "base"
    runtime = tmp_path / "runtime"
    base.mkdir()
    runtime.mkdir()
    weights = list(data_staging.EXPECTED_WEIGHT_DIRS)
    artifacts = [
        {"kind": "chip_scene", "name": "train-a"},
        {"kind": "chip_scene", "name": "test-a"},
        *(
            {"kind": "raster_scene", "name": scene_id}
            for scene_id in (*scope["dev8"], *scope["test"])
        ),
        *({"kind": "core_weight", "name": name} for name in weights),
        {"kind": "offline_environment", "name": "cp311-cu126"},
        {"kind": "labels", "name": "train.csv"},
    ]
    base_manifest = {"package_id": "base-id", "artifacts": artifacts}
    runtime_manifest = {
        "package_id": "runtime-id",
        "source": {"git_commit": "a" * 40},
        "artifacts": [],
    }
    (base / "manifest.json").write_text(json.dumps(base_manifest), encoding="utf-8")
    (runtime / "manifest.json").write_text(
        json.dumps(runtime_manifest), encoding="utf-8"
    )
    selected: list[tuple[str, str]] = []

    def fake_extract(_root: Path, artifact: dict, destination: Path) -> None:
        kind, name = str(artifact["kind"]), str(artifact["name"])
        selected.append((kind, name))
        if kind == "chip_scene":
            (destination / "data/chips" / name).mkdir(parents=True)
        elif kind == "raster_scene":
            (destination / "data/raw/xview3/GRD" / name).mkdir(parents=True)
        elif kind == "core_weight":
            (destination / "data/weights" / name).mkdir(parents=True)
        elif kind in {"labels", "offline_environment"}:
            raise AssertionError(f"pre-cohort campaign selected forbidden {kind}")

    def fake_labels(**kwargs: object) -> dict[str, object]:
        destination = Path(str(kwargs["staging"])) / data_staging.TRAINING_LABELS_EXPOSED_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("scene_id\ntrain-a\n", encoding="utf-8")
        return {"fixture": True}

    monkeypatch.setattr(data_staging, "split_scope", lambda *_a, **_k: scope)
    monkeypatch.setattr(
        data_staging, "_cohort_binding", lambda **_kwargs: ("train", None)
    )
    monkeypatch.setattr(data_staging, "_extract_artifact", fake_extract)
    monkeypatch.setattr(data_staging, "_runtime_labels_to_exposed", fake_labels)
    destination = tmp_path / "view"
    receipt = data_staging.stage_data_view(
        repo=repo,
        runs_root=runs,
        base_package_root=base,
        runtime_package_root=runtime,
        destination=destination,
        expected_git_sha="a" * 40,
        expected_base_package_id="base-id",
        expected_base_manifest_sha256=_digest(base / "manifest.json"),
        expected_runtime_package_id="runtime-id",
        expected_runtime_manifest_sha256=_digest(runtime / "manifest.json"),
        acceptance=False,
    )
    assert receipt["phase"] == "train"
    assert receipt["purpose"] == "campaign"
    assert ("chip_scene", "test-a") not in selected
    assert ("raster_scene", "test-a") not in selected
    assert not any(kind in {"labels", "offline_environment"} for kind, _ in selected)
    assert not (destination / "environment").exists()


def test_base_control_verifier_never_opens_data_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "base"
    bundle = root / "code/xview3.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"git-bundle-fixture")
    bundle_sha = _digest(bundle)
    bundle_sha1 = _digest(bundle, "sha1")
    forbidden = root / "data/rasters/test-a.tar.zst"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"held-out-raster")
    labels = root / "data/labels/train.csv.tar.zst"
    labels.parent.mkdir(parents=True)
    labels.write_bytes(b"combined-labels")
    manifest = {
        "package_id": "base-fixture",
        "package_type": "h100-source-handoff",
        "source": {"git_commit": "b" * 40, "environment_lock_sha256": "c" * 64},
        "artifacts": [
            {
                "kind": "git_bundle",
                "format": "file",
                "extraction_root": "code/xview3.bundle",
                "archive_sha256": bundle_sha,
                "parts": [
                    {
                        "path": "code/xview3.bundle",
                        "bytes": bundle.stat().st_size,
                        "sha256": bundle_sha,
                        "sha1": bundle_sha1,
                    }
                ],
            },
            {
                "kind": "raster_scene",
                "name": "test-a",
                "parts": [{"path": "data/rasters/test-a.tar.zst"}],
            },
            {
                "kind": "labels",
                "name": "train.csv",
                "parts": [{"path": "data/labels/train.csv.tar.zst"}],
            },
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sums_path = root / "SHA256SUMS"
    sums_path.write_text("transfer-verified-control\n", encoding="utf-8")
    ready = {
        "status": "READY",
        "package_id": "base-fixture",
        "git_commit": "b" * 40,
        "manifest": _record(manifest_path, root),
        "checksums": _record(sums_path, root),
    }
    (root / "READY.json").write_text(json.dumps(ready), encoding="utf-8")
    expected = {
        "schema": 1,
        "package_id": "base-fixture",
        "package_type": "h100-source-handoff",
        "source_git_commit": "b" * 40,
        "ready_sha256": _digest(root / "READY.json"),
        "manifest_sha256": _digest(manifest_path),
        "sha256sums_sha256": _digest(sums_path),
        "repo_bundle_sha256": bundle_sha,
        "environment_lock_sha256": "c" * 64,
    }
    monkeypatch.setattr(runtime_amendment, "BASE_REPO_BUNDLE_SHA256", bundle_sha)
    monkeypatch.setattr(runtime_amendment, "_expected_base_identity", lambda: expected)
    real_open = Path.open
    opened: list[str] = []

    def guarded_open(path: Path, *args: object, **kwargs: object):
        try:
            relative = path.absolute().relative_to(root.absolute()).as_posix()
        except ValueError:
            return real_open(path, *args, **kwargs)
        opened.append(relative)
        if relative.startswith("data/"):
            raise AssertionError(f"control verifier opened held-out data: {relative}")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert runtime_amendment.verify_base_payload_control(root) == expected
    assert set(opened) <= {
        "READY.json",
        "manifest.json",
        "SHA256SUMS",
        "code/xview3.bundle",
    }
    assert not any(path.startswith("data/") for path in opened)


def test_source_validation_never_dispatches_full_base_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_base = {
        "package_id": "base",
        "git_sha": "b" * 40,
        "manifest_sha256": "1" * 64,
        "ready_sha256": "2" * 64,
        "sha256sums_sha256": "3" * 64,
        "repo_bundle_sha256": "4" * 64,
    }
    expected_runtime = {
        "package_id": "runtime",
        "git_sha": "a" * 40,
        "manifest_sha256": "5" * 64,
        "ready_sha256": "6" * 64,
        "sha256sums_sha256": "7" * 64,
        "runtime_bundle_sha256": "8" * 64,
    }
    manifest = {
        "package_id": "runtime",
        "base_payload": {
            "package_id": "base",
            "source_git_commit": "b" * 40,
            "manifest_sha256": "1" * 64,
            "ready_sha256": "2" * 64,
            "sha256sums_sha256": "3" * 64,
            "repo_bundle_sha256": "4" * 64,
        },
        "source": {"git_commit": "a" * 40, "git_bundle_sha256": "8" * 64},
    }
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    for name, digest in (
        ("manifest.json", "5" * 64),
        ("READY.json", "6" * 64),
        ("SHA256SUMS", "7" * 64),
    ):
        (runtime_root / name).write_text(name, encoding="utf-8")
        monkeypatch.setattr(
            source_validation,
            "sha256_file",
            lambda path, mapping={
                "manifest.json": "5" * 64,
                "READY.json": "6" * 64,
                "SHA256SUMS": "7" * 64,
            }: mapping[path.name],
        )
    calls: list[Path] = []

    def control_verifier(base_root: Path):
        calls.append(base_root)
        return lambda package_root: manifest

    monkeypatch.setattr(
        runtime_amendment, "prepare_runtime_control_verifier", control_verifier
    )
    monkeypatch.setattr(
        runtime_amendment,
        "prepare_runtime_verifier",
        lambda _root: (_ for _ in ()).throw(AssertionError("full verifier called")),
    )
    base_root = tmp_path / "base"
    base_root.mkdir()
    assert source_validation.verify_transfer_bindings(
        base_payload_root=base_root,
        runtime_amendment_root=runtime_root,
        expected_base_payload=expected_base,
        expected_runtime_amendment=expected_runtime,
    ) == {
        "base_payload": expected_base,
        "runtime_amendment": expected_runtime,
    }
    assert calls == [base_root]


def test_normal_training_completion_requires_a_fresh_score_allocation() -> None:
    cell = SimpleNamespace(exp_id="cell")
    controller = campaign.Controller.__new__(campaign.Controller)
    controller.initialize = lambda: None
    controller.poll = lambda: None
    controller.preemption_seen = False
    controller.failure_seen = False
    controller.failure_allowed_ids = set()
    controller.failed_ids = set()
    controller.running = {}
    controller.cells = [cell]
    controller.complete_ids = {"cell"}
    controller.phase = "train"
    controller.args = SimpleNamespace(poll_seconds=0)
    events: list[str] = []
    controller.freeze_training_cohort = lambda: events.append("freeze")
    controller.write_manifest = lambda status=None: events.append(str(status))

    assert controller.run() == campaign.HOST_REQUEUE_EXIT_CODE
    assert events == ["freeze", "host-requeue-required"]


def test_checkpoint_tamper_fails_before_test_labels_or_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    checkpoint = tmp_path / "best.ckpt"
    torch.save({"epoch": 3}, checkpoint)
    binding = {
        "relative_path": "cell/checkpoints/best.ckpt",
        "sha256": sha256_file(checkpoint),
        "epoch": 3,
    }
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered-after-cohort")
    label_accesses: list[Path] = []

    def forbidden_label_access(path: Path, **_kwargs: object) -> dict[str, object]:
        label_accesses.append(path)
        raise AssertionError("TEST labels opened after checkpoint tamper")

    monkeypatch.setattr(data_staging, "score_labels_summary", forbidden_label_access)
    with pytest.raises(HeldoutContractError, match="drifted"):
        score_test_cohort.load_score_labels(
            checkpoint=checkpoint,
            checkpoint_binding=binding,
            labels_path=tmp_path / "test-labels.csv",
            splits_path=tmp_path / "splits.json",
        )
    assert label_accesses == []

    source = inspect.getsource(score_test_cohort.score_run)
    final_guard = source.rindex("validate_checkpoint_load_boundary(")
    model_load = source.index("HeatmapLitModule.load_from_checkpoint")
    assert final_guard < model_load
