"""Fail-closed public H100 result snapshot and paper-generation tests."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.analysis.h100_results import (
    FROZEN_PATHS,
    STRICT_BACKEND,
    SnapshotError,
    expected_cells,
    generate,
    import_complete_results,
    import_deadline_results,
    load_verified_references,
    load_snapshot,
    render_progress_log,
)


REPO = Path(__file__).resolve().parents[1]
ARMS = REPO / "configs/arms.yaml"
COMMITTED = REPO / "results/h100/h100_campaign_snapshot.json"
HEX = "a" * 64


def _read() -> dict:
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _deadline(tmp_path: Path, mutate=None):
    payload = _read()
    if mutate is not None:
        mutate(payload)
    path = _write(tmp_path / "snapshot.json", payload)
    return load_snapshot(path, ARMS)


def _verify(cell: dict, hardware_sha: str, *, f1: float = 0.8) -> None:
    cell.update(
        {
            "status": "DONE",
            "latest_epoch": 9,
            "duration_seconds": 100,
            "train_loss": 0.1,
            "development": {
                "f1": f1,
                "precision": f1,
                "recall": f1,
                "threshold": 0.5,
            },
            "evidence": {
                "verification": "contract-verified",
                "completion_marker_sha256": HEX,
                "runtime_provenance_sha256": HEX,
                "best_checkpoint_sha256": HEX,
                "last_checkpoint_sha256": HEX,
                "hardware_class_sha256": hardware_sha,
            },
        }
    )


def test_committed_deadline_snapshot_is_status_only_and_exact() -> None:
    snapshot = load_snapshot(COMMITTED, ARMS)
    assert snapshot.payload["profile"] == "deadline"
    assert snapshot.status_counts == {"DONE": 13, "STARTED": 8, "PENDING": 11}
    assert snapshot.verified_count == 0
    assert snapshot.reportable_fractions == ()
    assert all(cell.development is None for cell in snapshot.cells)
    assert all(cell.train_loss is None for cell in snapshot.cells)
    assert snapshot.payload["source"]["sha256"] == "d2c1226b2b141b34cb9f3d21b34239e6dc8c0df2b03959a28f2d0bcd68036d29"
    assert "V100" not in COMMITTED.read_text(encoding="utf-8")


def test_sanitized_progress_log_has_no_metrics_or_site_paths() -> None:
    log = render_progress_log(load_snapshot(COMMITTED, ARMS))
    assert "summary=done:13;started:8;pending:11" in log
    assert "vitrand-f100-s0|DONE|34|70260|status-only" in log
    for forbidden in ("train_loss", "dev_f1", "threshold", "/projects/", "Box"):
        assert forbidden not in log


def test_deadline_generation_exposes_only_sanitized_runtime_metadata(tmp_path: Path) -> None:
    output = tmp_path / "generated"
    generate(load_snapshot(COMMITTED, ARMS), output)
    assert (output / "h100_campaign_status.txt").is_file()
    table = (output / "h100_core_table.tex").read_text(encoding="utf-8")
    macros = (output / "h100_results.tex").read_text(encoding="utf-8")
    assert "Duration" in table
    assert "19h31m" in table
    assert r"\textemdash" in table
    assert "train_loss" not in table.lower()
    assert r"\HFullGridCompletefalse" in macros
    assert r"\HHasReportableCohortfalse" in macros


def test_snapshot_requires_exact_unique_32_cell_matrix(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="exactly the 32 unique"):
        _deadline(tmp_path, lambda payload: payload["cells"].pop())

    def duplicate(payload):
        payload["cells"][-1] = copy.deepcopy(payload["cells"][0])

    with pytest.raises(SnapshotError, match="exactly the 32 unique"):
        _deadline(tmp_path, duplicate)


@pytest.mark.parametrize("status", ["STARTED", "DONE"])
def test_unverified_or_running_metric_is_never_reportable(tmp_path: Path, status: str) -> None:
    def mutate(payload):
        cell = payload["cells"][0]
        cell["status"] = status
        cell["development"] = {
            "f1": 0.99,
            "precision": 0.99,
            "recall": 0.99,
            "threshold": 0.5,
        }

    with pytest.raises(SnapshotError, match="running cell|lacks contract-verified"):
        _deadline(tmp_path, mutate)


def test_snapshot_rejects_nonfinite_metric_and_hardware_drift(tmp_path: Path) -> None:
    def nonfinite(payload):
        hardware_sha = payload["campaign"]["hardware"]["class_sha256"]
        _verify(payload["cells"][0], hardware_sha)
        payload["cells"][0]["development"]["f1"] = float("nan")

    with pytest.raises(SnapshotError, match="must be finite"):
        _deadline(tmp_path, nonfinite)

    def v100(payload):
        payload["campaign"]["hardware"]["accelerator"] = "Tesla V100-SXM2-32GB"

    with pytest.raises(SnapshotError, match="H100 class"):
        _deadline(tmp_path, v100)


def test_snapshot_rejects_tf32_and_wrong_completion_hardware(tmp_path: Path) -> None:
    def tf32(payload):
        payload["campaign"]["strict_fp32"]["tf32_enabled"] = True

    with pytest.raises(SnapshotError, match="strict IEEE FP32"):
        _deadline(tmp_path, tf32)

    def hardware(payload):
        _verify(payload["cells"][0], "b" * 64)

    with pytest.raises(SnapshotError, match="different hardware class"):
        _deadline(tmp_path, hardware)


def test_fake_digest_only_status_source_cannot_publish_a_cohort(tmp_path: Path) -> None:
    def mutate(payload):
        hardware_sha = payload["campaign"]["hardware"]["class_sha256"]
        for cell in payload["cells"]:
            if cell["label_fraction"] == 100:
                _verify(cell, hardware_sha)

    with pytest.raises(
        SnapshotError,
        match="contract-verified cells require a verified reverse-result package source",
    ):
        _deadline(tmp_path, mutate)


def test_deadline_closes_only_fully_verified_fraction(tmp_path: Path) -> None:
    root, status_path, _ = _deadline_reverse_fixture(tmp_path)
    payload = import_deadline_results(
        extracted_root=root,
        status_snapshot=status_path,
        arms_config=ARMS,
        frozen_root=REPO,
        source_sha256="f" * 64,
        captured_utc="2026-08-13T00:00:00+00:00",
    )
    snapshot = load_snapshot(_write(tmp_path / "imported-deadline.json", payload), ARMS)
    assert snapshot.reportable_fractions == (100,)
    assert snapshot.verified_count == 8

    output = tmp_path / "generated"
    generated = generate(snapshot, output)
    assert set(generated) == {
        "h100_results.tex",
        "h100_core_table.tex",
        "h100_status_table.tex",
        "h100_figure.tex",
        "h100_completed_cohorts.pdf",
        "h100_generated_manifest.json",
    }
    macros = (output / "h100_results.tex").read_text(encoding="utf-8")
    assert r"\HOneHundredCompletetrue" in macros
    assert r"\HFullGridCompletefalse" in macros
    assert r"\HHasReportableCohorttrue" in macros
    assert r"\def\HClosedFractions{f100}" in macros
    assert r"\def\HDoneCount{8}" in macros
    assert "V100" not in macros
    assert r"\label{tab:h100-status}" in (output / "h100_status_table.tex").read_text()
    assert r"\label{fig:h100-results}" in (output / "h100_figure.tex").read_text()


def test_complete_profile_requires_all_32_verified(tmp_path: Path) -> None:
    def mutate(payload):
        payload["profile"] = "complete"

    with pytest.raises(SnapshotError, match="32 contract-verified"):
        _deadline(tmp_path, mutate)


def _complete_reverse_fixture(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "reverse/results"
    core = root / "core"
    provenance = root / "provenance"
    git_sha = "1" * 40
    hardware_class = {
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "total_memory_bytes": 80_000_000_000,
        "compute_capability": [9, 0],
        "driver_version": "590.1",
        "torch": "2.11.0+cu126",
        "cuda_build": "12.6",
    }
    ids = list(expected_cells(ARMS))
    campaign = {
        "schema": 2,
        "status": "complete",
        "campaign_id": "fixture-h100",
        "git_sha": git_sha,
        "accepted_hardware_class": hardware_class,
        "base_python": {"version": "3.11.13"},
        "strict_fp32": STRICT_BACKEND,
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "complete": ids,
        "training_complete": ids,
        "running": {},
    }
    _write(provenance / "campaign_manifest.json", campaign)
    fields = ["exp_id", "init", "track", "role", "label_frac", "seed", "precision", "git_sha", "dev_f1"]
    grid = provenance / "summary/grid.csv"
    grid.parent.mkdir(parents=True, exist_ok=True)
    with grid.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, metadata in enumerate(expected_cells(ARMS).values()):
            exp_id = metadata["exp_id"]
            f1 = 0.70 + index / 1000
            run = core / exp_id
            checkpoint_root = run / "checkpoints"
            checkpoint_root.mkdir(parents=True)
            best = checkpoint_root / "best.ckpt"
            last = checkpoint_root / "last.ckpt"
            best.write_bytes(("best-" + exp_id).encode())
            last.write_bytes(("last-" + exp_id).encode())
            best_sha = hashlib.sha256(best.read_bytes()).hexdigest()
            marker = {
                "exp_id": exp_id,
                "git_sha": git_sha,
                "precision": "32-true",
                "epochs_run": 10,
                "train_loss": 0.1,
                "best_dev_f1": f1,
                "best_dev": {"f1": f1, "precision": f1, "recall": f1, "threshold": 0.5},
                "best_checkpoint": {"relative_path": "checkpoints/best.ckpt", "sha256": best_sha},
            }
            runtime = {
                "exp_id": exp_id,
                "git_sha": git_sha,
                "accepted_hardware_class": hardware_class,
                "strict_fp32": STRICT_BACKEND,
                "accumulated_active_seconds": 100.0,
            }
            _write(run / "final_metrics.json", marker)
            _write(run / "runtime_provenance.json", runtime)
            writer.writerow(
                {
                    "exp_id": exp_id,
                    "init": metadata["init"],
                    "track": metadata["track"],
                    "role": metadata["role"],
                    "label_frac": metadata["label_fraction"] / 100,
                    "seed": 0,
                    "precision": "32-true",
                    "git_sha": git_sha,
                    "dev_f1": f1,
                }
            )
    return tmp_path / "reverse", git_sha


def _reference_receipt(tmp_path: Path, git_sha: str) -> Path:
    return _write(
        tmp_path / "reference-receipt.json",
        {
            "schema": 1,
            "code_sha": git_sha,
            "references": [
                {
                    "reference_id": "R2",
                    "exp_id": "r2-yolo",
                    "status": "DONE",
                    "development_f1": 0.7,
                    "completion_marker_sha256": "c" * 64,
                    "runtime_provenance_sha256": "d" * 64,
                },
                {
                    "reference_id": "R3",
                    "exp_id": "r3-locate-anything",
                    "status": "DONE",
                    "development_f1": 0.8,
                    "completion_marker_sha256": "e" * 64,
                    "runtime_provenance_sha256": "f" * 64,
                },
            ],
        },
    )


def test_import_complete_reverse_results_rebinds_every_artifact(tmp_path: Path) -> None:
    root, git_sha = _complete_reverse_fixture(tmp_path)
    receipt = _reference_receipt(tmp_path, git_sha)
    payload = import_complete_results(
        extracted_root=root,
        arms_config=ARMS,
        frozen_root=REPO,
        source_sha256="f" * 64,
        captured_utc="2026-08-13T00:00:00+00:00",
        reference_receipt=receipt,
    )
    assert payload["profile"] == "complete"
    assert payload["campaign"]["code_sha"] == git_sha
    assert len(payload["cells"]) == 32
    assert [item["reference_id"] for item in payload["references"]] == ["R2", "R3"]

    snapshot_path = _write(tmp_path / "complete-snapshot.json", payload)
    snapshot = load_snapshot(snapshot_path, ARMS)
    output = tmp_path / "complete-generated"
    generate(snapshot, output)
    macros = (output / "h100_results.tex").read_text(encoding="utf-8")
    assert r"\HFullGridCompletetrue" in macros
    assert r"\HHasReportableCohorttrue" in macros
    assert (output / "h100_label_efficiency.pdf").is_file()


def _deadline_reverse_fixture(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    root, git_sha = _complete_reverse_fixture(tmp_path)
    result_root = root / "results"
    campaign_path = result_root / "provenance/campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    completed = tuple(
        exp_id
        for exp_id, metadata in expected_cells(ARMS).items()
        if metadata["label_fraction"] == 100
    )
    campaign.update(
        {
            "status": "running",
            "complete": list(completed),
            "training_complete": list(completed),
            "running": {},
        }
    )
    _write(campaign_path, campaign)

    grid_path = result_root / "provenance/summary/grid.csv"
    with grid_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["exp_id"] in completed]
        fields = list(rows[0])
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    status = _read()
    public_hardware = {
        "accelerator": campaign["accepted_hardware_class"]["gpu_name"],
        "compute_capability": campaign["accepted_hardware_class"]["compute_capability"],
        "python": campaign["base_python"]["version"],
        "torch": campaign["accepted_hardware_class"]["torch"],
        "cuda": campaign["accepted_hardware_class"]["cuda_build"],
    }
    public_hardware["class_sha256"] = hashlib.sha256(
        json.dumps(public_hardware, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    status["campaign"]["campaign_id"] = campaign["campaign_id"]
    status["campaign"]["code_sha"] = git_sha
    status["campaign"]["hardware"] = public_hardware
    for cell in status["cells"]:
        if cell["exp_id"] in completed:
            cell.update(
                {
                    "status": "DONE",
                    "latest_epoch": 9,
                    "duration_seconds": 100,
                    "train_loss": None,
                    "development": None,
                    "evidence": {"verification": "status-only"},
                }
            )
        else:
            cell.update(
                {
                    "status": "PENDING",
                    "latest_epoch": None,
                    "duration_seconds": None,
                    "train_loss": None,
                    "development": None,
                    "evidence": {"verification": "status-only"},
                }
            )
    status_path = _write(tmp_path / "deadline-status.json", status)
    return root, status_path, completed


def test_import_deadline_rehashes_one_closed_fraction(tmp_path: Path) -> None:
    root, status_path, completed = _deadline_reverse_fixture(tmp_path)
    payload = import_deadline_results(
        extracted_root=root,
        status_snapshot=status_path,
        arms_config=ARMS,
        frozen_root=REPO,
        source_sha256="f" * 64,
        captured_utc="2026-08-13T00:00:00+00:00",
    )
    snapshot = load_snapshot(_write(tmp_path / "deadline-import.json", payload), ARMS)
    assert snapshot.payload["profile"] == "deadline"
    assert snapshot.verified_count == 8
    assert snapshot.reportable_fractions == (100,)
    assert {
        cell.exp_id for cell in snapshot.cells if cell.verified
    } == set(completed)

    output = tmp_path / "deadline-generated"
    generate(snapshot, output)
    assert (output / "h100_completed_cohorts.pdf").is_file()
    assert not (output / "h100_campaign_status.pdf").exists()
    assert not (output / "h100_label_efficiency.pdf").exists()


def test_import_deadline_rejects_fake_digest_seed_snapshot(tmp_path: Path) -> None:
    root, status_path, completed = _deadline_reverse_fixture(tmp_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    hardware_sha = status["campaign"]["hardware"]["class_sha256"]
    status["source"] = {
        "kind": "verified-reverse-result-package",
        "description": "digest-only fake",
        "sha256": "b" * 64,
    }
    for cell in status["cells"]:
        if cell["exp_id"] in completed:
            _verify(cell, hardware_sha)
    _write(status_path, status)

    with pytest.raises(SnapshotError, match="status snapshot must be status-only"):
        import_deadline_results(
            extracted_root=root,
            status_snapshot=status_path,
            arms_config=ARMS,
            frozen_root=REPO,
            source_sha256="f" * 64,
            captured_utc="2026-08-13T00:00:00+00:00",
        )


def test_import_deadline_rejects_missing_evidence_bytes(tmp_path: Path) -> None:
    root, status_path, completed = _deadline_reverse_fixture(tmp_path)
    first = completed[0]
    (root / "results/core" / first / "checkpoints/last.ckpt").unlink()
    with pytest.raises(SnapshotError, match="last checkpoint is absent"):
        import_deadline_results(
            extracted_root=root,
            status_snapshot=status_path,
            arms_config=ARMS,
            frozen_root=REPO,
            source_sha256="f" * 64,
            captured_utc="2026-08-13T00:00:00+00:00",
        )


def test_reference_receipt_requires_exact_verified_pair(tmp_path: Path) -> None:
    receipt = _reference_receipt(tmp_path, "1" * 40)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["references"].pop()
    _write(receipt, payload)
    with pytest.raises(SnapshotError, match="exactly R2 and R3"):
        load_verified_references(receipt, expected_code_sha="1" * 40)


def test_import_rejects_checkpoint_byte_drift(tmp_path: Path) -> None:
    root, _ = _complete_reverse_fixture(tmp_path)
    first = next(iter(expected_cells(ARMS)))
    (root / "results/core" / first / "checkpoints/best.ckpt").write_bytes(b"tampered")
    with pytest.raises(SnapshotError, match="best checkpoint hash mismatch"):
        import_complete_results(
            extracted_root=root,
            arms_config=ARMS,
            frozen_root=REPO,
            source_sha256="f" * 64,
            captured_utc="2026-08-13T00:00:00+00:00",
        )
