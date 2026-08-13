"""Truthful failed-grid reverse-handback regressions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import pytest

import scripts.handoff.package as handoff_package
from scripts.h100.contracts import Cell, EXTERNAL_CONTROLS_POLICY
from scripts.h100.reverse_results import command, validate_grid
from scripts.h100.lightning_contract import (
    CUDA_ACCELERATOR,
    PRECISION_PLUGIN,
    SINGLE_DEVICE_STRATEGY,
)
from scripts.h100.strict_fp32_probe import bind_child_probes
from scripts.handoff.package import (
    PackageError,
    _canonical_json,
    _validate_amended_result_provenance,
    _validate_artifact_schema,
)
from scripts.handoff.results import (
    _validate_campaign_completion_state,
    _validate_optional_final_results,
    _validate_owner_amendment,
)
from src.eval.final_authorization import (
    AUTHORIZATION_FILENAME,
    AUTHORIZED_OWNER,
    build_authorization,
    write_authorization,
)
from src.eval.heldout_contract import COHORT_FILENAME, TEST_RESULT_FILENAME
from src.eval.result_contract import sha256_file


_STRICT_FP32 = {
    "cuda_matmul_fp32_precision": "ieee",
    "cudnn_conv_fp32_precision": "ieee",
    "cudnn_rnn_fp32_precision": "ieee",
}


def _runtime_contract() -> dict[str, object]:
    return {
        "schema": 1,
        "status": "verified",
        "pre_trainer": {
            "schema": 1,
            "status": "verified",
            "stage": "pre-trainer",
            "precision": "32-true",
            "devices": 1,
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "strict_fp32": _STRICT_FP32,
            "autocast": {"global": False, "cuda": False, "cpu": False},
            "process": {
                "WORLD_SIZE": "unset",
                "SLURM_NTASKS": "unset",
                "effective_world_size": 1,
            },
            "model": {
                "floating_parameter_count": 2,
                "floating_parameter_dtypes": ["torch.float32"],
            },
        },
        "resolved_trainer": {
            "accelerator": CUDA_ACCELERATOR,
            "precision_plugin": PRECISION_PLUGIN,
            "precision": "32-true",
            "gradient_scaler": None,
            "strategy": SINGLE_DEVICE_STRATEGY,
            "root_device_type": "cuda",
            "root_device_index": 0,
            "num_devices": 1,
            "world_size": 1,
            "device_ids": [0],
            "gradient_accumulation": 1,
        },
    }


def _h100_hardware() -> tuple[dict[str, object], dict[str, object]]:
    devices = [
        {
            "index": index,
            "name": "NVIDIA H100 80GB HBM3",
            "uuid": f"GPU-H100-{index}",
            "compute_capability": [9, 0],
            "total_memory_bytes": 85_000_000_000,
        }
        for index in range(8)
    ]
    children = [
        {
            "device": {**device, "index": 0},
            "backend": _STRICT_FP32,
            "finite": True,
            "runtime_contract": _runtime_contract(),
        }
        for device in devices
    ]
    hardware = {
        "torch": "2.11.0+cu126",
        "cuda_build": "12.6",
        "driver_version": "590.1",
        "backend": _STRICT_FP32,
        "devices": devices,
        "child_probes": bind_child_probes(
            devices, children, expected_backend=_STRICT_FP32
        ),
    }
    hardware_class = {
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "total_memory_bytes": 85_000_000_000,
        "compute_capability": [9, 0],
        "driver_version": "590.1",
        "torch": "2.11.0+cu126",
        "cuda_build": "12.6",
    }
    return hardware, hardware_class


def _write_immutable_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)


def _data_view_fixture(
    *, root: Path, campaign_git_sha: str, evaluator_git_sha: str
) -> dict[str, object]:
    scenes = []
    for index in range(50):
        scene = f"scene-{index:02d}"
        scenes.append(
            {
                "scene_id": scene,
                "source_archive": {
                    "path": f"data/final-inputs/rasters/{scene}.tar.gz",
                    "bytes": 100 + index,
                    "sha256": hashlib.sha256(f"archive:{scene}".encode()).hexdigest(),
                },
                "rasters": [
                    {
                        "path": f"data/raw/xview3/GRD/{scene}/{name}",
                        "bytes": 10 + raster_index,
                        "sha256": hashlib.sha256(
                            f"raster:{scene}:{name}".encode()
                        ).hexdigest(),
                    }
                    for raster_index, name in enumerate(
                        ("VH_dB.tif", "VV_dB.tif", "bathymetry.tif")
                    )
                ],
            }
        )
    package_tail = "1" * 64
    return {
        "schema": 1,
        "status": "final-eval-data-view-staged",
        "package": {
            "package_id": (
                f"xview3-h100-final-eval-{evaluator_git_sha}-{package_tail}"
            ),
            "identity_sha256": "2" * 64,
            "ready_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "sha256sums_sha256": "5" * 64,
        },
        "source": {
            "branch": "sprint-8-final-eval-amendment",
            "git_bundle_ref": "refs/heads/sprint-8-final-eval-amendment",
            "git_commit": evaluator_git_sha,
            "required_campaign_commit": campaign_git_sha,
            "git_bundle_sha256": "6" * 64,
            "splits_sha256": "7" * 64,
        },
        "view": {
            "root": str(root),
            "repo": str(root / "repo"),
            "bundle": str(root / "code/xview3-final-eval.bundle"),
            "training_labels": {
                "path": "repo/data/raw/xview3/labels/train.csv",
                "bytes": 111,
                "sha256": "8" * 64,
            },
            "validation_labels": {
                "path": "repo/data/raw/xview3/labels/validation.csv",
                "bytes": 222,
                "sha256": "9" * 64,
                "access": "opaque-bytes-staged-not-semantically-read",
            },
            "scenes": scenes,
        },
    }


def _failed_grid(path: Path) -> None:
    fields = ("exp_id", "init", "label_frac", "test_f1", "monotonicity_ok")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(8):
            init_name = "beS1" if index == 6 else f"arm-{index}"
            scores = (
                (0.70, 0.80, 0.8521, 0.8221)
                if init_name == "beS1"
                else (0.70, 0.75, 0.80, 0.85)
            )
            for fraction, score in zip((0.1, 0.25, 0.5, 1.0), scores, strict=True):
                writer.writerow(
                    {
                        "exp_id": f"cell-{index}-{fraction}",
                        "init": init_name,
                        "label_frac": fraction,
                        "test_f1": score,
                        "monotonicity_ok": init_name != "beS1",
                    }
                )


def test_reverse_wrapper_requires_explicit_amendment_for_failed_grid(tmp_path):
    runs = tmp_path / "runs"
    grid = runs / "summary/grid.csv"
    _failed_grid(grid)
    amendment = runs / ".h100" / AUTHORIZATION_FILENAME

    with pytest.raises(RuntimeError, match="monotone TEST grid"):
        validate_grid(runs)
    assert validate_grid(runs, owner_amendment=amendment) == grid

    argv = command(
        repo=tmp_path / "repo",
        runs_root=runs,
        campaign_manifest=runs / ".h100/campaign_manifest.json",
        output_dir=tmp_path / "out",
        max_part_bytes=123,
        owner_amendment=amendment,
    )
    assert argv[-2:] == ["--owner-amendment", str(amendment)]


def test_failed_campaign_state_is_not_rewritten_as_complete():
    ids = [f"cell-{index:02d}" for index in range(32)]
    failed = {
        "status": "failed",
        "phase": "score-test",
        "cell_order": ids,
        "complete": list(reversed(ids)),
        "training_complete": ids,
        "test_complete": ids,
        "running": {},
        "fail_stop": {
            "engaged": True,
            "failed": [],
            "allowed_to_finish": [],
        },
        "events": [
            {
                "event": "grid_validation_failed",
                "error": "grid.csv monotonicity STOP for beS1",
                "utc": "2026-08-13T04:34:00+00:00",
            }
        ],
    }
    _validate_campaign_completion_state(failed, expected_order=ids, amended=True)

    false_green = json.loads(json.dumps(failed))
    false_green["status"] = "complete"
    with pytest.raises(PackageError, match="truthful 'failed'"):
        _validate_campaign_completion_state(
            false_green, expected_order=ids, amended=True
        )

    cell_failure = json.loads(json.dumps(failed))
    cell_failure["fail_stop"]["failed"] = [ids[0]]
    with pytest.raises(PackageError, match="exact scientific STOP"):
        _validate_campaign_completion_state(
            cell_failure, expected_order=ids, amended=True
        )

    validated_event = json.loads(json.dumps(failed))
    validated_event["events"].insert(0, {"event": "grid_validated"})
    with pytest.raises(PackageError, match="disclosed monotonicity STOP"):
        _validate_campaign_completion_state(
            validated_event, expected_order=ids, amended=True
        )


def _authorization_fixture(tmp_path: Path):
    runs = (tmp_path / "runs").resolve()
    meta = runs / ".h100"
    meta.mkdir(parents=True)
    campaign_git_sha = "a" * 40
    evaluator_git_sha = "b" * 40
    cohort_sha256 = "c" * 64
    fractions = (0.1, 0.25, 0.5, 1.0)
    cells = [
        Cell(
            init=f"arm-{arm}",
            short=f"cell{arm}",
            track="vit" if arm < 4 else "cnn",
            fraction=fraction,
        )
        for arm in range(8)
        for fraction in fractions
    ]
    ids = [cell.exp_id for cell in cells]
    terminal = {
        "event": "grid_validation_failed",
        "error": "grid.csv monotonicity STOP for beS1",
        "utc": "2026-08-13T04:34:00+00:00",
    }
    campaign = {
        "campaign_id": "fixture-h100",
        "git_sha": campaign_git_sha,
        "detector_sha256": "2" * 64,
        "status": "failed",
        "events": [terminal],
    }
    campaign_path = meta / "campaign_manifest.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    test_sha256 = {}
    for cell in cells:
        path = runs / cell.exp_id / TEST_RESULT_FILENAME
        path.parent.mkdir(parents=True)
        path.write_text(f"TEST:{cell.exp_id}\n", encoding="utf-8")
        test_sha256[cell.exp_id] = sha256_file(path)
    grid_audit = {
        "relative_path": "summary/grid.csv",
        "sha256": "1" * 64,
        "monotonicity_tolerance": 0.02,
        "monotonicity_ok": False,
        "violations": [
            {
                "init": "beS1",
                "from_fraction": 0.5,
                "to_fraction": 1.0,
                "from_test_f1": 0.8521,
                "to_test_f1": 0.8221,
                "drop": 0.03,
            }
        ],
    }
    controls = {
        "status": "reporting-controls-persisted",
        "h100_ready": {
            "relative_path": ".h100/H100_READY.json",
            "sha256": "d" * 64,
        },
        "cutover_ready": {
            "relative_path": ".h100/CUTOVER_READY.json",
            "sha256": "e" * 64,
        },
        "v100_diagnostic_isolation": {
            "relative_path": ".h100/V100_DIAGNOSTIC_ISOLATION.json",
            "sha256": "f" * 64,
        },
    }
    receipt = build_authorization(
        owner=AUTHORIZED_OWNER,
        created_utc="2026-08-13T05:00:00+00:00",
        campaign={
            "campaign_id": campaign["campaign_id"],
            "git_sha": campaign_git_sha,
            "runs_root": str(runs),
            "manifest": {
                "relative_path": ".h100/campaign_manifest.json",
                "sha256": sha256_file(campaign_path),
            },
            "terminal_event": terminal,
        },
        evaluator_git_sha=evaluator_git_sha,
        cohort={
            "relative_path": f".h100/{COHORT_FILENAME}",
            "sha256": cohort_sha256,
        },
        phase5_controls=controls,
        grid=grid_audit,
        test_results=[
            {
                "exp_id": cell.exp_id,
                "relative_path": f"{cell.exp_id}/{TEST_RESULT_FILENAME}",
                "sha256": test_sha256[cell.exp_id],
            }
            for cell in cells
        ],
        selected_cells=ids,
    )
    amendment_path = meta / AUTHORIZATION_FILENAME
    amendment_sha256 = write_authorization(amendment_path, receipt)
    context = {
        "repo": tmp_path / "repo",
        "evaluator_git_sha": evaluator_git_sha,
        "cohort_sha256": cohort_sha256,
        "cohort": {
            "cells": [
                {
                    "exp_id": cell.exp_id,
                    "best_checkpoint": {
                        "relative_path": f"{cell.exp_id}/checkpoints/best.ckpt",
                        "sha256": hashlib.sha256(
                            f"checkpoint:{cell.exp_id}".encode()
                        ).hexdigest(),
                        "epoch": 3,
                    },
                    "best_dev": {"threshold": 0.25, "epoch": 3},
                }
                for cell in cells
            ]
        },
        "h100_ready_sha256": "d" * 64,
        "cutover_ready_sha256": "e" * 64,
        "diagnostic_isolation": {"sha256": "f" * 64},
    }
    _hardware, accepted_class = _h100_hardware()
    context["strict_fp32"] = _STRICT_FP32
    context["accepted_hardware_class"] = accepted_class
    amendment = _validate_owner_amendment(
        amendment_path,
        runs_root=runs,
        campaign=campaign,
        context=context,
        cells=cells,
        grid_audit=grid_audit,
        test_metrics_sha256=test_sha256,
    )
    splits_path = Path(context["repo"]) / "data/splits.json"
    splits_path.parent.mkdir(parents=True)
    splits_path.write_text(
        json.dumps(
            {
                "splits": {
                    "eval_final": [f"scene-{index:02d}" for index in range(50)]
                }
            }
        ),
        encoding="utf-8",
    )
    assert amendment["sha256"] == amendment_sha256
    return runs, campaign, context, cells, grid_audit, test_sha256, amendment


def test_reverse_owner_receipt_and_complete_final_provenance_are_bound(tmp_path):
    (
        runs,
        campaign,
        context,
        cells,
        grid_audit,
        test_sha256,
        amendment,
    ) = _authorization_fixture(tmp_path)
    meta = runs / ".h100"
    owner_sha256 = amendment["sha256"]

    scene_ids = [f"scene-{index:02d}" for index in range(50)]
    data_view_path = meta / "FINAL_DATA_VIEW.json"
    data_view = _data_view_fixture(
        root=(tmp_path / "final-view").resolve(),
        campaign_git_sha=campaign["git_sha"],
        evaluator_git_sha=context["evaluator_git_sha"],
    )
    data_view_path.write_bytes(_canonical_json(data_view))
    data_view_path.chmod(0o444)
    staged_scenes = data_view["view"]["scenes"]
    data_view_binding = {
        "relative_path": ".h100/FINAL_DATA_VIEW.json",
        "sha256": sha256_file(data_view_path),
        "package": data_view["package"],
        "source": data_view["source"],
        "validation_labels": data_view["view"]["validation_labels"],
        "scenes": staged_scenes,
    }
    hardware, hardware_class = _h100_hardware()
    hardware_path = meta / "FINAL_H100_RUNTIME-123.json"
    _write_immutable_json(hardware_path, hardware)
    hardware_binding = {
        "relative_path": ".h100/FINAL_H100_RUNTIME-123.json",
        "sha256": sha256_file(hardware_path),
        "strict_fp32": _STRICT_FP32,
        "hardware_class": hardware_class,
    }
    lock_path = runs / "final_eval.lock"
    expected_tests = [
        {
            "exp_id": cell.exp_id,
            "relative_path": f"{cell.exp_id}/{TEST_RESULT_FILENAME}",
            "sha256": test_sha256[cell.exp_id],
        }
        for cell in cells
    ]
    _write_immutable_json(
        lock_path,
        {
            "lock_schema": 3,
            "started_utc": "2026-08-13T05:01:00+00:00",
            "policy": amendment["receipt"]["policy"],
            "campaign_git_sha": campaign["git_sha"],
            "evaluator_git_sha": context["evaluator_git_sha"],
            "detector_sha256": campaign["detector_sha256"],
            "cohort_sha256": context["cohort_sha256"],
            "grid": grid_audit,
            "selected_cells": [cell.exp_id for cell in cells],
            "eval_scene_count": 50,
            "owner_amendment": {
                "relative_path": amendment["relative_path"],
                "sha256": owner_sha256,
                "owner": amendment["receipt"]["owner"],
                "decision": amendment["receipt"]["decision"],
            },
            "test_results": expected_tests,
            "phase5_controls": amendment["receipt"]["phase5_controls"],
            "inference_precision": "32-true",
            "allocation_hardware": hardware_binding,
            "final_data_view": data_view_binding,
        },
    )
    normalized_path = meta / "FINAL_NORMALIZED_GROUND_TRUTH.json"
    normalized_bytes = json.dumps(
        {scene: [] for scene in scene_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    normalized_path.write_bytes(normalized_bytes + b"\n")
    normalized_path.chmod(0o444)
    normalized_sha256 = hashlib.sha256(normalized_bytes).hexdigest()
    _write_immutable_json(
        meta / "FINAL_GROUND_TRUTH_CONSUMED.json",
        {
            "consumption_schema": 1,
            "status": "verified-final-ground-truth-consumed",
            "consumed_utc": "2026-08-13T05:02:00+00:00",
            "campaign_git_sha": campaign["git_sha"],
            "evaluator_git_sha": context["evaluator_git_sha"],
            "owner_authorization_sha256": owner_sha256,
            "final_data_view": data_view_binding,
            "lock": {
                "relative_path": "final_eval.lock",
                "sha256": sha256_file(lock_path),
            },
            "normalized_ground_truth": {
                "relative_path": ".h100/FINAL_NORMALIZED_GROUND_TRUTH.json",
                "sha256": normalized_sha256,
                "stored_sha256": sha256_file(normalized_path),
            },
            "normalized_ground_truth_sha256": normalized_sha256,
            "validation_csv": {
                "row_count": 19_224,
                "scene_count": 50,
                "sha256": "8" * 64,
            },
        },
    )
    consumption_path = meta / "FINAL_GROUND_TRUTH_CONSUMED.json"
    zero_metric = {
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ignored_predictions": 0,
    }
    final_access = {
        "lock_sha256": sha256_file(lock_path),
        "consumption_sha256": sha256_file(consumption_path),
        "data_view_sha256": sha256_file(data_view_path),
    }
    result_sha256 = {}
    result_payloads = {}
    for cell in cells:
        record = next(
            item
            for item in context["cohort"]["cells"]
            if item["exp_id"] == cell.exp_id
        )
        payload = {
            "final_result_schema": 1,
            "created_utc": "2026-08-13T05:03:00+00:00",
            "exp_id": cell.exp_id,
            "init": cell.init,
            "label_frac": cell.fraction,
            "seed": 0,
            "study_design": "single-seed-0-point-estimate-no-uncertainty-estimate",
            "interpretation": (
                "descriptive-exploratory-post-test-owner-amendment;"
                "predeclared-test-monotonicity-failed"
            ),
            "threshold": record["best_dev"]["threshold"],
            "threshold_source": "best-dev-checkpoint-bound",
            "dev_epoch": record["best_dev"]["epoch"],
            "checkpoint": record["best_checkpoint"],
            "campaign_git_sha": campaign["git_sha"],
            "evaluator_git_sha": context["evaluator_git_sha"],
            "detector_sha256": campaign["detector_sha256"],
            "cohort_sha256": context["cohort_sha256"],
            "test_result_sha256": test_sha256[cell.exp_id],
            "owner_authorization_sha256": owner_sha256,
            "final_access": final_access,
            "test_monotonicity": {
                "ok": False,
                "tolerance": grid_audit["monotonicity_tolerance"],
                "violations": grid_audit["violations"],
            },
            "inference_precision": "32-true",
            "strict_fp32": _STRICT_FP32,
            "eval_scene_ids": scene_ids,
            "metrics": {
                **zero_metric,
                "dark_recall": 0.0,
                "dark_support": 0,
                "near_shore_f1": 0.0,
                "near_shore_support": 0,
            },
            "per_scene": {
                scene: {
                    "aggregate": zero_metric,
                    "slices": {
                        "dark": zero_metric,
                        "near_shore": zero_metric,
                    },
                    "matches": [],
                }
                for scene in scene_ids
            },
            "thresholded_predictions": {scene: [] for scene in scene_ids},
        }
        path = runs / cell.exp_id / "final_verified_metrics.json"
        _write_immutable_json(
            path,
            payload,
        )
        result_sha256[cell.exp_id] = sha256_file(path)
        result_payloads[cell.exp_id] = payload
    summary = runs / "summary/final_verified.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    from src.eval.final_eval import _summary_row

    rows = [
        _summary_row(
            cell=cell,
            payload=result_payloads[cell.exp_id],
            grid_audit=grid_audit,
            campaign_git_sha=campaign["git_sha"],
            evaluator_git_sha=context["evaluator_git_sha"],
            detector_sha256=campaign["detector_sha256"],
            cohort_sha256=context["cohort_sha256"],
            authorization_sha256=owner_sha256,
            final_result_sha256=result_sha256[cell.exp_id],
        )
        for cell in cells
    ]
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary.chmod(0o444)
    _write_immutable_json(
        meta / "FINAL_EVAL_COMPLETE.json",
        {
            "completion_schema": 1,
            "status": "all-32-verified-final-complete",
            "completed_utc": "2026-08-13T05:04:00+00:00",
            "interpretation": (
                "descriptive-exploratory-post-test-owner-amendment;"
                "predeclared-test-monotonicity-failed"
            ),
            "seed": 0,
            "study_design": "single-seed-0-point-estimate-no-uncertainty-estimate",
            "campaign_git_sha": campaign["git_sha"],
            "evaluator_git_sha": context["evaluator_git_sha"],
            "owner_authorization_sha256": owner_sha256,
            "lock_sha256": final_access["lock_sha256"],
            "consumption_sha256": final_access["consumption_sha256"],
            "data_view_sha256": final_access["data_view_sha256"],
            "summary": {
                "relative_path": "summary/final_verified.csv",
                "sha256": sha256_file(summary),
                "row_count": 32,
            },
            "final_results": [
                {
                    "exp_id": cell.exp_id,
                    "relative_path": (
                        f"{cell.exp_id}/final_verified_metrics.json"
                    ),
                    "sha256": result_sha256[cell.exp_id],
                }
                for cell in cells
            ],
            "allocation_hardware": json.loads(lock_path.read_text())[
                "allocation_hardware"
            ],
            "test_monotonicity": grid_audit,
        },
    )

    final = _validate_optional_final_results(
        runs_root=runs,
        campaign=campaign,
        context=context,
        cells=cells,
        grid_audit=grid_audit,
        amendment=amendment,
        test_metrics_sha256=test_sha256,
    )
    assert final is not None
    assert final["status"] == "complete"
    assert len(final["cell_result_sha256"]) == 32
    assert len(final["cell_paths"]) == 32
    assert normalized_path not in final["paths"]

    tampered = runs / cells[0].exp_id / "final_verified_metrics.json"
    tampered.chmod(0o644)
    payload = json.loads(tampered.read_text(encoding="utf-8"))
    payload["owner_authorization_sha256"] = "0" * 64
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    tampered.chmod(0o444)
    with pytest.raises(PackageError, match="final-result binding"):
        _validate_optional_final_results(
            runs_root=runs,
            campaign=campaign,
            context=context,
            cells=cells,
            grid_audit=grid_audit,
            amendment=amendment,
            test_metrics_sha256=test_sha256,
        )


def _receiver_fixture(tmp_path: Path):
    runs, campaign, context, cells, grid, tests, amendment = (
        _authorization_fixture(tmp_path)
    )
    del runs, campaign
    cell_ids = [cell.exp_id for cell in cells]
    result_digests = {
        cell: hashlib.sha256(f"final:{cell}".encode()).hexdigest()
        for cell in cell_ids
    }
    eval_scene_ids = [f"scene-{index:02d}" for index in range(50)]
    data_view = _data_view_fixture(
        root=(tmp_path / "receiver-final-view").resolve(),
        campaign_git_sha="a" * 40,
        evaluator_git_sha=context["evaluator_git_sha"],
    )
    final = {
        "status": "complete",
        "lock_sha256": "3" * 64,
        "data_view_sha256": hashlib.sha256(_canonical_json(data_view)).hexdigest(),
        "consumption_sha256": "5" * 64,
        "normalized_ground_truth_sha256": "6" * 64,
        "summary_sha256": "7" * 64,
        "completion_sha256": "8" * 64,
        "allocation_hardware": {
            "relative_path": ".h100/FINAL_H100_RUNTIME-123.json",
            "sha256": "9" * 64,
            "strict_fp32": _STRICT_FP32,
            "hardware_class": {"gpu_name": "NVIDIA H100 80GB HBM3"},
        },
        "data_view": data_view,
        "eval_scene_ids": eval_scene_ids,
        "cell_result_sha256": result_digests,
    }
    source = {
        "git_commit": context["evaluator_git_sha"],
        "campaign_id": "fixture-h100",
        "campaign_git_commit": "a" * 40,
        "evaluator_git_commit": context["evaluator_git_sha"],
        "campaign_status": "failed",
        "campaign_manifest_sha256": amendment["receipt"]["campaign"][
            "manifest"
        ]["sha256"],
        "training_cohort_sha256": context["cohort_sha256"],
        "h100_ready_sha256": context["h100_ready_sha256"],
        "cutover_ready_sha256": context["cutover_ready_sha256"],
        "v100_diagnostic_isolation": {
            "sha256": context["diagnostic_isolation"]["sha256"]
        },
        "summary_grid_sha256": grid["sha256"],
        "summary_grid_diagnostic": grid,
        "test_metrics_sha256": tests,
        "post_test_owner_amendment": amendment,
        "final_evaluation": final,
    }
    identity = {
        "schema": 3,
        "git_commit": context["evaluator_git_sha"],
        "campaign_git_commit": "a" * 40,
        "evaluator_git_commit": context["evaluator_git_sha"],
        "campaign_status": "failed",
        "post_test_owner_amendment": amendment,
        "final_evaluation": final,
    }
    members = {
        "results/provenance/FINAL_EVAL_OWNER_AMENDMENT.json": amendment["sha256"],
        "results/provenance/final/final_eval.lock": final["lock_sha256"],
        "results/provenance/final/FINAL_DATA_VIEW.json": final[
            "data_view_sha256"
        ],
        "results/provenance/final/FINAL_GROUND_TRUTH_CONSUMED.json": final[
            "consumption_sha256"
        ],
        "results/provenance/final/FINAL_EVAL_COMPLETE.json": final[
            "completion_sha256"
        ],
        "results/provenance/final/FINAL_H100_RUNTIME-123.json": final[
            "allocation_hardware"
        ]["sha256"],
        "results/provenance/final/final_verified.csv": final["summary_sha256"],
        **{
            f"results/provenance/final/cells/{cell}.json": digest
            for cell, digest in result_digests.items()
        },
    }
    return source, identity, members, cell_ids


def test_reverse_receiver_accepts_exact_complete_amended_provenance(tmp_path):
    source, identity, members, cells = _receiver_fixture(tmp_path)
    assert _validate_amended_result_provenance(
        source=source,
        result_identity=identity,
        provenance_members=members,
        cells=cells,
    )


def test_result_artifact_receiver_wires_schema3_amendment_end_to_end(
    tmp_path, monkeypatch
):
    source, identity, members, cells = _receiver_fixture(tmp_path)
    source.update(
        {
            "external_controls_policy": EXTERNAL_CONTROLS_POLICY,
            "source_validation_sha256": "0" * 64,
            "evaluation_ground_truth_sha256": "b" * 64,
            "acceptance_test_suite_sha256": "e" * 64,
            "runtime_provenance_sha256": {
                cell: hashlib.sha256(f"runtime:{cell}".encode()).hexdigest()
                for cell in cells
            },
        }
    )
    source["v100_diagnostic_isolation"] = {
        "sha256": "f" * 64,
        "receipt": {"cutover_ready_sha256": source["cutover_ready_sha256"]},
    }
    identity.update(
        {
            "external_controls_policy": EXTERNAL_CONTROLS_POLICY,
            "h100_ready_sha256": source["h100_ready_sha256"],
            "cutover_ready_sha256": source["cutover_ready_sha256"],
            "v100_diagnostic_isolation": source["v100_diagnostic_isolation"],
        }
    )
    base_member_digests = {
        "results/provenance/campaign_manifest.json": source[
            "campaign_manifest_sha256"
        ],
        "results/provenance/H100_READY.json": source["h100_ready_sha256"],
        "results/provenance/h100_runtime.json": "2" * 64,
        "results/provenance/throughput_projection.json": "3" * 64,
        "results/provenance/venv_build.json": "4" * 64,
        "results/provenance/CUTOVER_READY.json": source[
            "cutover_ready_sha256"
        ],
        "results/provenance/SOURCE_VALIDATED.json": source[
            "source_validation_sha256"
        ],
        "results/provenance/EVAL_GROUND_TRUTH_VALIDATED.json": source[
            "evaluation_ground_truth_sha256"
        ],
        "results/provenance/TRAINING_COHORT.json": source[
            "training_cohort_sha256"
        ],
        "results/provenance/HOST_HANDOFF_TESTS.json": "5" * 64,
        "results/provenance/PYTEST_ACCEPTANCE.json": source[
            "acceptance_test_suite_sha256"
        ],
        "results/provenance/V100_DIAGNOSTIC_ISOLATION.json": source[
            "v100_diagnostic_isolation"
        ]["sha256"],
        "results/provenance/slurm-smoke/SLURM_SMOKE_READY.json": "6" * 64,
        "results/provenance/slurm-smoke/SLURM_SMOKE_STATE.json": "7" * 64,
        "results/provenance/summary/grid.csv": source["summary_grid_sha256"],
        "results/provenance/acceptance-logs/pytest-handoff-host.log": "8" * 64,
        "results/provenance/acceptance-logs/pytest-venv-remaining.log": "9" * 64,
        "results/provenance/acceptance-logs/vit-fp32.log": "a" * 64,
        "results/provenance/acceptance-logs/cnn-200step-fp32.log": "b" * 64,
        "results/provenance/allocations/h100_runtime-123-r0.json": "c" * 64,
    }
    members.update(base_member_digests)
    source["campaign_provenance_member_sha256"] = members
    source["result_identity"] = identity

    artifacts = []
    for cell in cells:
        root = f"results/core/{cell}"
        artifacts.append(
            {
                "kind": "core_result",
                "name": cell,
                "extraction_root": root,
                "format": "tar.zst",
                "file_count": 8,
                "unpacked_bytes": 8,
                "parts": [{"path": f"{root}.tar.zst", "bytes": 1}],
                "member_sha256": {
                    f"{root}/final_metrics.json": "0" * 64,
                    f"{root}/test_metrics.json": source["test_metrics_sha256"][
                        cell
                    ],
                    f"{root}/config.yaml": "1" * 64,
                    f"{root}/metrics/metrics.csv": "2" * 64,
                    f"{root}/runtime_provenance.json": source[
                        "runtime_provenance_sha256"
                    ][cell],
                    f"{root}/checkpoints/best.ckpt": "3" * 64,
                    f"{root}/checkpoints/last.ckpt": "4" * 64,
                    f"{root}/log.txt": "5" * 64,
                },
            }
        )
    artifacts.append(
        {
            "kind": "campaign_provenance",
            "name": source["campaign_id"],
            "extraction_root": "results/provenance",
            "format": "tar.zst",
            "file_count": len(members),
            "unpacked_bytes": len(members),
            "parts": [
                {"path": "results/provenance/campaign.tar.zst", "bytes": 1}
            ],
            "member_sha256": members,
        }
    )
    maximum_part_bytes = 100
    package_root = tmp_path / "received-results"
    for artifact in artifacts:
        relative = artifact["parts"][0]["path"]
        physical = package_root / relative
        physical.parent.mkdir(parents=True, exist_ok=True)
        physical.write_bytes(b"x")
        part = {
            "path": relative,
            "bytes": 1,
            "sha256": hashlib.sha256(b"x").hexdigest(),
            "sha1": hashlib.sha1(b"x").hexdigest(),
        }
        artifact.update(
            {
                "parts": [part],
                "archive_bytes": 1,
                "archive_sha256": part["sha256"],
                "archive_sha1": part["sha1"],
            }
        )
    identity.update(
        {
            "maximum_physical_file_bytes": maximum_part_bytes,
            "artifact_digest_index": artifacts,
        }
    )
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    source["result_identity_sha256"] = identity_sha256
    package_id = (
        f"xview3-h100-results-{source['git_commit']}-{identity_sha256}"
    )
    manifest = {
        "format_version": handoff_package.FORMAT_VERSION,
        "package_type": "h100-core-results",
        "package_id": package_id,
        "source": source,
        "cells": cells,
        "contract": {
            "production": False,
            "core_only": True,
            "strict_fp32": True,
            "tf32": False,
            "maximum_physical_file_bytes": maximum_part_bytes,
        },
        "counts": {"core_result_archives": 32, "provenance_archives": 1},
        "artifacts": artifacts,
    }
    manifest_path = package_root / "manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    sums_path = package_root / "SHA256SUMS"
    sums_path.write_text(
        "".join(
            f"{artifact['parts'][0]['sha256']}  "
            f"{artifact['parts'][0]['path']}\n"
            for artifact in artifacts
        ),
        encoding="utf-8",
    )

    def control(path: Path, relative: str) -> dict[str, object]:
        data = path.read_bytes()
        return {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "sha1": hashlib.sha1(data).hexdigest(),
        }

    ready = {
        "format_version": handoff_package.FORMAT_VERSION,
        "status": "READY",
        "package_id": package_id,
        "git_commit": source["git_commit"],
        "manifest": control(manifest_path, "manifest.json"),
        "checksums": control(sums_path, "SHA256SUMS"),
    }
    (package_root / "READY.json").write_bytes(_canonical_json(ready))
    monkeypatch.setattr(
        handoff_package,
        "_inspect_tar_zst",
        lambda *args, **kwargs: None,
    )
    assert handoff_package._verify_fixture_package(package_root) == manifest


def _strip_amended_declarations(source, identity, _members):
    for key in (
        "campaign_git_commit",
        "evaluator_git_commit",
        "campaign_status",
        "summary_grid_diagnostic",
        "post_test_owner_amendment",
        "final_evaluation",
    ):
        source.pop(key, None)
    for key in (
        "campaign_git_commit",
        "evaluator_git_commit",
        "campaign_status",
        "post_test_owner_amendment",
        "final_evaluation",
    ):
        identity.pop(key, None)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (_strip_amended_declarations, "partial"),
        (lambda source, identity, members: source.pop("final_evaluation"), "partial"),
        (
            lambda source, identity, members: identity.update(
                campaign_status="complete"
            ),
            "status 'failed'",
        ),
        (
            lambda source, identity, members: source[
                "summary_grid_diagnostic"
            ].update(monotonicity_ok=True),
            "failed-grid diagnostic",
        ),
        (
            lambda source, identity, members: source["final_evaluation"][
                "data_view"
            ]["view"]["scenes"][0]["rasters"].pop(),
            "scene binding",
        ),
        (
            lambda source, identity, members: source["final_evaluation"][
                "data_view"
            ]["source"].update(branch="unreviewed-final-eval"),
            "package/source",
        ),
        (
            lambda source, identity, members: members.pop(
                "results/provenance/final/FINAL_EVAL_COMPLETE.json"
            ),
            "missing, extra, or digest-mismatched",
        ),
        (
            lambda source, identity, members: members.update(
                {
                    "results/provenance/final/cells/"
                    f"{next(iter(source['final_evaluation']['cell_result_sha256']))}.json": (
                        "0" * 64
                    )
                }
            ),
            "missing, extra, or digest-mismatched",
        ),
    ],
)
def test_reverse_receiver_rejects_partial_or_tampered_amendment(
    tmp_path, mutation, match
):
    source, identity, members, cells = _receiver_fixture(tmp_path)
    mutation(source, identity, members)
    with pytest.raises(PackageError, match=match):
        _validate_amended_result_provenance(
            source=source,
            result_identity=identity,
            provenance_members=members,
            cells=cells,
        )
