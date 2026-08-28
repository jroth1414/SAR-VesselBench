from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.handoff import bes2_results as results
from scripts.handoff.__main__ import _build_parser
from scripts.handoff.box import upload_package_with_verifier
from scripts.handoff.package import PackageError
from src.analysis.bes2_contract import (
    FROZEN_COMPARATOR_EXPERIMENTS,
    H100_CAMPAIGN_GIT_SHA,
    OPTIMIZATION_TRACE_STEPS,
    diagnostic_run_id,
    fraction_tag,
    stage_fractions,
    summarize_payload,
)
from test_h100_handoff import _Client


REPO = Path(__file__).resolve().parents[1]
SOURCE_SHA = "1" * 40
CREATED = "2026-08-27T00:00:00+00:00"
COUNTS = {
    "f10": {
        "bigearthnet_s2": (2, 1, 1),
        "cnn_random": (4, 1, 1),
        "first_conv_reset": (11, 4, 4),
    },
    "f50": {
        "bigearthnet_s2": (4, 1, 1),
        "cnn_random": (8, 1, 1),
        "first_conv_reset": (38, 7, 7),
    },
}


def _write(root: Path, logical: str, data: bytes) -> Path:
    path = root / logical
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def _best(counts: tuple[int, int, int]) -> dict[str, object]:
    tp, fp, fn = counts
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "epoch": 9,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": 0,
        "threshold": 0.5,
        "n_candidates": tp + fp,
    }


def _optimization_trace() -> dict[str, object]:
    group = {
        "parameter_count": 1,
        "parameter_norm_before": 1.0,
        "gradient_norm": 0.5,
        "gradient_to_parameter_norm": 0.5,
        "lr_min": 0.001,
        "lr_max": 0.001,
        "lr_scale_min": 0.65,
        "lr_scale_max": 0.65,
        "update_norm": 0.001,
        "relative_update_norm": 0.001,
    }
    first = {
        "name": "backbone.model.stem.0.weight",
        "parameter_count": 1,
        "parameter_norm_before": 1.0,
        "gradient_norm": 0.5,
        "gradient_to_parameter_norm": 0.5,
        "lr": 0.001,
        "lr_scale": 0.65,
        "update_norm": 0.001,
        "relative_update_norm": 0.001,
    }
    groups = {
        name: dict(group)
        for name in (
            "stem",
            "stage_0",
            "stage_1",
            "stage_2",
            "stage_3",
            "detector_head",
        )
    }
    return {
        "schema": 1,
        "milestones": list(OPTIMIZATION_TRACE_STEPS),
        "records": [
            {
                "optimizer_step": step,
                "groups": groups,
                "first_convolution": first,
            }
            for step in OPTIMIZATION_TRACE_STEPS
        ],
    }


def _comparator_receipt() -> dict[str, object]:
    fractions = {}
    for tag, specification in FROZEN_COMPARATOR_EXPERIMENTS.items():
        records = {}
        for role in ("bigearthnet_s2", "cnn_random"):
            best = _best(COUNTS[tag][role])
            exp_id = specification[role]
            records[role] = {
                "exp_id": exp_id,
                "result": {
                    "path": f"{exp_id}/final_metrics.json",
                    "sha256": hashlib.sha256(
                        f"{tag}/{role}".encode()
                    ).hexdigest(),
                },
                "best_dev_f1": best["f1"],
                "best_dev": best,
            }
        fractions[tag] = {
            "fraction": specification["fraction"],
            **records,
            "frozen_random_minus_s2_f1": (
                records["cnn_random"]["best_dev_f1"]
                - records["bigearthnet_s2"]["best_dev_f1"]
            ),
        }
    return {
        "schema": 1,
        "status": "frozen",
        "purpose": "bes2-h100-best-dev-numerical-comparators",
        "created_utc": CREATED,
        "campaign_git_sha": H100_CAMPAIGN_GIT_SHA,
        "detector_sha256": "d" * 64,
        "read_scope": {
            "final_metrics_json": sorted(
                f"{specification[role]}/final_metrics.json"
                for specification in FROZEN_COMPARATOR_EXPERIMENTS.values()
                for role in ("bigearthnet_s2", "cnn_random")
            ),
            "checkpoint_bytes": False,
            "training_cohort": False,
            "test_metrics": False,
            "verified_final": False,
        },
        "fractions": fractions,
    }


def _metric_payload(
    *,
    tag: str,
    variant: str,
    stage: str,
    fraction: float,
    readiness_sha256: str,
) -> dict[str, object]:
    counts_role = (
        "bigearthnet_s2"
        if variant == "current_replay"
        else "first_conv_reset"
    )
    best = _best(COUNTS[tag][counts_role])
    return {
        "diagnostic_metrics_schema": 2,
        "status": "complete",
        "purpose": "bes2-root-cause",
        "variant": variant,
        "stage": stage,
        "fraction": fraction,
        "created_utc": CREATED,
        "readiness": {
            "path": "/judy/diagnostic/.control/BES2_DIAGNOSTIC_READY.json",
            "sha256": readiness_sha256,
        },
        "training": {
            "variant": variant,
            "stage": stage,
            "fraction": fraction,
            "initialization": {
                "variant": variant,
                "stage": stage,
                "fraction": fraction,
            },
            "training_result": {
                "best_dev_f1": best["f1"],
                "best_dev": best,
            },
            "optimization_trace": _optimization_trace(),
        },
        "dev_evidence": {
            "selected_operating_point": best,
            "leave_one_dev_scene_out": {
                "scene": {
                    "threshold": 0.5,
                    "retained_seven": {"f1": best["f1"]},
                }
            },
        },
        "activation_statistics": {},
        "layer_drift": {},
        "runtime_provenance": {},
        "gpu_hours": 1.0,
    }


def _fixture_evidence(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    source = tmp_path / "evidence-source"
    source.mkdir()
    evidence: dict[str, Path] = {}

    comparator_receipt = _comparator_receipt()
    comparator_bytes = _json_bytes(comparator_receipt)
    comparator_binding = {
        "path": "/judy/diagnostic/.control/"
        + results.COMPARATOR_FILENAME,
        "sha256": hashlib.sha256(comparator_bytes).hexdigest(),
    }
    evidence[f"control/{results.COMPARATOR_FILENAME}"] = _write(
        source,
        f"control/{results.COMPARATOR_FILENAME}",
        comparator_bytes,
    )
    readiness = {
        "source": {"git_sha": SOURCE_SHA},
        "frozen_comparators": comparator_binding,
    }
    readiness_bytes = _json_bytes(readiness)
    readiness_sha256 = hashlib.sha256(readiness_bytes).hexdigest()
    evidence[f"control/{results.READY_FILENAME}"] = _write(
        source,
        f"control/{results.READY_FILENAME}",
        readiness_bytes,
    )

    metric_payloads: dict[tuple[str, str, str], dict[str, object]] = {}
    metric_bytes: dict[tuple[str, str, str], bytes] = {}
    for stage in results.RUN_STAGES:
        for fraction in stage_fractions(stage):
            tag = fraction_tag(fraction)
            for variant in results.DIAGNOSTIC_VARIANTS:
                payload = _metric_payload(
                    tag=tag,
                    variant=variant,
                    stage=stage,
                    fraction=fraction,
                    readiness_sha256=readiness_sha256,
                )
                metric_payloads[(tag, variant, stage)] = payload
                data = _json_bytes(payload)
                metric_bytes[(tag, variant, stage)] = data
                for filename in results.RUN_FILES:
                    logical = results._logical_run_file(
                        variant,
                        stage,
                        fraction,
                        filename,
                    )
                    if filename == "diagnostic_metrics.json":
                        artifact = data
                    elif filename == "training_metrics.json":
                        artifact = _json_bytes(payload["training"])
                    elif filename == "initialization.json":
                        artifact = _json_bytes(
                            payload["training"]["initialization"]
                        )
                    elif filename.endswith(".json"):
                        artifact = _json_bytes(
                            {
                                "variant": variant,
                                "stage": stage,
                                "fraction": fraction,
                            }
                        )
                    else:
                        artifact = (
                            f"{tag} {variant} {stage} controller log\n"
                        ).encode()
                    evidence[logical] = _write(source, logical, artifact)

    full_metrics = {}
    bindings = {}
    for tag, specification in FROZEN_COMPARATOR_EXPERIMENTS.items():
        fraction = float(specification["fraction"])
        full_metrics[tag] = {}
        bindings[tag] = {}
        for variant in results.DIAGNOSTIC_VARIANTS:
            full_metrics[tag][variant] = metric_payloads[
                (tag, variant, "full")
            ]
            bindings[tag][variant] = {
                "path": (
                    "/judy/diagnostic/"
                    + diagnostic_run_id(variant, "full", fraction)
                    + "/diagnostic_metrics.json"
                ),
                "sha256": hashlib.sha256(
                    metric_bytes[(tag, variant, "full")]
                ).hexdigest(),
            }
    root_cause = summarize_payload(
        full_metrics,
        run_bindings=bindings,
        comparator_receipt=comparator_receipt,
        comparator_binding=comparator_binding,
    )
    root_cause["created_utc"] = CREATED

    for name in results.CONTROL_JSON:
        if name in {
            results.ROOT_CAUSE_FILENAME,
            results.READY_FILENAME,
            results.COMPARATOR_FILENAME,
        }:
            continue
        evidence[f"control/{name}"] = _write(
            source,
            f"control/{name}",
            _json_bytes({}),
        )
    evidence[f"control/{results.ROOT_CAUSE_FILENAME}"] = _write(
        source,
        f"control/{results.ROOT_CAUSE_FILENAME}",
        _json_bytes(root_cause),
    )
    evidence[f"control/{results.REPORT_FILENAME}"] = _write(
        source,
        f"control/{results.REPORT_FILENAME}",
        results._markdown_report(root_cause).encode("utf-8"),
    )
    evidence["control/BES2_TEST_GATE.log"] = _write(
        source,
        "control/BES2_TEST_GATE.log",
        b"500 passed\n",
    )

    hardware = {
        "diagnostic_scope": "bes2-train-dev-only",
        "backend": {
            "cuda_matmul_fp32_precision": "ieee",
            "cudnn_conv_fp32_precision": "ieee",
            "cudnn_rnn_fp32_precision": "ieee",
        },
        "devices": [{"name": "NVIDIA H100 80GB HBM3"}],
    }
    evidence["runtime/h100_runtime-5001-r0.json"] = _write(
        source,
        "runtime/h100_runtime-5001-r0.json",
        _json_bytes(hardware),
    )
    for index, stage in enumerate(("audit", "probe", "full"), start=1):
        logical = f"slurm/xview3-bes2-{stage}-{5000 + index}.out"
        evidence[logical] = _write(
            source,
            logical,
            f"{stage} complete\n".encode(),
        )

    root_cause_path = evidence[f"control/{results.ROOT_CAUSE_FILENAME}"]
    summary = {
        "source_git_sha": SOURCE_SHA,
        "readiness_sha256": readiness_sha256,
        "root_cause_sha256": hashlib.sha256(
            root_cause_path.read_bytes()
        ).hexdigest(),
        "classification": root_cause["decision"]["aggregate"][
            "classification"
        ],
        "decision_status": root_cause["status"],
        "replacement_eligibility": root_cause["decision"]["aggregate"][
            "replacement_eligibility"
        ],
        "probe_gpu_hours": 2.0,
        "full_gpu_hours": 4.0,
        "total_diagnostic_gpu_hours": 6.0,
        "created_utc": CREATED,
    }
    _write(source, "ignored/checkpoints/best.ckpt", b"large checkpoint bytes")
    return evidence, summary


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_bes2_results_are_deterministic_evidence_only_and_ready_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, summary = _fixture_evidence(tmp_path)
    monkeypatch.setattr(
        results,
        "_collect_evidence",
        lambda **_kwargs: (evidence, summary),
    )
    outputs = []
    for name in ("out-a", "out-b"):
        output = tmp_path / name
        output.mkdir()
        outputs.append(
            results.build_bes2_results(
                repo=REPO,
                diagnostic_root=tmp_path / "evidence-source",
                original_runs_root=tmp_path / "original-runs",
                job_log_dir=tmp_path / "job-logs",
                output_dir=output,
                max_part_bytes=64,
            )
        )
    first, second = outputs
    assert first.name == second.name
    assert _tree(first) == _tree(second)

    manifest = results.verify_bes2_results(
        first, expected_source_git_sha=SOURCE_SHA
    )
    names = {artifact["name"] for artifact in manifest["artifacts"]}
    assert manifest["contract"]["evidence_only"] is True
    assert manifest["contract"]["diagnostic_run_count"] == 6
    assert manifest["contract"]["fraction_analysis"] == (
        "per-fraction-no-pooling"
    )
    assert manifest["contract"]["checkpoint_artifacts"] == 0
    assert manifest["contract"]["test_or_final_artifacts"] == 0
    assert manifest["counts"]["checkpoints"] == 0
    assert all(Path(name).suffix in results._ALLOWED_SUFFIXES for name in names)
    assert not any("checkpoint" in name.lower() for name in names)
    assert not any(path.endswith(".ckpt") for path in _tree(first))

    client = _Client(chunk_start_mode="none")
    receipt = tmp_path / "upload-receipt.json"
    upload_package_with_verifier(
        client,
        "0",
        first,
        repo_root=REPO,
        receipt_path=receipt,
        verifier=results.prepare_bes2_results_verifier(SOURCE_SHA),
        chunked_threshold=1,
    )
    stored = [name for action, name in client.events if action == "stored"]
    assert stored[-1] == "READY.json"
    assert receipt.is_file()

    choices = _build_parser()._subparsers._group_actions[0].choices
    assert {
        "build-bes2-results",
        "verify-bes2-results",
        "upload-bes2-results",
        "download-bes2-results",
    } <= set(choices)


def test_bes2_results_verifier_rejects_extra_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, summary = _fixture_evidence(tmp_path)
    monkeypatch.setattr(
        results,
        "_collect_evidence",
        lambda **_kwargs: (evidence, summary),
    )
    output = tmp_path / "output"
    output.mkdir()
    package = results.build_bes2_results(
        repo=REPO,
        diagnostic_root=tmp_path / "evidence-source",
        original_runs_root=tmp_path / "original-runs",
        job_log_dir=tmp_path / "job-logs",
        output_dir=output,
        max_part_bytes=1024,
    )
    (package / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PackageError, match="physical tree"):
        results.verify_bes2_results(
            package, expected_source_git_sha=SOURCE_SHA
        )


def test_bes2_results_secret_scan_rejects_runtime_credentials(
    tmp_path: Path,
) -> None:
    path = tmp_path / "controller.log"
    path.write_text("BOX_FOLDER_ID=secret-folder\n", encoding="utf-8")
    with pytest.raises(PackageError, match="secret"):
        results._scan(path)


def test_bes2_results_recomputes_causal_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence, summary = _fixture_evidence(tmp_path)
    root_cause_path = evidence[f"control/{results.ROOT_CAUSE_FILENAME}"]
    root_cause = json.loads(root_cause_path.read_text(encoding="utf-8"))
    root_cause["decision"]["aggregate"]["classification"] = "mixed-mechanism"
    root_cause_path.write_bytes(_json_bytes(root_cause))
    summary["root_cause_sha256"] = hashlib.sha256(
        root_cause_path.read_bytes()
    ).hexdigest()
    summary["classification"] = root_cause["decision"]["aggregate"][
        "classification"
    ]
    monkeypatch.setattr(
        results,
        "_collect_evidence",
        lambda **_kwargs: (evidence, summary),
    )
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(PackageError, match="not reproducible"):
        results.build_bes2_results(
            repo=REPO,
            diagnostic_root=tmp_path / "evidence-source",
            original_runs_root=tmp_path / "original-runs",
            job_log_dir=tmp_path / "job-logs",
            output_dir=output,
            max_part_bytes=1024,
        )
