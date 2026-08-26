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
    RANDOM_FROZEN_DEV_F1,
    S2_FROZEN_DEV_F1,
    summarize_payload,
)
from test_h100_handoff import _Client


REPO = Path(__file__).resolve().parents[1]
SOURCE_SHA = "1" * 40


def _write(root: Path, logical: str, data: bytes) -> Path:
    path = root / logical
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def _fixture_evidence(tmp_path: Path) -> tuple[dict[str, Path], dict]:
    source = tmp_path / "evidence-source"
    source.mkdir()
    evidence: dict[str, Path] = {}
    run_bytes: dict[tuple[str, str], bytes] = {}
    readiness = {"source": {"git_sha": SOURCE_SHA}}
    readiness_bytes = _json_bytes(readiness)
    readiness_sha256 = hashlib.sha256(readiness_bytes).hexdigest()
    evidence[f"control/{results.READY_FILENAME}"] = _write(
        source,
        f"control/{results.READY_FILENAME}",
        readiness_bytes,
    )
    replay_f1 = S2_FROZEN_DEV_F1
    reset_f1 = replay_f1 + 0.3 * (
        RANDOM_FROZEN_DEV_F1 - replay_f1
    )

    for stage in results.RUN_STAGES:
        for variant in results.DIAGNOSTIC_VARIANTS:
            for filename in results.RUN_FILES:
                logical = results._logical_run_file(variant, stage, filename)
                if filename == "diagnostic_metrics.json":
                    best_f1 = (
                        replay_f1
                        if variant == "current_replay"
                        else reset_f1
                    )
                    best_dev = {
                        "f1": best_f1,
                        "precision": best_f1,
                        "recall": best_f1,
                        "threshold": 0.5,
                    }
                    data = _json_bytes(
                        {
                            "diagnostic_metrics_schema": 1,
                            "status": "complete",
                            "purpose": "bes2-root-cause",
                            "variant": variant,
                            "stage": stage,
                            "created_utc": "2026-08-26T00:00:00+00:00",
                            "readiness": {
                                "path": "/judy/BES2_DIAGNOSTIC_READY.json",
                                "sha256": readiness_sha256,
                            },
                            "training": {
                                "training_result": {
                                    "best_dev_f1": best_f1,
                                    "best_dev": best_dev,
                                }
                            },
                            "dev_evidence": {
                                "selected_operating_point": best_dev,
                                "leave_one_dev_scene_out": {
                                    "scene": {
                                        "threshold": 0.5,
                                        "retained_seven": {
                                            "f1": best_f1,
                                        },
                                    }
                                },
                            },
                            "activation_statistics": {},
                            "layer_drift": {},
                            "runtime_provenance": {},
                            "gpu_hours": 1.0,
                        }
                    )
                    run_bytes[(variant, stage)] = data
                elif filename.endswith(".json"):
                    data = _json_bytes({"variant": variant, "stage": stage})
                else:
                    data = f"{variant} {stage} controller log\n".encode()
                evidence[logical] = _write(source, logical, data)

    bindings = {
        variant: {
            "path": (
                f"/judy/diagnostic/bes2-{variant}-full-f100-s0/"
                "diagnostic_metrics.json"
            ),
            "sha256": hashlib.sha256(
                run_bytes[(variant, "full")]
            ).hexdigest(),
        }
        for variant in results.DIAGNOSTIC_VARIANTS
    }
    full_metrics = {
        variant: json.loads(run_bytes[(variant, "full")])
        for variant in results.DIAGNOSTIC_VARIANTS
    }
    root_cause = summarize_payload(
        full_metrics["current_replay"],
        full_metrics["first_conv_reset"],
        replay_binding=bindings["current_replay"],
        reset_binding=bindings["first_conv_reset"],
    )
    root_cause["created_utc"] = "2026-08-26T00:00:00+00:00"
    control_payloads = {
        name: {}
        for name in results.CONTROL_JSON
        if name not in {results.ROOT_CAUSE_FILENAME, results.READY_FILENAME}
    }
    control_payloads[results.ROOT_CAUSE_FILENAME] = root_cause
    for name, payload in control_payloads.items():
        logical = f"control/{name}"
        evidence[logical] = _write(source, logical, _json_bytes(payload))
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
    hardware_logical = "runtime/h100_runtime-5001-r0.json"
    evidence[hardware_logical] = _write(
        source, hardware_logical, _json_bytes(hardware)
    )
    for index, stage in enumerate(("audit", "probe", "full"), start=1):
        logical = f"slurm/xview3-bes2-{stage}-{5000 + index}.out"
        evidence[logical] = _write(
            source, logical, f"{stage} complete\n".encode()
        )

    root_cause_path = evidence[f"control/{results.ROOT_CAUSE_FILENAME}"]
    readiness_path = evidence[f"control/{results.READY_FILENAME}"]
    summary = {
        "source_git_sha": SOURCE_SHA,
        "readiness_sha256": hashlib.sha256(
            readiness_path.read_bytes()
        ).hexdigest(),
        "root_cause_sha256": hashlib.sha256(
            root_cause_path.read_bytes()
        ).hexdigest(),
        "classification": root_cause["decision"]["classification"],
        "decision_status": "determinate",
        "replacement_eligibility": root_cause["decision"][
            "replacement_eligibility"
        ],
        "probe_gpu_hours": 8.0,
        "full_gpu_hours": 100.0,
        "total_pair_gpu_hours": 108.0,
        "created_utc": "2026-08-26T00:00:00+00:00",
    }
    # A large checkpoint may coexist on Judy but is never selected.
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
    root_cause["decision"]["classification"] = (
        "stem-conversion-explains-most-deficit"
    )
    root_cause_path.write_bytes(_json_bytes(root_cause))
    summary["root_cause_sha256"] = hashlib.sha256(
        root_cause_path.read_bytes()
    ).hexdigest()
    summary["classification"] = root_cause["decision"]["classification"]
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
