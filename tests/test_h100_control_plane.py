from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.h100 import contracts, cutover, operator_cutover
from scripts.handoff.box import (
    download_package_with_verifier,
    upload_package_with_verifier,
)
from scripts.handoff.control import (
    _build_control_package,
    control_package_identity,
    prepare_control_verifier,
    verify_control_package,
)
from scripts.handoff.package import (
    PackageError,
    _canonical_json,
    _hash_file,
    _physical_record,
)
from scripts.handoff.runtime_bootstrap import _generate_runtime_bootstrap
from test_h100_handoff import _Client

REPO = Path(__file__).resolve().parents[1]
DIAGNOSTIC_SCHEMA = REPO / "slurm/h100/V100_DIAGNOSTIC_ISOLATION.schema.json"


def _run(*argv: str, cwd: Path) -> str:
    return subprocess.run(
        list(argv), cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))


def _fixture_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-q", "-b", "fixture-control", cwd=repo)
    _run("git", "config", "user.name", "John Roth", cwd=repo)
    _run(
        "git",
        "config",
        "user.email",
        "jroth1414@users.noreply.github.com",
        cwd=repo,
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "fixture", cwd=repo)
    return repo, _run("git", "rev-parse", "HEAD", cwd=repo)


def _reference_sources(tmp_path: Path) -> dict[str, Path]:
    values = {
        "references/REFERENCE_CAMPAIGN.json": {"schema": 1},
        "references/locateanything-zs/final_metrics.json": {"reference": "R3"},
        "references/locateanything-zs/runtime_provenance.json": {"run": "R3"},
        "references/yolo26-f100/final_metrics.json": {"reference": "R2"},
        "references/yolo26-f100/runtime_provenance.json": {"run": "R2"},
    }
    result = {}
    for relative, value in values.items():
        source = tmp_path / "source" / relative
        _json(source, value)
        result[relative] = source
    return result


def _reference_bindings(commit: str) -> dict[str, str]:
    return {
        "evaluation_contract": "vessel-hm-positive-low-ignore-v2",
        "reference_git_sha": commit,
        "reference_campaign_id": "sprint7f-v100-references-20260804",
        "v100_core_git_sha": "48e10534a8c7baf0662acd548f52928da69f23c8",
        "v100_core_campaign_id": "fresh34-v100-fp32-20260726",
    }


def test_control_package_is_deterministic_closed_and_box_round_trips(tmp_path):
    repo, commit = _fixture_repo(tmp_path)
    sources = _reference_sources(tmp_path)
    bindings = _reference_bindings(commit)
    first_out = tmp_path / "first"
    second_out = tmp_path / "second"
    first_out.mkdir()
    second_out.mkdir()
    first = _build_control_package(
        kind="references",
        repo_root=repo,
        source_files=sources,
        bindings=bindings,
        output_dir=first_out,
        branch="fixture-control",
        production=False,
    )
    second = _build_control_package(
        kind="references",
        repo_root=repo,
        source_files=sources,
        bindings=bindings,
        output_dir=second_out,
        branch="fixture-control",
        production=False,
    )
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first.name == second.name
    assert first_files == second_files
    manifest = verify_control_package(
        first, expected_kind="references", expected_bindings=bindings
    )
    assert manifest["producer_git_sha"] == commit
    assert set(first_files) == set(sources) | {
        "manifest.json", "SHA256SUMS", "READY.json"
    }

    verifier = prepare_control_verifier("references", bindings)
    client = _Client(chunk_start_mode="none")
    receipt = tmp_path / "upload.json"
    upload_package_with_verifier(
        client,
        "0",
        first,
        repo_root=repo,
        receipt_path=receipt,
        verifier=verifier,
        chunked_threshold=1,
    )
    stored = [name for action, name in client.events if action == "stored"]
    assert stored[-1] == "READY.json"
    identity = control_package_identity(first, expected_kind="references")
    replacement_identity = {**identity, "ready_sha256": "0" * 64}
    with pytest.raises(PackageError, match="out-of-band identity"):
        control_package_identity(
            first,
            expected_kind="references",
            expected_bindings=bindings,
            expected_identity=replacement_identity,
        )
    downloaded = download_package_with_verifier(
        client,
        "0",
        tmp_path / "downloaded",
        repo_root=repo,
        expected_ready_sha256=identity["ready_sha256"],
        expected_manifest_sha256=identity["manifest_sha256"],
        expected_sha256sums_sha256=identity["sha256sums_sha256"],
        expected_package_id=identity["package_id"],
        verifier=verifier,
    )
    assert {
        path.relative_to(downloaded).as_posix(): path.read_bytes()
        for path in downloaded.rglob("*")
        if path.is_file()
    } == first_files
    assert not list(tmp_path.rglob("*.partial"))


def test_control_package_rejects_allowlist_tamper_and_credentials(tmp_path):
    repo, commit = _fixture_repo(tmp_path)
    sources = _reference_sources(tmp_path)
    bindings = _reference_bindings(commit)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(PackageError, match="exact allowlist"):
        _build_control_package(
            kind="references",
            repo_root=repo,
            source_files={key: value for key, value in sources.items() if "yolo" not in key},
            bindings=bindings,
            output_dir=output,
            branch="fixture-control",
            production=False,
        )
    secret = sources["references/yolo26-f100/final_metrics.json"]
    _json(
        secret,
        {"boxAppSettings": {"clientSecret": "must-never-cross"}},
    )
    with pytest.raises(PackageError, match="credential material"):
        _build_control_package(
            kind="references",
            repo_root=repo,
            source_files=sources,
            bindings=bindings,
            output_dir=output,
            branch="fixture-control",
            production=False,
        )

    _json(secret, {"reference": "R2"})
    package = _build_control_package(
        kind="references",
        repo_root=repo,
        source_files=sources,
        bindings=bindings,
        output_dir=output,
        branch="fixture-control",
        production=False,
    )
    target = package / "references/yolo26-f100/final_metrics.json"
    target.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(PackageError, match="mismatch"):
        verify_control_package(package, expected_kind="references")


def _metric(
    *, tp: int = 1, fp: int = 1, fn: int = 1, ignored: int = 0
) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored,
    }


def _score(
    member_key: str, members: list[str], *, positive_support: int
) -> dict[str, object]:
    quotient, remainder = divmod(positive_support, len(members))
    member_metrics = {
        member: _metric(
            tp=quotient + (1 if index < remainder else 0),
            fp=0,
            fn=0,
        )
        for index, member in enumerate(members)
    }
    aggregate = _metric(tp=positive_support, fp=0, fn=0)
    return {
        "aggregate": aggregate,
        "slices": {"dark": dict(aggregate), "near_shore": dict(aggregate)},
        member_key: {
            member: {
                "aggregate": dict(metric),
                "slices": {"dark": dict(metric), "near_shore": dict(metric)},
            }
            for member, metric in member_metrics.items()
        },
    }


def _provenance(git_sha: str, campaign_id: str) -> dict[str, object]:
    return {
        "campaign_id": campaign_id,
        "git_sha": git_sha,
        "environment_sha256": "1" * 64,
        "environment_lock_sha256": "2" * 64,
        "campaign_manifest_sha256": "3" * 64,
        "runtime_launcher_sha256": "4" * 64,
        "hardware": "Tesla V100-SXM2-32GB",
        "container_local_gpu": 0,
        "gpu_uuid": "GPU-fixture",
        "started_utc": "2026-08-04T00:00:00+00:00",
        "finished_utc": "2026-08-04T01:00:00+00:00",
        "elapsed_hours": 1.0,
        "gpu_hours": 1.0,
        "reference_precision": "float32",
    }


def test_cutover_reference_validators_require_corrected_schema2(tmp_path):
    reference_git = "9" * 40
    source_git = reference_git
    campaign_id = "fresh34-v100-fp32-20260726"
    provenance = _provenance(reference_git, campaign_id)

    r2 = tmp_path / "yolo26-f100"
    dev = _score(
        "per_scene",
        sorted(cutover.R2_DEV_SCENE_IDS),
        positive_support=cutover.R2_GT_COUNTS["dev"]["positive"],
    )
    test = _score(
        "per_scene",
        sorted(cutover.R2_TEST_SCENE_IDS),
        positive_support=cutover.R2_GT_COUNTS["test"]["positive"],
    )
    r2_metrics = {
        "result_schema": 2,
        "exp_id": "yolo26-f100",
        "reference": "R2",
        "source_git_sha": source_git,
        "scored_at": "2026-08-04T01:00:00+00:00",
        "inference_precision": "float32",
        "training_disposition": "preserved-best-pt-rescore-only",
        "checkpoint": {
            "relative_path": "weights/best.pt",
            "sha256": cutover.R2_CHECKPOINT_SHA256,
        },
        "ground_truth_contract": {
            **cutover.GT_CONTRACT,
            "dev_counts": cutover.R2_GT_COUNTS["dev"],
            "test_counts": cutover.R2_GT_COUNTS["test"],
        },
        "threshold_source": {
            "split": "dev",
            "checkpoint_sha256": cutover.R2_CHECKPOINT_SHA256,
        },
        "threshold": 0.5,
        "dev": dev,
        "test": test,
        "dev_f1": dev["aggregate"]["f1"],
        "test_f1": test["aggregate"]["f1"],
        "test_precision": test["aggregate"]["precision"],
        "test_recall": test["aggregate"]["recall"],
        "test_near_shore_f1": test["slices"]["near_shore"]["f1"],
    }
    _json(r2 / "final_metrics.json", r2_metrics)
    _json(r2 / "runtime_provenance.json", provenance)
    kwargs = {
        "expected_git_sha": reference_git,
        "expected_campaign_id": campaign_id,
    }
    cutover.validate_r2(r2, **kwargs)
    for split_name, scene_ids in (
        ("dev", cutover.R2_DEV_SCENE_IDS),
        ("test", cutover.R2_TEST_SCENE_IDS),
    ):
        valid_score = r2_metrics[split_name]
        r2_metrics[split_name] = _score(
            "per_scene",
            sorted(scene_ids),
            positive_support=cutover.R2_GT_COUNTS[split_name]["positive"] - 1,
        )
        _json(r2 / "final_metrics.json", r2_metrics)
        with pytest.raises(RuntimeError, match=f"R2 {split_name} aggregate positive"):
            cutover.validate_r2(r2, **kwargs)
        r2_metrics[split_name] = valid_score
    r2_metrics["checkpoint"]["sha256"] = "0" * 64
    _json(r2 / "final_metrics.json", r2_metrics)
    with pytest.raises(RuntimeError, match="best.pt binding"):
        cutover.validate_r2(r2, **kwargs)

    r3 = tmp_path / "locateanything-zs"
    dev_scenes = sorted(cutover.R2_DEV_SCENE_IDS)
    candidates = [
        {
            "chip": f"{dev_scenes[index % len(dev_scenes)]}/chip-{index}.npy",
            "sidecar_sha256": f"{index:064x}",
        }
        for index in range(cutover.R3_CANDIDATE_COUNT)
    ]
    entries = [
        candidates[(index * len(candidates)) // 200] for index in range(200)
    ]
    chips = [Path(entry["chip"]).with_suffix("").as_posix() for entry in entries]
    prompt_metrics = _score("per_chip", chips, positive_support=200)
    prompt_score = {
        "f1": prompt_metrics["aggregate"]["f1"],
        "precision": prompt_metrics["aggregate"]["precision"],
        "recall": prompt_metrics["aggregate"]["recall"],
        "threshold": 0.5,
        **prompt_metrics,
    }
    r3_metrics = {
        "result_schema": 2,
        "exp_id": "locateanything-zs",
        "reference": "R3",
        "source_git_sha": source_git,
        "scored_at": "2026-08-04T02:00:00+00:00",
        "model": {
            "id": "nvidia/LocateAnything-3B",
            "revision": cutover.R3_MODEL_REVISION,
            "payload_sha256": cutover.R3_MODEL_PAYLOAD_SHA256,
            "source_note_sha256": "5" * 64,
            "license_sha256": "6" * 64,
        },
        "precision": "bfloat16",
        "eval_contract_disposition": "full-rerun-under-corrected-contract",
        "legacy_result_reused": False,
        "ground_truth_contract": {
            **cutover.GT_CONTRACT,
            "sample_counts": {"positive": 200, "background": 10, "ignore": 5},
        },
        "sample": {
            "policy": "sorted-foreground-dev-chips-even-stride",
            "algorithm_version": "floor-index-v1",
            "candidate_count": cutover.R3_CANDIDATE_COUNT,
            "candidate_sha256": hashlib.sha256(
                json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "candidates": candidates,
            "n_chips": 200,
            "sha256": hashlib.sha256(
                json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "entries": entries,
        },
        "per_prompt": {
            prompt: json.loads(json.dumps(prompt_score))
            for prompt in ("ship", "vessel", "boat")
        },
        "best_prompt": "ship",
    }
    _json(r3 / "final_metrics.json", r3_metrics)
    r3_provenance = dict(provenance)
    r3_provenance["reference_precision"] = "bfloat16"
    _json(r3 / "runtime_provenance.json", r3_provenance)
    cutover.validate_r3(r3, **kwargs)
    bad_prompt_metrics = _score(
        "per_chip",
        chips,
        positive_support=r3_metrics["ground_truth_contract"]["sample_counts"][
            "positive"
        ]
        - 1,
    )
    bad_prompt = {
        "f1": bad_prompt_metrics["aggregate"]["f1"],
        "precision": bad_prompt_metrics["aggregate"]["precision"],
        "recall": bad_prompt_metrics["aggregate"]["recall"],
        "threshold": 0.5,
        **bad_prompt_metrics,
    }
    for prompt in ("ship", "vessel", "boat"):
        valid_prompt = r3_metrics["per_prompt"][prompt]
        r3_metrics["per_prompt"][prompt] = bad_prompt
        _json(r3 / "final_metrics.json", r3_metrics)
        with pytest.raises(RuntimeError, match=f"R3 {prompt} aggregate positive"):
            cutover.validate_r3(r3, **kwargs)
        r3_metrics["per_prompt"][prompt] = valid_prompt
    r3_metrics["legacy_result_reused"] = True
    _json(r3 / "final_metrics.json", r3_metrics)
    with pytest.raises(RuntimeError, match="identity/disposition"):
        cutover.validate_r3(r3, **kwargs)


def test_cutover_binds_reference_campaign_to_repo_and_core_identity(tmp_path):
    repo = tmp_path / "repo"
    (repo / "locks").mkdir(parents=True)
    (repo / "scripts").mkdir()
    lock = repo / "locks/env-v100node.txt"
    launcher = repo / "scripts/run_corrected_references.py"
    lock.write_text("Torch==2.11.0+cu126\n", encoding="utf-8")
    launcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    reference_git = "9" * 40
    core_git = "8" * 40
    manifest_path = tmp_path / "REFERENCE_CAMPAIGN.json"
    manifest = {
        "schema": 1,
        "campaign_role": "corrected-v100-references",
        "campaign_id": "reference-campaign",
        "core_campaign_id": "core-campaign",
        "core_git_sha": core_git,
        "git_sha": reference_git,
        "environment_sha256": hashlib.sha256(
            b"torch==2.11.0+cu126\n"
        ).hexdigest(),
        "environment_lock_sha256": _hash_file(lock),
        "runtime_launcher_sha256": _hash_file(launcher),
    }
    _json(manifest_path, manifest)
    record = cutover.validate_reference_campaign_manifest(
        manifest_path,
        expected_reference_git_sha=reference_git,
        expected_reference_campaign_id="reference-campaign",
        expected_v100_core_git_sha=core_git,
        expected_v100_core_campaign_id="core-campaign",
        repo_root=repo,
    )
    assert record == {
        "manifest": manifest,
        "manifest_sha256": _hash_file(manifest_path),
    }

    provenance = _provenance(reference_git, "reference-campaign")
    expected_hashes = {
        "environment_sha256": manifest["environment_sha256"],
        "environment_lock_sha256": manifest["environment_lock_sha256"],
        "campaign_manifest_sha256": record["manifest_sha256"],
        "runtime_launcher_sha256": manifest["runtime_launcher_sha256"],
    }
    provenance.update(expected_hashes)
    cutover.validate_reference_provenance(
        provenance,
        expected_git_sha=reference_git,
        expected_campaign_id="reference-campaign",
        expected_hashes=expected_hashes,
    )
    provenance["runtime_launcher_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="campaign manifest"):
        cutover.validate_reference_provenance(
            provenance,
            expected_git_sha=reference_git,
            expected_campaign_id="reference-campaign",
            expected_hashes=expected_hashes,
        )

    manifest["core_git_sha"] = "7" * 40
    _json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="bindings mismatch"):
        cutover.validate_reference_campaign_manifest(
            manifest_path,
            expected_reference_git_sha=reference_git,
            expected_reference_campaign_id="reference-campaign",
            expected_v100_core_git_sha=core_git,
            expected_v100_core_campaign_id="core-campaign",
            repo_root=repo,
        )


def _diagnostic_fixture(
    tmp_path: Path,
    *,
    current_remaining_v100_wall_hours: float = 2.5,
    v100_execution_status: str = contracts.V100_DIAGNOSTIC_RUNNING,
) -> dict[str, object]:
    h100_git = "1" * 40
    reference_git = "9" * 40
    core_git = "8" * 40
    h100_campaign = "h100-campaign"
    reference_campaign = "reference-campaign"
    core_campaign = "core-campaign"
    reference_manifest = {
        "schema": 1,
        "campaign_role": "corrected-v100-references",
        "campaign_id": reference_campaign,
        "core_campaign_id": core_campaign,
        "core_git_sha": core_git,
        "git_sha": reference_git,
        "environment_sha256": "1" * 64,
        "environment_lock_sha256": "2" * 64,
        "runtime_launcher_sha256": "4" * 64,
    }
    reference_manifest_sha256 = hashlib.sha256(
        _canonical_json(reference_manifest)
    ).hexdigest()
    external = (tmp_path / "external").absolute()
    external.mkdir()
    cutover_path = external / "CUTOVER_READY.json"
    cutover_payload = {
        "schema": 2,
        "status": "cutover-ready",
        "created_utc": "2026-08-04T00:00:00+00:00",
        "h100_campaign_id": h100_campaign,
        "acceptance": {
            "uuid": "acceptance-uuid",
            "source": {"git_sha": h100_git},
            "evaluation_ground_truth": {"sha256": "a" * 64},
            "base_payload": {"sha256sums_sha256": "b" * 64},
            "runtime_amendment": {"sha256sums_sha256": "c" * 64},
            "external_controls_policy": contracts.EXTERNAL_CONTROLS_POLICY,
            "projection": {
                "conservative_h100_wall_hours": 2.0,
            },
        },
        "cutover_forecast": {
            "conservative_h100_wall_hours": 2.0,
            "current_remaining_v100_wall_hours": current_remaining_v100_wall_hours,
            "v100_diagnostic_status": v100_execution_status,
            "h100_scientifically_mandatory": True,
        },
        "references_control": {
            "package_id": "xview3-control-references-fixture",
            "kind": "references",
            "direction": "v100-to-judy",
            "identity_sha256": "1" * 64,
            "producer_git_sha": h100_git,
            "manifest_sha256": "2" * 64,
            "ready_sha256": "3" * 64,
            "sha256sums_sha256": "4" * 64,
        },
        "reference_campaign": {
            "manifest": reference_manifest,
            "manifest_sha256": reference_manifest_sha256,
        },
        "references": {
            name: {
                "metrics": {"result_schema": 2},
                "metrics_sha256": "5" * 64,
                "provenance": {
                    "git_sha": reference_git,
                    "campaign_id": reference_campaign,
                    "environment_sha256": "1" * 64,
                    "environment_lock_sha256": "2" * 64,
                    "campaign_manifest_sha256": reference_manifest_sha256,
                    "runtime_launcher_sha256": "4" * 64,
                },
                "provenance_sha256": "6" * 64,
            }
            for name in ("r2", "r3")
        },
        "v100_action": "none; this guard never stops or signals V100 processes",
    }
    _json(cutover_path, cutover_payload)
    cutover_sha = _hash_file(cutover_path)
    attestation_path = external / "V100_DIAGNOSTIC_ISOLATION.json"
    attestation_payload = {
        "schema": 1,
        "status": "v100-diagnostic-isolated",
        "created_utc": "2026-08-04T01:00:00+00:00",
        "attestation": "external-human-operator",
        "cutover_ready_sha256": cutover_sha,
        "h100": {
            "acceptance_uuid": "acceptance-uuid",
            "git_sha": h100_git,
            "campaign_id": h100_campaign,
        },
        "v100": {
            "git_sha": core_git,
            "campaign_id": core_campaign,
            "execution_status": v100_execution_status,
            "diagnostic_status": "non-reportable-diagnostic",
        },
        "namespaces": {
            "v100_runs_root": "/nfs/v100/xview3/runs",
            "h100_runs_root": "/projects/geofam/jroth/xview3-h100-runs",
            "disjoint": True,
        },
        "h100_suppression": {
            "v100_completions_suppress_h100": False,
            "v100_checkpoints_resume_h100": False,
            "mixed_hardware_curve_allowed": False,
        },
        "post_diagnostic": {
            "safe_stop_archive": "optional-after-diagnostic",
            "required_before_h100_campaign": False,
        },
    }
    _json(attestation_path, attestation_payload)
    return {
        "cutover_ready": cutover_path,
        "cutover_ready_sha256": cutover_sha,
        "attestation": attestation_path,
        "attestation_sha256": _hash_file(attestation_path),
        "expected_h100_git_sha": h100_git,
        "expected_h100_campaign_id": h100_campaign,
        "expected_h100_runs_root": "/projects/geofam/jroth/xview3-h100-runs",
        "expected_reference_git_sha": reference_git,
        "expected_reference_campaign_id": reference_campaign,
        "expected_v100_core_git_sha": core_git,
        "expected_v100_core_campaign_id": core_campaign,
    }


def test_operator_diagnostic_isolation_is_immutable_and_never_requires_stop(tmp_path):
    fixture = _diagnostic_fixture(tmp_path)
    result = operator_cutover.validate_diagnostic_isolation(**fixture)
    assert result["status"] == "operator-diagnostic-isolation-validated"
    assert "stopped_utc" not in json.dumps(result)
    evidence = operator_cutover.persist_diagnostic_isolation_evidence(
        meta_root=(tmp_path / "meta").absolute(),
        cutover_ready=fixture["cutover_ready"],
        cutover_ready_sha256=fixture["cutover_ready_sha256"],
        attestation=fixture["attestation"],
        attestation_sha256=fixture["attestation_sha256"],
    )
    canonical = Path(evidence["v100_diagnostic_isolation"]["path"])
    assert stat.S_IMODE(canonical.stat().st_mode) == 0o444
    assert canonical.read_bytes() == fixture["attestation"].read_bytes()

    payload = json.loads(fixture["attestation"].read_text())
    payload["h100_suppression"]["v100_completions_suppress_h100"] = True
    altered = fixture["attestation"].with_name("altered.json")
    _json(altered, payload)
    with pytest.raises(RuntimeError, match="suppress, resume, or mix"):
        operator_cutover.validate_diagnostic_isolation(
            **{
                **fixture,
                "attestation": altered,
                "attestation_sha256": _hash_file(altered),
            }
        )


def test_operator_binds_attested_h100_namespace_to_actual_judy_runs_root(tmp_path):
    fixture = _diagnostic_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="canonical Judy H100_RUNS_ROOT"):
        operator_cutover.validate_diagnostic_isolation(
            **{
                **fixture,
                "expected_h100_runs_root": "/projects/geofam/jroth/other-h100-runs",
            }
        )


def test_operator_binds_cutover_to_exact_base_and_runtime_packages(tmp_path):
    fixture = _diagnostic_fixture(tmp_path)
    cutover_payload = json.loads(fixture["cutover_ready"].read_text())
    operator_cutover._validate_package_sha256sums_bindings(
        cutover_payload,
        expected_base_payload_sha256sums_sha256="b" * 64,
        expected_runtime_amendment_sha256sums_sha256="c" * 64,
    )
    with pytest.raises(RuntimeError, match="runtime-amendment"):
        operator_cutover._validate_package_sha256sums_bindings(
            cutover_payload,
            expected_base_payload_sha256sums_sha256="b" * 64,
            expected_runtime_amendment_sha256sums_sha256="d" * 64,
        )


def test_operator_accepts_completed_nonreportable_v100_when_remaining_is_zero(
    tmp_path,
):
    fixture = _diagnostic_fixture(
        tmp_path,
        current_remaining_v100_wall_hours=0.0,
        v100_execution_status=contracts.V100_DIAGNOSTIC_COMPLETE,
    )
    result = operator_cutover.validate_diagnostic_isolation(**fixture)
    assert result["attestation"]["v100"]["execution_status"] == (
        "complete-non-reportable-diagnostic"
    )


def test_operator_rejects_attestation_status_that_differs_from_cutover(tmp_path):
    fixture = _diagnostic_fixture(
        tmp_path,
        current_remaining_v100_wall_hours=0.0,
        v100_execution_status=contracts.V100_DIAGNOSTIC_COMPLETE,
    )
    payload = json.loads(fixture["attestation"].read_text())
    payload["v100"]["execution_status"] = contracts.V100_DIAGNOSTIC_RUNNING
    altered = fixture["attestation"].with_name("status-mismatch.json")
    _json(altered, payload)
    with pytest.raises(RuntimeError, match="identity/disposition mismatch"):
        operator_cutover.validate_diagnostic_isolation(
            **{
                **fixture,
                "attestation": altered,
                "attestation_sha256": _hash_file(altered),
            }
        )


def test_diagnostic_isolation_schema_is_closed_and_exact():
    schema = json.loads(DIAGNOSTIC_SCHEMA.read_text())
    assert schema["additionalProperties"] is False
    assert schema["properties"]["status"] == {
        "const": "v100-diagnostic-isolated"
    }
    assert set(schema["required"]) == operator_cutover.DIAGNOSTIC_ATTESTATION_KEYS
    assert set(schema["properties"]["h100"]["required"]) == (
        operator_cutover.DIAGNOSTIC_H100_KEYS
    )
    assert set(schema["properties"]["v100"]["required"]) == (
        operator_cutover.DIAGNOSTIC_V100_KEYS
    )
    execution_schema = schema["properties"]["v100"]["properties"][
        "execution_status"
    ]
    assert set(execution_schema["enum"]) == {
        contracts.V100_DIAGNOSTIC_RUNNING,
        contracts.V100_DIAGNOSTIC_COMPLETE,
    }


def test_standalone_runtime_bootstrap_pins_every_identity(tmp_path):
    repo, commit = _fixture_repo(tmp_path)
    package = tmp_path / "runtime-package"
    bundle = package / "code/xview3-runtime.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"fixture runtime bundle")
    bundle_sha = _hash_file(bundle)
    labels = package / "data/training-view/labels/train.csv"
    labels.parent.mkdir(parents=True)
    labels.write_bytes(b"scene_id,is_vessel,confidence\ntrain-scene,true,HIGH\n")
    labels_sha = _hash_file(labels)
    artifact = {
        "kind": "git_bundle",
        "name": "fixture-control",
        "format": "file",
        "extraction_root": "code/xview3-runtime.bundle",
        "file_count": 1,
        "unpacked_bytes": bundle.stat().st_size,
        "archive_bytes": bundle.stat().st_size,
        "archive_sha256": bundle_sha,
        "archive_sha1": _hash_file(bundle, "sha1"),
        "parts": [_physical_record(bundle, package)],
    }
    labels_artifact = {
        "kind": "training_labels",
        "name": "train-dev8.csv",
        "format": "file",
        "extraction_root": "data/training-view/labels/train.csv",
        "file_count": 1,
        "unpacked_bytes": labels.stat().st_size,
        "archive_bytes": labels.stat().st_size,
        "archive_sha256": labels_sha,
        "archive_sha1": _hash_file(labels, "sha1"),
        "parts": [_physical_record(labels, package)],
    }
    manifest = {
        "format_version": 2,
        "package_type": "h100-runtime-amendment",
        "package_id": f"xview3-h100-runtime-{commit}-{'a' * 64}",
        "source": {
            "branch": "fixture-control",
            "git_commit": commit,
            "git_bundle_sha256": bundle_sha,
        },
        "counts": {"git_bundles": 1, "training_label_artifacts": 1},
        "artifacts": [artifact, labels_artifact],
    }
    _json(package / "manifest.json", manifest)
    (package / "SHA256SUMS").write_text(
        f"{bundle_sha}  code/xview3-runtime.bundle\n"
        f"{labels_sha}  data/training-view/labels/train.csv\n",
        encoding="utf-8",
    )
    _json(
        package / "READY.json",
        {
            "format_version": 2,
            "status": "READY",
            "package_id": manifest["package_id"],
        },
    )
    assert (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines() == [
        f"{bundle_sha}  code/xview3-runtime.bundle",
        f"{labels_sha}  data/training-view/labels/train.csv",
    ]
    first = tmp_path / "pull-runtime.sh"
    receipt = _generate_runtime_bootstrap(
        repo_root=repo,
        runtime_package_root=package,
        output=first,
        verifier=lambda _root: manifest,
        production=False,
    )
    script = first.read_text()
    subprocess.run(["bash", "-n", str(first)], check=True)
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    for value in (
        manifest["package_id"],
        commit,
        bundle_sha,
        labels_sha,
        receipt["ready_sha256"],
        receipt["manifest_sha256"],
        receipt["sha256sums_sha256"],
    ):
        assert value in script
    for relative in (
        "code/xview3-runtime.bundle",
        "data/training-view/labels/train.csv",
    ):
        assert relative in script
    assert "for name in sorted(expected):" in script
    assert 'client.file(str(attr(remote[name], "id"))).download_to(handle)' in script
    assert ".partial" in script
    assert "refusing to overwrite existing path" in script
    assert "scripts.handoff" not in script
    second = tmp_path / "pull-runtime-second.sh"
    _generate_runtime_bootstrap(
        repo_root=repo,
        runtime_package_root=package,
        output=second,
        verifier=lambda _root: manifest,
        production=False,
    )
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(PackageError, match="output exists"):
        _generate_runtime_bootstrap(
            repo_root=repo,
            runtime_package_root=package,
            output=first,
            verifier=lambda _root: manifest,
            production=False,
        )
