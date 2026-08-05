"""Validate H100 readiness plus fresh R2/R3; never signal the V100 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from scripts.h100.acceptance import (
    EXPECTED_FRACTION_WORKLOAD,
    validate_hardware_runtime_contracts,
)
from scripts.h100.build_venv import EXPECTED_PYTHON_VERSION
from scripts.h100.contracts import (
    FROZEN_PATHS,
    V100_DIAGNOSTIC_COMPLETE,
    V100_DIAGNOSTIC_RUNNING,
    V100_DIAGNOSTIC_STATUSES,
    atomic_write_json,
    cutover_acceptance_bindings,
    sha256_file,
    staging_aware_wall_clock,
    validate_bound_cutover_forecast,
    validate_gpu_inventory,
)
from scripts.h100.data_staging import (
    validate_acceptance_data_view_binding,
)
from scripts.h100.host_test_gate import HOST_TESTS, validate_host_gate
from scripts.h100.slurm_smoke import make_bindings as make_smoke_bindings
from scripts.h100.slurm_smoke import validate_smoke_receipt
from scripts.h100.source_validation import validate_source_receipt
from src.references.runtime_provenance import (
    CAMPAIGN_MANIFEST_KEYS,
    CAMPAIGN_MANIFEST_SCHEMA,
    CAMPAIGN_ROLE,
    normalized_environment_lock_sha256,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
R2_CHECKPOINT_SHA256 = (
    "15520cb6cff9d4b01ed5c4a7e039fab763e8e5b0ca5b8e6bffd591ef0d7b8064"
)
R2_GT_COUNTS = {
    "dev": {"positive": 1479, "background": 804, "ignore": 441},
    "test": {"positive": 1165, "background": 420, "ignore": 325},
}
R3_MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
R3_CANDIDATE_COUNT = 1525
R3_MODEL_PAYLOAD_SHA256 = (
    "4b097656ce640f7b6cc446f873af4e8e3a489c8ed27fae5ed67337f48c8318f9"
)
GT_CONTRACT = {
    "version": 2,
    "positive": "is_vessel=true and confidence in {HIGH,MEDIUM}",
    "background": "is_vessel=false and confidence in {HIGH,MEDIUM}",
    "ignore": "confidence=LOW",
}
REFERENCE_HASH_FIELDS = (
    "environment_sha256",
    "environment_lock_sha256",
    "campaign_manifest_sha256",
    "runtime_launcher_sha256",
)
REFERENCE_PROVENANCE_KEYS = {
    "campaign_id",
    "git_sha",
    *REFERENCE_HASH_FIELDS,
    "hardware",
    "container_local_gpu",
    "gpu_uuid",
    "started_utc",
    "finished_utc",
    "elapsed_hours",
    "gpu_hours",
    "reference_precision",
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FROZEN_SPLITS = json.loads((_REPO_ROOT / "data/splits.json").read_text())["splits"]
R2_DEV_SCENE_IDS = frozenset(map(str, _FROZEN_SPLITS["dev"]))
R2_TEST_SCENE_IDS = frozenset(map(str, _FROZEN_SPLITS["test"]))


def write_new_cutover_ready(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise RuntimeError(
            "CUTOVER_READY already exists; archive or remove it only through the "
            "operator cutover procedure"
        )
    atomic_write_json(path, payload)
    path.chmod(0o444)


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite")
    return number


def validate_reference_provenance(
    payload: Mapping[str, object],
    *,
    expected_git_sha: str,
    expected_campaign_id: str,
    expected_hashes: Mapping[str, str] | None = None,
) -> None:
    if set(payload) != REFERENCE_PROVENANCE_KEYS:
        raise RuntimeError("reference provenance keys do not match schema")
    if payload.get("git_sha") != expected_git_sha:
        raise RuntimeError("reference provenance git SHA mismatch")
    if payload.get("campaign_id") != expected_campaign_id:
        raise RuntimeError("reference provenance campaign mismatch")
    if payload.get("hardware") != "Tesla V100-SXM2-32GB":
        raise RuntimeError("reference provenance is not from Tesla V100-SXM2-32GB")
    if not str(payload.get("gpu_uuid", "")).strip():
        raise RuntimeError("reference provenance lacks a GPU UUID")
    gpu_index = payload.get("container_local_gpu")
    if type(gpu_index) is not int or not 0 <= gpu_index < 8:
        raise RuntimeError("reference provenance has an invalid container-local GPU")
    if not payload.get("reference_precision"):
        raise RuntimeError("reference precision is absent")
    elapsed = _finite(payload.get("elapsed_hours"), "reference elapsed_hours")
    gpu_hours = _finite(payload.get("gpu_hours"), "reference gpu_hours")
    if elapsed <= 0 or not math.isclose(elapsed, gpu_hours):
        raise RuntimeError("reference elapsed_hours must be positive")
    started = _reference_timestamp(
        payload.get("started_utc"), "reference started_utc"
    )
    finished = _reference_timestamp(
        payload.get("finished_utc"), "reference finished_utc"
    )
    if finished < started:
        raise RuntimeError("reference finished_utc predates started_utc")
    invalid_hashes = [
        name
        for name in REFERENCE_HASH_FIELDS
        if not HEX64.fullmatch(str(payload.get(name, "")))
    ]
    if invalid_hashes:
        raise RuntimeError(
            "reference provenance lacks explicit 64-hex hashes: "
            + ", ".join(invalid_hashes)
        )
    if expected_hashes is not None:
        if set(expected_hashes) != set(REFERENCE_HASH_FIELDS):
            raise RuntimeError("expected reference provenance hashes are incomplete")
        mismatches = [
            name
            for name in REFERENCE_HASH_FIELDS
            if payload.get(name) != expected_hashes[name]
        ]
        if mismatches:
            raise RuntimeError(
                "reference provenance differs from campaign manifest: "
                + ", ".join(mismatches)
            )


def _reference_json(path: Path, label: str) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload, sha256_file(path)


def validate_reference_campaign_manifest(
    path: Path,
    *,
    expected_reference_git_sha: str,
    expected_reference_campaign_id: str,
    expected_v100_core_git_sha: str,
    expected_v100_core_campaign_id: str,
    repo_root: Path = _REPO_ROOT,
) -> dict[str, object]:
    """Bind transferred R2/R3 receipts to one exact reference execution."""

    manifest, manifest_sha256 = _reference_json(
        path, "corrected-reference campaign manifest"
    )
    lock = repo_root / "locks/env-v100node.txt"
    launcher = repo_root / "scripts/run_corrected_references.py"
    for source, label in ((lock, "V100 environment lock"), (launcher, "reference launcher")):
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"{label} must be a regular non-symlink file")
    expected = {
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_role": CAMPAIGN_ROLE,
        "campaign_id": expected_reference_campaign_id,
        "core_campaign_id": expected_v100_core_campaign_id,
        "core_git_sha": expected_v100_core_git_sha,
        "git_sha": expected_reference_git_sha,
        "environment_sha256": normalized_environment_lock_sha256(lock),
        "environment_lock_sha256": sha256_file(lock),
        "runtime_launcher_sha256": sha256_file(launcher),
    }
    if set(manifest) != set(CAMPAIGN_MANIFEST_KEYS):
        raise RuntimeError("corrected-reference campaign manifest schema is invalid")
    if manifest != expected:
        mismatches = sorted(
            key for key, value in expected.items() if manifest.get(key) != value
        )
        raise RuntimeError(
            "corrected-reference campaign manifest bindings mismatch: "
            + ", ".join(mismatches)
        )
    return {"manifest": manifest, "manifest_sha256": manifest_sha256}


def _reference_result(
    *,
    metrics_path: Path,
    provenance_path: Path,
    metrics: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "metrics": dict(metrics),
        "metrics_sha256": sha256_file(metrics_path),
        "provenance": dict(provenance),
        "provenance_sha256": sha256_file(provenance_path),
    }


def _reference_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} timestamp is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} timestamp lacks a timezone")
    return parsed


def _metric_record(value: object, label: str) -> Mapping[str, object]:
    keys = {"f1", "precision", "recall", "tp", "fp", "fn", "ignored_predictions"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"{label} metric schema is invalid")
    for key in ("f1", "precision", "recall"):
        number = _finite(value[key], f"{label}.{key}")
        if not 0.0 <= number <= 1.0:
            raise RuntimeError(f"{label}.{key} is outside [0,1]")
    for key in ("tp", "fp", "fn", "ignored_predictions"):
        count = value[key]
        if type(count) is not int or count < 0:
            raise RuntimeError(f"{label}.{key} must be a nonnegative integer")
    tp = int(value["tp"])
    fp = int(value["fp"])
    fn = int(value["fn"])
    expected_precision = tp / (tp + fp) if tp + fp else 0.0
    expected_recall = tp / (tp + fn) if tp + fn else 0.0
    expected_f1 = (
        2.0 * expected_precision * expected_recall
        / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    for key, expected in (
        ("precision", expected_precision),
        ("recall", expected_recall),
        ("f1", expected_f1),
    ):
        if not math.isclose(
            float(value[key]), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(f"{label}.{key} is inconsistent with TP/FP/FN")
    return value


def _require_metric_sum(
    aggregate: Mapping[str, object],
    members: list[Mapping[str, object]],
    *,
    label: str,
) -> None:
    for key in ("tp", "fp", "fn", "ignored_predictions"):
        if int(aggregate[key]) != sum(int(member[key]) for member in members):
            raise RuntimeError(f"{label}.{key} does not equal the member sum")


def _score_record(
    value: object,
    *,
    label: str,
    member_key: str,
    expected_members: int | None = None,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "aggregate", "slices", member_key
    }:
        raise RuntimeError(f"{label} score schema is invalid")
    _metric_record(value["aggregate"], f"{label}.aggregate")
    slices = value["slices"]
    if not isinstance(slices, Mapping) or set(slices) != {"dark", "near_shore"}:
        raise RuntimeError(f"{label} slices are invalid")
    for name, record in slices.items():
        _metric_record(record, f"{label}.slices.{name}")
    members = value[member_key]
    if not isinstance(members, Mapping) or not members:
        raise RuntimeError(f"{label}.{member_key} is empty or invalid")
    if expected_members is not None and len(members) != expected_members:
        raise RuntimeError(
            f"{label}.{member_key} count is {len(members)}, expected {expected_members}"
        )
    member_aggregates: list[Mapping[str, object]] = []
    member_slice_metrics: dict[str, list[Mapping[str, object]]] = {
        "dark": [],
        "near_shore": [],
    }
    for name, record in members.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"{label}.{member_key} has an invalid name")
        if not isinstance(record, Mapping) or set(record) != {"aggregate", "slices"}:
            raise RuntimeError(f"{label}.{member_key}.{name} schema is invalid")
        member_aggregates.append(
            _metric_record(
                record["aggregate"], f"{label}.{member_key}.{name}.aggregate"
            )
        )
        member_slices = record["slices"]
        if (
            not isinstance(member_slices, Mapping)
            or set(member_slices) != {"dark", "near_shore"}
        ):
            raise RuntimeError(f"{label}.{member_key}.{name} slices are invalid")
        for slice_name, metric in member_slices.items():
            member_slice_metrics[slice_name].append(
                _metric_record(
                    metric, f"{label}.{member_key}.{name}.slices.{slice_name}"
                )
            )
    _require_metric_sum(value["aggregate"], member_aggregates, label=f"{label}.aggregate")
    for slice_name in ("dark", "near_shore"):
        _require_metric_sum(
            slices[slice_name],
            member_slice_metrics[slice_name],
            label=f"{label}.slices.{slice_name}",
        )
    return value


def _gt_counts(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "positive", "background", "ignore"
    }:
        raise RuntimeError(f"{label} GT counts are invalid")
    if any(type(count) is not int or count < 0 for count in value.values()):
        raise RuntimeError(f"{label} GT counts must be nonnegative integers")


def validate_r2(run_dir: Path, **provenance_kwargs) -> dict:
    metrics_path = run_dir / "final_metrics.json"
    provenance_path = run_dir / "runtime_provenance.json"
    metrics, _metrics_sha256 = _reference_json(metrics_path, "R2 final metrics")
    expected_keys = {
        "result_schema", "exp_id", "reference", "source_git_sha", "scored_at",
        "inference_precision", "training_disposition", "checkpoint",
        "ground_truth_contract", "threshold_source", "threshold", "dev", "test",
        "dev_f1", "test_f1", "test_precision", "test_recall",
        "test_near_shore_f1",
    }
    if set(metrics) != expected_keys or metrics.get("result_schema") != 2:
        raise RuntimeError("R2 must use the exact schema-2 result contract")
    expected_identity = {
        "exp_id": "yolo26-f100",
        "reference": "R2",
        "inference_precision": "float32",
        "training_disposition": "preserved-best-pt-rescore-only",
    }
    if any(metrics.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("R2 identity/disposition is invalid")
    if metrics.get("source_git_sha") != provenance_kwargs.get("expected_git_sha"):
        raise RuntimeError("R2 corrected scoring Git SHA is not provenance-bound")
    _reference_timestamp(metrics.get("scored_at"), "R2 scored_at")
    checkpoint = metrics.get("checkpoint")
    if checkpoint != {
        "relative_path": "weights/best.pt",
        "sha256": R2_CHECKPOINT_SHA256,
    }:
        raise RuntimeError("R2 preserved best.pt binding is invalid")
    threshold_source = metrics.get("threshold_source")
    if threshold_source != {
        "split": "dev",
        "checkpoint_sha256": R2_CHECKPOINT_SHA256,
    }:
        raise RuntimeError("R2 threshold is not bound to the preserved best.pt")
    expected_gt = {
        **GT_CONTRACT,
        "dev_counts": R2_GT_COUNTS["dev"],
        "test_counts": R2_GT_COUNTS["test"],
    }
    if metrics.get("ground_truth_contract") != expected_gt:
        raise RuntimeError("R2 corrected ground-truth counts/contract mismatch")
    threshold = _finite(metrics.get("threshold"), "R2 threshold")
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError("R2 threshold is outside [0,1]")
    dev = _score_record(
        metrics.get("dev"), label="R2.dev", member_key="per_scene",
        expected_members=23,
    )
    test = _score_record(
        metrics.get("test"), label="R2.test", member_key="per_scene",
        expected_members=16,
    )
    if set(dev["per_scene"]) != R2_DEV_SCENE_IDS:
        raise RuntimeError("R2 dev per-scene IDs differ from the frozen dev split")
    if set(test["per_scene"]) != R2_TEST_SCENE_IDS:
        raise RuntimeError("R2 test per-scene IDs differ from the frozen test split")
    for split_name, scored in (("dev", dev), ("test", test)):
        declared_positive = expected_gt[f"{split_name}_counts"]["positive"]
        aggregate = scored["aggregate"]
        observed_positive = int(aggregate["tp"]) + int(aggregate["fn"])
        if observed_positive != declared_positive:
            raise RuntimeError(
                f"R2 {split_name} aggregate positive support "
                "differs from the declared ground-truth count"
            )
    aliases = {
        "dev_f1": dev["aggregate"]["f1"],
        "test_f1": test["aggregate"]["f1"],
        "test_precision": test["aggregate"]["precision"],
        "test_recall": test["aggregate"]["recall"],
        "test_near_shore_f1": test["slices"]["near_shore"]["f1"],
    }
    if any(
        not math.isclose(_finite(metrics.get(key), f"R2 {key}"), float(value))
        for key, value in aliases.items()
    ):
        raise RuntimeError("R2 legacy metric aliases differ from schema-2 metrics")
    provenance, _provenance_sha256 = _reference_json(
        provenance_path, "R2 runtime provenance"
    )
    validate_reference_provenance(provenance, **provenance_kwargs)
    if provenance.get("reference_precision") != "float32":
        raise RuntimeError("R2 runtime provenance precision mismatch")
    return _reference_result(
        metrics_path=metrics_path,
        provenance_path=provenance_path,
        metrics=metrics,
        provenance=provenance,
    )


def validate_r3(run_dir: Path, **provenance_kwargs) -> dict:
    metrics_path = run_dir / "final_metrics.json"
    provenance_path = run_dir / "runtime_provenance.json"
    metrics, _metrics_sha256 = _reference_json(metrics_path, "R3 final metrics")
    expected_keys = {
        "result_schema", "exp_id", "reference", "source_git_sha", "scored_at",
        "model", "precision", "eval_contract_disposition",
        "legacy_result_reused", "ground_truth_contract", "sample",
        "per_prompt", "best_prompt",
    }
    if set(metrics) != expected_keys or metrics.get("result_schema") != 2:
        raise RuntimeError("R3 must use the exact schema-2 result contract")
    expected_identity = {
        "exp_id": "locateanything-zs",
        "reference": "R3",
        "precision": "bfloat16",
        "eval_contract_disposition": "full-rerun-under-corrected-contract",
        "legacy_result_reused": False,
    }
    if any(metrics.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("R3 identity/disposition is invalid")
    if metrics.get("source_git_sha") != provenance_kwargs.get("expected_git_sha"):
        raise RuntimeError("R3 corrected scoring Git SHA is not provenance-bound")
    _reference_timestamp(metrics.get("scored_at"), "R3 scored_at")
    model = metrics.get("model")
    if (
        not isinstance(model, Mapping)
        or set(model) != {
            "id", "revision", "payload_sha256", "source_note_sha256",
            "license_sha256"
        }
        or model.get("id") != "nvidia/LocateAnything-3B"
        or model.get("revision") != R3_MODEL_REVISION
        or model.get("payload_sha256") != R3_MODEL_PAYLOAD_SHA256
        or HEX64.fullmatch(str(model.get("source_note_sha256", ""))) is None
        or HEX64.fullmatch(str(model.get("license_sha256", ""))) is None
    ):
        raise RuntimeError("R3 model revision/license/source binding is invalid")
    gt_contract = metrics.get("ground_truth_contract")
    if not isinstance(gt_contract, Mapping) or set(gt_contract) != {
        *GT_CONTRACT, "sample_counts"
    }:
        raise RuntimeError("R3 ground-truth contract schema is invalid")
    if any(gt_contract.get(key) != value for key, value in GT_CONTRACT.items()):
        raise RuntimeError("R3 ground-truth contract is not corrected v2")
    sample_counts = gt_contract.get("sample_counts")
    _gt_counts(sample_counts, "R3 sample")
    expected_sample_positive = int(sample_counts["positive"])
    sample = metrics.get("sample")
    if not isinstance(sample, Mapping) or set(sample) != {
        "policy", "algorithm_version", "candidate_count", "candidate_sha256",
        "candidates", "n_chips", "sha256", "entries"
    }:
        raise RuntimeError("R3 sample schema is invalid")
    if (
        sample.get("policy") != "sorted-foreground-dev-chips-even-stride"
        or sample.get("algorithm_version") != "floor-index-v1"
        or sample.get("candidate_count") != R3_CANDIDATE_COUNT
        or HEX64.fullmatch(str(sample.get("candidate_sha256", ""))) is None
        or sample.get("n_chips") != 200
        or HEX64.fullmatch(str(sample.get("sha256", ""))) is None
    ):
        raise RuntimeError("R3 sample identity is invalid")
    candidates = sample.get("candidates")
    entries = sample.get("entries")
    if not isinstance(candidates, list) or len(candidates) != R3_CANDIDATE_COUNT:
        raise RuntimeError("R3 candidate manifest must contain exactly 1,525 entries")
    if not isinstance(entries, list) or len(entries) != 200:
        raise RuntimeError("R3 sample must contain exactly 200 chip entries")
    for entry in candidates:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"chip", "sidecar_sha256"}
            or not isinstance(entry.get("chip"), str)
            or HEX64.fullmatch(str(entry.get("sidecar_sha256", ""))) is None
        ):
            raise RuntimeError("R3 candidate entry is invalid")
    if len({str(entry["chip"]) for entry in candidates}) != len(candidates):
        raise RuntimeError("R3 candidate chip identities are not unique")
    computed_candidates = hashlib.sha256(
        json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if computed_candidates != sample.get("candidate_sha256"):
        raise RuntimeError("R3 candidate manifest digest mismatch")
    expected_entries = [
        candidates[(index * len(candidates)) // 200] for index in range(200)
    ]
    if entries != expected_entries:
        raise RuntimeError("R3 sample is not the declared floor-index-v1 selection")
    chips: list[str] = []
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"chip", "sidecar_sha256"}
            or not isinstance(entry.get("chip"), str)
            or HEX64.fullmatch(str(entry.get("sidecar_sha256", ""))) is None
        ):
            raise RuntimeError("R3 sample entry is invalid")
        chip = str(entry["chip"])
        relative = Path(chip)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npy":
            raise RuntimeError("R3 sample chip path is unsafe")
        chips.append(relative.with_suffix("").as_posix())
    if len(set(chips)) != len(chips):
        raise RuntimeError("R3 sample chip identities are not unique")
    candidate_scenes = {Path(str(entry["chip"])).parts[0] for entry in candidates}
    sample_scenes = {Path(str(entry["chip"])).parts[0] for entry in entries}
    if candidate_scenes != R2_DEV_SCENE_IDS or sample_scenes != R2_DEV_SCENE_IDS:
        raise RuntimeError("R3 candidate/sample scenes differ from the frozen dev split")
    computed_sample = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if computed_sample != sample.get("sha256"):
        raise RuntimeError("R3 sample digest mismatch")
    prompts = metrics.get("per_prompt")
    if not isinstance(prompts, Mapping) or set(prompts) != {"ship", "vessel", "boat"}:
        raise RuntimeError("R3 prompt set must be exactly ship/vessel/boat")
    for prompt, result in prompts.items():
        if not isinstance(result, Mapping) or set(result) != {
            "f1", "precision", "recall", "threshold", "aggregate", "slices",
            "per_chip",
        }:
            raise RuntimeError(f"R3 {prompt} result schema is invalid")
        scored = _score_record(
            {
                "aggregate": result["aggregate"],
                "slices": result["slices"],
                "per_chip": result["per_chip"],
            },
            label=f"R3.{prompt}",
            member_key="per_chip",
            expected_members=200,
        )
        if set(scored["per_chip"]) != set(chips):
            raise RuntimeError(f"R3 {prompt} per-chip sample binding mismatch")
        aggregate = scored["aggregate"]
        if int(aggregate["tp"]) + int(aggregate["fn"]) != expected_sample_positive:
            raise RuntimeError(
                f"R3 {prompt} aggregate positive support differs from "
                "ground_truth_contract.sample_counts.positive"
            )
        for key in ("f1", "precision", "recall"):
            value = _finite(result.get(key), f"R3 {prompt}.{key}")
            if not math.isclose(value, float(scored["aggregate"][key])):
                raise RuntimeError(f"R3 {prompt}.{key} alias mismatch")
        threshold = _finite(result.get("threshold"), f"R3 {prompt}.threshold")
        if not 0.0 <= threshold <= 1.0:
            raise RuntimeError(f"R3 {prompt}.threshold is outside [0,1]")
    best = metrics.get("best_prompt")
    if best not in prompts:
        raise RuntimeError("R3 best_prompt is invalid")
    if float(prompts[best]["f1"]) != max(float(result["f1"]) for result in prompts.values()):
        raise RuntimeError("R3 best_prompt does not have maximal F1")
    provenance, _provenance_sha256 = _reference_json(
        provenance_path, "R3 runtime provenance"
    )
    validate_reference_provenance(provenance, **provenance_kwargs)
    if provenance.get("reference_precision") != "bfloat16":
        raise RuntimeError("R3 runtime provenance precision mismatch")
    return _reference_result(
        metrics_path=metrics_path,
        provenance_path=provenance_path,
        metrics=metrics,
        provenance=provenance,
    )


def validate_h100_ready(
    path: Path,
    *,
    expected_git_sha: str,
    expected_venv_sha256: str,
    expected_venv_build_sha256: str,
    expected_base_python_sha256: str,
    expected_base_python_runtime_sha256: str,
    expected_wheelhouse_sha256: str,
    expected_base_extraction_receipt_sha256: str,
    expected_base_payload: Mapping[str, str],
    expected_runtime_amendment: Mapping[str, str],
    expected_frozen_sha256: Mapping[str, str],
    expected_smoke_receipt: Mapping[str, object],
    expected_smoke_sha256: str,
) -> dict:
    path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("H100_READY must be a regular non-symlink")
    payload = json.loads(path.read_text())
    if payload.get("schema") != 2 or payload.get("status") != "ready":
        raise RuntimeError("H100 acceptance marker is not schema-2 ready")
    if payload.get("source", {}).get("git_sha") != expected_git_sha:
        raise RuntimeError("H100-ready git SHA mismatch")
    accepted_venv = payload.get("venv")
    if not isinstance(accepted_venv, Mapping):
        raise RuntimeError("H100-ready native-venv binding is absent")
    if set(accepted_venv) != {
        "path",
        "sha256",
        "venv_build_sha256",
        "base_python",
        "wheelhouse",
        "staged_data_view",
    } or not Path(str(accepted_venv.get("path", ""))).is_absolute():
        raise RuntimeError(
            "H100-ready native-venv binding schema is not canonical"
        )
    if accepted_venv.get("sha256") != expected_venv_sha256:
        raise RuntimeError("H100-ready native-venv tree digest mismatch")
    if accepted_venv.get("venv_build_sha256") != expected_venv_build_sha256:
        raise RuntimeError("H100-ready native-venv build receipt mismatch")
    base_python = accepted_venv.get("base_python")
    base_python_runtime = (
        base_python.get("runtime") if isinstance(base_python, Mapping) else None
    )
    if (
        not isinstance(base_python, Mapping)
        or base_python.get("version") != EXPECTED_PYTHON_VERSION
        or base_python.get("executable_sha256") != expected_base_python_sha256
        or not isinstance(base_python_runtime, Mapping)
        or base_python_runtime.get("sha256")
        != expected_base_python_runtime_sha256
    ):
        raise RuntimeError("H100-ready base-Python identity mismatch")
    wheelhouse = accepted_venv.get("wheelhouse")
    if (
        not isinstance(wheelhouse, Mapping)
        or set(wheelhouse)
        != {"identity", "artifacts", "base_extraction", "reverified_after_build"}
        or not isinstance(wheelhouse.get("artifacts"), Mapping)
        or wheelhouse.get("reverified_after_build") is not True
    ):
        raise RuntimeError(
            "H100-ready wheelhouse/base-extraction schema is not canonical"
        )
    wheelhouse_identity = (
        wheelhouse.get("identity") if isinstance(wheelhouse, Mapping) else None
    )
    base_extraction = (
        wheelhouse.get("base_extraction")
        if isinstance(wheelhouse, Mapping)
        else None
    )
    if (
        not isinstance(base_extraction, Mapping)
        or set(base_extraction) != {"sha256", "receipt"}
    ):
        raise RuntimeError(
            "H100-ready wheelhouse/base-extraction schema is not canonical"
        )
    extraction_receipt = (
        base_extraction.get("receipt")
        if isinstance(base_extraction, Mapping)
        else None
    )
    if (
        not isinstance(wheelhouse_identity, Mapping)
        or wheelhouse_identity.get("sha256") != expected_wheelhouse_sha256
        or not isinstance(base_extraction, Mapping)
        or base_extraction.get("sha256")
        != expected_base_extraction_receipt_sha256
        or not isinstance(extraction_receipt, Mapping)
        or extraction_receipt.get("package_id")
        != expected_base_payload.get("package_id")
        or extraction_receipt.get("manifest_sha256")
        != expected_base_payload.get("manifest_sha256")
        or extraction_receipt.get("wheelhouse") != wheelhouse_identity
    ):
        raise RuntimeError("H100-ready wheelhouse/base-extraction identity mismatch")
    validate_acceptance_data_view_binding(
        accepted_venv.get("staged_data_view"),
        repo=_REPO_ROOT,
        expected_git_sha=expected_git_sha,
        expected_base_package_id=str(expected_base_payload.get("package_id", "")),
        expected_base_manifest_sha256=str(
            expected_base_payload.get("manifest_sha256", "")
        ),
        expected_runtime_package_id=str(expected_runtime_amendment.get("package_id", "")),
        expected_runtime_manifest_sha256=str(
            expected_runtime_amendment.get("manifest_sha256", "")
        ),
    )
    if payload.get("base_payload") != dict(expected_base_payload):
        raise RuntimeError("H100-ready base-payload bindings mismatch")
    if payload.get("runtime_amendment") != dict(expected_runtime_amendment):
        raise RuntimeError("H100-ready runtime-amendment bindings mismatch")
    if payload.get("source", {}).get("frozen_sha256") != dict(expected_frozen_sha256):
        raise RuntimeError("H100-ready frozen-file hash bindings mismatch")
    source_binding = payload.get("source_validation")
    expected_source_path = path.parent / "SOURCE_VALIDATED.json"
    if (
        not isinstance(source_binding, Mapping)
        or set(source_binding) != {"path", "sha256", "receipt"}
        or source_binding.get("path") != str(expected_source_path)
        or not HEX64.fullmatch(str(source_binding.get("sha256", "")))
    ):
        raise RuntimeError("H100-ready source-validation binding is invalid")
    source_receipt = validate_source_receipt(
        expected_source_path,
        expected_sha256=str(source_binding["sha256"]),
        expected_git_sha=expected_git_sha,
        expected_hashes=expected_frozen_sha256,
        expected_base_payload=expected_base_payload,
        expected_runtime_amendment=expected_runtime_amendment,
    )
    if source_binding.get("receipt") != source_receipt:
        raise RuntimeError("H100-ready embedded source-validation receipt differs")

    evaluation_binding = payload.get("evaluation_ground_truth")
    expected_evaluation_path = path.parent / "EVAL_GROUND_TRUTH_VALIDATED.json"
    if (
        not isinstance(evaluation_binding, Mapping)
        or set(evaluation_binding) != {"path", "sha256", "receipt"}
        or evaluation_binding.get("path") != str(expected_evaluation_path)
        or not HEX64.fullmatch(str(evaluation_binding.get("sha256", "")))
        or expected_evaluation_path.is_symlink()
        or not expected_evaluation_path.is_file()
        or stat.S_IMODE(expected_evaluation_path.stat().st_mode) != 0o444
        or sha256_file(expected_evaluation_path) != evaluation_binding.get("sha256")
    ):
        raise RuntimeError(
            "H100-ready evaluation-ground-truth binding is invalid"
        )
    evaluation_receipt = json.loads(expected_evaluation_path.read_text())
    if (
        not isinstance(evaluation_receipt, Mapping)
        or evaluation_binding.get("receipt") != evaluation_receipt
    ):
        raise RuntimeError(
            "H100-ready embedded evaluation-ground-truth receipt differs"
        )

    try:
        uuid.UUID(str(payload.get("acceptance_uuid")))
    except (ValueError, AttributeError) as exc:
        raise RuntimeError("H100-ready acceptance UUID is absent or invalid") from exc
    strict_fp32 = payload.get("strict_fp32")
    hardware = payload.get("hardware")
    expected_strict_fp32 = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }
    if (
        strict_fp32 != expected_strict_fp32
        or not isinstance(hardware, Mapping)
        or hardware.get("backend") != expected_strict_fp32
    ):
        raise RuntimeError("H100-ready marker does not assert strict IEEE FP32")
    validate_gpu_inventory(hardware.get("devices", []))
    validate_hardware_runtime_contracts(hardware)
    for key in ("torch", "cuda_build", "driver_version"):
        if not str(hardware.get(key, "")).strip():
            raise RuntimeError(f"H100-ready hardware lacks {key}")
    gates = payload.get("gates")
    required_gates = {
        "pytest_seconds",
        "hardware_probe_seconds",
        "vit_gate_seconds",
        "cnn_200step_seconds",
    }
    if not isinstance(gates, Mapping) or set(gates) != required_gates:
        raise RuntimeError("H100-ready acceptance gate evidence is incomplete")
    if any(_finite(gates[key], f"H100 gate {key}") < 0 for key in required_gates):
        raise RuntimeError("H100-ready gate durations cannot be negative")
    test_binding = payload.get("test_suite")
    expected_test_path = path.parent / "PYTEST_ACCEPTANCE.json"
    if (
        not isinstance(test_binding, Mapping)
        or set(test_binding) != {"path", "sha256", "receipt"}
        or test_binding.get("path") != str(expected_test_path)
        or not HEX64.fullmatch(str(test_binding.get("sha256", "")))
        or expected_test_path.is_symlink()
        or not expected_test_path.is_file()
        or sha256_file(expected_test_path) != test_binding.get("sha256")
    ):
        raise RuntimeError("H100-ready aggregate test-suite binding is invalid")
    test_suite = json.loads(expected_test_path.read_text())
    if test_binding.get("receipt") != test_suite or set(test_suite) != {
        "schema",
        "status",
        "source_validation_sha256",
        "coverage",
        "host_handoff",
        "venv_remaining",
        "aggregate_duration_seconds",
    }:
        raise RuntimeError("H100-ready aggregate test-suite receipt differs")
    if (
        test_suite.get("schema") != 2
        or test_suite.get("status") != "passed"
        or test_suite.get("source_validation_sha256") != source_binding["sha256"]
        or test_suite.get("coverage")
        != {
            "host": HOST_TESTS,
            "venv": "all pytest collection except the two host-only files",
            "aggregate": "entire repository pytest suite",
        }
    ):
        raise RuntimeError("aggregate test-suite scope/source binding is invalid")
    host_slice = test_suite.get("host_handoff")
    expected_host_path = path.parent / "HOST_HANDOFF_TESTS.json"
    if (
        not isinstance(host_slice, Mapping)
        or set(host_slice) != {"receipt_path", "receipt_sha256", "receipt"}
        or host_slice.get("receipt_path") != str(expected_host_path)
    ):
        raise RuntimeError("aggregate host-test slice binding is invalid")
    host_receipt = validate_host_gate(
        expected_host_path,
        expected_sha256=str(host_slice.get("receipt_sha256", "")),
        expected_source_validation_sha256=str(source_binding["sha256"]),
    )
    if host_slice.get("receipt") != host_receipt:
        raise RuntimeError("aggregate embedded host-test receipt differs")
    venv_slice = test_suite.get("venv_remaining")
    expected_venv_log = path.parent / "acceptance-logs/pytest-venv-remaining.log"
    if not isinstance(venv_slice, Mapping) or set(venv_slice) != {
        "command",
        "duration_seconds",
        "log",
    }:
        raise RuntimeError("aggregate venv-test slice is invalid")
    venv_log = venv_slice.get("log")
    if (
        venv_slice.get("command")
        != ["-m", "pytest", "-q", *(f"--ignore={item}" for item in HOST_TESTS)]
        or not isinstance(venv_log, Mapping)
        or set(venv_log) != {"path", "sha256"}
        or venv_log.get("path") != str(expected_venv_log)
        or expected_venv_log.is_symlink()
        or not expected_venv_log.is_file()
        or not HEX64.fullmatch(str(venv_log.get("sha256", "")))
        or sha256_file(expected_venv_log) != venv_log.get("sha256")
    ):
        raise RuntimeError("aggregate venv-test log/scope binding is invalid")
    aggregate_seconds = _finite(
        test_suite.get("aggregate_duration_seconds"), "aggregate pytest duration"
    )
    expected_seconds = _finite(
        host_receipt.get("duration_seconds"), "host pytest duration"
    ) + _finite(venv_slice.get("duration_seconds"), "venv pytest duration")
    if (
        aggregate_seconds <= 0
        or not math.isclose(aggregate_seconds, expected_seconds)
        or not math.isclose(float(gates["pytest_seconds"]), aggregate_seconds)
    ):
        raise RuntimeError("aggregate pytest durations are inconsistent")
    projection = payload.get("projection")
    projection_fields = (
        "steps_per_second",
        "expected_gpu_hours",
        "ceiling_gpu_hours",
        "expected_wall_hours_ideal",
        "ceiling_wall_hours_ideal",
        "conservative_h100_wall_hours",
        "remaining_v100_wall_hours",
        "staging_seconds",
        "staging_hours_per_allocation",
        "allocation_wall_hours",
        "signal_lead_hours",
        "usable_training_hours_per_allocation",
        "training_wall_hours_before_staging",
    )
    if not isinstance(projection, Mapping):
        raise RuntimeError("H100-ready throughput projection is absent")
    expected_steps = {
        label: int(item["steps_per_epoch"])
        for label, item in EXPECTED_FRACTION_WORKLOAD.items()
    }
    if (
        projection.get("fraction_workload") != EXPECTED_FRACTION_WORKLOAD
        or projection.get("steps_per_epoch") != expected_steps
        or projection.get("grid_steps_per_epoch") != 8 * sum(expected_steps.values())
    ):
        raise RuntimeError("H100-ready projection does not bind the exact workload")
    values = {key: _finite(projection.get(key), f"projection {key}") for key in projection_fields}
    if any(value <= 0 for value in values.values()):
        raise RuntimeError("H100-ready projection values must be positive")
    try:
        recomputed_wall = staging_aware_wall_clock(
            training_wall_hours=values["training_wall_hours_before_staging"],
            staging_seconds=values["staging_seconds"],
            allocation_wall_hours=values["allocation_wall_hours"],
            signal_lead_hours=values["signal_lead_hours"],
        )
    except ValueError as exc:
        raise RuntimeError("H100-ready staging projection is invalid") from exc
    projected_allocations = projection.get("projected_allocation_count")
    if (
        type(projected_allocations) is not int
        or projected_allocations != recomputed_wall["projected_allocation_count"]
        or any(
            not math.isclose(values[key], float(recomputed_wall[key]))
            for key in (
                "staging_hours_per_allocation",
                "usable_training_hours_per_allocation",
                "training_wall_hours_before_staging",
                "conservative_h100_wall_hours",
            )
        )
    ):
        raise RuntimeError("H100-ready staging/wall-clock projection is inconsistent")
    if values["conservative_h100_wall_hours"] >= values["remaining_v100_wall_hours"]:
        raise RuntimeError("H100-ready projection no longer proves a faster cutover")
    smoke = payload.get("slurm_smoke")
    if not isinstance(smoke, Mapping):
        raise RuntimeError("H100-ready Slurm smoke binding is absent")
    if smoke.get("sha256") != expected_smoke_sha256:
        raise RuntimeError("H100-ready Slurm smoke receipt hash mismatch")
    if smoke.get("receipt") != expected_smoke_receipt:
        raise RuntimeError("H100-ready Slurm smoke receipt payload mismatch")
    return payload


def validate_current_v100_advantage(
    ready: Mapping[str, object],
    current_remaining_v100_wall_hours: float,
    *,
    current_v100_diagnostic_status: str,
) -> dict[str, object]:
    """Recheck cutover timing or bind a completed non-reportable diagnostic."""

    projection = ready.get("projection")
    if not isinstance(projection, Mapping):
        raise RuntimeError("H100-ready throughput projection is absent")
    current = _finite(
        current_remaining_v100_wall_hours, "current remaining V100 wall hours"
    )
    conservative = _finite(
        projection.get("conservative_h100_wall_hours"),
        "conservative H100 wall hours",
    )
    accepted_v100 = _finite(
        projection.get("remaining_v100_wall_hours"),
        "acceptance remaining V100 wall hours",
    )
    if current < 0 or conservative <= 0 or accepted_v100 <= 0:
        raise RuntimeError(
            "H100/acceptance forecasts must be positive and current V100 "
            "remaining hours must be nonnegative"
        )
    if conservative >= accepted_v100:
        raise RuntimeError(
            "H100 cutover rejected: the original acceptance comparison no "
            "longer proves an H100 advantage"
        )
    if current_v100_diagnostic_status not in V100_DIAGNOSTIC_STATUSES:
        raise RuntimeError("current V100 diagnostic status is invalid")
    if current > 0 and (
        current_v100_diagnostic_status != V100_DIAGNOSTIC_RUNNING
        or conservative >= current
    ):
        raise RuntimeError(
            "H100 cutover rejected: the current V100 forecast is no longer slower"
        )
    if current == 0 and current_v100_diagnostic_status != V100_DIAGNOSTIC_COMPLETE:
        raise RuntimeError(
            "zero remaining V100 hours require explicit "
            "complete-non-reportable-diagnostic status"
        )
    return {
        "conservative_h100_wall_hours": conservative,
        "acceptance_remaining_v100_wall_hours": accepted_v100,
        "current_remaining_v100_wall_hours": current,
        "v100_diagnostic_status": current_v100_diagnostic_status,
        "h100_scientifically_mandatory": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100-ready", type=Path, required=True)
    parser.add_argument("--references-package-root", type=Path, required=True)
    parser.add_argument("--expected-references-package-id", required=True)
    parser.add_argument("--expected-references-producer-git-sha", required=True)
    parser.add_argument("--expected-references-identity-sha256", required=True)
    parser.add_argument("--expected-references-manifest-sha256", required=True)
    parser.add_argument("--expected-references-ready-sha256", required=True)
    parser.add_argument("--expected-references-sha256sums-sha256", required=True)
    parser.add_argument("--expected-h100-git-sha", required=True)
    parser.add_argument("--expected-h100-campaign-id", required=True)
    parser.add_argument("--expected-reference-git-sha", required=True)
    parser.add_argument("--expected-v100-core-git-sha", required=True)
    parser.add_argument("--expected-v100-core-campaign-id", required=True)
    parser.add_argument("--expected-venv-sha256", required=True)
    parser.add_argument("--expected-venv-build-sha256", required=True)
    parser.add_argument("--expected-base-python-sha256", required=True)
    parser.add_argument("--expected-base-python-runtime-sha256", required=True)
    parser.add_argument("--expected-wheelhouse-sha256", required=True)
    parser.add_argument(
        "--expected-base-extraction-receipt-sha256", required=True
    )
    parser.add_argument("--expected-base-payload-package-id", required=True)
    parser.add_argument("--expected-base-payload-git-sha", required=True)
    parser.add_argument("--expected-base-payload-manifest-sha256", required=True)
    parser.add_argument("--expected-base-payload-ready-sha256", required=True)
    parser.add_argument("--expected-base-payload-sha256sums-sha256", required=True)
    parser.add_argument("--expected-base-payload-repo-bundle-sha256", required=True)
    parser.add_argument("--expected-runtime-amendment-package-id", required=True)
    parser.add_argument("--expected-runtime-amendment-git-sha", required=True)
    parser.add_argument("--expected-runtime-amendment-manifest-sha256", required=True)
    parser.add_argument("--expected-runtime-amendment-ready-sha256", required=True)
    parser.add_argument("--expected-runtime-amendment-sha256sums-sha256", required=True)
    parser.add_argument("--expected-runtime-amendment-bundle-sha256", required=True)
    parser.add_argument(
        "--expected-frozen-sha256",
        action="append",
        required=True,
        help="repeat in FROZEN_PATHS order",
    )
    parser.add_argument("--smoke-ready", type=Path, required=True)
    parser.add_argument("--expected-reference-campaign-id", required=True)
    parser.add_argument(
        "--current-remaining-v100-wall-hours", type=float, required=True
    )
    parser.add_argument(
        "--current-v100-diagnostic-status",
        choices=sorted(V100_DIAGNOSTIC_STATUSES),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_frozen_sha256) != len(FROZEN_PATHS):
        parser.error(
            f"--expected-frozen-sha256 must be repeated {len(FROZEN_PATHS)} times"
        )

    expected_frozen = dict(
        zip(FROZEN_PATHS, args.expected_frozen_sha256, strict=True)
    )
    expected_base_payload = {
        "package_id": args.expected_base_payload_package_id,
        "git_sha": args.expected_base_payload_git_sha,
        "manifest_sha256": args.expected_base_payload_manifest_sha256,
        "ready_sha256": args.expected_base_payload_ready_sha256,
        "sha256sums_sha256": args.expected_base_payload_sha256sums_sha256,
        "repo_bundle_sha256": args.expected_base_payload_repo_bundle_sha256,
    }
    expected_runtime_amendment = {
        "package_id": args.expected_runtime_amendment_package_id,
        "git_sha": args.expected_runtime_amendment_git_sha,
        "manifest_sha256": args.expected_runtime_amendment_manifest_sha256,
        "ready_sha256": args.expected_runtime_amendment_ready_sha256,
        "sha256sums_sha256": args.expected_runtime_amendment_sha256sums_sha256,
        "runtime_bundle_sha256": args.expected_runtime_amendment_bundle_sha256,
    }
    smoke_bindings = make_smoke_bindings(
        git_sha=args.expected_h100_git_sha,
        detector_sha256=expected_frozen["configs/detector.yaml"],
        venv_sha256=args.expected_venv_sha256,
        venv_build_sha256=args.expected_venv_build_sha256,
        base_python_sha256=args.expected_base_python_sha256,
        base_python_runtime_sha256=(
            args.expected_base_python_runtime_sha256
        ),
        wheelhouse_sha256=args.expected_wheelhouse_sha256,
        base_extraction_receipt_sha256=(
            args.expected_base_extraction_receipt_sha256
        ),
        base_payload=expected_base_payload,
        runtime_amendment=expected_runtime_amendment,
    )
    smoke = validate_smoke_receipt(
        args.smoke_ready,
        expected_bindings=smoke_bindings,
    )
    smoke_sha256 = sha256_file(args.smoke_ready)

    from scripts.handoff.control import control_package_identity

    reference_bindings = {
        "evaluation_contract": "vessel-hm-positive-low-ignore-v2",
        "reference_git_sha": args.expected_reference_git_sha,
        "reference_campaign_id": args.expected_reference_campaign_id,
        "v100_core_git_sha": args.expected_v100_core_git_sha,
        "v100_core_campaign_id": args.expected_v100_core_campaign_id,
    }
    references_root = args.references_package_root.absolute()
    expected_references_control = {
        "package_id": args.expected_references_package_id,
        "kind": "references",
        "direction": "v100-to-judy",
        "producer_git_sha": args.expected_references_producer_git_sha,
        "identity_sha256": args.expected_references_identity_sha256,
        "manifest_sha256": args.expected_references_manifest_sha256,
        "ready_sha256": args.expected_references_ready_sha256,
        "sha256sums_sha256": args.expected_references_sha256sums_sha256,
    }
    references_control = control_package_identity(
        references_root,
        expected_kind="references",
        expected_bindings=reference_bindings,
        expected_identity=expected_references_control,
    )
    if args.expected_references_producer_git_sha != args.expected_h100_git_sha:
        raise RuntimeError(
            "references producer Git SHA differs from the corrected-scoring SHA"
        )
    if args.expected_reference_git_sha != args.expected_h100_git_sha:
        raise RuntimeError(
            "corrected-reference Git SHA differs from the accepted H100 source SHA"
        )
    reference_campaign = validate_reference_campaign_manifest(
        references_root / "references/REFERENCE_CAMPAIGN.json",
        expected_reference_git_sha=args.expected_reference_git_sha,
        expected_reference_campaign_id=args.expected_reference_campaign_id,
        expected_v100_core_git_sha=args.expected_v100_core_git_sha,
        expected_v100_core_campaign_id=args.expected_v100_core_campaign_id,
    )
    campaign_manifest = reference_campaign["manifest"]
    if not isinstance(campaign_manifest, Mapping):
        raise AssertionError("validated reference campaign manifest is not a mapping")
    expected_reference_hashes = {
        "environment_sha256": str(campaign_manifest["environment_sha256"]),
        "environment_lock_sha256": str(
            campaign_manifest["environment_lock_sha256"]
        ),
        "campaign_manifest_sha256": str(reference_campaign["manifest_sha256"]),
        "runtime_launcher_sha256": str(
            campaign_manifest["runtime_launcher_sha256"]
        ),
    }
    provenance_kwargs = {
        "expected_git_sha": args.expected_reference_git_sha,
        "expected_campaign_id": args.expected_reference_campaign_id,
        "expected_hashes": expected_reference_hashes,
    }
    ready = validate_h100_ready(
        args.h100_ready,
        expected_git_sha=args.expected_h100_git_sha,
        expected_venv_sha256=args.expected_venv_sha256,
        expected_venv_build_sha256=args.expected_venv_build_sha256,
        expected_base_python_sha256=args.expected_base_python_sha256,
        expected_base_python_runtime_sha256=(
            args.expected_base_python_runtime_sha256
        ),
        expected_wheelhouse_sha256=args.expected_wheelhouse_sha256,
        expected_base_extraction_receipt_sha256=(
            args.expected_base_extraction_receipt_sha256
        ),
        expected_base_payload=expected_base_payload,
        expected_runtime_amendment=expected_runtime_amendment,
        expected_frozen_sha256=expected_frozen,
        expected_smoke_receipt=smoke,
        expected_smoke_sha256=smoke_sha256,
    )
    r2 = validate_r2(
        references_root / "references/yolo26-f100", **provenance_kwargs
    )
    r3 = validate_r3(
        references_root / "references/locateanything-zs", **provenance_kwargs
    )
    scoring_shas = {
        references_control.get("producer_git_sha"),
        r2["metrics"].get("source_git_sha"),
        r3["metrics"].get("source_git_sha"),
    }
    if scoring_shas != {args.expected_h100_git_sha}:
        raise RuntimeError(
            "reference package/R2/R3 corrected-scoring Git SHA mismatch"
        )
    cutover_forecast = validate_current_v100_advantage(
        ready,
        args.current_remaining_v100_wall_hours,
        current_v100_diagnostic_status=(
            args.current_v100_diagnostic_status
        ),
    )
    marker = {
        "schema": 2,
        "status": "cutover-ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "h100_campaign_id": args.expected_h100_campaign_id,
        "h100_ready": ready,
        "acceptance": cutover_acceptance_bindings(ready),
        "references_control": references_control,
        "reference_campaign": reference_campaign,
        "cutover_forecast": cutover_forecast,
        "references": {
            "r2": r2,
            "r3": r3,
        },
        "v100_action": "none; this guard never stops or signals V100 processes",
    }
    validate_bound_cutover_forecast(marker)
    write_new_cutover_ready(args.output, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
