"""Evidence-only schema-2 Box return package for the BES2 Judy diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Callable

from scripts.h100.bes2_diagnostic import (
    EXECUTION_COMPLETE,
    EXECUTION_STATE,
    PROBE_COMPLETE,
    TEST_RECEIPT,
    _test_receipt_valid,
)
from src.analysis.bes2_contract import (
    BES2ContractError,
    assert_no_final_consumption,
    canonical_json,
    hash_binding,
    read_regular_json,
    sha256_file,
    summarize_payload,
    validate_diagnostic_metrics,
    validate_diagnostic_metrics_payload,
    validate_training_metrics,
)
from src.analysis.bes2_root_cause import (
    AUDIT_FILENAME,
    READY_FILENAME,
    REPORT_FILENAME,
    ROOT_CAUSE_FILENAME,
    SAMPLE_FILENAME,
    _markdown_report,
    validate_readiness,
)
from src.models.bes2_diagnostic import DIAGNOSTIC_VARIANTS

from .bes2_amendment import _artifact_paths
from .package import (
    PackageError,
    _hash_file,
    _inside_repository_worktrees,
    _package_regular_files,
    _parse_sha256sums,
    _physical_record,
    _plain_artifact,
    _require_no_symlink_components,
    _validate_control_record,
    _validate_safe_path,
    _write_bytes,
)

FORMAT_VERSION = 2
PACKAGE_TYPE = "bes2-diagnostic-results"
RUN_STAGES = ("probe", "full")
CONTROL_JSON = (
    READY_FILENAME,
    AUDIT_FILENAME,
    SAMPLE_FILENAME,
    TEST_RECEIPT,
    PROBE_COMPLETE,
    EXECUTION_COMPLETE,
    EXECUTION_STATE,
    ROOT_CAUSE_FILENAME,
)
CONTROL_TEXT = ("BES2_TEST_GATE.log", REPORT_FILENAME)
RUN_FILES = (
    "diagnostic_metrics.json",
    "training_metrics.json",
    "initialization.json",
    "controller.log",
)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HARDWARE = re.compile(r"h100_runtime-[1-9][0-9]*-r[0-9]+[.]json")
_SLURM_LOG = re.compile(
    r"[^/]+-bes2-(audit|probe|full)-[1-9][0-9]*[.]out"
)
_ALLOWED_SUFFIXES = frozenset({".json", ".md", ".log", ".out"})
_FORBIDDEN_PATH_TOKENS = (
    "checkpoint",
    ".ckpt",
    "test_metrics",
    "final_metrics",
    "training_cohort",
    "final_eval",
)
_SECRET = re.compile(
    rb"(?:BOX_(?:JWT_CONFIG|FOLDER_ID)|access_token|private_key)"
    rb"[\"']?\s*(?::|=)|-----BEGIN (?:RSA )?PRIVATE KEY-----|"
    rb"https?://[^\s]*box[.]com",
    re.IGNORECASE,
)

PackageVerifier = Callable[[Path], dict[str, object]]


def _git_sha(repo: Path) -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackageError("cannot resolve BES2 result source SHA") from exc


def _run_dir(root: Path, variant: str, stage: str) -> Path:
    return root / f"bes2-{variant}-{stage}-f100-s0"


def _logical_run_file(variant: str, stage: str, filename: str) -> str:
    return f"diagnostic_runs/{variant}/{stage}/{filename}"


def _regular_file(path: Path, description: str) -> Path:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PackageError(f"{description} must be a nonempty regular file: {path}")
    return path


def _json_bytes(data: bytes, description: str) -> dict[str, object]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"invalid packaged {description}") from exc
    if not isinstance(payload, dict):
        raise PackageError(f"packaged {description} is not a JSON object")
    return payload


def _scan(path: Path) -> None:
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise PackageError(f"nonallowlisted BES2 result suffix: {path}")
    tail = b""
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            window = tail + block
            if _SECRET.search(window):
                raise PackageError(f"secret or Box endpoint in BES2 evidence: {path}")
            tail = window[-256:]


def _stage_marker(
    path: Path,
    *,
    root: Path,
    stage: str,
    readiness_sha256: str,
) -> dict[str, object]:
    payload = read_regular_json(path, f"BES2 {stage} completion")
    expected_purpose = f"bes2-{stage}-pair"
    variants = payload.get("variants")
    required = {
        "schema",
        "status",
        "purpose",
        "created_utc",
        "readiness_sha256",
        "variants",
        "gpu_hours",
    }
    if stage == "full":
        required |= {
            "probe_pair",
            "root_cause",
            "report",
            "total_pair_gpu_hours",
        }
    if (
        set(payload) != required
        or payload.get("schema") != 1
        or payload.get("status") != "complete"
        or payload.get("purpose") != expected_purpose
        or payload.get("readiness_sha256") != readiness_sha256
        or not isinstance(variants, Mapping)
        or set(variants) != set(DIAGNOSTIC_VARIANTS)
    ):
        raise PackageError(f"BES2 {stage} completion marker is invalid")
    validated_metrics: dict[str, dict[str, object]] = {}
    for variant in DIAGNOSTIC_VARIANTS:
        metrics_path = (
            _run_dir(root, variant, stage) / "diagnostic_metrics.json"
        )
        metrics = validate_diagnostic_metrics(
            metrics_path,
            expected_variant=variant,
            expected_stage=stage,
        )
        readiness = metrics.get("readiness")
        if (
            not isinstance(readiness, Mapping)
            or readiness.get("sha256") != readiness_sha256
        ):
            raise PackageError(
                f"{variant}/{stage} does not bind the packaged readiness"
            )
        expected = hash_binding(
            metrics_path,
            relative_to=root,
        )
        if variants.get(variant) != expected:
            raise PackageError(f"BES2 {stage} marker metric binding drifted")
        validated_metrics[variant] = metrics
    expected_gpu_hours = sum(
        float(validated_metrics[variant]["gpu_hours"])
        for variant in DIAGNOSTIC_VARIANTS
    )
    value = payload.get("gpu_hours")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        or not math.isclose(
            float(value), expected_gpu_hours, rel_tol=1.0e-12, abs_tol=1.0e-12
        )
    ):
        raise PackageError(f"BES2 {stage} marker GPU-hours are invalid")
    if stage == "full":
        expected_bindings = {
            "probe_pair": hash_binding(
                root / ".control" / PROBE_COMPLETE,
                relative_to=root,
            ),
            "root_cause": hash_binding(
                root / ".control" / ROOT_CAUSE_FILENAME,
                relative_to=root,
            ),
            "report": hash_binding(
                root / ".control" / REPORT_FILENAME,
                relative_to=root,
            ),
        }
        if any(
            payload.get(name) != binding
            for name, binding in expected_bindings.items()
        ):
            raise PackageError("BES2 full completion binding drifted")
    return payload


def _hardware_json(path: Path) -> None:
    payload = read_regular_json(path, "BES2 H100 allocation inventory")
    devices = payload.get("devices")
    backend = payload.get("backend")
    if (
        not isinstance(devices, list)
        or len(devices) not in (1, 2)
        or payload.get("diagnostic_scope") != "bes2-train-dev-only"
        or any(
            not isinstance(device, Mapping)
            or "H100" not in str(device.get("name"))
            for device in devices
        )
        or backend
        != {
            "cuda_matmul_fp32_precision": "ieee",
            "cudnn_conv_fp32_precision": "ieee",
            "cudnn_rnn_fp32_precision": "ieee",
        }
    ):
        raise PackageError(f"invalid strict-H100 allocation evidence: {path}")


def _collect_evidence(
    *,
    repo: Path,
    diagnostic_root: Path,
    original_runs_root: Path,
    job_log_dir: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    source_sha = _git_sha(repo)
    if not _HEX40.fullmatch(source_sha):
        raise PackageError("BES2 result source is not a full Git SHA")
    assert_no_final_consumption(original_runs_root)
    root = _require_no_symlink_components(diagnostic_root, leaf="directory")
    control = root / ".control"
    readiness_path = control / READY_FILENAME
    readiness = validate_readiness(
        readiness_path,
        expected_source_sha=source_sha,
        diagnostic_root=root,
    )
    readiness_sha256 = sha256_file(readiness_path)
    _test_receipt_valid(control / TEST_RECEIPT, source_sha)

    metrics: dict[tuple[str, str], dict[str, object]] = {}
    evidence: dict[str, Path] = {}
    for filename in CONTROL_JSON:
        evidence[f"control/{filename}"] = _regular_file(
            control / filename, f"BES2 control {filename}"
        )
    for filename in CONTROL_TEXT:
        evidence[f"control/{filename}"] = _regular_file(
            control / filename, f"BES2 control {filename}"
        )

    for stage in RUN_STAGES:
        for variant in DIAGNOSTIC_VARIANTS:
            run = _run_dir(root, variant, stage)
            metric_path = run / "diagnostic_metrics.json"
            metric = validate_diagnostic_metrics(
                metric_path,
                expected_variant=variant,
                expected_stage=stage,
            )
            training, _checkpoint = validate_training_metrics(
                run / "training_metrics.json",
                expected_variant=variant,
                expected_stage=stage,
                candidate_floor=0.05,
            )
            if metric.get("training") != training:
                raise PackageError(
                    f"{variant}/{stage} diagnostic/training evidence differs"
                )
            initialization = read_regular_json(
                run / "initialization.json",
                f"{variant}/{stage} initialization",
            )
            if training.get("initialization") != initialization:
                raise PackageError(
                    f"{variant}/{stage} initialization binding drifted"
                )
            metrics[(variant, stage)] = metric
            for filename in RUN_FILES:
                evidence[_logical_run_file(variant, stage, filename)] = (
                    _regular_file(run / filename, f"{variant}/{stage} {filename}")
                )

    probe_marker = _stage_marker(
        control / PROBE_COMPLETE,
        root=root,
        stage="probe",
        readiness_sha256=readiness_sha256,
    )
    full_marker = _stage_marker(
        control / EXECUTION_COMPLETE,
        root=root,
        stage="full",
        readiness_sha256=readiness_sha256,
    )
    if full_marker.get("probe_pair") != hash_binding(
        control / PROBE_COMPLETE, relative_to=root
    ):
        raise PackageError("full completion does not bind the probe pair")
    expected_total = float(probe_marker["gpu_hours"]) + float(
        full_marker["gpu_hours"]
    )
    if not math.isclose(
        float(full_marker.get("total_pair_gpu_hours", float("nan"))),
        expected_total,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise PackageError("BES2 full completion total GPU-hours drifted")

    replay_path = _run_dir(
        root, "current_replay", "full"
    ) / "diagnostic_metrics.json"
    reset_path = _run_dir(
        root, "first_conv_reset", "full"
    ) / "diagnostic_metrics.json"
    expected_summary = summarize_payload(
        metrics[("current_replay", "full")],
        metrics[("first_conv_reset", "full")],
        replay_binding=hash_binding(replay_path),
        reset_binding=hash_binding(reset_path),
    )
    observed_summary = read_regular_json(
        control / ROOT_CAUSE_FILENAME, "BES2 root-cause decision"
    )
    expected_summary["created_utc"] = observed_summary.get("created_utc")
    if expected_summary != observed_summary:
        raise PackageError("BES2 root-cause JSON is not reproducible")
    report = (control / REPORT_FILENAME).read_text(encoding="utf-8")
    classification = str(observed_summary["decision"]["classification"])
    if report != _markdown_report(observed_summary):
        raise PackageError("BES2 Markdown report is not reproducible from JSON")

    hardware_paths = sorted(control.glob("h100_runtime-*.json"))
    if not hardware_paths:
        raise PackageError("BES2 result lacks strict-H100 allocation evidence")
    for path in hardware_paths:
        if not _HARDWARE.fullmatch(path.name):
            raise PackageError(f"invalid BES2 hardware filename: {path.name}")
        _hardware_json(path)
        evidence[f"runtime/{path.name}"] = _regular_file(
            path, "BES2 hardware inventory"
        )

    logs = _require_no_symlink_components(job_log_dir, leaf="directory")
    observed_modes: set[str] = set()
    for path in sorted(logs.glob("*-bes2-*.out")):
        match = _SLURM_LOG.fullmatch(path.name)
        if match is None:
            raise PackageError(f"invalid BES2 Slurm log filename: {path.name}")
        observed_modes.add(match.group(1))
        evidence[f"slurm/{path.name}"] = _regular_file(path, "BES2 Slurm log")
    if observed_modes != {"audit", "probe", "full"}:
        raise PackageError("BES2 result requires audit/probe/full Slurm logs")

    for logical, path in evidence.items():
        relative = PurePosixPath(logical)
        _validate_safe_path(relative)
        lower = logical.lower()
        if (
            path.suffix.lower() not in _ALLOWED_SUFFIXES
            or any(token in lower for token in _FORBIDDEN_PATH_TOKENS)
            or path.is_symlink()
        ):
            raise PackageError(f"forbidden BES2 result evidence: {logical}")
        _scan(path)
    return evidence, {
        "source_git_sha": source_sha,
        "readiness_sha256": readiness_sha256,
        "root_cause_sha256": sha256_file(control / ROOT_CAUSE_FILENAME),
        "classification": classification,
        "decision_status": observed_summary["status"],
        "replacement_eligibility": observed_summary["decision"][
            "replacement_eligibility"
        ],
        "probe_gpu_hours": probe_marker["gpu_hours"],
        "full_gpu_hours": full_marker["gpu_hours"],
        "total_pair_gpu_hours": full_marker["total_pair_gpu_hours"],
        "created_utc": observed_summary["created_utc"],
    }


def _contract(max_part_bytes: int) -> dict[str, object]:
    if (
        isinstance(max_part_bytes, bool)
        or not isinstance(max_part_bytes, int)
        or max_part_bytes <= 0
    ):
        raise PackageError("BES2 result max part size must be positive")
    return {
        "schema": 2,
        "production": True,
        "scope": "train111-fixed-dev8-only",
        "evidence_only": True,
        "checkpoint_artifacts": 0,
        "test_or_final_artifacts": 0,
        "credentials": 0,
        "variants": list(DIAGNOSTIC_VARIANTS),
        "stages": list(RUN_STAGES),
        "allowed_suffixes": sorted(_ALLOWED_SUFFIXES),
        "maximum_physical_file_bytes": max_part_bytes,
    }

def _expected_logical_names(names: set[str]) -> None:
    base = {
        *(f"control/{name}" for name in CONTROL_JSON),
        *(f"control/{name}" for name in CONTROL_TEXT),
        *(
            _logical_run_file(variant, stage, filename)
            for stage in RUN_STAGES
            for variant in DIAGNOSTIC_VARIANTS
            for filename in RUN_FILES
        ),
    }
    remainder = names - base
    if not base <= names or not remainder:
        raise PackageError("BES2 result logical evidence inventory is incomplete")
    runtime = {name for name in remainder if name.startswith("runtime/")}
    slurm = {name for name in remainder if name.startswith("slurm/")}
    if runtime | slurm != remainder or not runtime or not slurm:
        raise PackageError("BES2 result contains nonallowlisted logical evidence")
    if any(
        not _HARDWARE.fullmatch(PurePosixPath(name).name)
        for name in runtime
    ):
        raise PackageError("BES2 packaged runtime evidence name is invalid")
    matches = [
        _SLURM_LOG.fullmatch(PurePosixPath(name).name) for name in slurm
    ]
    if (
        any(match is None for match in matches)
        or {match.group(1) for match in matches if match is not None}
        != {"audit", "probe", "full"}
    ):
        raise PackageError("BES2 packaged Slurm evidence is incomplete")


def _verify(
    package_root: Path,
    *,
    expected_source_git_sha: str | None,
) -> dict[str, object]:
    root = _require_no_symlink_components(package_root, leaf="directory")
    manifest = read_regular_json(root / "manifest.json", "BES2 result manifest")
    ready = read_regular_json(root / "READY.json", "BES2 result READY")
    sums = _require_no_symlink_components(root / "SHA256SUMS", leaf="file")
    _validate_control_record(root, ready.get("manifest", {}), "manifest.json")
    _validate_control_record(root, ready.get("checksums", {}), "SHA256SUMS")

    expected_manifest = {
        "format_version",
        "package_type",
        "package_id",
        "created_utc",
        "contract",
        "source",
        "result_identity",
        "result_identity_sha256",
        "counts",
        "artifacts",
    }
    contract = manifest.get("contract")
    source = manifest.get("source")
    identity = manifest.get("result_identity")
    artifacts = manifest.get("artifacts")
    if (
        set(manifest) != expected_manifest
        or manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("package_type") != PACKAGE_TYPE
        or not isinstance(contract, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(identity, Mapping)
        or not isinstance(artifacts, list)
        or not artifacts
    ):
        raise PackageError("BES2 result manifest schema is invalid")
    maximum = contract.get("maximum_physical_file_bytes")
    if not isinstance(maximum, int) or dict(contract) != _contract(maximum):
        raise PackageError("BES2 result package contract is invalid")
    source_sha = source.get("git_commit")
    if (
        not isinstance(source_sha, str)
        or not _HEX40.fullmatch(source_sha)
        or (
            expected_source_git_sha is not None
            and source_sha != expected_source_git_sha
        )
    ):
        raise PackageError("BES2 result source Git SHA is invalid")

    names = [
        str(item.get("name", ""))
        for item in artifacts
        if isinstance(item, Mapping)
    ]
    if len(names) != len(artifacts) or len(names) != len(set(names)):
        raise PackageError("BES2 result artifact names are not unique")
    _expected_logical_names(set(names))
    digest_index = []
    expected_sums: dict[str, str] = {}
    logical_payloads: dict[str, bytes] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise PackageError("BES2 result artifact is not an object")
        name = str(artifact.get("name", ""))
        lower = name.lower()
        if (
            PurePosixPath(name).suffix.lower() not in _ALLOWED_SUFFIXES
            or any(token in lower for token in _FORBIDDEN_PATH_TOKENS)
        ):
            raise PackageError(f"forbidden packaged BES2 evidence: {name}")
        paths = _artifact_paths(
            root,
            artifact,
            maximum=maximum,
            expected_kind="diagnostic_evidence",
            expected_name=name,
            expected_root=f"evidence/{name}",
        )
        data = b"".join(path.read_bytes() for path in paths)
        if _SECRET.search(data):
            raise PackageError(
                f"secret or Box endpoint in packaged evidence: {name}"
            )
        logical_payloads[name] = data
        digest_index.append(
            {
                "path": name,
                "bytes": int(artifact["archive_bytes"]),
                "sha256": str(artifact["archive_sha256"]),
            }
        )
        for part in artifact["parts"]:
            expected_sums[str(part["path"])] = str(part["sha256"])

    expected_identity = {
        "schema": 2,
        "source_git_sha": source_sha,
        "readiness_sha256": source.get("readiness_sha256"),
        "root_cause_sha256": source.get("root_cause_sha256"),
        "classification": source.get("classification"),
        "decision_status": source.get("decision_status"),
        "replacement_eligibility": source.get("replacement_eligibility"),
        "probe_gpu_hours": source.get("probe_gpu_hours"),
        "full_gpu_hours": source.get("full_gpu_hours"),
        "total_pair_gpu_hours": source.get("total_pair_gpu_hours"),
        "evidence_digest_index": digest_index,
    }
    identity_sha = hashlib.sha256(canonical_json(expected_identity)).hexdigest()
    package_id = f"xview3-bes2-results-{source_sha}-{identity_sha}"
    if (
        dict(identity) != expected_identity
        or manifest.get("result_identity_sha256") != identity_sha
        or manifest.get("package_id") != package_id
        or source.get("root_cause_sha256")
        != hashlib.sha256(
            logical_payloads[f"control/{ROOT_CAUSE_FILENAME}"]
        ).hexdigest()
        or source.get("readiness_sha256")
        != hashlib.sha256(
            logical_payloads[f"control/{READY_FILENAME}"]
        ).hexdigest()
    ):
        raise PackageError("BES2 result content identity is invalid")
    expected_ready = {
        "format_version",
        "status",
        "package_id",
        "git_commit",
        "result_identity_sha256",
        "manifest",
        "checksums",
    }
    if (
        set(ready) != expected_ready
        or ready.get("format_version") != FORMAT_VERSION
        or ready.get("status") != "READY"
        or ready.get("package_id") != package_id
        or ready.get("git_commit") != source_sha
        or ready.get("result_identity_sha256") != identity_sha
    ):
        raise PackageError("BES2 result READY identity is invalid")

    root_cause = _json_bytes(
        logical_payloads[f"control/{ROOT_CAUSE_FILENAME}"],
        "root-cause JSON",
    )
    readiness = _json_bytes(
        logical_payloads[f"control/{READY_FILENAME}"],
        "readiness JSON",
    )
    decision = root_cause.get("decision")
    ready_source = readiness.get("source")
    runs = root_cause.get("runs")
    if (
        not isinstance(decision, Mapping)
        or not isinstance(ready_source, Mapping)
        or not isinstance(runs, Mapping)
    ):
        raise PackageError("BES2 packaged decision/readiness schema is invalid")
    if (
        root_cause.get("status") != source.get("decision_status")
        or decision.get("classification") != source.get("classification")
        or decision.get("replacement_eligibility")
        != source.get("replacement_eligibility")
        or ready_source.get("git_sha") != source_sha
    ):
        raise PackageError("BES2 packaged decision/readiness binding is invalid")
    evidence_hash = {
        item["path"]: item["sha256"] for item in digest_index
    }
    packaged_metrics: dict[tuple[str, str], dict[str, object]] = {}
    readiness_sha256 = source.get("readiness_sha256")
    for stage in RUN_STAGES:
        for variant in DIAGNOSTIC_VARIANTS:
            logical = _logical_run_file(
                variant, stage, "diagnostic_metrics.json"
            )
            metrics_payload = _json_bytes(
                logical_payloads[logical],
                f"{variant}/{stage} diagnostic metrics",
            )
            try:
                validated = validate_diagnostic_metrics_payload(
                    metrics_payload,
                    expected_variant=variant,
                    expected_stage=stage,
                )
            except BES2ContractError as exc:
                raise PackageError(
                    f"invalid packaged {variant}/{stage} diagnostic metrics"
                ) from exc
            metrics_readiness = validated.get("readiness")
            if (
                not isinstance(metrics_readiness, Mapping)
                or metrics_readiness.get("sha256") != readiness_sha256
            ):
                raise PackageError(
                    f"{variant}/{stage} readiness binding differs from package"
                )
            packaged_metrics[(variant, stage)] = validated

    for variant in DIAGNOSTIC_VARIANTS:
        binding = runs.get(variant)
        if not isinstance(binding, Mapping) or set(binding) != {
            "path",
            "sha256",
        }:
            raise PackageError("BES2 root-cause run binding is invalid")
        expected_path = _logical_run_file(
            variant, "full", "diagnostic_metrics.json"
        )
        expected_suffix = (
            f"bes2-{variant}-full-f100-s0/diagnostic_metrics.json"
        )
        if (
            binding.get("sha256") != evidence_hash[expected_path]
            or not str(binding.get("path", "")).endswith(expected_suffix)
        ):
            raise PackageError("BES2 root-cause run hash differs from package")
    try:
        expected_root_cause = summarize_payload(
            packaged_metrics[("current_replay", "full")],
            packaged_metrics[("first_conv_reset", "full")],
            replay_binding=dict(runs["current_replay"]),
            reset_binding=dict(runs["first_conv_reset"]),
        )
    except BES2ContractError as exc:
        raise PackageError(
            "packaged BES2 metrics cannot produce a cause decision"
        ) from exc
    expected_root_cause["created_utc"] = root_cause.get("created_utc")
    if expected_root_cause != root_cause:
        raise PackageError(
            "BES2 packaged root-cause decision is not reproducible"
        )
    try:
        packaged_report = logical_payloads[
            f"control/{REPORT_FILENAME}"
        ].decode("utf-8")
    except UnicodeError as exc:
        raise PackageError("packaged BES2 report is not UTF-8") from exc
    if packaged_report != _markdown_report(root_cause):
        raise PackageError(
            "packaged BES2 Markdown report is not reproducible"
        )
    if _parse_sha256sums(sums) != expected_sums:
        raise PackageError("BES2 result SHA256SUMS differs from artifacts")
    expected_files = set(expected_sums) | {
        "manifest.json",
        "SHA256SUMS",
        "READY.json",
    }
    if _package_regular_files(root) != expected_files:
        raise PackageError("BES2 result physical tree is not exact")
    if manifest.get("counts") != {
        "evidence_files": len(artifacts),
        "json": sum(name.endswith(".json") for name in names),
        "markdown": sum(name.endswith(".md") for name in names),
        "logs": sum(
            name.endswith(".log") or name.endswith(".out") for name in names
        ),
        "checkpoints": 0,
    }:
        raise PackageError("BES2 result evidence counts are invalid")
    return manifest


def prepare_bes2_results_verifier(
    expected_source_git_sha: str | None = None,
) -> PackageVerifier:
    if (
        expected_source_git_sha is not None
        and not _HEX40.fullmatch(expected_source_git_sha)
    ):
        raise PackageError("expected BES2 result Git SHA is malformed")

    def verifier(package_root: Path) -> dict[str, object]:
        return _verify(
            package_root,
            expected_source_git_sha=expected_source_git_sha,
        )

    return verifier


def verify_bes2_results(
    package_root: Path,
    *,
    expected_source_git_sha: str | None = None,
) -> dict[str, object]:
    return prepare_bes2_results_verifier(expected_source_git_sha)(package_root)


def build_bes2_results(
    *,
    repo: Path,
    diagnostic_root: Path,
    original_runs_root: Path,
    job_log_dir: Path,
    output_dir: Path,
    max_part_bytes: int,
) -> Path:
    source_repo = _require_no_symlink_components(repo, leaf="directory")
    output = _require_no_symlink_components(output_dir, leaf="directory")
    root = _require_no_symlink_components(diagnostic_root, leaf="directory")
    if _inside_repository_worktrees(output, source_repo):
        raise PackageError("BES2 result output must be outside repository worktrees")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise PackageError("BES2 result output must be outside diagnostic root")
    evidence, summary = _collect_evidence(
        repo=source_repo,
        diagnostic_root=root,
        original_runs_root=original_runs_root,
        job_log_dir=job_log_dir,
    )
    contract = _contract(max_part_bytes)
    staging = Path(tempfile.mkdtemp(prefix=".xview3-bes2-results-", dir=output))
    try:
        artifacts = []
        digest_index = []
        for logical, source_path in sorted(evidence.items()):
            destination = staging / "evidence" / logical
            destination.parent.mkdir(parents=True, exist_ok=True)
            before = {
                "bytes": source_path.stat().st_size,
                "sha256": _hash_file(source_path),
            }
            shutil.copyfile(source_path, destination)
            if (
                destination.stat().st_size != before["bytes"]
                or _hash_file(destination) != before["sha256"]
                or source_path.stat().st_size != before["bytes"]
                or _hash_file(source_path) != before["sha256"]
            ):
                raise PackageError(f"BES2 evidence changed while packaging: {logical}")
            artifact = _plain_artifact(
                destination,
                staging,
                kind="diagnostic_evidence",
                name=logical,
                extraction_root=PurePosixPath("evidence") / logical,
                max_part_bytes=max_part_bytes,
            )
            artifacts.append(artifact)
            digest_index.append(
                {
                    "path": logical,
                    "bytes": before["bytes"],
                    "sha256": before["sha256"],
                }
            )
        source = {
            "git_commit": summary["source_git_sha"],
            "readiness_sha256": summary["readiness_sha256"],
            "root_cause_sha256": summary["root_cause_sha256"],
            "classification": summary["classification"],
            "decision_status": summary["decision_status"],
            "replacement_eligibility": summary["replacement_eligibility"],
            "probe_gpu_hours": summary["probe_gpu_hours"],
            "full_gpu_hours": summary["full_gpu_hours"],
            "total_pair_gpu_hours": summary["total_pair_gpu_hours"],
        }
        identity = {
            "schema": 2,
            "source_git_sha": source["git_commit"],
            "readiness_sha256": source["readiness_sha256"],
            "root_cause_sha256": source["root_cause_sha256"],
            "classification": source["classification"],
            "decision_status": source["decision_status"],
            "replacement_eligibility": source["replacement_eligibility"],
            "probe_gpu_hours": source["probe_gpu_hours"],
            "full_gpu_hours": source["full_gpu_hours"],
            "total_pair_gpu_hours": source["total_pair_gpu_hours"],
            "evidence_digest_index": digest_index,
        }
        identity_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
        package_id = (
            f"xview3-bes2-results-{source['git_commit']}-{identity_sha}"
        )
        counts = {
            "evidence_files": len(artifacts),
            "json": sum(name.endswith(".json") for name in evidence),
            "markdown": sum(name.endswith(".md") for name in evidence),
            "logs": sum(
                name.endswith(".log") or name.endswith(".out")
                for name in evidence
            ),
            "checkpoints": 0,
        }
        manifest = {
            "format_version": FORMAT_VERSION,
            "package_type": PACKAGE_TYPE,
            "package_id": package_id,
            "created_utc": summary["created_utc"],
            "contract": contract,
            "source": source,
            "result_identity": identity,
            "result_identity_sha256": identity_sha,
            "counts": counts,
            "artifacts": artifacts,
        }
        _write_bytes(staging / "manifest.json", canonical_json(manifest))
        checksum_text = "".join(
            f"{part['sha256']}  {part['path']}\n"
            for artifact in artifacts
            for part in artifact["parts"]
        )
        _write_bytes(staging / "SHA256SUMS", checksum_text.encode("utf-8"))
        ready = {
            "format_version": FORMAT_VERSION,
            "status": "READY",
            "package_id": package_id,
            "git_commit": source["git_commit"],
            "result_identity_sha256": identity_sha,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        # READY is deliberately the final package write.
        _write_bytes(staging / "READY.json", canonical_json(ready))
        verify_bes2_results(
            staging,
            expected_source_git_sha=str(source["git_commit"]),
        )
        destination = output / package_id
        if os.path.lexists(destination):
            raise PackageError(f"BES2 result destination exists: {destination}")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
