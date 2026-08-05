"""Fail-closed runtime provenance for corrected V100 reference executions.

R2 and R3 are separate reference experiments, not members of the diagnostic
V100 core campaign.  This module binds each corrected execution to a clean
source commit, a dedicated reference-campaign manifest, immutable runtime
artifacts, and the one V100 made visible to the process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


EXPECTED_HARDWARE = "Tesla V100-SXM2-32GB"
EXPECTED_COMPUTE_CAPABILITY = (7, 0)
EXPECTED_TORCH = "2.11.0+cu126"
EXPECTED_CUDA = "12.6"
CAMPAIGN_ROLE = "corrected-v100-references"
CAMPAIGN_MANIFEST_SCHEMA = 1
CAMPAIGN_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "campaign_role",
        "campaign_id",
        "core_campaign_id",
        "core_git_sha",
        "git_sha",
        "environment_sha256",
        "environment_lock_sha256",
        "runtime_launcher_sha256",
    }
)
PROVENANCE_KEYS = {
    "campaign_id",
    "git_sha",
    "environment_sha256",
    "environment_lock_sha256",
    "campaign_manifest_sha256",
    "runtime_launcher_sha256",
    "hardware",
    "container_local_gpu",
    "gpu_uuid",
    "started_utc",
    "finished_utc",
    "elapsed_hours",
    "gpu_hours",
    "reference_precision",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UNLOCKED_RUNTIME_PACKAGES = {"jhu-xview3", "pip", "setuptools"}


def sha256_file(path: Path, chunk_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _safe_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _canonical_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _environment_lock_lines(path: Path) -> set[str]:
    lines = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise RuntimeError(f"environment lock contains an unsupported entry: {line}")
        name, version = line.split("==", 1)
        lines.add(f"{_canonical_package(name)}=={version}")
    if not lines:
        raise RuntimeError("environment lock contains no pinned packages")
    return lines


def normalized_environment_lock_sha256(lock_path: Path) -> str:
    """Hash the canonical package set encoded by a reference environment lock."""

    normalized = "\n".join(sorted(_environment_lock_lines(lock_path))) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _runtime_environment_lines() -> set[str]:
    lines = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            raise RuntimeError("installed distribution lacks a package name")
        canonical = _canonical_package(str(name))
        if canonical in _UNLOCKED_RUNTIME_PACKAGES:
            continue
        lines.add(f"{canonical}=={distribution.version}")
    return lines


def validate_runtime_environment(lock_path: Path) -> str:
    """Require the actual interpreter environment to match the pinned lock."""

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("corrected references require Python 3.11")
    expected = _environment_lock_lines(lock_path)
    observed = _runtime_environment_lines()
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            "reference runtime does not match the environment lock; "
            f"missing={missing}, extra={extra}"
        )
    return normalized_environment_lock_sha256(lock_path)


def _clean_git_sha(repo: Path) -> str:
    repo = repo.resolve()
    sha = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if _GIT_SHA.fullmatch(sha) is None:
        raise RuntimeError("reference source did not resolve to a full Git SHA")
    status = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo}",
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError("reference source worktree is not clean")
    return sha


@dataclass(frozen=True)
class ReferenceRuntimeInputs:
    repo: Path
    campaign_id: str
    core_campaign_id: str
    core_git_sha: str
    git_sha: str
    environment_sha256: str
    environment_lock: Path
    environment_lock_sha256: str
    campaign_manifest: Path
    campaign_manifest_sha256: str
    runtime_launcher: Path
    runtime_launcher_sha256: str

    def revalidate(self) -> None:
        """Reject source or provenance-artifact drift before publication."""

        if _clean_git_sha(self.repo) != self.git_sha:
            raise RuntimeError("reference source Git SHA changed during execution")
        if validate_runtime_environment(self.environment_lock) != self.environment_sha256:
            raise RuntimeError("reference runtime environment changed during execution")
        observed = {
            "environment lock": sha256_file(
                _safe_regular_file(self.environment_lock, "environment lock")
            ),
            "campaign manifest": sha256_file(
                _safe_regular_file(self.campaign_manifest, "campaign manifest")
            ),
            "runtime launcher": sha256_file(
                _safe_regular_file(self.runtime_launcher, "runtime launcher")
            ),
        }
        expected = {
            "environment lock": self.environment_lock_sha256,
            "campaign manifest": self.campaign_manifest_sha256,
            "runtime launcher": self.runtime_launcher_sha256,
        }
        drifted = [name for name, digest in observed.items() if digest != expected[name]]
        if drifted:
            raise RuntimeError(
                "reference provenance artifact changed during execution: "
                + ", ".join(drifted)
            )


@dataclass(frozen=True)
class ReferenceExecution:
    inputs: ReferenceRuntimeInputs
    reference_precision: str
    hardware: str
    container_local_gpu: int
    gpu_uuid: str
    started_utc: str
    started_monotonic_ns: int


def add_runtime_provenance_arguments(parser: argparse.ArgumentParser) -> None:
    """Add inputs required only by a production corrected-reference run."""

    parser.add_argument("--expected-git-sha")
    parser.add_argument("--reference-campaign-id")
    parser.add_argument("--core-campaign-id")
    parser.add_argument("--core-git-sha")
    parser.add_argument("--environment-sha256")
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--runtime-launcher", type=Path)
    parser.add_argument("--results-root", type=Path)


def _required_arg(args: argparse.Namespace, name: str) -> object:
    value = getattr(args, name, None)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(
            f"production corrected-reference run requires --{name.replace('_', '-')}"
        )
    return value


def load_runtime_inputs(
    args: argparse.Namespace,
    *,
    repo: Path,
    required: bool,
) -> ReferenceRuntimeInputs | None:
    """Validate explicit campaign/runtime bindings, or return None for smoke."""

    if not required:
        return None
    expected_git_sha = str(_required_arg(args, "expected_git_sha"))
    campaign_id = str(_required_arg(args, "reference_campaign_id"))
    core_campaign_id = str(_required_arg(args, "core_campaign_id"))
    core_git_sha = str(_required_arg(args, "core_git_sha"))
    environment_sha256 = str(_required_arg(args, "environment_sha256"))
    environment_lock = Path(_required_arg(args, "environment_lock"))
    campaign_manifest = Path(_required_arg(args, "campaign_manifest"))
    runtime_launcher = Path(_required_arg(args, "runtime_launcher"))
    _required_arg(args, "results_root")

    if _GIT_SHA.fullmatch(expected_git_sha) is None:
        raise RuntimeError("--expected-git-sha must be a lowercase 40-hex SHA")
    if _SHA256.fullmatch(environment_sha256) is None:
        raise RuntimeError("--environment-sha256 must be a lowercase 64-hex digest")
    if _CAMPAIGN_ID.fullmatch(campaign_id) is None:
        raise RuntimeError("--reference-campaign-id is invalid")
    if _CAMPAIGN_ID.fullmatch(core_campaign_id) is None:
        raise RuntimeError("--core-campaign-id is invalid")
    if _GIT_SHA.fullmatch(core_git_sha) is None:
        raise RuntimeError("--core-git-sha must be a lowercase 40-hex SHA")
    if campaign_id == core_campaign_id:
        raise RuntimeError(
            "corrected references require a new campaign ID distinct from the "
            "diagnostic V100 core campaign"
        )

    current_git_sha = _clean_git_sha(repo)
    if current_git_sha != expected_git_sha:
        raise RuntimeError(
            f"reference source Git SHA mismatch: {current_git_sha} != {expected_git_sha}"
        )
    environment_lock = _safe_regular_file(environment_lock, "environment lock")
    campaign_manifest = _safe_regular_file(campaign_manifest, "campaign manifest")
    runtime_launcher = _safe_regular_file(runtime_launcher, "runtime launcher")
    environment_lock_sha256 = sha256_file(environment_lock)
    observed_environment_sha256 = validate_runtime_environment(environment_lock)
    if observed_environment_sha256 != environment_sha256:
        raise RuntimeError(
            "--environment-sha256 does not match the normalized installed runtime"
        )
    runtime_launcher_sha256 = sha256_file(runtime_launcher)
    campaign_manifest_sha256 = sha256_file(campaign_manifest)

    try:
        manifest = json.loads(campaign_manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("reference campaign manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError("reference campaign manifest must be a JSON object")
    expected_manifest = {
        "schema": CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_role": CAMPAIGN_ROLE,
        "campaign_id": campaign_id,
        "core_campaign_id": core_campaign_id,
        "core_git_sha": core_git_sha,
        "git_sha": current_git_sha,
        "environment_sha256": environment_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "runtime_launcher_sha256": runtime_launcher_sha256,
    }
    if set(manifest) != CAMPAIGN_MANIFEST_KEYS:
        raise RuntimeError(
            "reference campaign manifest must contain the exact schema-1 fields; "
            f"observed={sorted(manifest)}"
        )
    mismatches = {
        key: (expected, manifest.get(key))
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "reference campaign manifest bindings are invalid: "
            + ", ".join(sorted(mismatches))
        )

    return ReferenceRuntimeInputs(
        repo=repo.resolve(),
        campaign_id=campaign_id,
        core_campaign_id=core_campaign_id,
        core_git_sha=core_git_sha,
        git_sha=current_git_sha,
        environment_sha256=environment_sha256,
        environment_lock=environment_lock,
        environment_lock_sha256=environment_lock_sha256,
        campaign_manifest=campaign_manifest,
        campaign_manifest_sha256=campaign_manifest_sha256,
        runtime_launcher=runtime_launcher,
        runtime_launcher_sha256=runtime_launcher_sha256,
    )


def _query_gpu(index: int) -> tuple[str, str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=name,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().splitlines()
    if len(output) != 1 or "," not in output[0]:
        raise RuntimeError("nvidia-smi did not return one unambiguous GPU identity")
    name, uuid = (part.strip() for part in output[0].split(",", 1))
    if name != EXPECTED_HARDWARE or not uuid.startswith("GPU-"):
        raise RuntimeError("reference execution is not on the required V100 hardware")
    return name, uuid


def probe_gpu_runtime(torch_module, *, device: str) -> tuple[str, int, str]:
    """Resolve one numeric container-local V100 and validate torch/CUDA."""

    if device not in {"cuda", "cuda:0"}:
        raise RuntimeError("corrected references require the single visible cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"[0-7]", visible) is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must name exactly one numeric container-local GPU"
        )
    container_local_gpu = int(visible)
    cuda = torch_module.cuda
    if not cuda.is_available() or cuda.device_count() != 1:
        raise RuntimeError("reference execution must see exactly one CUDA device")
    if str(torch_module.__version__) != EXPECTED_TORCH:
        raise RuntimeError("reference torch runtime does not match the pinned environment")
    if str(torch_module.version.cuda) != EXPECTED_CUDA:
        raise RuntimeError("reference CUDA runtime does not match the pinned environment")
    if tuple(cuda.get_device_capability(0)) != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError("reference GPU compute capability is not sm_70")
    if "sm_70" not in set(cuda.get_arch_list()):
        raise RuntimeError("reference torch build does not include sm_70")
    hardware, gpu_uuid = _query_gpu(container_local_gpu)
    if cuda.get_device_name(0) != hardware:
        raise RuntimeError("torch and nvidia-smi disagree on the assigned GPU")
    return hardware, container_local_gpu, gpu_uuid


def begin_reference_execution(
    inputs: ReferenceRuntimeInputs,
    *,
    reference_precision: str,
    device: str,
    torch_module,
) -> ReferenceExecution:
    if reference_precision not in {"float32", "bfloat16"}:
        raise RuntimeError("reference precision is invalid")
    inputs.revalidate()
    hardware, local_gpu, gpu_uuid = probe_gpu_runtime(torch_module, device=device)
    return ReferenceExecution(
        inputs=inputs,
        reference_precision=reference_precision,
        hardware=hardware,
        container_local_gpu=local_gpu,
        gpu_uuid=gpu_uuid,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
        started_monotonic_ns=time.monotonic_ns(),
    )


def finish_reference_execution(
    execution: ReferenceExecution,
    *,
    device: str,
    torch_module,
) -> dict[str, object]:
    """Synchronize, revalidate bindings, and return the exact cutover schema."""

    torch_module.cuda.synchronize()
    execution.inputs.revalidate()
    hardware, local_gpu, gpu_uuid = probe_gpu_runtime(torch_module, device=device)
    if (
        hardware,
        local_gpu,
        gpu_uuid,
    ) != (
        execution.hardware,
        execution.container_local_gpu,
        execution.gpu_uuid,
    ):
        raise RuntimeError("reference GPU identity changed during execution")
    elapsed_hours = max(
        (time.monotonic_ns() - execution.started_monotonic_ns) / 3_600_000_000_000,
        1 / 3_600_000_000_000,
    )
    if not math.isfinite(elapsed_hours) or elapsed_hours <= 0:
        raise RuntimeError("reference elapsed time is invalid")
    payload = {
        "campaign_id": execution.inputs.campaign_id,
        "git_sha": execution.inputs.git_sha,
        "environment_sha256": execution.inputs.environment_sha256,
        "environment_lock_sha256": execution.inputs.environment_lock_sha256,
        "campaign_manifest_sha256": execution.inputs.campaign_manifest_sha256,
        "runtime_launcher_sha256": execution.inputs.runtime_launcher_sha256,
        "hardware": hardware,
        "container_local_gpu": local_gpu,
        "gpu_uuid": gpu_uuid,
        "started_utc": execution.started_utc,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="microseconds"
        ),
        "elapsed_hours": elapsed_hours,
        "gpu_hours": elapsed_hours,
        "reference_precision": execution.reference_precision,
    }
    if set(payload) != PROVENANCE_KEYS:
        raise AssertionError("internal reference provenance schema drift")
    return payload


def publish_reference_result(
    result_dir: Path,
    *,
    metrics: Mapping[str, object],
    provenance: Mapping[str, object],
) -> tuple[Path, Path]:
    """Atomically publish provenance first and final metrics as commit marker.

    A corrected campaign uses a fresh result directory.  Refusing replacement
    prevents an old metrics file from ever being paired with new provenance.
    Each JSON file is written by durable temp-file replacement; metrics is
    published last, so its presence means the provenance sidecar is complete.
    """

    from scripts.h100.contracts import atomic_write_json

    result_dir = Path(result_dir)
    if result_dir.is_symlink():
        raise RuntimeError("reference result directory must not be a symlink")
    result_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = result_dir / "runtime_provenance.json"
    metrics_path = result_dir / "final_metrics.json"
    existing = [path for path in (provenance_path, metrics_path) if path.exists() or path.is_symlink()]
    if existing:
        raise RuntimeError(
            "corrected reference output already exists; use a fresh campaign result "
            "directory: " + ", ".join(str(path) for path in existing)
        )
    if set(provenance) != PROVENANCE_KEYS:
        raise RuntimeError("reference provenance does not match the exact cutover schema")
    atomic_write_json(provenance_path, dict(provenance))
    provenance_path.chmod(0o444)
    atomic_write_json(metrics_path, dict(metrics))
    metrics_path.chmod(0o444)
    return metrics_path, provenance_path


def result_directory(args: argparse.Namespace, exp_id: str) -> Path:
    root = Path(_required_arg(args, "results_root"))
    if root.is_symlink():
        raise RuntimeError("--results-root must not be a symlink")
    result_dir = root / exp_id
    if result_dir.is_symlink():
        raise RuntimeError("reference result directory must not be a symlink")
    existing = [
        result_dir / name
        for name in ("runtime_provenance.json", "final_metrics.json")
        if (result_dir / name).exists() or (result_dir / name).is_symlink()
    ]
    if existing:
        raise RuntimeError(
            "corrected reference output already exists; use a fresh campaign result "
            "directory: " + ", ".join(str(path) for path in existing)
        )
    return result_dir
