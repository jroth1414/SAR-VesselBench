"""Hardware-neutral runtime provenance for the independent R2/R3 references.

Reference results are outside the controlled eight-arm curves, but they still
need reproducible source, environment, hardware, and precision records.  This
module enforces those bindings without assuming a scheduler, host, GPU model,
or launcher.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.runtime.io import atomic_write_json, sha256_file


PROVENANCE_SCHEMA = 1
PROVENANCE_KEYS = frozenset(
    {
        "schema",
        "git_sha",
        "environment_sha256",
        "environment_lock_sha256",
        "hardware",
        "started_utc",
        "finished_utc",
        "elapsed_hours",
        "gpu_hours",
        "reference_precision",
    }
)
HARDWARE_KEYS = frozenset(
    {
        "accelerator",
        "compute_capability",
        "device",
        "device_index",
        "torch_version",
        "cuda_version",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CUDA_DEVICE = re.compile(r"^cuda(?::([0-9]+))?$")
_UNLOCKED_RUNTIME_PACKAGES = {"jhu-xview3", "pip", "setuptools"}


def _safe_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _canonical_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _environment_lock_lines(path: Path) -> set[str]:
    lines: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
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
    """Hash the canonical package set encoded by an environment lock."""

    normalized = "\n".join(sorted(_environment_lock_lines(lock_path))) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _runtime_environment_lines() -> set[str]:
    lines: set[str] = set()
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
    """Require the active Python 3.11 environment to match the pinned lock."""

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("reference execution requires Python 3.11")
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
    git_sha: str
    environment_sha256: str
    environment_lock: Path
    environment_lock_sha256: str

    def revalidate(self) -> None:
        """Reject source, environment, or lock drift before publication."""

        if _clean_git_sha(self.repo) != self.git_sha:
            raise RuntimeError("reference source Git SHA changed during execution")
        if validate_runtime_environment(self.environment_lock) != self.environment_sha256:
            raise RuntimeError("reference runtime environment changed during execution")
        observed = sha256_file(
            _safe_regular_file(self.environment_lock, "environment lock")
        )
        if observed != self.environment_lock_sha256:
            raise RuntimeError("reference environment lock changed during execution")


@dataclass(frozen=True)
class ReferenceExecution:
    inputs: ReferenceRuntimeInputs
    reference_precision: str
    hardware: Mapping[str, object]
    started_utc: str
    started_monotonic_ns: int


def add_runtime_provenance_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the explicit bindings required by a reportable reference run."""

    parser.add_argument("--expected-git-sha")
    parser.add_argument("--environment-sha256")
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--results-root", type=Path)


def _required_arg(args: argparse.Namespace, name: str) -> object:
    value = getattr(args, name, None)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(
            f"reportable reference run requires --{name.replace('_', '-')}"
        )
    return value


def load_runtime_inputs(
    args: argparse.Namespace,
    *,
    repo: Path,
    required: bool,
) -> ReferenceRuntimeInputs | None:
    """Validate explicit source/environment bindings, or return ``None`` for smoke."""

    if not required:
        return None
    expected_git_sha = str(_required_arg(args, "expected_git_sha"))
    environment_sha256 = str(_required_arg(args, "environment_sha256"))
    environment_lock = Path(_required_arg(args, "environment_lock"))
    _required_arg(args, "results_root")

    if _GIT_SHA.fullmatch(expected_git_sha) is None:
        raise RuntimeError("--expected-git-sha must be a lowercase 40-hex SHA")
    if _SHA256.fullmatch(environment_sha256) is None:
        raise RuntimeError("--environment-sha256 must be a lowercase 64-hex digest")

    current_git_sha = _clean_git_sha(repo)
    if current_git_sha != expected_git_sha:
        raise RuntimeError(
            f"reference source Git SHA mismatch: {current_git_sha} != {expected_git_sha}"
        )
    environment_lock = _safe_regular_file(environment_lock, "environment lock")
    observed_environment_sha256 = validate_runtime_environment(environment_lock)
    if observed_environment_sha256 != environment_sha256:
        raise RuntimeError(
            "--environment-sha256 does not match the normalized installed runtime"
        )

    return ReferenceRuntimeInputs(
        repo=repo.resolve(),
        git_sha=current_git_sha,
        environment_sha256=environment_sha256,
        environment_lock=environment_lock,
        environment_lock_sha256=sha256_file(environment_lock),
    )


def _device_index(device: str) -> int:
    match = _CUDA_DEVICE.fullmatch(device)
    if match is None:
        raise RuntimeError("reference device must be 'cuda' or 'cuda:N'")
    return int(match.group(1) or 0)


def probe_gpu_runtime(torch_module, *, device: str) -> dict[str, object]:
    """Validate and describe the requested CUDA device without naming a GPU class."""

    index = _device_index(device)
    cuda = torch_module.cuda
    if not cuda.is_available() or cuda.device_count() <= index:
        raise RuntimeError(f"reference CUDA device is unavailable: cuda:{index}")
    capability = tuple(cuda.get_device_capability(index))
    if (
        len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
        or any(value < 0 for value in capability)
    ):
        raise RuntimeError("reference GPU compute capability is invalid")
    accelerator = str(cuda.get_device_name(index)).strip()
    torch_version = str(torch_module.__version__).strip()
    cuda_version = str(torch_module.version.cuda or "").strip()
    if not accelerator or not torch_version or not cuda_version:
        raise RuntimeError("reference CUDA runtime identity is incomplete")
    compiled_arches = set(cuda.get_arch_list())
    required_arch = f"sm_{capability[0]}{capability[1]}"
    if compiled_arches and required_arch not in compiled_arches:
        raise RuntimeError(
            f"reference torch build does not include assigned GPU architecture {required_arch}"
        )
    hardware: dict[str, object] = {
        "accelerator": accelerator,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "device": f"cuda:{index}",
        "device_index": index,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
    }
    if set(hardware) != HARDWARE_KEYS:
        raise AssertionError("internal reference hardware schema drift")
    return hardware


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
    return ReferenceExecution(
        inputs=inputs,
        reference_precision=reference_precision,
        hardware=probe_gpu_runtime(torch_module, device=device),
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
        started_monotonic_ns=time.monotonic_ns(),
    )


def finish_reference_execution(
    execution: ReferenceExecution,
    *,
    device: str,
    torch_module,
) -> dict[str, object]:
    """Synchronize, revalidate all bindings, and return portable provenance."""

    torch_module.cuda.synchronize(_device_index(device))
    execution.inputs.revalidate()
    hardware = probe_gpu_runtime(torch_module, device=device)
    if hardware != execution.hardware:
        raise RuntimeError("reference GPU identity changed during execution")
    elapsed_hours = max(
        (time.monotonic_ns() - execution.started_monotonic_ns) / 3_600_000_000_000,
        1 / 3_600_000_000_000,
    )
    if not math.isfinite(elapsed_hours) or elapsed_hours <= 0:
        raise RuntimeError("reference elapsed time is invalid")
    payload: dict[str, object] = {
        "schema": PROVENANCE_SCHEMA,
        "git_sha": execution.inputs.git_sha,
        "environment_sha256": execution.inputs.environment_sha256,
        "environment_lock_sha256": execution.inputs.environment_lock_sha256,
        "hardware": hardware,
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
    """Atomically publish provenance first and final metrics as commit marker."""

    result_dir = Path(result_dir)
    if result_dir.is_symlink():
        raise RuntimeError("reference result directory must not be a symlink")
    result_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = result_dir / "runtime_provenance.json"
    metrics_path = result_dir / "final_metrics.json"
    existing = [
        path
        for path in (provenance_path, metrics_path)
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise RuntimeError(
            "reference output already exists; use a fresh result directory: "
            + ", ".join(str(path) for path in existing)
        )
    if set(provenance) != PROVENANCE_KEYS:
        raise RuntimeError("reference provenance does not match the exact schema")
    hardware = provenance.get("hardware")
    if not isinstance(hardware, Mapping) or set(hardware) != HARDWARE_KEYS:
        raise RuntimeError("reference hardware provenance does not match the exact schema")
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
            "reference output already exists; use a fresh result directory: "
            + ", ".join(str(path) for path in existing)
        )
    return result_dir
