#!/usr/bin/env python3
"""Isolated launcher for the corrected Sprint 7f V100 references.

This launcher is deliberately independent of the live V100 core controller.
It creates one immutable reference-campaign manifest and dispatches exactly one
R2 or R3 production run at a time into a fresh external results namespace.
The operator must obtain a separate GPU lease before invoking ``r2`` or ``r3``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import yaml

from scripts.h100.contracts import atomic_write_json
from src.references import runtime_provenance


REPO = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).resolve()
CAMPAIGN_MANIFEST_NAME = "REFERENCE_CAMPAIGN.json"
RESULTS_DIRECTORY_NAME = "results"
REFERENCE_IDS = {
    "r2": "yolo26-f100",
    "r3": "locateanything-zs",
}
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _outside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return True
    return False


def _absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"{label} must be an absolute path")
    return path


def _regular_file(path: Path, label: str) -> Path:
    path = _absolute(path, label)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve()


def _directory(path: Path, label: str) -> Path:
    path = _absolute(path, label)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be a physical directory: {path}")
    return path.resolve()


def _external_file(path: Path, repo: Path, label: str) -> Path:
    resolved = _regular_file(path, label)
    if not _outside(resolved, repo.resolve()):
        raise RuntimeError(f"{label} must be outside the reference checkout")
    return resolved


def _external_directory(path: Path, repo: Path, label: str) -> Path:
    resolved = _directory(path, label)
    if not _outside(resolved, repo.resolve()):
        raise RuntimeError(f"{label} must be outside the reference checkout")
    return resolved


def _validate_sha(value: str, *, label: str, pattern: re.Pattern[str]) -> str:
    if pattern.fullmatch(value) is None:
        width = 40 if pattern is _GIT_SHA else 64
        raise RuntimeError(f"{label} must be a lowercase {width}-hex value")
    return value


def _validate_campaign_bindings(args: argparse.Namespace, repo: Path) -> str:
    reference_campaign = str(args.reference_campaign_id)
    core_campaign = str(args.core_campaign_id)
    if _CAMPAIGN_ID.fullmatch(reference_campaign) is None:
        raise RuntimeError("--reference-campaign-id is invalid")
    if _CAMPAIGN_ID.fullmatch(core_campaign) is None:
        raise RuntimeError("--core-campaign-id is invalid")
    if reference_campaign == core_campaign:
        raise RuntimeError(
            "the corrected reference campaign ID must differ from the live "
            "diagnostic core campaign ID"
        )
    expected_git_sha = _validate_sha(
        str(args.expected_git_sha), label="--expected-git-sha", pattern=_GIT_SHA
    )
    _validate_sha(str(args.core_git_sha), label="--core-git-sha", pattern=_GIT_SHA)
    _validate_sha(
        str(args.environment_sha256),
        label="--environment-sha256",
        pattern=_SHA256,
    )
    current = runtime_provenance._clean_git_sha(repo)  # one shared clean-tree gate
    if current != expected_git_sha:
        raise RuntimeError(
            f"reference source Git SHA mismatch: {current} != {expected_git_sha}"
        )
    return current


def _environment_lock(path: Path, repo: Path) -> Path:
    observed = _regular_file(path, "--environment-lock")
    expected = (repo / "locks/env-v100node.txt").resolve()
    if observed != expected:
        raise RuntimeError(
            "--environment-lock must be this checkout's locks/env-v100node.txt"
        )
    return observed


def _manifest_path(path: Path, repo: Path, *, must_exist: bool) -> Path:
    path = _absolute(path, "--campaign-manifest")
    if path.name != CAMPAIGN_MANIFEST_NAME:
        raise RuntimeError(
            f"--campaign-manifest must be named {CAMPAIGN_MANIFEST_NAME}"
        )
    parent = _external_directory(
        path.parent, repo, "reference campaign manifest parent"
    )
    normalized = parent / CAMPAIGN_MANIFEST_NAME
    if must_exist:
        return _external_file(normalized, repo, "--campaign-manifest")
    if normalized.exists() or normalized.is_symlink():
        raise RuntimeError(
            "reference campaign manifest already exists; use a fresh campaign root"
        )
    return normalized


def _reference_manifest_payload(
    args: argparse.Namespace,
    *,
    repo: Path,
) -> dict[str, object]:
    git_sha = _validate_campaign_bindings(args, repo)
    lock = _environment_lock(Path(args.environment_lock), repo)
    observed_environment = runtime_provenance.validate_runtime_environment(lock)
    if observed_environment != args.environment_sha256:
        raise RuntimeError(
            "--environment-sha256 does not match the normalized installed runtime"
        )
    launcher = _regular_file(LAUNCHER_PATH, "corrected-reference launcher")
    payload: dict[str, object] = {
        "schema": runtime_provenance.CAMPAIGN_MANIFEST_SCHEMA,
        "campaign_role": runtime_provenance.CAMPAIGN_ROLE,
        "campaign_id": args.reference_campaign_id,
        "core_campaign_id": args.core_campaign_id,
        "core_git_sha": args.core_git_sha,
        "git_sha": git_sha,
        "environment_sha256": args.environment_sha256,
        "environment_lock_sha256": runtime_provenance.sha256_file(lock),
        "runtime_launcher_sha256": runtime_provenance.sha256_file(launcher),
    }
    if set(payload) != runtime_provenance.CAMPAIGN_MANIFEST_KEYS:
        raise AssertionError("internal reference campaign manifest schema drift")
    return payload


def create_campaign_manifest(
    args: argparse.Namespace,
    *,
    repo: Path = REPO,
) -> Path:
    destination = _manifest_path(
        Path(args.campaign_manifest), repo, must_exist=False
    )
    payload = _reference_manifest_payload(args, repo=repo)
    atomic_write_json(destination, payload)
    destination.chmod(0o444)
    print(f"reference_campaign_manifest={destination}")
    print(f"reference_campaign_manifest_sha256={runtime_provenance.sha256_file(destination)}")
    return destination


def _validate_data_config(path: Path, repo: Path) -> Path:
    config_path = _external_file(path, repo, "--data-config")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        configured = payload["paths"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise RuntimeError("--data-config is not a valid xView3 data config") from exc
    if not isinstance(configured, Mapping):
        raise RuntimeError("--data-config paths must be a mapping")

    requirements = {
        "raw_xview3": "directory",
        "chips": "directory",
        "splits": "file",
        "stats": "file",
    }
    resolved: dict[str, Path] = {}
    for key, kind in requirements.items():
        value = configured.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise RuntimeError(
                f"--data-config paths.{key} must be an absolute path"
            )
        candidate = Path(value)
        if kind == "file" and not candidate.is_file():
            raise RuntimeError(f"--data-config paths.{key} is not a file: {candidate}")
        if kind == "directory" and not candidate.is_dir():
            raise RuntimeError(
                f"--data-config paths.{key} is not a directory: {candidate}"
            )
        resolved[key] = candidate.resolve()

    frozen = {
        "splits": (repo / "data/splits.json").resolve(),
        "stats": (repo / "data/stats.json").resolve(),
    }
    for key, expected in frozen.items():
        if resolved[key] != expected:
            raise RuntimeError(
                f"--data-config paths.{key} must use this clean checkout's "
                f"frozen data/{key}.json"
            )
    if not (resolved["raw_xview3"] / "labels/train.csv").is_file():
        raise RuntimeError("xView3 train.csv is absent from paths.raw_xview3")
    return config_path


def _results_root(path: Path, manifest: Path, repo: Path) -> Path:
    root = _external_directory(path, repo, "--results-root")
    expected = (manifest.parent / RESULTS_DIRECTORY_NAME).resolve()
    if root != expected:
        raise RuntimeError(
            f"--results-root must be the campaign-local directory {expected}"
        )
    return root


def _r2_weights(path: Path, repo: Path) -> Path:
    from src.references.yolo26_ref import EXPECTED_R2_BEST_SHA256

    checkpoint = _regular_file(path, "--r2-weights")
    expected = (repo / "runs/yolo26-f100/weights/best.pt").resolve()
    if checkpoint != expected:
        raise RuntimeError(
            "--r2-weights must be a physical copy at this fresh checkout's "
            "runs/yolo26-f100/weights/best.pt"
        )
    observed = runtime_provenance.sha256_file(checkpoint)
    if observed != EXPECTED_R2_BEST_SHA256:
        raise RuntimeError(
            f"preserved R2 best.pt SHA-256 mismatch: {observed} != "
            f"{EXPECTED_R2_BEST_SHA256}"
        )
    return checkpoint


def _r3_weights(path: Path) -> Path:
    weights = _directory(path, "--r3-weights")
    if not (weights / "SOURCE.note").is_file() or not (weights / "LICENSE").is_file():
        raise RuntimeError("--r3-weights lacks SOURCE.note or LICENSE")
    return weights


def _common_reference_arguments(
    args: argparse.Namespace,
    *,
    manifest: Path,
    lock: Path,
    results_root: Path,
) -> list[str]:
    return [
        "--expected-git-sha",
        str(args.expected_git_sha),
        "--reference-campaign-id",
        str(args.reference_campaign_id),
        "--core-campaign-id",
        str(args.core_campaign_id),
        "--core-git-sha",
        str(args.core_git_sha),
        "--environment-sha256",
        str(args.environment_sha256),
        "--environment-lock",
        str(lock),
        "--campaign-manifest",
        str(manifest),
        "--runtime-launcher",
        str(LAUNCHER_PATH),
        "--results-root",
        str(results_root),
    ]


def build_reference_command(
    args: argparse.Namespace,
    *,
    data_config: Path,
    manifest: Path,
    lock: Path,
    results_root: Path,
    repo: Path = REPO,
) -> tuple[list[str], str]:
    common = _common_reference_arguments(
        args, manifest=manifest, lock=lock, results_root=results_root
    )
    if args.action == "r2":
        weights = _r2_weights(Path(args.r2_weights), repo)
        command = [
            sys.executable,
            "-m",
            "src.references.yolo26_ref",
            "score",
            "--config",
            str(data_config),
            "--weights",
            str(weights),
            "--device",
            "cuda",
            *common,
        ]
    elif args.action == "r3":
        weights = _r3_weights(Path(args.r3_weights))
        command = [
            sys.executable,
            "-m",
            "src.references.locateanything_zs",
            "--config",
            str(data_config),
            "--weights",
            str(weights),
            "--n-chips",
            "200",
            "--device",
            "cuda",
            *common,
        ]
    else:  # protected by argparse and retained for direct function callers
        raise RuntimeError(f"unsupported corrected reference: {args.action!r}")
    return command, REFERENCE_IDS[args.action]


@contextmanager
def _exclusive_campaign_lock(campaign_root: Path, action: str) -> Iterator[None]:
    lock_path = campaign_root / ".corrected-references.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"cannot open corrected-reference lock: {lock_path}") from exc
    with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another corrected R2/R3 process already owns this campaign"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump({"action": action, "pid": os.getpid()}, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_published_result(
    result_dir: Path,
    *,
    expected_campaign_id: str,
    expected_git_sha: str,
) -> tuple[Path, Path]:
    provenance_path = _regular_file(
        result_dir / "runtime_provenance.json", "published runtime provenance"
    )
    metrics_path = _regular_file(
        result_dir / "final_metrics.json", "published final metrics"
    )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("corrected-reference output is not valid JSON") from exc
    if (
        provenance.get("campaign_id") != expected_campaign_id
        or provenance.get("git_sha") != expected_git_sha
        or metrics.get("source_git_sha") != expected_git_sha
    ):
        raise RuntimeError("corrected-reference output source/campaign binding is invalid")
    return metrics_path, provenance_path


def run_reference(
    args: argparse.Namespace,
    *,
    repo: Path = REPO,
    runner: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
) -> tuple[Path, Path]:
    _validate_campaign_bindings(args, repo)
    lock = _environment_lock(Path(args.environment_lock), repo)
    manifest = _manifest_path(Path(args.campaign_manifest), repo, must_exist=True)
    results_root = _results_root(Path(args.results_root), manifest, repo)
    data_config = _validate_data_config(Path(args.data_config), repo)

    runtime_args = Namespace(
        **vars(args),
        runtime_launcher=LAUNCHER_PATH,
    )
    runtime_provenance.load_runtime_inputs(runtime_args, repo=repo, required=True)
    command, exp_id = build_reference_command(
        args,
        data_config=data_config,
        manifest=manifest,
        lock=lock,
        results_root=results_root,
        repo=repo,
    )
    result_dir = results_root / exp_id
    if result_dir.exists() or result_dir.is_symlink():
        raise RuntimeError(
            f"fresh corrected-reference result directory already exists: {result_dir}"
        )

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested_gpu = str(args.gpu)
    if inherited not in (None, "", requested_gpu):
        raise RuntimeError(
            "inherited CUDA_VISIBLE_DEVICES disagrees with --gpu; do not remap a lease"
        )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = requested_gpu
    env["PYTHONNOUSERSITE"] = "1"

    with _exclusive_campaign_lock(manifest.parent, args.action):
        print(f"corrected_reference_command={shlex.join(command)}", flush=True)
        runner(command, cwd=repo, env=env, check=True)
        outputs = _validate_published_result(
            result_dir,
            expected_campaign_id=args.reference_campaign_id,
            expected_git_sha=args.expected_git_sha,
        )
    print(f"corrected_reference_status=success reference={exp_id}")
    print(f"final_metrics={outputs[0]}")
    print(f"runtime_provenance={outputs[1]}")
    return outputs


def _add_campaign_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--reference-campaign-id", required=True)
    parser.add_argument("--core-campaign-id", required=True)
    parser.add_argument("--core-git-sha", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    _add_campaign_arguments(parser)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    manifest = commands.add_parser(
        "manifest", help="create the immutable schema-1 reference campaign manifest"
    )
    _add_campaign_arguments(manifest)

    r2 = commands.add_parser(
        "r2", help="rescore the preserved R2 best.pt under the corrected contract"
    )
    _add_run_arguments(r2)
    r2.add_argument("--r2-weights", type=Path, required=True)

    r3 = commands.add_parser(
        "r3", help="rerun the pinned R3 zero-shot reference under the corrected contract"
    )
    _add_run_arguments(r3)
    r3.add_argument("--r3-weights", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "manifest":
            create_campaign_manifest(args)
        else:
            run_reference(args)
    except subprocess.CalledProcessError as exc:
        return exc.returncode if 0 < exc.returncode < 126 else 1
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
