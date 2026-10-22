"""Phase-separated H100 data staging around the immutable training cohort.

The Sprint 7d base package stores each chip and raster scene separately, but
stores all pre-final-evaluation labels in one archive.  Sprint 7f therefore
carries a small source-built TRAIN+fixed-DEV8 label artifact in its runtime
amendment.  A training allocation never extracts the combined label archive or
any TEST scene.  After the all-32 ``TRAINING_COHORT.json`` validates, a fresh
allocation may instead extract the TEST view and the combined labels.

This module is host/CPU-only.  It intentionally imports no torch or CUDA code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from scripts.h100.contracts import load_cells, sha256_file
from scripts.handoff.package import (
    PackageError,
    _artifact_part_paths,
    _extract_tar_zst,
    _hash_paths,
    _safe_output_path,
)
from src.eval.ground_truth_audit import scene_ids_sha256
from src.eval.ground_truth_audit import (
    audit_ground_truth_dataset,
    validate_ground_truth_audit_receipt,
)
from src.eval.heldout_contract import COHORT_FILENAME, validate_training_cohort


DATA_VIEW_SCHEMA = 1
REPLACEMENT_DATA_VIEW_SCHEMA = 2
DATA_VIEW_RECEIPT = "H100_DATA_VIEW.json"
TRAINING_LABELS_ARTIFACT_PATH = "data/training-view/labels/train.csv"
TRAINING_LABELS_EXPOSED_PATH = "data/raw/xview3/labels/train.csv"
TRAINING_VIEW_CONTRACT = "train111-fixed-dev8-no-test-v1"
SCORE_VIEW_CONTRACT = "frozen-cohort-test16-v1"
EXPECTED_TRAIN_SCENES = 111
EXPECTED_DEV_SCENES = 23
EXPECTED_DEV_TRAINING_SCENES = 8
EXPECTED_TEST_SCENES = 16
EXPECTED_TRAINING_LABEL_ROWS = 13_911
EXPECTED_SCORE_LABEL_ROWS = 15_079
EXPECTED_WEIGHT_DIRS = (
    "bigearthnet_s1",
    "bigearthnet_s2",
    "imagenet_cnn_fcmae_ft_in1k",
    "imagenet_vit_augreg_in1k",
    "sarmae",
    "satdino",
)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _load_json(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{description} must be a regular non-symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must contain a JSON object")
    return payload


def split_scope(splits_path: Path, *, production: bool = True) -> dict[str, tuple[str, ...]]:
    """Return the exact train, fixed-DEV8, and TEST scene identities."""

    payload = _load_json(splits_path, "frozen split")
    try:
        splits = payload["splits"]
        train = tuple(map(str, splits["train"]))
        dev = tuple(map(str, splits["dev"]))
        test = tuple(map(str, splits["test"]))
        eval_final = tuple(map(str, splits["eval_final"]))
    except (KeyError, TypeError) as exc:
        raise RuntimeError("frozen split structure is invalid") from exc
    groups = {"train": train, "dev": dev, "test": test, "eval_final": eval_final}
    if any(not scene_id for values in groups.values() for scene_id in values):
        raise RuntimeError("frozen split contains an empty scene ID")
    if any(len(values) != len(set(values)) for values in groups.values()):
        raise RuntimeError("frozen split contains duplicate scene IDs")
    owners: dict[str, str] = {}
    for name, values in groups.items():
        for scene_id in values:
            previous = owners.setdefault(scene_id, name)
            if previous != name:
                raise RuntimeError(
                    f"scene {scene_id!r} occurs in both {previous} and {name}"
                )
    if production and (
        len(train) != EXPECTED_TRAIN_SCENES
        or len(dev) != EXPECTED_DEV_SCENES
        or len(test) != EXPECTED_TEST_SCENES
    ):
        raise RuntimeError("frozen train/dev/TEST scene counts are not 111/23/16")
    if len(dev) < EXPECTED_DEV_TRAINING_SCENES:
        raise RuntimeError("frozen DEV split has fewer than eight scenes")
    return {
        "train": tuple(sorted(train)),
        "dev8": tuple(sorted(dev)[:EXPECTED_DEV_TRAINING_SCENES]),
        "test": tuple(sorted(test)),
        "eval_final": tuple(sorted(eval_final)),
    }


def filter_training_labels(
    source: Path,
    destination: Path,
    *,
    splits_path: Path,
    production: bool = True,
) -> dict[str, object]:
    """Write the deterministic TRAIN+DEV8 subset used before cohort freeze."""

    scope = split_scope(splits_path, production=production)
    allowed = set(scope["train"]) | set(scope["dev8"])
    forbidden = set(scope["test"]) | set(scope["eval_final"])
    raw = source.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    except UnicodeDecodeError as exc:
        raise RuntimeError("source train.csv is not valid UTF-8") from exc
    if not reader.fieldnames or "scene_id" not in reader.fieldnames:
        raise RuntimeError("source train.csv lacks scene_id")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=reader.fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    rows = 0
    observed: set[str] = set()
    for row in reader:
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id:
            raise RuntimeError("source train.csv contains an empty scene_id")
        if scene_id in allowed:
            writer.writerow(row)
            rows += 1
            observed.add(scene_id)
    if observed != allowed:
        raise RuntimeError(
            "source labels do not cover exact TRAIN+DEV8 scope: "
            f"missing={sorted(allowed - observed)[:8]}"
        )
    if observed & forbidden:
        raise RuntimeError("TEST/eval-final labels entered the training view")
    data = output.getvalue().encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(data)
    return training_labels_summary(
        destination,
        splits_path=splits_path,
        production=production,
        expected_rows=rows,
    )


def training_labels_summary(
    path: Path,
    *,
    splits_path: Path,
    production: bool = True,
    expected_rows: int | None = None,
) -> dict[str, object]:
    """Validate that a label artifact exposes exactly TRAIN+fixed-DEV8 rows."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("training-view labels must be a regular non-symlink")
    scope = split_scope(splits_path, production=production)
    allowed = set(scope["train"]) | set(scope["dev8"])
    forbidden = set(scope["test"]) | set(scope["eval_final"])
    raw = path.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    except UnicodeDecodeError as exc:
        raise RuntimeError("training-view labels are not valid UTF-8") from exc
    if not reader.fieldnames or "scene_id" not in reader.fieldnames:
        raise RuntimeError("training-view labels lack scene_id")
    rows = 0
    observed: set[str] = set()
    for row in reader:
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id:
            raise RuntimeError("training-view labels contain an empty scene_id")
        if scene_id not in allowed:
            raise RuntimeError(
                f"training-view labels contain forbidden scene {scene_id!r}"
            )
        observed.add(scene_id)
        rows += 1
    if observed != allowed:
        raise RuntimeError(
            "training-view labels do not cover exact TRAIN+DEV8 scope: "
            f"missing={sorted(allowed - observed)[:8]}"
        )
    if observed & forbidden:
        raise RuntimeError("TEST/eval-final labels entered the training view")
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError("training-view label row count changed during validation")
    if production and rows != EXPECTED_TRAINING_LABEL_ROWS:
        raise RuntimeError("training-view labels do not contain exactly 13,911 rows")
    scene_ids = sorted(observed)
    return {
        "artifact_path": TRAINING_LABELS_ARTIFACT_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "row_count": rows,
        "scene_count": len(scene_ids),
        "scene_ids": scene_ids,
        "scene_ids_sha256": scene_ids_sha256(scene_ids),
        "train_scene_ids_sha256": scene_ids_sha256(scope["train"]),
        "dev8_scene_ids": list(scope["dev8"]),
        "dev8_scene_ids_sha256": scene_ids_sha256(scope["dev8"]),
        "forbidden_test_scene_ids_sha256": scene_ids_sha256(scope["test"]),
        "contract": TRAINING_VIEW_CONTRACT,
    }


def filter_score_labels(
    source: Path,
    destination: Path,
    *,
    splits_path: Path,
    production: bool = True,
) -> dict[str, object]:
    """After cohort validation, publish only TRAIN+TEST rows to scorers."""

    scope = split_scope(splits_path, production=production)
    allowed = set(scope["train"]) | set(scope["test"])
    raw = source.read_bytes()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    except UnicodeDecodeError as exc:
        raise RuntimeError("source train.csv is not valid UTF-8") from exc
    if not reader.fieldnames or "scene_id" not in reader.fieldnames:
        raise RuntimeError("source train.csv lacks scene_id")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=reader.fieldnames, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in reader:
        if str(row.get("scene_id", "")).strip() in allowed:
            writer.writerow(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(output.getvalue().encode("utf-8"))
    return score_labels_summary(
        destination, splits_path=splits_path, production=production
    )


def score_labels_summary(
    path: Path,
    *,
    splits_path: Path,
    production: bool = True,
) -> dict[str, object]:
    """Require exact TRAIN+TEST rows in the post-cohort scorer view."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("score-view labels must be a regular non-symlink")
    scope = split_scope(splits_path, production=production)
    allowed = set(scope["train"]) | set(scope["test"])
    forbidden = set(scope["dev8"]) | set(scope["eval_final"])
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if not reader.fieldnames or "scene_id" not in reader.fieldnames:
        raise RuntimeError("score-view labels lack scene_id")
    rows = 0
    observed: set[str] = set()
    for row in reader:
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id or scene_id not in allowed:
            raise RuntimeError(f"score-view labels contain forbidden scene {scene_id!r}")
        observed.add(scene_id)
        rows += 1
    if observed != allowed or observed & forbidden:
        raise RuntimeError("score-view labels do not cover exact TRAIN+TEST scope")
    if production and rows != EXPECTED_SCORE_LABEL_ROWS:
        raise RuntimeError("score-view labels do not contain exactly 15,079 rows")
    scene_ids = sorted(observed)
    return {
        "exposed_path": TRAINING_LABELS_EXPOSED_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "row_count": rows,
        "scene_count": len(scene_ids),
        "scene_ids_sha256": scene_ids_sha256(scene_ids),
        "train_scene_ids_sha256": scene_ids_sha256(scope["train"]),
        "test_scene_ids_sha256": scene_ids_sha256(scope["test"]),
        "contract": SCORE_VIEW_CONTRACT,
    }


def _artifact(manifest: Mapping[str, object], *, kind: str, name: str) -> dict:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("package manifest lacks artifacts")
    matches = [
        item
        for item in artifacts
        if isinstance(item, dict) and item.get("kind") == kind and item.get("name") == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"package requires exactly one {kind}/{name} artifact")
    return matches[0]


def _extract_artifact(package_root: Path, artifact: Mapping[str, object], destination: Path) -> None:
    try:
        parts = _artifact_part_paths(package_root, artifact)
        expected_bytes = int(artifact["archive_bytes"])
        expected_sha256 = str(artifact["archive_sha256"])
        extraction = PurePosixPath(str(artifact["extraction_root"]))
    except (KeyError, TypeError, ValueError, PackageError) as exc:
        raise RuntimeError("package artifact record is invalid") from exc
    if sum(path.stat().st_size for path in parts) != expected_bytes:
        raise RuntimeError("package artifact physical size mismatch")
    if _hash_paths(parts, "sha256") != expected_sha256:
        raise RuntimeError("package artifact SHA-256 mismatch")
    if artifact.get("format") == "tar.zst":
        _extract_tar_zst(
            parts,
            destination,
            extraction_root=extraction,
            expected_file_count=int(artifact["file_count"]),
            expected_unpacked_bytes=int(artifact["unpacked_bytes"]),
        )
        return
    if artifact.get("format") != "file" or int(artifact.get("file_count", 0)) != 1:
        raise RuntimeError("unsupported staged artifact format")
    output = _safe_output_path(destination, extraction.as_posix())
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"artifact extraction would overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)


def _runtime_labels_to_exposed(
    *, runtime_root: Path, runtime_manifest: Mapping[str, object], staging: Path, repo: Path
) -> dict[str, object]:
    artifact = _artifact(runtime_manifest, kind="training_labels", name="train-dev8.csv")
    _extract_artifact(runtime_root, artifact, staging)
    private_path = staging / TRAINING_LABELS_ARTIFACT_PATH
    summary = training_labels_summary(
        private_path,
        splits_path=repo / "data/splits.json",
        production=True,
    )
    expected = runtime_manifest.get("training_view")
    if not isinstance(expected, Mapping) or expected.get("labels") != summary:
        raise RuntimeError("runtime training-label artifact binding mismatch")
    exposed = staging / TRAINING_LABELS_EXPOSED_PATH
    exposed.parent.mkdir(parents=True, exist_ok=True)
    os.replace(private_path, exposed)
    private_parent = staging / "data/training-view"
    shutil.rmtree(private_parent)
    return summary


def _cohort_binding(
    *, repo: Path, runs_root: Path, expected_git_sha: str
) -> tuple[str, dict[str, str] | None]:
    import yaml

    cohort_path = runs_root / ".h100" / COHORT_FILENAME
    if cohort_path.exists() or cohort_path.is_symlink():
        detector = yaml.safe_load((repo / "configs/detector.yaml").read_text())
        cohort, digest = validate_training_cohort(
            path=cohort_path,
            cells=load_cells(repo),
            runs_root=runs_root,
            git_sha=expected_git_sha,
            detector_sha256=sha256_file(repo / "configs/detector.yaml"),
            candidate_floor=float(detector["decode"]["candidate_floor"]),
        )
        if cohort.get("cell_count") != 32:
            raise RuntimeError("canonical training cohort does not contain 32 cells")
        return "score-test", {"path": str(cohort_path.absolute()), "sha256": digest}
    leaked = [
        cell.exp_id
        for cell in load_cells(repo)
        if (runs_root / cell.exp_id / "test_metrics.json").exists()
        or (runs_root / cell.exp_id / "test_metrics.json").is_symlink()
    ]
    if leaked:
        raise RuntimeError("TEST result exists without the all-32 training cohort")
    return "train", None


def replacement_data_binding(control: Mapping[str, object]) -> dict[str, object]:
    """Return the pathless code-to-accepted-data binding used by new receipts."""

    try:
        binding = {
            "accepted_h100_ready_sha256": control["accepted_h100_ready_sha256"],
            "accepted_data_view_sha256": control["accepted_data_view"]["sha256"],
            "data_runtime_package": dict(control["accepted_data_runtime"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("replacement data binding is incomplete") from exc
    _validate_replacement_binding(binding)
    return binding


def _validate_replacement_binding(binding: Mapping[str, object]) -> None:
    if not isinstance(binding, Mapping):
        raise RuntimeError("replacement data binding schema is invalid")
    runtime = binding.get("data_runtime_package")
    runtime_keys = {
        "package_id", "git_sha", "manifest_sha256", "ready_sha256",
        "sha256sums_sha256", "runtime_bundle_sha256",
    }
    if (
        set(binding) != {
            "accepted_h100_ready_sha256", "accepted_data_view_sha256",
            "data_runtime_package",
        }
        or not isinstance(runtime, Mapping)
        or set(runtime) != runtime_keys
        or not isinstance(runtime.get("package_id"), str)
        or not runtime["package_id"]
    ):
        raise RuntimeError("replacement data binding schema is invalid")
    digests = [
        binding["accepted_h100_ready_sha256"], binding["accepted_data_view_sha256"],
        *(runtime[key] for key in runtime_keys - {"package_id", "git_sha"}),
    ]
    hashes = [(value, 64) for value in digests] + [(runtime["git_sha"], 40)]
    if any(
        not isinstance(value, str) or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
        for value, length in hashes
    ):
        raise RuntimeError("replacement data binding hashes are invalid")


def _validate_receipt_version(
    receipt: Mapping[str, object], expected: Mapping[str, object] | None
) -> None:
    if expected is None:
        valid = receipt.get("schema") == DATA_VIEW_SCHEMA and "replacement" not in receipt
    else:
        _validate_replacement_binding(expected)
        valid = (
            receipt.get("schema") == REPLACEMENT_DATA_VIEW_SCHEMA
            and receipt.get("replacement") == dict(expected)
        )
    if not valid:
        raise RuntimeError("H100 data-view replacement binding mismatch")


def _canonical_replacement_root(path: Path, description: str) -> Path:
    if (
        not path.is_absolute() or path.absolute() != path.resolve()
        or any(item.is_symlink() for item in (path, *path.parents))
        or (path.exists() and not path.is_dir())
    ):
        raise RuntimeError(f"{description} must be a canonical non-symlink directory")
    return path.absolute()


def validate_replacement_namespaces(
    *, runs_root: Path, original_runs_root: Path,
) -> tuple[Path, Path]:
    """Validate isolation before creating state, submitting, or looking up a cohort.

    The new root may be absent before first submission. The canonical original
    root must exist; callers separately prove its accepted H100_READY digest.
    This function reads no original campaign marker or held-out material.
    """

    new_runs = _canonical_replacement_root(runs_root, "replacement runs root")
    original = _canonical_replacement_root(original_runs_root, "original runs root")
    if not original.is_dir():
        raise RuntimeError("original runs root must be an existing directory")
    if new_runs.is_relative_to(original) or original.is_relative_to(new_runs):
        raise RuntimeError("replacement and original runs roots must be disjoint")
    _canonical_replacement_root(new_runs / ".h100", "replacement control root")
    if (new_runs / ".h100" / COHORT_FILENAME).is_symlink():
        raise RuntimeError("replacement cohort cannot be a symlink")
    return new_runs, original


def _replacement_data_source(
    *, repo: Path, runs_root: Path, original_runs_root: Path,
    runtime: Mapping[str, object], control: Mapping[str, object],
    data_runtime_root: Path, expected_base: Mapping[str, str],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    from scripts.handoff.replacement_amendment import validate_replacement_control

    control = validate_replacement_control(control)
    _new_runs, original = validate_replacement_namespaces(
        runs_root=runs_root, original_runs_root=original_runs_root,
    )
    old_root = _canonical_replacement_root(data_runtime_root, "accepted data-runtime root")
    ready_path = original / ".h100/H100_READY.json"
    if any(item.is_symlink() for item in (ready_path, *ready_path.parents)):
        raise RuntimeError("original H100_READY must have no symlink components")
    ready = _load_json(ready_path, "original H100_READY")
    binding = replacement_data_binding(control)
    if sha256_file(ready_path) != binding["accepted_h100_ready_sha256"]:
        raise RuntimeError("original runs root does not bind accepted H100_READY")
    if ready.get("evaluation_ground_truth", {}).get("path") != str(original / ".h100/EVAL_GROUND_TRUTH_VALIDATED.json"):
        raise RuntimeError("accepted H100_READY belongs to a different original namespace")
    if runtime.get("amendment_identity") != control or runtime.get("source") != control["source"]:
        raise RuntimeError("replacement runtime manifest/control binding mismatch")
    old_identity = control["accepted_data_runtime"]
    accepted_view = control["accepted_data_view"]
    old_view = ready.get("venv", {}).get("staged_data_view", {})
    if (
        ready.get("runtime_amendment") != old_identity
        or ready.get("base_payload") != control["accepted_h100"]["base_payload"]
        or old_view.get("sha256") != accepted_view["sha256"]
        or old_view.get("receipt") != accepted_view["receipt"]
        or ready.get("evaluation_ground_truth", {}).get("sha256")
        != control["evaluation_ground_truth"]["sha256"]
        or ready.get("evaluation_ground_truth", {}).get("receipt")
        != control["evaluation_ground_truth"]["receipt"]
    ):
        raise RuntimeError("accepted H100 data/runtime controls differ from original")
    accepted = validate_acceptance_data_view_binding(
        old_view, repo=repo, expected_git_sha=old_identity["git_sha"],
        expected_base_package_id=expected_base["package_id"],
        expected_base_manifest_sha256=expected_base["manifest_sha256"],
        expected_runtime_package_id=old_identity["package_id"],
        expected_runtime_manifest_sha256=old_identity["manifest_sha256"],
    )
    for relative, key in (
        ("manifest.json", "manifest_sha256"), ("READY.json", "ready_sha256"),
        ("SHA256SUMS", "sha256sums_sha256"),
    ):
        path = old_root / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != old_identity[key]:
            raise RuntimeError(f"accepted data-runtime {relative} binding mismatch")
    old_runtime = _load_json(old_root / "manifest.json", "accepted data-runtime manifest")
    if (
        old_runtime.get("package_id") != old_identity["package_id"]
        or old_runtime.get("source", {}).get("git_commit") != old_identity["git_sha"]
        or old_runtime.get("source", {}).get("git_bundle_sha256")
        != old_identity["runtime_bundle_sha256"]
        or old_runtime.get("evaluation_ground_truth") != control["evaluation_ground_truth"]["receipt"]
    ):
        raise RuntimeError("accepted data-runtime source/audit identity mismatch")
    return old_runtime, binding, accepted


def stage_data_view(
    *,
    repo: Path,
    runs_root: Path,
    base_package_root: Path,
    runtime_package_root: Path,
    destination: Path,
    expected_git_sha: str,
    expected_base_package_id: str,
    expected_base_manifest_sha256: str,
    expected_runtime_package_id: str,
    expected_runtime_manifest_sha256: str,
    acceptance: bool = False,
    replacement_control: Mapping[str, object] | None = None,
    data_runtime_package_root: Path | None = None,
    original_runs_root: Path | None = None,
) -> dict[str, object]:
    """Atomically publish the only data view visible to one allocation."""

    requested_runs_root = runs_root
    repo = repo.resolve()
    runs_root = runs_root.resolve()
    base_root = base_package_root.resolve()
    runtime_root = runtime_package_root.resolve()
    destination = destination.absolute()
    if os.path.lexists(destination):
        raise RuntimeError(f"data-view destination must start absent: {destination}")
    if sha256_file(base_root / "manifest.json") != expected_base_manifest_sha256:
        raise RuntimeError("base-package manifest SHA-256 mismatch")
    if sha256_file(runtime_root / "manifest.json") != expected_runtime_manifest_sha256:
        raise RuntimeError("runtime-package manifest SHA-256 mismatch")
    base = _load_json(base_root / "manifest.json", "base-package manifest")
    runtime = _load_json(runtime_root / "manifest.json", "runtime-package manifest")
    if base.get("package_id") != expected_base_package_id:
        raise RuntimeError("base-package ID mismatch")
    if runtime.get("package_id") != expected_runtime_package_id:
        raise RuntimeError("runtime-package ID mismatch")
    source = runtime.get("source")
    if not isinstance(source, Mapping) or source.get("git_commit") != expected_git_sha:
        raise RuntimeError("runtime package Git identity mismatch")
    replacement_args = (replacement_control, data_runtime_package_root, original_runs_root)
    replacement = None
    accepted_view = None
    data_runtime, data_runtime_root = runtime, runtime_root
    if any(value is not None for value in replacement_args):
        if any(value is None for value in replacement_args):
            raise RuntimeError("replacement staging requires control, data runtime, and original runs root")
        data_runtime, replacement, accepted_view = _replacement_data_source(
            repo=repo, runs_root=requested_runs_root, original_runs_root=original_runs_root,
            runtime=runtime, control=replacement_control,
            data_runtime_root=data_runtime_package_root,
            expected_base={
                "package_id": expected_base_package_id,
                "manifest_sha256": expected_base_manifest_sha256,
            },
        )
        data_runtime_root = data_runtime_package_root
    elif isinstance(runtime.get("amendment_identity"), Mapping) and (
        "accepted_data_runtime" in runtime["amendment_identity"]
    ):
        raise RuntimeError("replacement runtime requires explicit accepted data bindings")
    scope = split_scope(repo / "data/splits.json", production=True)
    phase, cohort = _cohort_binding(
        repo=repo, runs_root=runs_root, expected_git_sha=expected_git_sha
    )
    if acceptance and phase != "train":
        raise RuntimeError("H100 acceptance requires an absent training cohort")

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    selected: list[dict[str, object]] = []
    try:
        # TEST scoring is whole-scene inference and has no reason to expose
        # held-out chip archives, even after the cohort barrier.
        wanted_chips = set(scope["train"] if phase == "train" else ())
        wanted_rasters = set(scope["dev8"] if phase == "train" else scope["test"])
        artifacts = base.get("artifacts")
        if not isinstance(artifacts, list):
            raise RuntimeError("base-package manifest lacks artifacts")
        for raw in artifacts:
            if not isinstance(raw, dict):
                raise RuntimeError("base-package artifact is not an object")
            kind = raw.get("kind")
            name = str(raw.get("name", ""))
            wanted = (
                (kind == "chip_scene" and name in wanted_chips)
                or (kind == "raster_scene" and name in wanted_rasters)
                or kind == "core_weight"
                or (phase == "train" and acceptance and kind == "offline_environment")
                or (phase == "score-test" and kind == "labels")
            )
            if wanted:
                _extract_artifact(base_root, raw, staging)
                selected.append(
                    {
                        "kind": kind,
                        "name": name,
                        "archive_sha256": raw.get("archive_sha256"),
                        "extraction_root": raw.get("extraction_root"),
                    }
                )
        if phase == "train":
            labels = _runtime_labels_to_exposed(
                runtime_root=data_runtime_root,
                runtime_manifest=data_runtime,
                staging=staging,
                repo=repo,
            )
            label_source = "runtime-train-dev8-artifact"
        else:
            labels_artifact = _artifact(base, kind="labels", name="train.csv")
            labels_path = staging / TRAINING_LABELS_EXPOSED_PATH
            if not labels_path.is_file():
                raise RuntimeError("score-test view lacks combined labels")
            source_sha256 = sha256_file(labels_path)
            source_bytes = labels_path.stat().st_size
            source_audit = data_runtime.get("evaluation_ground_truth")
            if not isinstance(source_audit, Mapping):
                raise RuntimeError("runtime package lacks source GT audit metadata")
            validate_ground_truth_audit_receipt(
                source_audit,
                splits_json=repo / "data/splits.json",
                expected_train_csv_sha256=source_sha256,
                expected_train_csv_bytes=source_bytes,
            )
            recomputed_audit = audit_ground_truth_dataset(
                train_csv=labels_path,
                splits_json=repo / "data/splits.json",
            )
            if recomputed_audit != source_audit:
                raise RuntimeError("post-cohort combined-label audit differs from source")
            filtered = labels_path.with_name(".train-test.filtered.csv")
            labels = filter_score_labels(
                labels_path,
                filtered,
                splits_path=repo / "data/splits.json",
                production=True,
            )
            os.replace(filtered, labels_path)
            labels.update(
                {
                    "source_combined_sha256": source_sha256,
                    "source_combined_bytes": source_bytes,
                    "source_archive_sha256": labels_artifact.get("archive_sha256"),
                    "source_audit_sha256": hashlib.sha256(
                        _canonical_json(source_audit)
                    ).hexdigest(),
                }
            )
            label_source = "post-cohort-audited-base-train-test-filter"

        for required in ("data/chips", "data/raw/xview3/GRD", "data/weights"):
            (staging / required).mkdir(parents=True, exist_ok=True)
        exposed_chips = sorted(
            path.name for path in (staging / "data/chips").iterdir() if path.is_dir()
        )
        exposed_rasters = sorted(
            path.name
            for path in (staging / "data/raw/xview3/GRD").iterdir()
            if path.is_dir()
        )
        if exposed_chips != sorted(wanted_chips) or exposed_rasters != sorted(wanted_rasters):
            raise RuntimeError("staged scene inventory differs from the phase allowlist")
        if phase == "train" and (
            set(exposed_chips) & set(scope["test"])
            or set(exposed_rasters) & set(scope["test"])
        ):
            raise RuntimeError("TEST scene leaked into the training data view")
        weight_root = staging / "data/weights"
        weight_entries = list(weight_root.iterdir())
        exposed_weights = sorted(
            path.name
            for path in weight_entries
            if path.is_dir() and not path.is_symlink()
        )
        if exposed_weights != list(EXPECTED_WEIGHT_DIRS) or any(
            not path.is_dir() or path.is_symlink() for path in weight_entries
        ):
            raise RuntimeError("staged weight inventory is not the exact six directories")
        receipt_labels = {
            "source": label_source,
            "exposed_path": TRAINING_LABELS_EXPOSED_PATH,
            **labels,
        }
        if accepted_view is not None and phase == "train":
            accepted_artifacts = [
                item for item in accepted_view["selected_artifacts"]
                if acceptance or item["kind"] != "offline_environment"
            ]
            if receipt_labels != accepted_view["labels"] or selected != accepted_artifacts:
                raise RuntimeError("replacement training view differs from accepted data identities")
        receipt: dict[str, object] = {
            "schema": DATA_VIEW_SCHEMA if replacement is None else REPLACEMENT_DATA_VIEW_SCHEMA,
            "status": "ready",
            "phase": phase,
            "purpose": "acceptance" if acceptance else "campaign",
            "contract": TRAINING_VIEW_CONTRACT if phase == "train" else SCORE_VIEW_CONTRACT,
            "git_sha": expected_git_sha,
            "base_package": {
                "package_id": expected_base_package_id,
                "manifest_sha256": expected_base_manifest_sha256,
            },
            "runtime_package": {
                "package_id": expected_runtime_package_id,
                "manifest_sha256": expected_runtime_manifest_sha256,
            },
            "training_cohort": cohort,
            "labels": receipt_labels,
            "chips": exposed_chips,
            "rasters": exposed_rasters,
            "weights": exposed_weights,
            "selected_artifacts": selected,
        }
        if replacement is not None:
            receipt["replacement"] = replacement
        receipt_path = staging / DATA_VIEW_RECEIPT
        receipt_path.write_bytes(_canonical_json(receipt))
        receipt_path.chmod(0o444)
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def validate_acceptance_data_view_binding(
    binding: Mapping[str, object],
    *,
    repo: Path,
    expected_git_sha: str,
    expected_base_package_id: str,
    expected_base_manifest_sha256: str,
    expected_runtime_package_id: str,
    expected_runtime_manifest_sha256: str,
    expected_replacement_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the immutable acceptance-view binding after scratch is gone.

    Acceptance physically validates the allocation-private view before writing
    H100_READY.json. Later control-plane and result-packaging steps cannot
    reopen that reconstructible scratch tree, so they validate this embedded
    canonical receipt instead. Keep this pure: it must not require the staged
    path to remain present after allocation cleanup.
    """

    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "sha256",
        "receipt",
    }:
        raise RuntimeError("H100 acceptance data-view binding is invalid")
    raw_path = binding.get("path")
    digest = binding.get("sha256")
    receipt = binding.get("receipt")
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not isinstance(receipt, Mapping)
        or hashlib.sha256(_canonical_json(receipt)).hexdigest() != digest
    ):
        raise RuntimeError("H100 acceptance data-view binding is invalid")

    scope = split_scope(repo / "data/splits.json", production=True)
    expected_chips = list(scope["train"])
    expected_rasters = list(scope["dev8"])
    expected_label_scenes = sorted((*scope["train"], *scope["dev8"]))
    expected_base = {
        "package_id": expected_base_package_id,
        "manifest_sha256": expected_base_manifest_sha256,
    }
    expected_runtime = {
        "package_id": expected_runtime_package_id,
        "manifest_sha256": expected_runtime_manifest_sha256,
    }
    expected_receipt_keys = {
        "schema",
        "status",
        "phase",
        "purpose",
        "contract",
        "git_sha",
        "base_package",
        "runtime_package",
        "training_cohort",
        "labels",
        "chips",
        "rasters",
        "weights",
        "selected_artifacts",
    }
    _validate_receipt_version(receipt, expected_replacement_binding)
    if expected_replacement_binding is not None:
        expected_receipt_keys.add("replacement")
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("status") != "ready"
        or receipt.get("phase") != "train"
        or receipt.get("purpose") != "acceptance"
        or receipt.get("contract") != TRAINING_VIEW_CONTRACT
        or receipt.get("git_sha") != expected_git_sha
        or receipt.get("base_package") != expected_base
        or receipt.get("runtime_package") != expected_runtime
        or receipt.get("training_cohort") is not None
        or receipt.get("chips") != expected_chips
        or receipt.get("rasters") != expected_rasters
        or receipt.get("weights") != list(EXPECTED_WEIGHT_DIRS)
    ):
        raise RuntimeError("H100 acceptance data-view receipt binding mismatch")

    labels = receipt.get("labels")
    expected_label_keys = {
        "source",
        "exposed_path",
        "artifact_path",
        "sha256",
        "bytes",
        "row_count",
        "scene_count",
        "scene_ids",
        "scene_ids_sha256",
        "train_scene_ids_sha256",
        "dev8_scene_ids",
        "dev8_scene_ids_sha256",
        "forbidden_test_scene_ids_sha256",
        "contract",
    }
    label_sha256 = labels.get("sha256") if isinstance(labels, Mapping) else None
    label_bytes = labels.get("bytes") if isinstance(labels, Mapping) else None
    if (
        not isinstance(labels, Mapping)
        or set(labels) != expected_label_keys
        or labels.get("source") != "runtime-train-dev8-artifact"
        or labels.get("exposed_path") != TRAINING_LABELS_EXPOSED_PATH
        or labels.get("artifact_path") != TRAINING_LABELS_ARTIFACT_PATH
        or not isinstance(label_sha256, str)
        or len(label_sha256) != 64
        or any(character not in "0123456789abcdef" for character in label_sha256)
        or not isinstance(label_bytes, int)
        or isinstance(label_bytes, bool)
        or label_bytes <= 0
        or labels.get("row_count") != EXPECTED_TRAINING_LABEL_ROWS
        or labels.get("scene_count") != len(expected_label_scenes)
        or labels.get("scene_ids") != expected_label_scenes
        or labels.get("scene_ids_sha256") != scene_ids_sha256(expected_label_scenes)
        or labels.get("train_scene_ids_sha256") != scene_ids_sha256(scope["train"])
        or labels.get("dev8_scene_ids") != expected_rasters
        or labels.get("dev8_scene_ids_sha256") != scene_ids_sha256(scope["dev8"])
        or labels.get("forbidden_test_scene_ids_sha256")
        != scene_ids_sha256(scope["test"])
        or labels.get("contract") != TRAINING_VIEW_CONTRACT
    ):
        raise RuntimeError("H100 acceptance data-view label binding mismatch")

    selected = receipt.get("selected_artifacts")
    if not isinstance(selected, list) or any(
        not isinstance(item, Mapping)
        or set(item) != {"kind", "name", "archive_sha256", "extraction_root"}
        or not isinstance(item.get("name"), str)
        or not item.get("name")
        or not isinstance(item.get("archive_sha256"), str)
        or len(str(item.get("archive_sha256"))) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(item.get("archive_sha256"))
        )
        or not isinstance(item.get("extraction_root"), str)
        or not item.get("extraction_root")
        for item in selected
    ):
        raise RuntimeError("H100 acceptance selected-artifact binding is invalid")
    selected_pairs = [(str(item["kind"]), str(item["name"])) for item in selected]
    expected_pairs = {
        *(("chip_scene", name) for name in expected_chips),
        *(("raster_scene", name) for name in expected_rasters),
        *(("core_weight", name) for name in EXPECTED_WEIGHT_DIRS),
    }
    offline = [pair for pair in selected_pairs if pair[0] == "offline_environment"]
    non_environment = {
        pair for pair in selected_pairs if pair[0] != "offline_environment"
    }
    if (
        len(selected_pairs) != len(set(selected_pairs))
        or non_environment != expected_pairs
        or len(offline) != 1
        or len(selected_pairs) != len(expected_pairs) + 1
    ):
        raise RuntimeError("H100 acceptance selected-artifact inventory mismatch")
    return dict(receipt)


def validate_data_view(
    path: Path,
    *,
    repo: Path,
    runs_root: Path,
    expected_sha256: str,
    expected_git_sha: str,
    expected_phase: str,
    expected_purpose: str,
    expected_base_package_id: str,
    expected_base_manifest_sha256: str,
    expected_runtime_package_id: str,
    expected_runtime_manifest_sha256: str,
    expected_replacement_binding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Revalidate a staged view without opening any excluded asset."""

    receipt_path = path / DATA_VIEW_RECEIPT
    if sha256_file(receipt_path) != expected_sha256:
        raise RuntimeError("H100 data-view receipt SHA-256 mismatch")
    receipt = _load_json(receipt_path, "H100 data-view receipt")
    _validate_receipt_version(receipt, expected_replacement_binding)
    if (
        receipt.get("status") != "ready"
        or receipt.get("phase") != expected_phase
        or receipt.get("purpose") != expected_purpose
        or receipt.get("git_sha") != expected_git_sha
        or receipt.get("base_package")
        != {
            "package_id": expected_base_package_id,
            "manifest_sha256": expected_base_manifest_sha256,
        }
        or receipt.get("runtime_package")
        != {
            "package_id": expected_runtime_package_id,
            "manifest_sha256": expected_runtime_manifest_sha256,
        }
    ):
        raise RuntimeError("H100 data-view receipt binding mismatch")
    environment = path / "environment"
    if expected_purpose == "acceptance":
        if expected_phase != "train" or not (environment / "wheelhouse").is_dir():
            raise RuntimeError("acceptance data view lacks the offline wheelhouse")
    elif expected_purpose == "campaign":
        if environment.exists() or environment.is_symlink():
            raise RuntimeError("campaign data view unexpectedly contains the wheelhouse")
    else:
        raise RuntimeError("unsupported H100 data-view purpose")
    scope = split_scope(repo / "data/splits.json", production=True)
    expected_chips = sorted(scope["train"] if expected_phase == "train" else ())
    expected_rasters = sorted(scope["dev8"] if expected_phase == "train" else scope["test"])
    actual_chips = sorted(
        item.name for item in (path / "data/chips").iterdir() if item.is_dir()
    )
    actual_rasters = sorted(
        item.name
        for item in (path / "data/raw/xview3/GRD").iterdir()
        if item.is_dir()
    )
    if receipt.get("chips") != expected_chips or actual_chips != expected_chips:
        raise RuntimeError("H100 data-view chip inventory mismatch")
    if receipt.get("rasters") != expected_rasters or actual_rasters != expected_rasters:
        raise RuntimeError("H100 data-view raster inventory mismatch")
    if expected_phase == "train":
        summary = training_labels_summary(
            path / TRAINING_LABELS_EXPOSED_PATH,
            splits_path=repo / "data/splits.json",
            production=True,
        )
        labels = receipt.get("labels")
        if not isinstance(labels, Mapping) or labels != {
            "source": "runtime-train-dev8-artifact",
            "exposed_path": TRAINING_LABELS_EXPOSED_PATH,
            **summary,
        }:
            raise RuntimeError("H100 training-label view binding mismatch")
        if receipt.get("training_cohort") is not None:
            raise RuntimeError("training data view unexpectedly binds a cohort")
    else:
        phase, cohort = _cohort_binding(
            repo=repo, runs_root=runs_root, expected_git_sha=expected_git_sha
        )
        if phase != "score-test" or receipt.get("training_cohort") != cohort:
            raise RuntimeError("score-test data view lacks the canonical cohort binding")
        labels = receipt.get("labels")
        if not isinstance(labels, Mapping):
            raise RuntimeError("score-test data view lacks label binding")
        summary = score_labels_summary(
            path / TRAINING_LABELS_EXPOSED_PATH,
            splits_path=repo / "data/splits.json",
            production=True,
        )
        if (
            labels.get("source") != "post-cohort-audited-base-train-test-filter"
            or labels.get("exposed_path") != TRAINING_LABELS_EXPOSED_PATH
            or any(labels.get(key) != value for key, value in summary.items())
            or not isinstance(labels.get("source_combined_sha256"), str)
            or labels.get("source_combined_bytes") != 11_134_981
            or not isinstance(labels.get("source_archive_sha256"), str)
            or not isinstance(labels.get("source_audit_sha256"), str)
        ):
            raise RuntimeError("score-test data-view label binding mismatch")
    weight_root = path / "data/weights"
    entries = list(weight_root.iterdir())
    actual_weights = sorted(
        item.name for item in entries if item.is_dir() and not item.is_symlink()
    )
    if (
        receipt.get("weights") != list(EXPECTED_WEIGHT_DIRS)
        or actual_weights != list(EXPECTED_WEIGHT_DIRS)
        or any(not item.is_dir() or item.is_symlink() for item in entries)
    ):
        raise RuntimeError("H100 data-view weight inventory mismatch")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--base-package-root", type=Path, required=True)
    parser.add_argument("--runtime-package-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--expected-base-package-id", required=True)
    parser.add_argument("--expected-base-manifest-sha256", required=True)
    parser.add_argument("--expected-runtime-package-id", required=True)
    parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--replacement-control", type=Path)
    parser.add_argument("--data-runtime-package-root", type=Path)
    parser.add_argument("--original-runs-root", type=Path)
    args = parser.parse_args(argv)
    receipt = stage_data_view(
        repo=args.repo,
        runs_root=args.runs_root,
        base_package_root=args.base_package_root,
        runtime_package_root=args.runtime_package_root,
        destination=args.destination,
        expected_git_sha=args.expected_git_sha,
        expected_base_package_id=args.expected_base_package_id,
        expected_base_manifest_sha256=args.expected_base_manifest_sha256,
        expected_runtime_package_id=args.expected_runtime_package_id,
        expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
        acceptance=args.acceptance,
        replacement_control=(
            _load_json(args.replacement_control, "replacement amendment control")
            if args.replacement_control is not None else None
        ),
        data_runtime_package_root=args.data_runtime_package_root,
        original_runs_root=args.original_runs_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
