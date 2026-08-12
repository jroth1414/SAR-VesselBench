"""Validate and render the public H100 campaign-result snapshot.

The public paper consumes one small, machine-readable snapshot rather than a
training directory.  This module deliberately treats operator status output as
status-only evidence: a development metric becomes reportable only after its
completion marker, runtime provenance, and best/last checkpoints are all
SHA-256 bound.  That rule prevents an in-progress value or a V100 diagnostic
from silently entering a table or curve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SNAPSHOT_SCHEMA = 1
METRIC_SCOPE = "corrected-development-selection"
PROFILES = frozenset({"deadline", "complete"})
STATUSES = frozenset({"DONE", "STARTED", "PENDING"})
VERIFICATIONS = frozenset({"status-only", "contract-verified"})
SOURCE_KINDS = frozenset({"operator-status-screenshot", "verified-reverse-result-package"})
FROZEN_PATHS = (
    "configs/detector.yaml",
    "src/eval/scorer.py",
    "data/splits.json",
    "data/stats.json",
    "data/lsssdd_split.json",
)
STRICT_BACKEND = {
    "cuda_matmul_fp32_precision": "ieee",
    "cudnn_conv_fp32_precision": "ieee",
    "cudnn_rnn_fp32_precision": "ieee",
}
FRACTION_NAMES = {10: "Ten", 25: "TwentyFive", 50: "Fifty", 100: "Hundred"}
FRACTION_SCENES = {10: 12, 25: 28, 50: 56, 100: 111}
TRACK_NAMES = {"vit": "ViT", "cnn": "CNN"}
ROLE_NAMES = {
    "floor": "Random",
    "optical": "Optical",
    "sar": "SAR",
    "imagenet": "ImageNet",
}
ROLE_MACROS = {
    "floor": "Random",
    "optical": "Optical",
    "sar": "Sar",
    "imagenet": "ImageNet",
}
ROLE_STYLES = {
    "floor": ("#555555", "x", "--"),
    "optical": ("#1baf7a", "^", "-"),
    "sar": ("#eb6834", "s", "-"),
    "imagenet": ("#2a78d6", "o", "-"),
}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class SnapshotError(ValueError):
    """The H100 snapshot is incomplete, mixed, or not provenance-bound."""


@dataclass(frozen=True)
class Cell:
    exp_id: str
    init: str
    track: str
    role: str
    label_fraction: int
    seed: int
    status: str
    latest_epoch: int | None
    duration_seconds: int | None
    train_loss: float | None
    development: dict[str, float] | None
    evidence: dict[str, str]

    @property
    def verified(self) -> bool:
        return self.evidence["verification"] == "contract-verified"


@dataclass(frozen=True)
class Snapshot:
    payload: dict[str, Any]
    cells: tuple[Cell, ...]
    reportable_fractions: tuple[int, ...]

    @property
    def status_counts(self) -> Counter[str]:
        return Counter(cell.status for cell in self.cells)

    @property
    def verified_count(self) -> int:
        return sum(cell.verified for cell in self.cells)


def _closed(value: object, keys: set[str], description: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        observed = set(value) if isinstance(value, Mapping) else set()
        raise SnapshotError(
            f"{description} keys do not match the closed schema; "
            f"missing={sorted(keys - observed)}, unexpected={sorted(observed - keys)}"
        )
    return dict(value)


def _sha(value: object, description: str, *, length: int = 64) -> str:
    text = str(value)
    pattern = _HEX40 if length == 40 else _HEX64
    if pattern.fullmatch(text) is None:
        raise SnapshotError(f"{description} must be lowercase SHA-{length * 4}")
    return text


def _finite(
    value: object,
    description: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise SnapshotError(f"{description} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"{description} must be finite") from exc
    if not math.isfinite(result):
        raise SnapshotError(f"{description} must be finite")
    if minimum is not None and result < minimum:
        raise SnapshotError(f"{description} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise SnapshotError(f"{description} must be <= {maximum}")
    return result


def _nonnegative_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SnapshotError(f"{description} must be a non-negative integer")
    return value


def _utc(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise SnapshotError(f"{description} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SnapshotError(f"{description} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SnapshotError(f"{description} must include a UTC offset")
    return value


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def expected_cells(arms_config: str | Path) -> dict[str, dict[str, Any]]:
    """Derive the exact seed-0 8x4 matrix from ``configs/arms.yaml``."""

    path = Path(arms_config)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        arms = config["arms"]
        fractions = [int(round(float(item) * 100)) for item in config["label_fracs"]]
        seeds = list(config["seeds"]["core"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise SnapshotError(f"cannot read the active arm matrix: {path}") from exc
    if set(fractions) != set(FRACTION_NAMES) or seeds != [0] or len(arms) != 8:
        raise SnapshotError("active arm manifest is not the exact seed-0 8x4 matrix")
    result: dict[str, dict[str, Any]] = {}
    for init, raw in arms.items():
        metadata = _closed(raw, {"short", "track", "role"}, f"arm {init}")
        for fraction in fractions:
            exp_id = f"{metadata['short']}-f{fraction}-s0"
            result[exp_id] = {
                "exp_id": exp_id,
                "init": str(init),
                "track": str(metadata["track"]),
                "role": str(metadata["role"]),
                "label_fraction": fraction,
                "seed": 0,
            }
    if len(result) != 32:
        raise SnapshotError("active arm manifest does not resolve to 32 unique cells")
    return result


def _validate_hardware(value: object) -> dict[str, Any]:
    hardware = _closed(
        value,
        {
            "accelerator",
            "compute_capability",
            "python",
            "torch",
            "cuda",
            "class_sha256",
        },
        "campaign.hardware",
    )
    accelerator = str(hardware["accelerator"])
    if "H100" not in accelerator.upper() or "V100" in accelerator.upper():
        raise SnapshotError("campaign hardware must be one H100 class, never V100")
    if hardware["compute_capability"] != [9, 0]:
        raise SnapshotError("campaign H100 compute capability must be [9, 0]")
    if str(hardware["python"]) != "3.11.13":
        raise SnapshotError("campaign Python must be the approved 3.11.13 runtime")
    if str(hardware["torch"]) != "2.11.0+cu126" or str(hardware["cuda"]) != "12.6":
        raise SnapshotError("campaign Torch/CUDA must be 2.11.0+cu126 / 12.6")
    identity = {key: hardware[key] for key in hardware if key != "class_sha256"}
    if _sha(hardware["class_sha256"], "hardware class") != _canonical_sha(identity):
        raise SnapshotError("campaign hardware class SHA-256 is inconsistent")
    return hardware


def _validate_development(value: object, exp_id: str) -> dict[str, float]:
    result = _closed(
        value,
        {"f1", "precision", "recall", "threshold"},
        f"{exp_id}.development",
    )
    return {
        key: _finite(result[key], f"{exp_id}.development.{key}", minimum=0, maximum=1)
        for key in ("f1", "precision", "recall", "threshold")
    }


def _validate_evidence(value: object, exp_id: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise SnapshotError(f"{exp_id}.evidence must be an object")
    verification = value.get("verification")
    if verification not in VERIFICATIONS:
        raise SnapshotError(f"{exp_id}.evidence verification is invalid")
    if verification == "status-only":
        return _closed(value, {"verification"}, f"{exp_id}.evidence")
    result = _closed(
        value,
        {
            "verification",
            "completion_marker_sha256",
            "runtime_provenance_sha256",
            "best_checkpoint_sha256",
            "last_checkpoint_sha256",
            "hardware_class_sha256",
        },
        f"{exp_id}.evidence",
    )
    for key in set(result) - {"verification"}:
        result[key] = _sha(result[key], f"{exp_id}.evidence.{key}")
    return result


def _validate_cell(
    value: object,
    expected: Mapping[str, Any],
    *,
    hardware_sha256: str,
) -> Cell:
    raw = _closed(
        value,
        {
            "exp_id",
            "init",
            "track",
            "role",
            "label_fraction",
            "seed",
            "status",
            "latest_epoch",
            "duration_seconds",
            "train_loss",
            "development",
            "evidence",
        },
        f"cell {expected['exp_id']}",
    )
    for key in ("exp_id", "init", "track", "role", "label_fraction", "seed"):
        if raw[key] != expected[key]:
            raise SnapshotError(f"{expected['exp_id']} metadata mismatch for {key}")
    status = str(raw["status"])
    if status not in STATUSES:
        raise SnapshotError(f"{expected['exp_id']} status is invalid")
    evidence = _validate_evidence(raw["evidence"], expected["exp_id"])
    verified = evidence["verification"] == "contract-verified"
    if verified and evidence["hardware_class_sha256"] != hardware_sha256:
        raise SnapshotError(f"{expected['exp_id']} is bound to a different hardware class")
    if not verified and raw["train_loss"] is not None:
        raise SnapshotError(
            f"{expected['exp_id']} train loss lacks contract-verified completion evidence"
        )

    if status == "PENDING":
        if any(
            raw[name] is not None
            for name in ("latest_epoch", "duration_seconds", "train_loss", "development")
        ) or verified:
            raise SnapshotError(f"{expected['exp_id']} pending cell contains result evidence")
        epoch = duration = None
        train_loss = development = None
    else:
        epoch = _nonnegative_int(raw["latest_epoch"], f"{expected['exp_id']}.latest_epoch")
        duration = _nonnegative_int(
            raw["duration_seconds"], f"{expected['exp_id']}.duration_seconds"
        )
        train_loss = (
            None
            if raw["train_loss"] is None
            else _finite(raw["train_loss"], f"{expected['exp_id']}.train_loss", minimum=0)
        )
        if status == "STARTED":
            if raw["development"] is not None or verified:
                raise SnapshotError(
                    f"{expected['exp_id']} running cell may not publish a development metric"
                )
            development = None
        elif verified:
            if train_loss is None:
                raise SnapshotError(f"{expected['exp_id']} verified train loss is absent")
            if duration <= 0:
                raise SnapshotError(f"{expected['exp_id']} verified duration must be positive")
            development = _validate_development(raw["development"], expected["exp_id"])
        else:
            if raw["development"] is not None:
                raise SnapshotError(
                    f"{expected['exp_id']} metric lacks contract-verified completion evidence"
                )
            development = None

    return Cell(
        **{key: expected[key] for key in ("exp_id", "init", "track", "role", "label_fraction", "seed")},
        status=status,
        latest_epoch=epoch,
        duration_seconds=duration,
        train_loss=train_loss,
        development=development,
        evidence=evidence,
    )


def load_snapshot(snapshot: str | Path, arms_config: str | Path) -> Snapshot:
    path = Path(snapshot)
    if path.is_symlink() or not path.is_file():
        raise SnapshotError(f"snapshot is not a regular non-symlink file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"snapshot is not valid JSON: {path}") from exc
    payload = _closed(
        raw,
        {
            "schema",
            "profile",
            "metric_scope",
            "captured_utc",
            "campaign",
            "source",
            "cells",
            "references",
        },
        "snapshot",
    )
    if payload["schema"] != SNAPSHOT_SCHEMA:
        raise SnapshotError(f"snapshot schema must be {SNAPSHOT_SCHEMA}")
    if payload["profile"] not in PROFILES:
        raise SnapshotError("snapshot profile must be deadline or complete")
    if payload["metric_scope"] != METRIC_SCOPE:
        raise SnapshotError("snapshot may report corrected development-selection only")
    _utc(payload["captured_utc"], "snapshot.captured_utc")

    campaign = _closed(
        payload["campaign"],
        {"campaign_id", "code_sha", "frozen_sha256", "hardware", "strict_fp32", "recipe"},
        "campaign",
    )
    if not str(campaign["campaign_id"]).strip():
        raise SnapshotError("campaign_id must be non-empty")
    _sha(campaign["code_sha"], "campaign.code_sha", length=40)
    frozen = _closed(campaign["frozen_sha256"], set(FROZEN_PATHS), "campaign.frozen_sha256")
    for relative in FROZEN_PATHS:
        _sha(frozen[relative], f"campaign.frozen_sha256[{relative}]")
    repository = Path(arms_config).resolve().parent.parent
    for relative in FROZEN_PATHS:
        artifact = repository / relative
        if artifact.is_symlink() or not artifact.is_file():
            raise SnapshotError(f"frozen artifact is absent or symlinked: {relative}")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != frozen[relative]:
            raise SnapshotError(
                f"campaign frozen SHA-256 differs from the public repository: {relative}"
            )
    hardware = _validate_hardware(campaign["hardware"])
    strict = _closed(
        campaign["strict_fp32"],
        {"precision", "tf32_enabled", "backend"},
        "campaign.strict_fp32",
    )
    if (
        strict["precision"] != "32-true"
        or strict["tf32_enabled"] is not False
        or strict["backend"] != STRICT_BACKEND
    ):
        raise SnapshotError("campaign is not strict IEEE FP32 with TF32 disabled")
    recipe = _closed(
        campaign["recipe"],
        {"micro_batch", "gradient_accumulation", "effective_batch", "seed"},
        "campaign.recipe",
    )
    if recipe != {
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "seed": 0,
    }:
        raise SnapshotError("campaign recipe is not the approved shared batch-16 seed-0 recipe")

    source = _closed(payload["source"], {"kind", "description", "sha256"}, "source")
    if source["kind"] not in SOURCE_KINDS or not str(source["description"]).strip():
        raise SnapshotError("snapshot source kind or description is invalid")
    _sha(source["sha256"], "source.sha256")

    expected = expected_cells(arms_config)
    raw_cells = payload["cells"]
    if not isinstance(raw_cells, list):
        raise SnapshotError("snapshot cells must be a list")
    ids = [item.get("exp_id") if isinstance(item, Mapping) else None for item in raw_cells]
    if len(ids) != 32 or len(set(ids)) != 32 or set(ids) != set(expected):
        raise SnapshotError("snapshot must contain exactly the 32 unique core cell IDs")
    cells = tuple(
        _validate_cell(
            item,
            expected[str(item["exp_id"])],
            hardware_sha256=str(hardware["class_sha256"]),
        )
        for item in raw_cells
    )
    if any(cell.verified for cell in cells) and source["kind"] != "verified-reverse-result-package":
        raise SnapshotError(
            "contract-verified cells require a verified reverse-result package source"
        )
    by_fraction = {
        fraction: [cell for cell in cells if cell.label_fraction == fraction]
        for fraction in FRACTION_NAMES
    }
    reportable = tuple(
        fraction
        for fraction in sorted(FRACTION_NAMES)
        if len(by_fraction[fraction]) == 8
        and all(cell.status == "DONE" and cell.verified for cell in by_fraction[fraction])
    )
    if payload["profile"] == "complete" and (
        len(reportable) != 4 or any(cell.status != "DONE" for cell in cells)
    ):
        raise SnapshotError("complete profile requires 32 contract-verified DONE cells")

    references = payload["references"]
    if not isinstance(references, list):
        raise SnapshotError("references must be a list")
    if references:
        reference_ids = []
        for item in references:
            record = _closed(
                item,
                {
                    "reference_id",
                    "exp_id",
                    "status",
                    "development_f1",
                    "completion_marker_sha256",
                    "runtime_provenance_sha256",
                },
                "reference",
            )
            reference_ids.append(record["reference_id"])
            if record["status"] != "DONE":
                raise SnapshotError("references may be included only after verified completion")
            _finite(record["development_f1"], "reference development_f1", minimum=0, maximum=1)
            _sha(record["completion_marker_sha256"], "reference completion marker")
            _sha(record["runtime_provenance_sha256"], "reference runtime provenance")
        if sorted(reference_ids) != ["R2", "R3"]:
            raise SnapshotError("references must be the exact separately validated R2/R3 pair")
    return Snapshot(payload=payload, cells=cells, reportable_fractions=reportable)


def _hash_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SnapshotError(f"result evidence is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def render_progress_log(snapshot: Snapshot) -> str:
    """Return a path-free, metric-free public campaign status transcript."""

    campaign = snapshot.payload["campaign"]
    counts = snapshot.status_counts
    lines = [
        "H100 CAMPAIGN STATUS (SANITIZED)",
        f"captured_utc={snapshot.payload['captured_utc']}",
        f"campaign_id={campaign['campaign_id']}",
        f"code_sha={campaign['code_sha']}",
        f"hardware={campaign['hardware']['accelerator']}",
        f"precision={campaign['strict_fp32']['precision']};tf32_enabled=false",
        f"summary=done:{counts['DONE']};started:{counts['STARTED']};pending:{counts['PENDING']}",
        f"contract_verified={snapshot.verified_count}",
        "",
        "exp_id|status|latest_epoch|duration_seconds|evidence",
    ]
    for cell in snapshot.cells:
        epoch = "-" if cell.latest_epoch is None else str(cell.latest_epoch)
        duration = "-" if cell.duration_seconds is None else str(cell.duration_seconds)
        lines.append(
            f"{cell.exp_id}|{cell.status}|{epoch}|{duration}|{cell.evidence['verification']}"
        )
    return "\n".join(lines) + "\n"


def _tex(value: object) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        text = text.replace(old, new)
    return text


def _macro(name: str, value: object) -> str:
    return rf"\def\{name}{{{value}}}"


def _cell_macro(cell: Cell) -> str:
    return (
        f"{TRACK_NAMES[cell.track]}{ROLE_MACROS[cell.role]}"
        f"{FRACTION_NAMES[cell.label_fraction]}"
    )


def _format_duration(seconds: int | None) -> str:
    if seconds is None:
        return r"\textemdash"
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes = remainder // 60
    if days:
        suffix = f"{days}d{hours:02d}h"
        return suffix if minutes == 0 else f"{suffix}{minutes:02d}m"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def _render_macros(snapshot: Snapshot) -> str:
    campaign = snapshot.payload["campaign"]
    hardware = campaign["hardware"]
    strict = campaign["strict_fp32"]
    counts = snapshot.status_counts
    fractions = ", ".join(f"f{value}" for value in snapshot.reportable_fractions) or "none"
    lines = [
        "% GENERATED by src.analysis.h100_results -- do not edit.",
        r"\newif\ifHOneHundredComplete",
        (
            r"\HOneHundredCompletetrue"
            if 100 in snapshot.reportable_fractions
            else r"\HOneHundredCompletefalse"
        ),
        r"\newif\ifHFullGridComplete",
        (
            r"\HFullGridCompletetrue"
            if snapshot.payload["profile"] == "complete"
            else r"\HFullGridCompletefalse"
        ),
        r"\newif\ifHHasReportableCohort",
        (
            r"\HHasReportableCohorttrue"
            if snapshot.reportable_fractions
            else r"\HHasReportableCohortfalse"
        ),
        _macro("HProfileLabel", _tex(snapshot.payload["profile"].title())),
        _macro("HDoneCount", counts["DONE"]),
        _macro("HStartedCount", counts["STARTED"]),
        _macro("HPendingCount", counts["PENDING"]),
        _macro("HCapturedUTC", _tex(snapshot.payload["captured_utc"])),
        _macro("HClosedFractions", _tex(fractions)),
        _macro("HHardware", _tex(hardware["accelerator"])),
        _macro("HCodeRevision", campaign["code_sha"]),
        _macro("HResultProfile", _tex(snapshot.payload["profile"])),
        _macro("HResultMetricScope", _tex(snapshot.payload["metric_scope"])),
        _macro("HResultCaptureUTC", _tex(snapshot.payload["captured_utc"])),
        _macro("HResultCampaignID", _tex(campaign["campaign_id"])),
        _macro("HResultCodeSHA", campaign["code_sha"]),
        _macro("HResultHardware", _tex(hardware["accelerator"])),
        _macro("HResultPythonVersion", _tex(hardware["python"])),
        _macro("HResultTorchVersion", _tex(hardware["torch"])),
        _macro("HResultCudaVersion", _tex(hardware["cuda"])),
        _macro("HResultPrecision", _tex(strict["precision"])),
        _macro("HResultDoneCount", counts["DONE"]),
        _macro("HResultStartedCount", counts["STARTED"]),
        _macro("HResultPendingCount", counts["PENDING"]),
        _macro("HResultVerifiedCount", snapshot.verified_count),
        _macro("HResultReportableFractionCount", len(snapshot.reportable_fractions)),
        _macro("HResultReportableFractions", _tex(fractions)),
        _macro(
            "HResultEvidenceScope",
            _tex("contract-verified completed cohorts only"),
        ),
    ]
    for cell in snapshot.cells:
        suffix = _cell_macro(cell)
        value = (
            rf"\ensuremath{{{cell.development['f1']:.3f}}}"
            if cell.development is not None and cell.verified
            else r"\textemdash"
        )
        lines.append(_macro(f"HDevF{suffix}", value))
        lines.append(_macro(f"HStatus{suffix}", _tex(cell.status.title())))
    return "\n".join(lines) + "\n"


def _render_core_table(snapshot: Snapshot) -> str:
    rows = []
    for cell in sorted(snapshot.cells, key=lambda item: (item.track, item.role, item.label_fraction)):
        f1 = f"{cell.development['f1']:.3f}" if cell.development is not None and cell.verified else r"\textemdash"
        evidence = "verified" if cell.verified else "status only"
        epoch = str(cell.latest_epoch) if cell.latest_epoch is not None else r"\textemdash"
        duration = _format_duration(cell.duration_seconds)
        rows.append(
            f"{TRACK_NAMES[cell.track]} & {ROLE_NAMES[cell.role]} & "
            f"{cell.label_fraction}\\% ({FRACTION_SCENES[cell.label_fraction]}) & "
            f"{cell.status.title()} & {evidence} & {epoch} & {duration} & {f1} \\\\"
        )
    return "\n".join(
        [
            "% GENERATED by src.analysis.h100_results -- do not edit.",
            r"\begin{table}[htbp]",
            r"\caption{H100 core-grid status. Development F1 is shown only for contract-verified completions; dashes are intentionally non-reportable.}",
            r"\label{tab:h100-core-status}",
            r"\centering\scriptsize",
            r"\begin{tabular}{@{}llllrrrr@{}}",
            r"\toprule",
            r"Track & Initialization & Fraction (scenes) & Status & Evidence & Epoch & Duration & Dev F1 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _render_status_table(snapshot: Snapshot) -> str:
    counts = snapshot.status_counts
    fractions = ", ".join(f"f{item}" for item in snapshot.reportable_fractions) or "None"
    return "\n".join(
        [
            "% GENERATED by src.analysis.h100_results -- do not edit.",
            r"\begin{table}[htbp]",
            r"\caption{H100 campaign evidence at the frozen manuscript cutoff.}",
            r"\label{tab:h100-status}",
            r"\centering",
            r"\begin{tabular}{@{}rrrrl@{}}",
            r"\toprule",
            r"Done & Started & Pending & Contract verified & Reportable fractions \\",
            r"\midrule",
            f"{counts['DONE']} & {counts['STARTED']} & {counts['PENDING']} & "
            f"{snapshot.verified_count} & {_tex(fractions)} \\\\ ".rstrip(),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _save_figure(snapshot: Snapshot, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.pdf")
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    if snapshot.reportable_fractions:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0), sharey=True, layout="constrained")
        for ax, track in zip(axes, ("vit", "cnn")):
            for role in ("floor", "optical", "sar", "imagenet"):
                selected = sorted(
                    (
                        cell
                        for cell in snapshot.cells
                        if cell.track == track
                        and cell.role == role
                        and cell.label_fraction in snapshot.reportable_fractions
                        and cell.verified
                    ),
                    key=lambda cell: cell.label_fraction,
                )
                color, marker, linestyle = ROLE_STYLES[role]
                ax.plot(
                    [FRACTION_SCENES[cell.label_fraction] for cell in selected],
                    [cell.development["f1"] for cell in selected],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.2,
                    markersize=4,
                    label=ROLE_NAMES[role],
                )
            ax.set_title("ViT-B/16" if track == "vit" else "ConvNeXt-V2-Base")
            ax.set_xlabel("Training scenes")
            ax.set_xticks(
                [FRACTION_SCENES[value] for value in snapshot.reportable_fractions]
            )
            ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Development F1")
        fig.suptitle(
            "H100 label efficiency" if snapshot.payload["profile"] == "complete"
            else "H100 completion-closed cohorts"
        )
        axes[1].legend(fontsize=7, frameon=False)
    else:
        order = sorted(
            snapshot.cells,
            key=lambda cell: (
                0 if cell.track == "vit" else 1,
                ("floor", "optical", "sar", "imagenet").index(cell.role),
                cell.label_fraction,
            ),
        )
        arms = []
        for cell in order:
            label = f"{TRACK_NAMES[cell.track]} {ROLE_NAMES[cell.role]}"
            if label not in arms:
                arms.append(label)
        lookup = {(f"{TRACK_NAMES[c.track]} {ROLE_NAMES[c.role]}", c.label_fraction): c for c in snapshot.cells}
        values = []
        for arm in arms:
            row = []
            for fraction in sorted(FRACTION_NAMES):
                cell = lookup[(arm, fraction)]
                row.append(3 if cell.verified else {"PENDING": 0, "STARTED": 1, "DONE": 2}[cell.status])
            values.append(row)
        from matplotlib.colors import ListedColormap

        fig, ax = plt.subplots(figsize=(5.5, 2.65), layout="constrained")
        ax.imshow(values, aspect="auto", cmap=ListedColormap(["#dddddd", "#f2b84b", "#77b881", "#176b3a"]), vmin=0, vmax=3)
        ax.set_xticks(range(4), [f"f{x}" for x in sorted(FRACTION_NAMES)])
        ax.set_yticks(range(len(arms)), arms)
        ax.set_xlabel("Training-label fraction")
        ax.set_title("H100 campaign status (metrics require contract verification)")
        ax.set_xticks([x - 0.5 for x in range(1, 4)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(arms))], minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.legend(
            handles=[
                Patch(color="#dddddd", label="Pending"),
                Patch(color="#f2b84b", label="Started"),
                Patch(color="#77b881", label="Done, status only"),
                Patch(color="#176b3a", label="Contract verified"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=4,
            frameon=False,
            fontsize=7,
        )
    fig.savefig(
        temporary,
        bbox_inches="tight",
        metadata={"Creator": "src.analysis.h100_results", "CreationDate": None},
    )
    plt.close(fig)
    os.replace(temporary, destination)


def generate(
    snapshot: Snapshot,
    output_dir: str | Path,
    progress_log: str | Path | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    if output.is_symlink():
        raise SnapshotError("generated output directory may not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise SnapshotError("generated output path is not a directory")

    _atomic_text(output / "h100_results.tex", _render_macros(snapshot))
    _atomic_text(output / "h100_core_table.tex", _render_core_table(snapshot))
    _atomic_text(output / "h100_status_table.tex", _render_status_table(snapshot))
    if snapshot.payload["profile"] == "complete":
        figure_name = "h100_label_efficiency.pdf"
    elif snapshot.reportable_fractions:
        figure_name = "h100_completed_cohorts.pdf"
    else:
        figure_name = "h100_campaign_status.pdf"
    _save_figure(snapshot, output / figure_name)
    for stale_name in (
        "h100_campaign_status.pdf",
        "h100_completed_cohorts.pdf",
        "h100_label_efficiency.pdf",
    ):
        if stale_name != figure_name:
            (output / stale_name).unlink(missing_ok=True)
    if snapshot.payload["profile"] == "complete":
        caption = "Contract-verified H100 development-selection label-efficiency results."
    elif snapshot.reportable_fractions:
        caption = "Corrected development-selection F1 for completion-closed H100 cohorts only."
    else:
        caption = (
            "H100 campaign status at the manuscript cutoff. Status-only completions "
            "are not treated as reportable metrics."
        )
    wrapper = "\n".join(
        [
            "% GENERATED by src.analysis.h100_results -- do not edit.",
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=\linewidth]{{{figure_name}}}",
            rf"\caption{{{caption}}}",
            r"\label{fig:h100-results}",
            r"\end{figure}",
            "",
        ]
    )
    _atomic_text(output / "h100_figure.tex", wrapper)
    progress_log_path = (
        output / "h100_campaign_status.txt" if progress_log is None else Path(progress_log)
    )
    _atomic_text(progress_log_path, render_progress_log(snapshot))

    managed = [
        "h100_results.tex",
        "h100_core_table.tex",
        "h100_status_table.tex",
        "h100_figure.tex",
        figure_name,
    ]
    manifest = {
        "schema": 1,
        "profile": snapshot.payload["profile"],
        "snapshot_sha256": _canonical_sha(snapshot.payload),
        "reportable_fractions": list(snapshot.reportable_fractions),
        "campaign_status_sha256": _hash_file(progress_log_path),
        "files": {name: _hash_file(output / name) for name in managed},
    }
    _atomic_text(
        output / "h100_generated_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return {**manifest["files"], "h100_generated_manifest.json": _hash_file(output / "h100_generated_manifest.json")}


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SnapshotError(f"{description} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{description} is not a JSON object: {path}")
    return value


def _run_evidence_path(run: Path, relative: object, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SnapshotError(f"{description} path is absent")
    candidate_relative = Path(relative)
    if (
        candidate_relative.is_absolute()
        or candidate_relative.as_posix() != relative
        or any(part in {"", ".", ".."} for part in candidate_relative.parts)
    ):
        raise SnapshotError(f"{description} path is unsafe")
    candidate = run.joinpath(*candidate_relative.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise SnapshotError(f"{description} is absent or symlinked")
    try:
        candidate.resolve(strict=True).relative_to(run.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise SnapshotError(f"{description} path escapes its run directory") from exc
    return candidate


def load_verified_references(
    path: str | Path, *, expected_code_sha: str
) -> list[dict[str, Any]]:
    """Load the separately closed R2/R3 reference-result receipt.

    Reference results are optional and never participate in the 32-cell core
    curves. If supplied, the receipt must contain exactly one verified DONE
    record for each of R2 and R3, bound to the same public code revision.
    """

    receipt = _closed(
        _load_json_object(Path(path), "reference-result receipt"),
        {"schema", "code_sha", "references"},
        "reference-result receipt",
    )
    if receipt["schema"] != 1:
        raise SnapshotError("reference-result receipt schema is not 1")
    if _sha(receipt["code_sha"], "reference-result code_sha", length=40) != expected_code_sha:
        raise SnapshotError("reference-result receipt code SHA mismatch")
    records = receipt["references"]
    if not isinstance(records, list) or len(records) != 2:
        raise SnapshotError("reference-result receipt must contain exactly R2 and R3")

    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        item = _closed(
            record,
            {
                "reference_id",
                "exp_id",
                "status",
                "development_f1",
                "completion_marker_sha256",
                "runtime_provenance_sha256",
            },
            f"reference-result receipt entry {index}",
        )
        if item["status"] != "DONE":
            raise SnapshotError("reference-result receipt contains a non-DONE record")
        normalized.append(
            {
                "reference_id": str(item["reference_id"]),
                "exp_id": str(item["exp_id"]),
                "status": "DONE",
                "development_f1": _finite(
                    item["development_f1"],
                    f"reference-result receipt entry {index}.development_f1",
                    minimum=0,
                    maximum=1,
                ),
                "completion_marker_sha256": _sha(
                    item["completion_marker_sha256"],
                    f"reference-result receipt entry {index}.completion_marker_sha256",
                ),
                "runtime_provenance_sha256": _sha(
                    item["runtime_provenance_sha256"],
                    f"reference-result receipt entry {index}.runtime_provenance_sha256",
                ),
            }
        )
    normalized.sort(key=lambda item: item["reference_id"])
    if [item["reference_id"] for item in normalized] != ["R2", "R3"]:
        raise SnapshotError("reference-result receipt must contain the exact R2/R3 pair")
    if len({item["exp_id"] for item in normalized}) != 2:
        raise SnapshotError("reference-result receipt contains duplicate experiment identities")
    return normalized


def _reverse_context(
    extracted_root: str | Path,
) -> tuple[Path, Path, dict[str, Any], str, Mapping[str, Any], dict[str, Any], dict[str, int]]:
    root = Path(extracted_root)
    if (root / "results").is_dir():
        root = root / "results"
    if root.is_symlink() or not root.is_dir():
        raise SnapshotError("reverse result root is absent, invalid, or symlinked")
    provenance = root / "provenance"
    core = root / "core"
    campaign = _load_json_object(provenance / "campaign_manifest.json", "campaign manifest")
    if campaign.get("schema") != 2:
        raise SnapshotError("reverse campaign is not schema 2")
    code_sha = _sha(campaign.get("git_sha"), "campaign git_sha", length=40)
    hardware_class = campaign.get("accepted_hardware_class")
    if not isinstance(hardware_class, Mapping):
        raise SnapshotError("campaign accepted hardware class is absent")
    base_python = campaign.get("base_python")
    hardware = {
        "accelerator": str(hardware_class.get("gpu_name", "")),
        "compute_capability": hardware_class.get("compute_capability"),
        "python": str(base_python.get("version", ""))
        if isinstance(base_python, Mapping)
        else "",
        "torch": str(hardware_class.get("torch", "")),
        "cuda": str(hardware_class.get("cuda_build", "")),
    }
    hardware["class_sha256"] = _canonical_sha(hardware)
    _validate_hardware(hardware)
    if campaign.get("strict_fp32") != STRICT_BACKEND:
        raise SnapshotError("reverse campaign is not strict IEEE FP32")
    if campaign.get("precision") != "32-true":
        raise SnapshotError("reverse campaign precision is not 32-true")
    recipe = {
        "micro_batch": campaign.get("micro_batch"),
        "gradient_accumulation": campaign.get("gradient_accumulation"),
        "effective_batch": campaign.get("effective_batch"),
        "seed": 0,
    }
    if recipe != {
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "seed": 0,
    }:
        raise SnapshotError("reverse campaign recipe is not the shared batch-16 seed-0 recipe")
    return root, core, campaign, code_sha, hardware_class, hardware, recipe


def _reverse_grid(
    root: Path,
    expected_ids: set[str],
) -> dict[str, dict[str, str]]:
    grid_path = root / "provenance" / "summary" / "grid.csv"
    if grid_path.is_symlink():
        raise SnapshotError("reverse campaign summary/grid.csv may not be symlinked")
    try:
        with grid_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SnapshotError("reverse campaign summary/grid.csv is absent") from exc
    ids = [row.get("exp_id") for row in rows]
    if (
        any(not isinstance(exp_id, str) or exp_id not in expected_ids for exp_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise SnapshotError("reverse grid contains an unknown or duplicate cell identity")
    return {str(row["exp_id"]): row for row in rows}


def _verified_reverse_cell(
    *,
    row: Mapping[str, str],
    metadata: Mapping[str, Any],
    core: Path,
    code_sha: str,
    hardware_class: Mapping[str, Any],
    hardware_sha256: str,
) -> dict[str, Any]:
    exp_id = str(metadata["exp_id"])
    try:
        metadata_matches = (
            row.get("exp_id") == exp_id
            and row.get("init") == metadata["init"]
            and row.get("track") == metadata["track"]
            and row.get("role") == metadata["role"]
            and math.isclose(
                float(row.get("label_frac", "nan")),
                metadata["label_fraction"] / 100,
                abs_tol=1e-12,
            )
            and int(row.get("seed", -1)) == 0
            and row.get("precision") == "32-true"
            and row.get("git_sha") == code_sha
        )
    except (TypeError, ValueError):
        metadata_matches = False
    if not metadata_matches:
        raise SnapshotError(f"{exp_id} reverse-grid metadata mismatch")

    run = core / exp_id
    marker_path = run / "final_metrics.json"
    provenance_path = run / "runtime_provenance.json"
    marker = _load_json_object(marker_path, f"{exp_id} completion marker")
    runtime = _load_json_object(provenance_path, f"{exp_id} runtime provenance")
    if (
        marker.get("exp_id") != exp_id
        or marker.get("git_sha") != code_sha
        or marker.get("precision") != "32-true"
        or runtime.get("exp_id") != exp_id
        or runtime.get("git_sha") != code_sha
        or runtime.get("accepted_hardware_class") != hardware_class
        or runtime.get("strict_fp32") != STRICT_BACKEND
    ):
        raise SnapshotError(f"{exp_id} completion/runtime bindings mismatch")
    best_dev = marker.get("best_dev")
    if not isinstance(best_dev, Mapping):
        raise SnapshotError(f"{exp_id} best development result is absent")
    dev_f1 = _finite(row.get("dev_f1"), f"{exp_id} grid dev_f1", minimum=0, maximum=1)
    marker_f1 = _finite(
        marker.get("best_dev_f1"), f"{exp_id} marker best_dev_f1", minimum=0, maximum=1
    )
    best_dev_f1 = _finite(
        best_dev.get("f1"), f"{exp_id} best_dev.f1", minimum=0, maximum=1
    )
    if not (
        math.isclose(dev_f1, marker_f1, abs_tol=1e-12)
        and math.isclose(dev_f1, best_dev_f1, abs_tol=1e-12)
    ):
        raise SnapshotError(f"{exp_id} grid/marker development F1 mismatch")
    best_binding = marker.get("best_checkpoint")
    if not isinstance(best_binding, Mapping):
        raise SnapshotError(f"{exp_id} best checkpoint binding is absent")
    best_path = _run_evidence_path(
        run, best_binding.get("relative_path"), f"{exp_id} best checkpoint"
    )
    last_path = _run_evidence_path(run, "checkpoints/last.ckpt", f"{exp_id} last checkpoint")
    best_sha = _hash_file(best_path)
    last_sha = _hash_file(last_path)
    if best_sha != best_binding.get("sha256"):
        raise SnapshotError(f"{exp_id} best checkpoint hash mismatch")
    active_seconds = _finite(
        runtime.get("accumulated_active_seconds"),
        f"{exp_id} accumulated_active_seconds",
        minimum=0,
    )
    duration_seconds = int(round(active_seconds))
    if duration_seconds <= 0:
        raise SnapshotError(f"{exp_id} verified duration must be positive")
    epochs_run = _nonnegative_int(marker.get("epochs_run"), f"{exp_id}.epochs_run")
    if epochs_run == 0:
        raise SnapshotError(f"{exp_id}.epochs_run must be positive for a DONE cell")
    return {
        **metadata,
        "status": "DONE",
        "latest_epoch": epochs_run - 1,
        "duration_seconds": duration_seconds,
        "train_loss": _finite(marker.get("train_loss"), f"{exp_id}.train_loss", minimum=0),
        "development": {
            key: _finite(
                best_dev.get(key), f"{exp_id}.best_dev.{key}", minimum=0, maximum=1
            )
            for key in ("f1", "precision", "recall", "threshold")
        },
        "evidence": {
            "verification": "contract-verified",
            "completion_marker_sha256": _hash_file(marker_path),
            "runtime_provenance_sha256": _hash_file(provenance_path),
            "best_checkpoint_sha256": best_sha,
            "last_checkpoint_sha256": last_sha,
            "hardware_class_sha256": hardware_sha256,
        },
    }


def _public_campaign(
    *,
    campaign: Mapping[str, Any],
    code_sha: str,
    hardware: Mapping[str, Any],
    recipe: Mapping[str, int],
    frozen_root: str | Path,
) -> dict[str, Any]:
    campaign_id = str(campaign.get("campaign_id", "")).strip()
    if not campaign_id:
        raise SnapshotError("reverse campaign_id is absent")
    frozen_root_path = Path(frozen_root)
    return {
        "campaign_id": campaign_id,
        "code_sha": code_sha,
        "frozen_sha256": {
            relative: _hash_file(frozen_root_path / relative) for relative in FROZEN_PATHS
        },
        "hardware": dict(hardware),
        "strict_fp32": {
            "precision": "32-true",
            "tf32_enabled": False,
            "backend": STRICT_BACKEND,
        },
        "recipe": dict(recipe),
    }


def import_complete_results(
    *,
    extracted_root: str | Path,
    arms_config: str | Path,
    frozen_root: str | Path,
    source_sha256: str,
    captured_utc: str,
    reference_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Normalize an already verified and extracted H100 reverse package.

    ``extracted_root`` contains the reverse archives' common ``results/`` tree.
    This importer independently rebinds every published development value to
    its marker, runtime receipt, and actual best/last checkpoint bytes.
    """

    root = Path(extracted_root)
    if (root / "results").is_dir():
        root = root / "results"
    provenance = root / "provenance"
    core = root / "core"
    campaign = _load_json_object(provenance / "campaign_manifest.json", "campaign manifest")
    if campaign.get("schema") != 2 or campaign.get("status") != "complete":
        raise SnapshotError("reverse campaign is not schema-2 complete")
    code_sha = _sha(campaign.get("git_sha"), "campaign git_sha", length=40)
    hardware_class = campaign.get("accepted_hardware_class")
    if not isinstance(hardware_class, Mapping):
        raise SnapshotError("campaign accepted hardware class is absent")
    hardware = {
        "accelerator": str(hardware_class.get("gpu_name", "")),
        "compute_capability": hardware_class.get("compute_capability"),
        "python": str((campaign.get("base_python") or {}).get("version", ""))
        if isinstance(campaign.get("base_python"), Mapping)
        else "",
        "torch": str(hardware_class.get("torch", "")),
        "cuda": str(hardware_class.get("cuda_build", "")),
    }
    hardware["class_sha256"] = _canonical_sha(hardware)
    _validate_hardware(hardware)
    strict_backend = campaign.get("strict_fp32")
    if strict_backend != STRICT_BACKEND:
        raise SnapshotError("reverse campaign is not strict IEEE FP32")
    if campaign.get("precision") != "32-true":
        raise SnapshotError("reverse campaign precision is not 32-true")

    expected = expected_cells(arms_config)
    expected_ids = set(expected)
    for key in ("complete", "training_complete"):
        values = campaign.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 32
            or set(values) != expected_ids
        ):
            raise SnapshotError(f"reverse campaign {key} is not the exact 32-cell grid")
    if campaign.get("running") != {}:
        raise SnapshotError("reverse campaign still has running cells")
    grid_path = provenance / "summary" / "grid.csv"
    try:
        with grid_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SnapshotError("reverse campaign summary/grid.csv is absent") from exc
    if len(rows) != 32 or {row.get("exp_id") for row in rows} != set(expected):
        raise SnapshotError("reverse grid is not the exact 32-cell matrix")
    cells = []
    for row in rows:
        exp_id = str(row["exp_id"])
        metadata = expected[exp_id]
        if (
            row.get("init") != metadata["init"]
            or row.get("track") != metadata["track"]
            or row.get("role") != metadata["role"]
            or float(row.get("label_frac", "nan")) != metadata["label_fraction"] / 100
            or int(row.get("seed", -1)) != 0
            or row.get("precision") != "32-true"
            or row.get("git_sha") != code_sha
        ):
            raise SnapshotError(f"{exp_id} reverse-grid metadata mismatch")
        run = core / exp_id
        marker_path = run / "final_metrics.json"
        provenance_path = run / "runtime_provenance.json"
        marker = _load_json_object(marker_path, f"{exp_id} completion marker")
        runtime = _load_json_object(provenance_path, f"{exp_id} runtime provenance")
        if (
            marker.get("exp_id") != exp_id
            or marker.get("git_sha") != code_sha
            or marker.get("precision") != "32-true"
            or runtime.get("exp_id") != exp_id
            or runtime.get("git_sha") != code_sha
            or runtime.get("accepted_hardware_class") != hardware_class
            or runtime.get("strict_fp32") != STRICT_BACKEND
        ):
            raise SnapshotError(f"{exp_id} completion/runtime bindings mismatch")
        best_dev = marker.get("best_dev")
        if not isinstance(best_dev, Mapping):
            raise SnapshotError(f"{exp_id} best development result is absent")
        dev_f1 = _finite(row.get("dev_f1"), f"{exp_id} grid dev_f1", minimum=0, maximum=1)
        marker_f1 = _finite(
            marker.get("best_dev_f1"), f"{exp_id} marker best_dev_f1", minimum=0, maximum=1
        )
        best_dev_f1 = _finite(
            best_dev.get("f1"), f"{exp_id} best_dev.f1", minimum=0, maximum=1
        )
        if not (
            math.isclose(dev_f1, marker_f1, abs_tol=1e-12)
            and math.isclose(dev_f1, best_dev_f1, abs_tol=1e-12)
        ):
            raise SnapshotError(f"{exp_id} grid/marker development F1 mismatch")
        best_binding = marker.get("best_checkpoint")
        if not isinstance(best_binding, Mapping):
            raise SnapshotError(f"{exp_id} best checkpoint binding is absent")
        best_path = _run_evidence_path(
            run, best_binding.get("relative_path"), f"{exp_id} best checkpoint"
        )
        last_path = _run_evidence_path(
            run, "checkpoints/last.ckpt", f"{exp_id} last checkpoint"
        )
        best_sha = _hash_file(best_path)
        last_sha = _hash_file(last_path)
        if best_sha != best_binding.get("sha256"):
            raise SnapshotError(f"{exp_id} best checkpoint hash mismatch")
        active_seconds = _finite(
            runtime.get("accumulated_active_seconds"),
            f"{exp_id} accumulated_active_seconds",
            minimum=0,
        )
        cells.append(
            {
                **metadata,
                "status": "DONE",
                "latest_epoch": _nonnegative_int(marker.get("epochs_run"), f"{exp_id}.epochs_run") - 1,
                "duration_seconds": int(round(active_seconds)),
                "train_loss": _finite(marker.get("train_loss"), f"{exp_id}.train_loss", minimum=0),
                "development": {
                    key: _finite(best_dev.get(key), f"{exp_id}.best_dev.{key}", minimum=0, maximum=1)
                    for key in ("f1", "precision", "recall", "threshold")
                },
                "evidence": {
                    "verification": "contract-verified",
                    "completion_marker_sha256": _hash_file(marker_path),
                    "runtime_provenance_sha256": _hash_file(provenance_path),
                    "best_checkpoint_sha256": best_sha,
                    "last_checkpoint_sha256": last_sha,
                    "hardware_class_sha256": hardware["class_sha256"],
                },
            }
        )
    frozen_root_path = Path(frozen_root)
    frozen = {relative: _hash_file(frozen_root_path / relative) for relative in FROZEN_PATHS}
    references = (
        []
        if reference_receipt is None
        else load_verified_references(reference_receipt, expected_code_sha=code_sha)
    )
    return {
        "schema": SNAPSHOT_SCHEMA,
        "profile": "complete",
        "metric_scope": METRIC_SCOPE,
        "captured_utc": _utc(captured_utc, "captured_utc"),
        "campaign": {
            "campaign_id": str(campaign.get("campaign_id", "")),
            "code_sha": code_sha,
            "frozen_sha256": frozen,
            "hardware": hardware,
            "strict_fp32": {
                "precision": "32-true",
                "tf32_enabled": False,
                "backend": STRICT_BACKEND,
            },
            "recipe": {
                "micro_batch": campaign.get("micro_batch"),
                "gradient_accumulation": campaign.get("gradient_accumulation"),
                "effective_batch": campaign.get("effective_batch"),
                "seed": 0,
            },
        },
        "source": {
            "kind": "verified-reverse-result-package",
            "description": "H100 reverse-result handback",
            "sha256": _sha(source_sha256, "source_sha256"),
        },
        "cells": cells,
        "references": references,
    }


def import_deadline_results(
    *,
    extracted_root: str | Path,
    status_snapshot: str | Path,
    arms_config: str | Path,
    frozen_root: str | Path,
    source_sha256: str,
    captured_utc: str,
    reference_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Import a completion-closed deadline cohort from real evidence bytes.

    The status snapshot supplies only the 32-cell progress ledger. It must be
    entirely status-only; digest-shaped strings are not accepted as a shortcut
    to result evidence. Every ``DONE`` cell is then reconstructed from the
    reverse package after this process opens and SHA-256 hashes its completion
    marker, runtime provenance, and actual best/last checkpoints.
    """

    root, core, campaign, code_sha, hardware_class, hardware, recipe = _reverse_context(
        extracted_root
    )
    campaign_status = campaign.get("status")
    if campaign_status not in {"running", "failed", "host-requeue-required", "complete"}:
        raise SnapshotError("reverse deadline campaign has an unknown status")

    expected = expected_cells(arms_config)
    expected_ids = set(expected)
    completion_sets: dict[str, set[str]] = {}
    for key in ("complete", "training_complete"):
        values = campaign.get(key)
        if (
            not isinstance(values, list)
            or len(values) != len(set(values))
            or not set(values).issubset(expected_ids)
        ):
            raise SnapshotError(f"reverse campaign {key} is not a unique core-cell subset")
        completion_sets[key] = set(values)
    if completion_sets["complete"] != completion_sets["training_complete"]:
        raise SnapshotError("reverse campaign completion ledgers disagree")
    completed_ids = completion_sets["training_complete"]
    if not completed_ids:
        raise SnapshotError("reverse deadline package contains no completed core evidence")
    if len(completed_ids) == 32:
        raise SnapshotError("a 32-cell reverse package must use import-complete")
    if campaign_status == "complete":
        raise SnapshotError("reverse campaign claims complete without the exact 32-cell grid")
    running = campaign.get("running")
    if not isinstance(running, Mapping):
        raise SnapshotError("reverse campaign running ledger is not an object")
    running_ids = set(running)
    if not running_ids.issubset(expected_ids) or running_ids & completed_ids:
        raise SnapshotError("reverse campaign running ledger is inconsistent")

    rows = _reverse_grid(root, expected_ids)
    if set(rows) != completed_ids:
        raise SnapshotError(
            "reverse deadline grid must contain exactly the completion-closed evidence cells"
        )

    status = load_snapshot(status_snapshot, arms_config)
    if status.payload["profile"] != "deadline":
        raise SnapshotError("deadline import requires a deadline status snapshot")
    if status.verified_count != 0:
        raise SnapshotError(
            "deadline status snapshot must be status-only; evidence is rehashed from bytes"
        )
    status_campaign = status.payload["campaign"]
    public_campaign = _public_campaign(
        campaign=campaign,
        code_sha=code_sha,
        hardware=hardware,
        recipe=recipe,
        frozen_root=frozen_root,
    )
    for key in ("campaign_id", "code_sha", "hardware", "strict_fp32", "recipe"):
        if status_campaign[key] != public_campaign[key]:
            raise SnapshotError(f"deadline status snapshot campaign mismatch for {key}")
    if status_campaign["frozen_sha256"] != public_campaign["frozen_sha256"]:
        raise SnapshotError("deadline status snapshot frozen-artifact mismatch")

    status_records = {
        str(item["exp_id"]): item for item in status.payload["cells"]
    }
    status_done = {
        exp_id for exp_id, item in status_records.items() if item["status"] == "DONE"
    }
    if status_done != completed_ids:
        raise SnapshotError(
            "deadline status DONE cells do not match the reverse completion ledger"
        )

    cells = []
    for exp_id in expected:
        if exp_id in completed_ids:
            cells.append(
                _verified_reverse_cell(
                    row=rows[exp_id],
                    metadata=expected[exp_id],
                    core=core,
                    code_sha=code_sha,
                    hardware_class=hardware_class,
                    hardware_sha256=str(hardware["class_sha256"]),
                )
            )
        else:
            status_record = dict(status_records[exp_id])
            status_record["evidence"] = dict(status_record["evidence"])
            cells.append(status_record)

    references = (
        []
        if reference_receipt is None
        else load_verified_references(reference_receipt, expected_code_sha=code_sha)
    )
    return {
        "schema": SNAPSHOT_SCHEMA,
        "profile": "deadline",
        "metric_scope": METRIC_SCOPE,
        "captured_utc": _utc(captured_utc, "captured_utc"),
        "campaign": public_campaign,
        "source": {
            "kind": "verified-reverse-result-package",
            "description": "H100 completion-closed reverse-result handback",
            "sha256": _sha(source_sha256, "source_sha256"),
        },
        "cells": cells,
        "references": references,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("generate", help="validate a snapshot and render paper inputs")
    render.add_argument("--snapshot", type=Path, required=True)
    render.add_argument("--arms-config", type=Path, default=Path("configs/arms.yaml"))
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument(
        "--progress-log",
        type=Path,
        default=Path("results/h100/logs/campaign_status.txt"),
    )
    render.add_argument("--validate-only", action="store_true")
    importers = {
        "import-complete": subparsers.add_parser(
            "import-complete", help="normalize a complete reverse-result package"
        ),
        "import-deadline": subparsers.add_parser(
            "import-deadline", help="normalize a completion-closed deadline package"
        ),
    }
    for command, ingest in importers.items():
        ingest.add_argument("--extracted-root", type=Path, required=True)
        ingest.add_argument("--arms-config", type=Path, default=Path("configs/arms.yaml"))
        ingest.add_argument("--frozen-root", type=Path, default=Path("."))
        ingest.add_argument("--source-sha256", required=True)
        ingest.add_argument("--captured-utc", required=True)
        ingest.add_argument("--reference-receipt", type=Path)
        ingest.add_argument("--output", type=Path, required=True)
        if command == "import-deadline":
            ingest.add_argument("--status-snapshot", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        snapshot = load_snapshot(args.snapshot, args.arms_config)
        if not args.validate_only:
            generate(snapshot, args.output_dir, args.progress_log)
        print(
            json.dumps(
                {
                    "profile": snapshot.payload["profile"],
                    "done": snapshot.status_counts["DONE"],
                    "started": snapshot.status_counts["STARTED"],
                    "pending": snapshot.status_counts["PENDING"],
                    "verified": snapshot.verified_count,
                    "reportable_fractions": list(snapshot.reportable_fractions),
                },
                sort_keys=True,
            )
        )
        return 0
    common = {
        "extracted_root": args.extracted_root,
        "arms_config": args.arms_config,
        "frozen_root": args.frozen_root,
        "source_sha256": args.source_sha256,
        "captured_utc": args.captured_utc,
        "reference_receipt": args.reference_receipt,
    }
    normalized = (
        import_complete_results(**common)
        if args.command == "import-complete"
        else import_deadline_results(status_snapshot=args.status_snapshot, **common)
    )
    _atomic_text(args.output, json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    # Re-open through the public schema before claiming success.
    load_snapshot(args.output, args.arms_config)
    print(json.dumps({"output": str(args.output), "cells": 32, "profile": normalized["profile"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
