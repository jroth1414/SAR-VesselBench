"""Stage a delivered H100 run tree as an evidence tree for ``heldout_results``.

The output mirrors ``results/h100/evidence``:

* ``TRAINING_COHORT.json``, each cell's ``final_metrics.json``, its
  ``metrics.csv``, any marker-bound ``terminal_recovery.json``, its
  ``test_metrics.json`` and, when present, ``final_verified_metrics.json``
  are copied byte-exact (each is hash-bound by the cohort, the TEST rebuild
  or ``FINAL_EVAL_COMPLETE.json``);
* ``runtime_provenance.json`` is copied with each private cluster prefix
  replaced, and ``REDACTIONS.json`` records each original SHA-256 and the
  replacement strings (never the private prefixes themselves);
* ``EVAL_GROUND_TRUTH_VALIDATED.json`` and, when present,
  ``FINAL_EVAL_COMPLETE.json`` come from the run tree's ``.h100``.

A byte-exact file that carries a private prefix cannot be both exact and
redacted, so staging stops instead of writing it. The output directory must
not exist yet. Prefixes are given on the command line, so this file names no
site paths.

Usage:
  python scripts/stage_evidence.py --run-root <runs tree> --out <new dir> \
      --redact <private prefix>=<replacement> [--redact ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

COHORT = "TRAINING_COHORT.json"
AUDIT = "EVAL_GROUND_TRUTH_VALIDATED.json"
EXACT_FILES = (("final_metrics.json", "final_metrics.json"), ("metrics/metrics.csv", "metrics.csv"))
# A zero-step terminal resume leaves no Lightning curve; its marker-bound record
# stands in, and heldout_results decides whether the substitution is valid.
OPTIONAL_EXACT = {"metrics/metrics.csv"}
# Hash-bound elsewhere (marker, TEST rebuild, FINAL_EVAL_COMPLETE), so never redacted.
EXTRA_EXACT_FILES = ("terminal_recovery.json", "test_metrics.json", "final_verified_metrics.json")
REQUIRED_EXTRA = {"test_metrics.json"}
REDACTED_FILES = ("runtime_provenance.json",)
REQUIRED_REDACTED = {"runtime_provenance.json"}
FINAL_COMPLETE = "FINAL_EVAL_COMPLETE.json"


LOG_HEAD_LINES = 60
LOG_TAIL_LINES = 120
LOG_KEY_PREFIX = "logs/h100_excerpts"

Redactions = Sequence[tuple[str, str]]


class StagingError(RuntimeError):
    """The delivered run tree cannot be staged without breaking a binding."""


def _redact(text: str, redactions: Redactions) -> tuple[str, list[str]]:
    """Apply every (private prefix, replacement) pair; return the text and the replacements used."""

    used = []
    for prefix, replacement in redactions:
        if prefix in text:
            text = text.replace(prefix, replacement)
            used.append(replacement)
    return text, used


def excerpt_log(raw: bytes, exp_id: str, redactions: Redactions) -> tuple[str, str]:
    """Head/tail excerpt of a training log; returns (unredacted, redacted) text.

    Progress-bar carriage returns count as line breaks, as in the August
    excerpts. The header binds the full log by line count and SHA-256.
    """

    lines = raw.decode("utf-8", errors="replace").splitlines()
    header = [
        f"# Excerpt of the full H100 training log for {exp_id}.",
        f"# Full log: {len(lines)} lines, sha256 {_sha256(raw)} (operator archive).",
        "# Private site paths are replaced by: "
        + ", ".join(sorted({replacement for _, replacement in redactions}))
        + ".",
    ]
    if len(lines) > LOG_HEAD_LINES + LOG_TAIL_LINES:
        elided = len(lines) - LOG_HEAD_LINES - LOG_TAIL_LINES
        body = lines[:LOG_HEAD_LINES] + [f"[... {elided} lines elided ...]"] + lines[-LOG_TAIL_LINES:]
    else:
        body = lines
    head = "\n".join(header) + "\n"
    text = "\n".join(body) + "\n"
    return head + text, head + _redact(text, redactions)[0]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_exact(source: Path, target: Path, redactions: Redactions) -> bytes:
    data = source.read_bytes()
    if any(prefix.encode("utf-8") in data for prefix, _ in redactions):
        raise StagingError(f"byte-exact file carries a private prefix: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def _copy_redacted(source: Path, target: Path, redactions: Redactions, record: dict, key: str) -> None:
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    text, used = _redact(data.decode("utf-8"), redactions)
    if used:
        record[key] = {"original_sha256": _sha256(data), "replaced_with": used}
        data = text.encode("utf-8")
    target.write_bytes(data)


def stage(
    run_root: Path,
    out: Path,
    *,
    redactions: Redactions,
    logs_out: Path | None = None,
) -> dict[str, object]:
    if not redactions:
        raise StagingError("at least one --redact prefix=replacement pair is required")
    if out.exists():
        raise StagingError(f"output directory already exists: {out}")
    if logs_out is not None and logs_out.exists():
        raise StagingError(f"log excerpt directory already exists: {logs_out}")
    control = run_root / ".h100"
    cohort_bytes = (control / COHORT).read_bytes()
    cohort = json.loads(cohort_bytes)
    cells = cohort.get("cells")
    if not isinstance(cells, list) or len(cells) != cohort.get("cell_count"):
        raise StagingError("cohort cell list disagrees with its cell_count")

    work = out.with_name(out.name + ".staging")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    record: dict[str, dict[str, object]] = {}
    try:
        _copy_exact(control / COHORT, work / COHORT, redactions)
        _copy_exact(control / AUDIT, work / AUDIT, redactions)
        if (control / FINAL_COMPLETE).is_file():
            _copy_exact(control / FINAL_COMPLETE, work / FINAL_COMPLETE, redactions)
        for cell in cells:
            exp_id = str(cell["exp_id"])
            binding = cell["completion_marker"]
            cell_source = run_root / Path(binding["relative_path"]).parent
            if cell_source.name != exp_id:
                raise StagingError(f"{exp_id}: marker path does not name its cell: {binding}")
            for relative, staged_name in EXACT_FILES:
                if relative in OPTIONAL_EXACT and not (cell_source / relative).is_file():
                    continue
                data = _copy_exact(cell_source / relative, work / exp_id / staged_name, redactions)
                if staged_name == "final_metrics.json" and _sha256(data) != binding["sha256"]:
                    raise StagingError(f"{exp_id}: marker does not hash to its cohort binding")
            for name in EXTRA_EXACT_FILES:
                if (cell_source / name).is_file():
                    _copy_exact(cell_source / name, work / exp_id / name, redactions)
                elif name in REQUIRED_EXTRA:
                    raise StagingError(f"{exp_id}: missing {name}")
            for name in REDACTED_FILES:
                source = cell_source / name
                if not source.is_file():
                    if name in REQUIRED_REDACTED:
                        raise StagingError(f"{exp_id}: missing {name}")
                    continue
                key = f"{exp_id}/{name}"
                _copy_redacted(source, work / key, redactions, record, key)
        excerpts: dict[str, str] = {}
        if logs_out is not None:
            for cell in cells:
                exp_id = str(cell["exp_id"])
                raw = (run_root / "logs" / "h100" / f"{exp_id}.log").read_bytes()
                original, redacted = excerpt_log(raw, exp_id, redactions)
                excerpts[exp_id] = redacted
                if original != redacted:
                    record[f"{LOG_KEY_PREFIX}/{exp_id}.log"] = {
                        "original_sha256": _sha256(original.encode("utf-8")),
                        "replaced_with": _redact(original, redactions)[1],
                    }
        manifest = {
            "files": dict(sorted(record.items())),
            "note": (
                "Sanitized copies replace private cluster paths with the listed replacement "
                "strings. final_metrics.json, metrics.csv, and TRAINING_COHORT.json are "
                "byte-exact; the cohort hash-binds each final_metrics.json."
            ),
            "replacements": sorted({replacement for _, replacement in redactions}),
        }
        (work / "REDACTIONS.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        work.rename(out)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    if logs_out is not None:
        logs_out.mkdir(parents=True)
        for exp_id, text in excerpts.items():
            (logs_out / f"{exp_id}.log").write_bytes(text.encode("utf-8"))
    return {"cells": len(cells), "redacted_files": len(record), "cohort_sha256": _sha256(cohort_bytes)}


def _pair(value: str) -> tuple[str, str]:
    prefix, sep, replacement = value.partition("=")
    if not sep or not prefix or not replacement:
        raise argparse.ArgumentTypeError("expected <private prefix>=<replacement>")
    return prefix, replacement


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--redact", required=True, action="append", type=_pair, metavar="PREFIX=REPLACEMENT")
    parser.add_argument("--logs-out", type=Path, help="write redacted head/tail log excerpts here")
    args = parser.parse_args(argv)
    try:
        summary = stage(args.run_root, args.out, redactions=args.redact, logs_out=args.logs_out)
    except (StagingError, OSError, KeyError, ValueError) as exc:
        print(f"stage_evidence: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
