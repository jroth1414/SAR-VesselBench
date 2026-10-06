"""Stage a delivered H100 run tree as an evidence tree for ``heldout_results``.

The output mirrors ``results/h100/evidence``:

* ``TRAINING_COHORT.json``, each cell's ``final_metrics.json``, its
  ``metrics.csv`` and any marker-bound ``terminal_recovery.json`` are copied
  byte-exact (the cohort hash-binds every marker);
* ``runtime_provenance.json``, ``test_metrics.json`` and, when present,
  ``final_verified_metrics.json`` are copied with the private cluster prefix
  replaced, and ``REDACTIONS.json`` records each original SHA-256;
* ``EVAL_GROUND_TRUTH_VALIDATED.json`` comes from the run tree's ``.h100``.

A byte-exact file that carries the private prefix cannot be both exact and
redacted, so staging stops instead of writing it. The output directory must
not exist yet.

Usage:
  python scripts/stage_evidence.py --run-root <runs tree> --out <new dir>
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
EXTRA_EXACT_FILES = ("terminal_recovery.json",)
REDACTED_FILES = ("runtime_provenance.json", "test_metrics.json", "final_verified_metrics.json")
REQUIRED_REDACTED = {"runtime_provenance.json", "test_metrics.json"}
DEFAULT_PREFIX = "/projects/geofam"
DEFAULT_REPLACEMENT = "/cluster-site-redacted"


class StagingError(RuntimeError):
    """The delivered run tree cannot be staged without breaking a binding."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_exact(source: Path, target: Path, prefix: bytes) -> bytes:
    data = source.read_bytes()
    if prefix in data:
        raise StagingError(f"byte-exact file carries the private prefix: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def _copy_redacted(
    source: Path, target: Path, prefix: bytes, replacement: bytes, record: dict, key: str
) -> None:
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if prefix in data:
        record[key] = {"original_sha256": _sha256(data), "replaced": prefix.decode("utf-8")}
        data = data.replace(prefix, replacement)
    target.write_bytes(data)


def stage(
    run_root: Path,
    out: Path,
    *,
    prefix: str = DEFAULT_PREFIX,
    replacement: str = DEFAULT_REPLACEMENT,
) -> dict[str, object]:
    if out.exists():
        raise StagingError(f"output directory already exists: {out}")
    control = run_root / ".h100"
    prefix_bytes, replacement_bytes = prefix.encode("utf-8"), replacement.encode("utf-8")
    cohort_bytes = (control / COHORT).read_bytes()
    cohort = json.loads(cohort_bytes)
    cells = cohort.get("cells")
    if not isinstance(cells, list) or len(cells) != cohort.get("cell_count"):
        raise StagingError("cohort cell list disagrees with its cell_count")

    work = out.with_name(out.name + ".staging")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    redactions: dict[str, dict[str, str]] = {}
    try:
        _copy_exact(control / COHORT, work / COHORT, prefix_bytes)
        _copy_exact(control / AUDIT, work / AUDIT, prefix_bytes)
        for record in cells:
            exp_id = str(record["exp_id"])
            binding = record["completion_marker"]
            cell_source = run_root / Path(binding["relative_path"]).parent
            if cell_source.name != exp_id:
                raise StagingError(f"{exp_id}: marker path does not name its cell: {binding}")
            for relative, staged_name in EXACT_FILES:
                if relative in OPTIONAL_EXACT and not (cell_source / relative).is_file():
                    continue
                data = _copy_exact(cell_source / relative, work / exp_id / staged_name, prefix_bytes)
                if staged_name == "final_metrics.json" and _sha256(data) != binding["sha256"]:
                    raise StagingError(f"{exp_id}: marker does not hash to its cohort binding")
            for name in EXTRA_EXACT_FILES:
                if (cell_source / name).is_file():
                    _copy_exact(cell_source / name, work / exp_id / name, prefix_bytes)
            for name in REDACTED_FILES:
                source = cell_source / name
                if not source.is_file():
                    if name in REQUIRED_REDACTED:
                        raise StagingError(f"{exp_id}: missing {name}")
                    continue
                key = f"{exp_id}/{name}"
                _copy_redacted(source, work / key, prefix_bytes, replacement_bytes, redactions, key)
        manifest = {
            "files": dict(sorted(redactions.items())),
            "note": (
                "Sanitized copies replace the private cluster prefix. final_metrics.json, "
                "metrics.csv, and TRAINING_COHORT.json are byte-exact; the cohort hash-binds "
                "each final_metrics.json."
            ),
            "redacted_prefix": prefix,
            "replacement": replacement,
        }
        (work / "REDACTIONS.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        work.rename(out)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return {"cells": len(cells), "redacted_files": len(redactions), "cohort_sha256": _sha256(cohort_bytes)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--replacement", default=DEFAULT_REPLACEMENT)
    args = parser.parse_args(argv)
    try:
        summary = stage(args.run_root, args.out, prefix=args.prefix, replacement=args.replacement)
    except (StagingError, OSError, KeyError, ValueError) as exc:
        print(f"stage_evidence: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
