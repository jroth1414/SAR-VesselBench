"""Delegate the verified reverse result package to the shared handoff lane."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


def validate_grid(runs_root: Path) -> Path:
    grid = runs_root / "summary/grid.csv"
    try:
        with grid.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RuntimeError("reverse handback requires runs/summary/grid.csv") from exc
    if len(rows) != 32 or len({row.get("exp_id") for row in rows}) != 32:
        raise RuntimeError("reverse handback requires exactly 32 unique grid rows")
    for row in rows:
        try:
            finite = math.isfinite(float(row["test_f1"]))
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite or row.get("monotonicity_ok", "").lower() != "true":
            raise RuntimeError("reverse handback requires finite test_f1 and monotonicity_ok")
    return grid


def command(
    *,
    repo: Path,
    runs_root: Path,
    campaign_manifest: Path,
    output: Path,
    max_part_bytes: int,
) -> list[str]:
    if max_part_bytes <= 0:
        raise ValueError("--max-part-bytes must come from a positive site preflight")
    return [
        sys.executable,
        "-m",
        "scripts.handoff",
        "build-results",
        "--repo",
        str(repo),
        "--runs-root",
        str(runs_root),
        "--campaign-manifest",
        str(campaign_manifest),
        "--output",
        str(output),
        "--max-part-bytes",
        str(max_part_bytes),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-part-bytes", type=int, required=True)
    args = parser.parse_args()
    validate_grid(args.runs_root.resolve())
    subprocess.run(
        command(
            repo=args.repo.resolve(),
            runs_root=args.runs_root.resolve(),
            campaign_manifest=args.campaign_manifest.resolve(),
            output=args.output.resolve(),
            max_part_bytes=args.max_part_bytes,
        ),
        cwd=args.repo.resolve(),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
