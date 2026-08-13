"""Delegate the verified reverse result package to the shared handoff lane."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


def validate_grid(runs_root: Path, *, owner_amendment: Path | None = None) -> Path:
    grid = runs_root / "summary/grid.csv"
    try:
        with grid.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise RuntimeError("reverse handback requires runs/summary/grid.csv") from exc
    if len(rows) != 32 or len({row.get("exp_id") for row in rows}) != 32:
        raise RuntimeError("reverse handback requires exactly 32 unique grid rows")
    points: dict[str, list[tuple[float, float]]] = {}
    flags: dict[str, list[str]] = {}
    for row in rows:
        try:
            score = float(row["test_f1"])
            fraction = float(row["label_frac"])
            finite = math.isfinite(score) and math.isfinite(fraction)
        except (KeyError, TypeError, ValueError):
            finite = False
            score = fraction = math.nan
        flag = row.get("monotonicity_ok", "").casefold()
        init_name = str(row.get("init", ""))
        if not finite or flag not in {"true", "false"} or not init_name:
            raise RuntimeError("reverse handback grid row is malformed")
        points.setdefault(init_name, []).append((fraction, score))
        flags.setdefault(init_name, []).append(flag)
    violating: set[str] = set()
    for init_name, values in points.items():
        ordered = sorted(values)
        if [fraction for fraction, _score in ordered] != [0.1, 0.25, 0.5, 1.0]:
            raise RuntimeError("reverse handback grid fractions are incomplete")
        if any(
            current < previous - 0.02
            for (_fraction, previous), (_next_fraction, current) in zip(
                ordered, ordered[1:]
            )
        ):
            violating.add(init_name)
    for init_name, observed in flags.items():
        expected = "false" if init_name in violating else "true"
        if len(observed) != 4 or set(observed) != {expected}:
            raise RuntimeError(
                "reverse handback grid flags disagree with the recomputed diagnostic"
            )
    if owner_amendment is None and violating:
        raise RuntimeError("reverse handback requires a monotone TEST grid")
    if owner_amendment is not None and not violating:
        raise RuntimeError(
            "owner-amended reverse handback requires the truthful failed TEST grid"
        )
    return grid


def command(
    *,
    repo: Path,
    runs_root: Path,
    campaign_manifest: Path,
    output_dir: Path,
    max_part_bytes: int,
    owner_amendment: Path | None = None,
) -> list[str]:
    if max_part_bytes <= 0:
        raise ValueError("--max-part-bytes must come from a positive site preflight")
    argv = [
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
        "--output-dir",
        str(output_dir),
        "--max-part-bytes",
        str(max_part_bytes),
    ]
    if owner_amendment is not None:
        argv.extend(["--owner-amendment", str(owner_amendment)])
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="parent for the automatically named content-addressed package",
    )
    parser.add_argument("--max-part-bytes", type=int, required=True)
    parser.add_argument(
        "--owner-amendment",
        type=Path,
        help="canonical immutable failed-grid owner amendment receipt",
    )
    args = parser.parse_args()
    amendment = args.owner_amendment.absolute() if args.owner_amendment else None
    validate_grid(args.runs_root.resolve(), owner_amendment=amendment)
    subprocess.run(
        command(
            repo=args.repo.resolve(),
            runs_root=args.runs_root.resolve(),
            campaign_manifest=args.campaign_manifest.resolve(),
            output_dir=args.output_dir.resolve(),
            max_part_bytes=args.max_part_bytes,
            owner_amendment=amendment,
        ),
        cwd=args.repo.resolve(),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
