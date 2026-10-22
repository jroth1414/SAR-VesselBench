"""Write TRAIN_LABEL_PROFILE.json into an evidence tree.

The profile counts the frozen training split's labels under the audited
contract, in particular how many positives lie within 2 km of shore. It
refuses to write unless the labels CSV and splits file hash to the inputs
that the tree's EVAL_GROUND_TRUTH_VALIDATED.json audit receipt binds.

Usage (as a module, so the checkout's own ``src`` is imported):
  python -m scripts.profile_train_labels --train-csv data/raw/xview3/labels/train.csv \
      --evidence-root results/h100/evidence
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.eval.ground_truth import classify_label
from src.eval.ground_truth_audit import (
    NEAR_SHORE_KM,
    REQUIRED_COLUMNS,
    _empty_counts,
    _flat_counts,
    _is_near_shore,
    _load_splits,
    _sha256,
    scene_ids_sha256,
)

PROFILE_NAME = "TRAIN_LABEL_PROFILE.json"
PROFILE_SCHEMA = 1


def profile_split_labels(
    *, train_csv: str | Path, splits_json: str | Path, split: str = "train"
) -> dict[str, object]:
    """Count one frozen split's labels under the audited contract.

    Unlike the audit, this has no expected counts: it describes the split (for
    example, how many training positives lie within 2 km of shore) and binds
    both inputs by SHA-256 so a consumer can tie it to the audit receipt.
    """

    train_path, splits_path = Path(train_csv), Path(splits_json)
    train_raw, splits_raw = train_path.read_bytes(), splits_path.read_bytes()
    scene_ids = _load_splits(splits_raw)[split]
    wanted = set(scene_ids)
    counts = _empty_counts()
    seen: set[str] = set()
    reader = csv.DictReader(io.StringIO(train_raw.decode("utf-8-sig"), newline=""))
    missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
    if missing_columns:
        raise ValueError(f"train CSV is missing columns: {sorted(missing_columns)}")
    for line_number, row in enumerate(reader, start=2):
        if row["scene_id"] not in wanted:
            continue
        try:
            category = classify_label(row)
            near_shore = _is_near_shore(row.get("distance_from_shore_km"))
        except ValueError as error:
            raise ValueError(f"invalid train CSV label at line {line_number}: {error}") from error
        seen.add(row["scene_id"])
        counts["rows"] = int(counts["rows"]) + 1
        counts[category] = int(counts[category]) + 1
        if near_shore:
            near = counts["near_shore"]
            assert isinstance(near, dict)
            near["rows"] = int(near["rows"]) + 1
            near[category] = int(near[category]) + 1
    missing_scenes = sorted(wanted - seen)
    if missing_scenes:
        raise ValueError(f"split {split!r} has scenes with no train CSV rows: {missing_scenes}")
    return {
        "profile_schema": PROFILE_SCHEMA,
        "contract": {
            "positive": "is_vessel=true and confidence in {HIGH,MEDIUM}",
            "background": "is_vessel=false and confidence in {HIGH,MEDIUM}",
            "ignore": "confidence=LOW",
            "near_shore_km_lte": NEAR_SHORE_KM,
            "scene_ids_hash_encoding": "canonical-json-array-utf8-v1",
        },
        "inputs": {
            "train_csv": {"name": train_path.name, "bytes": len(train_raw), "sha256": _sha256(train_raw)},
            "splits_json": {"name": splits_path.name, "bytes": len(splits_raw), "sha256": _sha256(splits_raw)},
        },
        split: {
            "scene_ids_sha256": scene_ids_sha256(scene_ids),
            **_flat_counts(scene_count=len(scene_ids), counts=counts),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--splits-json", default=Path("data/splits.json"), type=Path)
    parser.add_argument("--evidence-root", default=Path("results/h100/evidence"), type=Path)
    args = parser.parse_args(argv)
    audit = json.loads((args.evidence_root / "EVAL_GROUND_TRUTH_VALIDATED.json").read_text(encoding="utf-8"))
    profile = profile_split_labels(train_csv=args.train_csv, splits_json=args.splits_json, split="train")
    for name in ("train_csv", "splits_json"):
        if profile["inputs"][name]["sha256"] != audit["inputs"][name]["sha256"]:
            print(f"profile_train_labels: {name} does not match the audit receipt", file=sys.stderr)
            return 1
    out = args.evidence_root / PROFILE_NAME
    out.write_bytes((json.dumps(profile, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    train = profile["train"]
    print(f"wrote {out}: {train['positive']} positives, {train['near_shore_positive']} within 2 km of shore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
