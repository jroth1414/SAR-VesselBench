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
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.eval.ground_truth_audit import profile_split_labels

PROFILE_NAME = "TRAIN_LABEL_PROFILE.json"


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
