"""Deterministic, standard-library audit of the evaluation GT source contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from src.eval.ground_truth import classify_label

AUDIT_SCHEMA = 1
NEAR_SHORE_KM = 2.0
SCOPE_ORDER = ("dev8", "dev23", "test")
REQUIRED_COLUMNS = frozenset(
    {
        "scene_id",
        "confidence",
        "is_vessel",
        "distance_from_shore_km",
    }
)

# Source-acceptance values measured only from train.csv and frozen study IDs.
EXPECTED_SCOPE_COUNTS: Mapping[str, Mapping[str, int]] = {
    "dev8": {
        "scene_count": 8,
        "rows": 742,
        "positive": 517,
        "background": 107,
        "ignore": 118,
        "near_shore_rows": 6,
        "near_shore_positive": 4,
        "near_shore_background": 1,
        "near_shore_ignore": 1,
    },
    "dev23": {
        "scene_count": 23,
        "rows": 2724,
        "positive": 1479,
        "background": 804,
        "ignore": 441,
        "near_shore_rows": 13,
        "near_shore_positive": 10,
        "near_shore_background": 1,
        "near_shore_ignore": 2,
    },
    "test": {
        "scene_count": 16,
        "rows": 1910,
        "positive": 1165,
        "background": 420,
        "ignore": 325,
        "near_shore_rows": 4,
        "near_shore_positive": 2,
        "near_shore_background": 0,
        "near_shore_ignore": 2,
    },
}


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scene_ids_sha256(scene_ids: Sequence[str]) -> str:
    """Hash an exact, ordered scene-ID list using canonical JSON encoding."""

    return _sha256(_canonical_json(list(scene_ids)))


def _load_splits(raw: bytes) -> dict[str, list[str]]:
    try:
        payload = json.loads(raw)
        source = payload["splits"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("splits JSON must contain an object named 'splits'") from error
    if not isinstance(source, dict):
        raise ValueError("splits JSON field 'splits' must be an object")
    required = ("train", "dev", "test", "eval_final")
    result: dict[str, list[str]] = {}
    for name in required:
        raw_ids = source.get(name)
        if not isinstance(raw_ids, list) or not all(
            isinstance(scene_id, str) and scene_id for scene_id in raw_ids
        ):
            raise ValueError(f"split {name!r} must be a list of non-empty strings")
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(f"split {name!r} contains duplicate scene IDs")
        result[name] = list(raw_ids)
    owners: dict[str, str] = {}
    for name in required:
        for scene_id in result[name]:
            previous = owners.setdefault(scene_id, name)
            if previous != name:
                raise ValueError(
                    f"scene {scene_id!r} occurs in both {previous!r} and {name!r}"
                )
    if len(result["dev"]) < 8:
        raise ValueError("dev split must contain at least eight scenes")
    return result


def _is_near_shore(value: object) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    try:
        shore = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid distance_from_shore_km value: {value!r}") from error
    if math.isnan(shore):
        return False
    if not math.isfinite(shore):
        raise ValueError(f"non-finite distance_from_shore_km value: {value!r}")
    return shore <= NEAR_SHORE_KM


def _empty_counts() -> dict[str, object]:
    return {
        "rows": 0,
        "positive": 0,
        "background": 0,
        "ignore": 0,
        "near_shore": {
            "rows": 0,
            "positive": 0,
            "background": 0,
            "ignore": 0,
        },
    }


def _flat_counts(*, scene_count: int, counts: Mapping[str, object]) -> dict[str, int]:
    near = counts["near_shore"]
    assert isinstance(near, Mapping)
    return {
        "scene_count": scene_count,
        "rows": int(counts["rows"]),
        "positive": int(counts["positive"]),
        "background": int(counts["background"]),
        "ignore": int(counts["ignore"]),
        "near_shore_rows": int(near["rows"]),
        "near_shore_positive": int(near["positive"]),
        "near_shore_background": int(near["background"]),
        "near_shore_ignore": int(near["ignore"]),
    }


def audit_ground_truth_dataset(
    *,
    train_csv: str | Path,
    splits_json: str | Path,
    expected_counts: Mapping[str, Mapping[str, int]] = EXPECTED_SCOPE_COUNTS,
) -> dict[str, object]:
    """Build and validate the deterministic pre-final-evaluation GT receipt."""

    train_path = Path(train_csv)
    splits_path = Path(splits_json)
    train_raw = train_path.read_bytes()
    splits_raw = splits_path.read_bytes()
    splits = _load_splits(splits_raw)

    scope_ids = {
        "dev8": sorted(splits["dev"])[:8],
        "dev23": list(splits["dev"]),
        "test": list(splits["test"]),
    }
    scope_sets = {name: set(scene_ids) for name, scene_ids in scope_ids.items()}
    counts = {name: _empty_counts() for name in SCOPE_ORDER}
    seen_scenes = {name: set() for name in SCOPE_ORDER}

    text = train_raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
    if missing_columns:
        raise ValueError(f"train CSV is missing columns: {sorted(missing_columns)}")

    for line_number, row in enumerate(reader, start=2):
        scene_id = row["scene_id"]
        memberships = [name for name in SCOPE_ORDER if scene_id in scope_sets[name]]
        for name in memberships:
            try:
                category = classify_label(row)
                near_shore = _is_near_shore(row.get("distance_from_shore_km"))
            except ValueError as error:
                raise ValueError(
                    f"invalid train CSV label at line {line_number}: {error}"
                ) from error
            scope_count = counts[name]
            scope_count["rows"] = int(scope_count["rows"]) + 1
            scope_count[category] = int(scope_count[category]) + 1
            seen_scenes[name].add(scene_id)
            if near_shore:
                near = scope_count["near_shore"]
                assert isinstance(near, dict)
                near["rows"] = int(near["rows"]) + 1
                near[category] = int(near[category]) + 1

    scopes: dict[str, object] = {}
    normalized_expected: dict[str, dict[str, int]] = {}
    for name in SCOPE_ORDER:
        missing_scenes = sorted(scope_sets[name] - seen_scenes[name])
        if missing_scenes:
            raise ValueError(f"scope {name!r} has scenes with no train CSV rows: {missing_scenes}")
        actual = _flat_counts(scene_count=len(scope_ids[name]), counts=counts[name])
        try:
            expected = {key: int(value) for key, value in expected_counts[name].items()}
        except KeyError as error:
            raise ValueError(f"expected counts are missing scope {name!r}") from error
        if actual != expected:
            differences = {
                key: {"expected": expected.get(key), "actual": actual.get(key)}
                for key in sorted(set(expected) | set(actual))
                if expected.get(key) != actual.get(key)
            }
            raise ValueError(f"ground-truth count mismatch for {name}: {differences}")
        normalized_expected[name] = expected
        scopes[name] = {
            "scene_ids": scope_ids[name],
            "scene_ids_sha256": scene_ids_sha256(scope_ids[name]),
            "counts": counts[name],
            "expected_counts": expected,
            "matches_expected": True,
        }

    return {
        "audit_schema": AUDIT_SCHEMA,
        "contract": {
            "positive": "is_vessel=true and confidence in {HIGH,MEDIUM}",
            "background": "is_vessel=false and confidence in {HIGH,MEDIUM}",
            "ignore": "confidence=LOW",
            "near_shore_km_lte": NEAR_SHORE_KM,
            "scene_ids_hash_encoding": "canonical-json-array-utf8-v1",
        },
        "inputs": {
            "train_csv": {
                "name": train_path.name,
                "bytes": len(train_raw),
                "sha256": _sha256(train_raw),
            },
            "splits_json": {
                "name": splits_path.name,
                "bytes": len(splits_raw),
                "sha256": _sha256(splits_raw),
            },
        },
        "scopes": scopes,
        "expected_counts": normalized_expected,
        "verified": True,
    }


def audit_ground_truth_scope(
    *,
    train_csv: str | Path,
    splits_json: str | Path,
    scope: str,
    expected_counts: Mapping[str, Mapping[str, int]] = EXPECTED_SCOPE_COUNTS,
) -> dict[str, int]:
    """Audit one declared scope without requiring rows from any other scope."""

    if scope not in SCOPE_ORDER:
        raise ValueError(f"unsupported ground-truth audit scope: {scope!r}")
    splits = _load_splits(Path(splits_json).read_bytes())
    scene_ids = sorted(splits["dev"])[:8] if scope == "dev8" else list(splits[scope])
    wanted = set(scene_ids)
    counts = _empty_counts()
    seen: set[str] = set()
    with Path(train_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"train CSV is missing columns: {sorted(missing_columns)}")
        for line_number, row in enumerate(reader, start=2):
            scene_id = row["scene_id"]
            if scene_id not in wanted:
                continue
            try:
                category = classify_label(row)
                near_shore = _is_near_shore(row.get("distance_from_shore_km"))
            except ValueError as error:
                raise ValueError(
                    f"invalid train CSV label at line {line_number}: {error}"
                ) from error
            counts["rows"] = int(counts["rows"]) + 1
            counts[category] = int(counts[category]) + 1
            seen.add(scene_id)
            if near_shore:
                near = counts["near_shore"]
                assert isinstance(near, dict)
                near["rows"] = int(near["rows"]) + 1
                near[category] = int(near[category]) + 1
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(f"scope {scope!r} has scenes with no label rows: {missing}")
    actual = _flat_counts(scene_count=len(scene_ids), counts=counts)
    expected = {key: int(value) for key, value in expected_counts[scope].items()}
    if actual != expected:
        raise ValueError(f"ground-truth count mismatch for {scope}: {actual} != {expected}")
    return actual


def validate_ground_truth_audit_receipt(
    receipt: Mapping[str, object],
    *,
    splits_json: str | Path,
    expected_train_csv_sha256: str,
    expected_train_csv_bytes: int,
    expected_counts: Mapping[str, Mapping[str, int]] = EXPECTED_SCOPE_COUNTS,
) -> dict[str, object]:
    """Validate a source-built audit without opening the held-out label file.

    A transfer package can carry this metadata receipt after its builder audits
    the immutable label member. The consumer binds the receipt back to that
    member's SHA-256/size and the frozen split bytes; it does not need to parse
    TEST rows before the all-training cohort exists.
    """

    splits_path = Path(splits_json)
    splits_raw = splits_path.read_bytes()
    splits = _load_splits(splits_raw)
    if not isinstance(receipt, Mapping):
        raise ValueError("ground-truth audit receipt must be an object")
    if set(receipt) != {
        "audit_schema",
        "contract",
        "inputs",
        "scopes",
        "expected_counts",
        "verified",
    }:
        raise ValueError("ground-truth audit receipt keys are invalid")
    if receipt.get("audit_schema") != AUDIT_SCHEMA or receipt.get("verified") is not True:
        raise ValueError("ground-truth audit receipt is not verified schema 1")
    expected_contract = {
        "positive": "is_vessel=true and confidence in {HIGH,MEDIUM}",
        "background": "is_vessel=false and confidence in {HIGH,MEDIUM}",
        "ignore": "confidence=LOW",
        "near_shore_km_lte": NEAR_SHORE_KM,
        "scene_ids_hash_encoding": "canonical-json-array-utf8-v1",
    }
    if receipt.get("contract") != expected_contract:
        raise ValueError("ground-truth audit contract is invalid")
    expected_inputs = {
        "train_csv": {
            "name": "train.csv",
            "bytes": expected_train_csv_bytes,
            "sha256": expected_train_csv_sha256,
        },
        "splits_json": {
            "name": splits_path.name,
            "bytes": len(splits_raw),
            "sha256": _sha256(splits_raw),
        },
    }
    if receipt.get("inputs") != expected_inputs:
        raise ValueError("ground-truth audit input binding is invalid")

    normalized_expected = {
        name: {key: int(value) for key, value in expected_counts[name].items()}
        for name in SCOPE_ORDER
    }
    if receipt.get("expected_counts") != normalized_expected:
        raise ValueError("ground-truth audit expected-count binding is invalid")
    scope_ids = {
        "dev8": sorted(splits["dev"])[:8],
        "dev23": list(splits["dev"]),
        "test": list(splits["test"]),
    }
    scopes = receipt.get("scopes")
    if not isinstance(scopes, Mapping) or set(scopes) != set(SCOPE_ORDER):
        raise ValueError("ground-truth audit scopes are invalid")
    for name in SCOPE_ORDER:
        record = scopes.get(name)
        if not isinstance(record, Mapping) or set(record) != {
            "scene_ids",
            "scene_ids_sha256",
            "counts",
            "expected_counts",
            "matches_expected",
        }:
            raise ValueError(f"ground-truth audit scope {name!r} is invalid")
        counts = record.get("counts")
        if not isinstance(counts, Mapping) or set(counts) != {
            "rows",
            "positive",
            "background",
            "ignore",
            "near_shore",
        }:
            raise ValueError(f"ground-truth audit counts for {name!r} are invalid")
        near = counts.get("near_shore")
        if not isinstance(near, Mapping) or set(near) != {
            "rows",
            "positive",
            "background",
            "ignore",
        }:
            raise ValueError(f"ground-truth audit near-shore counts for {name!r} are invalid")
        actual = _flat_counts(scene_count=len(scope_ids[name]), counts=counts)
        if (
            record.get("scene_ids") != scope_ids[name]
            or record.get("scene_ids_sha256") != scene_ids_sha256(scope_ids[name])
            or record.get("expected_counts") != normalized_expected[name]
            or record.get("matches_expected") is not True
            or actual != normalized_expected[name]
        ):
            raise ValueError(f"ground-truth audit scope {name!r} binding is invalid")
    return dict(receipt)


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite audit receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(_canonical_json(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = audit_ground_truth_dataset(
        train_csv=args.train_csv,
        splits_json=args.splits_json,
    )
    if args.output is None:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        _write_receipt(args.output, receipt)
        print(json.dumps({"output": str(args.output), "verified": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
