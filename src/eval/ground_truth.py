"""Canonical xView3 label-to-scorer ground-truth conversion.

The frozen scorer intentionally operates on already-classified ground-truth
points. This module is the single boundary that applies the study's label
contract before rows reach that scorer:

* explicit vessel HIGH/MEDIUM rows are positives;
* explicit non-vessel HIGH/MEDIUM rows are background and are omitted; and
* LOW rows are retained as ignore points regardless of ``is_vessel``.

Malformed confidence or vessel fields fail closed instead of silently
changing the evaluation target set.
"""

from __future__ import annotations

import math
from typing import Literal, Mapping, Sequence

from src.eval.scorer import GroundTruthPoint

GSD_M = 10.0
VALID_CONFIDENCES = frozenset({"HIGH", "MEDIUM", "LOW"})
SOURCE_CASE = {"ais": "AIS", "ais/manual": "AIS/Manual", "manual": "Manual"}

LabelCategory = Literal["positive", "background", "ignore"]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def normalize_confidence(value: object) -> str:
    """Return canonical HIGH/MEDIUM/LOW, rejecting missing or unknown values."""

    if _is_missing(value):
        raise ValueError("confidence is missing")
    if not isinstance(value, str):
        raise ValueError(f"confidence must be a string, got {type(value).__name__}")
    confidence = value.strip().upper()
    if confidence not in VALID_CONFIDENCES:
        raise ValueError(f"unknown confidence value: {value!r}")
    return confidence


def parse_vessel_boolean(value: object) -> bool:
    """Parse the only accepted canonical representations of ``is_vessel``.

    Python booleans, NumPy booleans, and case-insensitive ``true``/``false``
    strings are accepted. Numbers and convenience spellings such as yes/no
    are deliberately rejected because their coercion can corrupt GT counts.
    """

    if isinstance(value, bool):
        return value
    value_type = type(value)
    if value_type.__module__ == "numpy" and value_type.__name__ == "bool_":
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    if _is_missing(value):
        raise ValueError("is_vessel is missing")
    raise ValueError(f"unparseable is_vessel value: {value!r}")


def classify_label(row: Mapping[str, object]) -> LabelCategory:
    """Classify one label row under the canonical evaluation GT contract."""

    confidence = normalize_confidence(row.get("confidence"))
    if confidence == "LOW":
        return "ignore"
    return "positive" if parse_vessel_boolean(row.get("is_vessel")) else "background"


def normalize_source(source: object) -> str | None:
    """Normalize known xView3 source spelling while preserving other values."""

    if _is_missing(source):
        return None
    text = str(source)
    return SOURCE_CASE.get(text.casefold(), text)


def _required_finite_float(value: object, *, field: str) -> float:
    if _is_missing(value):
        raise ValueError(f"{field} is missing")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _optional_finite_float(value: object, *, field: str) -> float | None:
    if _is_missing(value):
        return None
    return _required_finite_float(value, field=field)


def ground_truth_from_labels(
    rows: Sequence[Mapping[str, object]],
) -> list[GroundTruthPoint]:
    """Convert one scene's xView3 rows to inputs for the frozen scorer."""

    points: list[GroundTruthPoint] = []
    for index, row in enumerate(rows):
        try:
            confidence = normalize_confidence(row.get("confidence"))
            category = classify_label(row)
            if category == "background":
                continue
            points.append(
                GroundTruthPoint(
                    x_m=_required_finite_float(
                        row.get("detect_scene_column"),
                        field="detect_scene_column",
                    )
                    * GSD_M,
                    y_m=_required_finite_float(
                        row.get("detect_scene_row"),
                        field="detect_scene_row",
                    )
                    * GSD_M,
                    confidence=confidence,
                    source=normalize_source(row.get("source")),
                    distance_from_shore_km=_optional_finite_float(
                        row.get("distance_from_shore_km"),
                        field="distance_from_shore_km",
                    ),
                )
            )
        except ValueError as error:
            raise ValueError(f"invalid evaluation label at row {index}: {error}") from error
    return points
