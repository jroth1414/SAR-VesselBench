"""Poster numbers come from the validated evidence and render as legal TeX macros."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.heldout_results import validate_evidence
from src.analysis.poster_figures import macro_name, number_strings, poster_numbers, scene_details

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "results/h100/evidence"


@pytest.fixture(scope="module")
def numbers() -> dict:
    validated = validate_evidence(
        EVIDENCE,
        arms_config=REPO / "configs/arms.yaml",
        detector_config=REPO / "configs/detector.yaml",
        splits_config=REPO / "data/splits.json",
    )
    return poster_numbers(validated, rerun=None, details=scene_details(EVIDENCE, validated))


def test_headline_numbers_match_the_evidence(numbers: dict) -> None:
    shown = number_strings(numbers)
    assert shown["test_cnn_sar_minus_opt_10"] == "+0.080"
    assert shown["test_cnn_sar_minus_opt_100"] == "−0.018"
    assert shown["final_vessels"] == "8,642" and shown["gpu_hours"] == "567.6"
    assert shown["mean_test_minus_final"] == "0.306"
    assert (shown["train_near_shore_pct"], shown["final_near_shore_pct"]) == ("0.4%", "31%")
    assert shown["best_cell_near_shore_predictions"] == "34" and shown["best_cell_predictions"] == "4,302"


def test_every_twelve_scene_gain_has_an_interval_above_zero(numbers: dict) -> None:
    assert numbers["gain12_comparisons"] == 12 and numbers["gain12_ci_above_zero"] == 12
    ci = numbers["final_cnn_sar_minus_opt_100_ci"]
    assert ci["low"] <= numbers["final_cnn_sar_minus_opt_100"] <= ci["high"] < 0


def test_every_macro_name_is_letters_only(numbers: dict) -> None:
    names = [macro_name(key) for key in number_strings(numbers)]
    assert all(name.isalpha() for name in names)
    assert len(set(names)) == len(names)
    assert macro_name("final_f1_max") == "PNFinalFOneMax"
    assert macro_name("test_gain12_min") == "PNTestGainTwelveMin"
