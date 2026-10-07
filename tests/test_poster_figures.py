"""Poster numbers come from the validated evidence and render as legal TeX macros."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis.heldout_results import validate_evidence
from src.analysis.poster_figures import macro_name, number_strings, poster_numbers

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def numbers() -> dict:
    validated = validate_evidence(
        REPO / "results/h100/evidence",
        arms_config=REPO / "configs/arms.yaml",
        detector_config=REPO / "configs/detector.yaml",
        splits_config=REPO / "data/splits.json",
    )
    return poster_numbers(validated, rerun=None)


def test_headline_numbers_match_the_evidence(numbers: dict) -> None:
    shown = number_strings(numbers)
    assert shown["final_vit_best12"] == "0.523" and shown["final_vit_random_111"] == "0.522"
    assert shown["test_cnn_sar_minus_opt_10"] == "+0.080"
    assert shown["test_cnn_sar_minus_opt_100"] == "−0.018"
    assert shown["final_vessels"] == "8,642" and shown["gpu_hours"] == "567.6"
    assert shown["mean_test_minus_final"] == "0.306"


def test_every_macro_name_is_letters_only(numbers: dict) -> None:
    names = [macro_name(key) for key in number_strings(numbers)]
    assert all(name.isalpha() for name in names)
    assert len(set(names)) == len(names)
    assert macro_name("final_f1_max") == "PNFinalFOneMax"
    assert macro_name("test_gain12_min") == "PNTestGainTwelveMin"
