"""Guard the amended 32-core + 2-reference experiment manifest."""

from pathlib import Path

import yaml

from src.runtime.experiment import (
    EFFECTIVE_BATCH,
    EXPECTED_PRECISION,
    GRADIENT_ACCUMULATION,
    MICRO_BATCH,
    load_cells,
)


EXPECTED_SHORTS = {
    "vitrand", "satdino", "sarmae", "vitin1k",
    "cnnrand", "beS2", "beS1", "cnnin1k",
}
EXPECTED_FRACS = (0.1, 0.25, 0.5, 1.0)
EXPECTED_CORE_IDS = {
    f"{short}-f{int(round(frac * 100))}-s0"
    for short in EXPECTED_SHORTS
    for frac in EXPECTED_FRACS
}


def test_manifest_is_exactly_34_experiments():
    config = yaml.safe_load(Path("configs/arms.yaml").read_text())
    assert set(config["arms"]) == {
        "vit_random", "satdino_b", "sarmae_b", "vit_imagenet",
        "cnn_random", "bigearthnet_s2", "bigearthnet_s1", "cnn_imagenet",
    }
    assert config["seeds"] == {"core": [0], "reruns": [], "rerun_fracs": []}
    cells = [
        f"{meta['short']}-f{int(round(frac * 100))}-s{seed}"
        for meta in config["arms"].values()
        for frac in config["label_fracs"]
        for seed in config["seeds"]["core"]
    ]
    references = {item["exp_id"] for item in config["references"].values()}
    assert tuple(config["label_fracs"]) == EXPECTED_FRACS
    assert set(cells) == EXPECTED_CORE_IDS
    assert references == {"yolo26-f100", "locateanything-zs"}
    assert len(EXPECTED_CORE_IDS | references) == 34
    assert not any(cell.startswith(("vitsup-", "cnnsup-")) for cell in cells)


def test_tracks_have_four_distinct_roles():
    config = yaml.safe_load(Path("configs/arms.yaml").read_text())
    for track in ("vit", "cnn"):
        roles = {
            arm["role"] for arm in config["arms"].values()
            if arm["track"] == track
        }
        assert roles == {"floor", "optical", "sar", "imagenet"}


def test_neutral_matrix_helper_enumerates_the_exact_core_manifest():
    cells = load_cells(Path(__file__).resolve().parents[1])
    assert {cell.exp_id for cell in cells} == EXPECTED_CORE_IDS
    assert len(cells) == 32
    assert [(cell.track, cell.fraction) for cell in cells[:4]] == [
        ("cnn", 1.0), ("cnn", 1.0), ("cnn", 1.0), ("cnn", 1.0),
    ]


def test_neutral_runtime_exposes_the_frozen_training_recipe():
    assert EXPECTED_PRECISION == "32-true"
    assert MICRO_BATCH == 16
    assert GRADIENT_ACCUMULATION == 1
    assert EFFECTIVE_BATCH == 16
