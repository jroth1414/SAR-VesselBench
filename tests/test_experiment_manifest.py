"""Guard the amended 32-core + 2-reference experiment manifest."""

import json
from pathlib import Path

import pytest
import yaml

from scripts import run_grid_node, run_grid_queue


EXPECTED_SHORTS = {
    "vitrand",
    "satdino",
    "sarmae",
    "vitin1k",
    "cnnrand",
    "beS2",
    "beS1",
    "cnnin1k",
}
EXPECTED_FRACS = (0.1, 0.25, 0.5, 1.0)
EXPECTED_CORE_IDS = {
    f"{short}-f{int(round(frac * 100))}-s0"
    for short in EXPECTED_SHORTS
    for frac in EXPECTED_FRACS
}


def test_manifest_is_exactly_34_experiments():
    config = yaml.safe_load(Path("configs/arms.yaml").read_text())

    expected_arms = {
        "vit_random",
        "satdino_b",
        "sarmae_b",
        "vit_imagenet",
        "cnn_random",
        "bigearthnet_s2",
        "bigearthnet_s1",
        "cnn_imagenet",
    }
    assert set(config["arms"]) == expected_arms
    assert config["seeds"] == {
        "core": [0],
        "reruns": [],
        "rerun_fracs": [],
    }

    cells = [
        f"{meta['short']}-f{int(round(frac * 100))}-s{seed}"
        for meta in config["arms"].values()
        for frac in config["label_fracs"]
        for seed in config["seeds"]["core"]
    ]
    references = {
        reference["exp_id"] for reference in config["references"].values()
    }

    assert tuple(config["label_fracs"]) == EXPECTED_FRACS
    assert set(cells) == EXPECTED_CORE_IDS
    assert references == {"yolo26-f100", "locateanything-zs"}
    assert len(EXPECTED_CORE_IDS | references) == 34
    assert not any(cell.startswith(("vitsup-", "cnnsup-")) for cell in cells)


def test_tracks_have_four_distinct_roles():
    config = yaml.safe_load(Path("configs/arms.yaml").read_text())
    for track in ("vit", "cnn"):
        roles = {
            arm["role"]
            for arm in config["arms"].values()
            if arm["track"] == track
        }
        assert roles == {"floor", "optical", "sar", "imagenet"}


def test_runners_enumerate_the_exact_core_manifest():
    dev_ids = {
        run_grid_node.exp_id(item["init"], item["frac"])
        for item in run_grid_queue.build_queue()
    }
    node_cells = run_grid_node.matrix_cells()
    node_ids = {
        run_grid_node.exp_id(init, frac)
        for init, frac in node_cells
    }

    assert dev_ids == EXPECTED_CORE_IDS
    assert node_ids == EXPECTED_CORE_IDS
    assert len(node_cells) == 32
    assert node_cells[:2] == [
        ("vit_imagenet", 0.1),
        ("cnn_imagenet", 0.1),
    ]


def test_node_runner_rejects_unsafe_hardware_args():
    run_grid_node.validate_hardware_args([0, 1, 2, 3, 4], 8)
    with pytest.raises(ValueError, match="duplicate"):
        run_grid_node.validate_hardware_args([0, 0], 8)
    with pytest.raises(ValueError, match="divisor"):
        run_grid_node.validate_hardware_args([0], 3)


def test_runner_completion_markers_are_value_checked(tmp_path):
    init, frac = "vit_imagenet", 0.1
    exp = run_grid_node.exp_id(init, frac)
    run_dir = tmp_path / exp
    run_dir.mkdir()
    final = run_dir / "final_metrics.json"
    final.write_text(
        json.dumps({"exp_id": exp, "best_dev_f1": 0.5}),
        newline="\n",
    )

    assert run_grid_node.cell_done(init, frac, tmp_path)
    assert run_grid_queue.cell_done(exp, tmp_path)

    final.write_text(
        json.dumps({"exp_id": "wrong-id", "best_dev_f1": 0.5}),
        newline="\n",
    )
    with pytest.raises(RuntimeError, match="contents"):
        run_grid_node.cell_done(init, frac, tmp_path)
    with pytest.raises(RuntimeError, match="contents"):
        run_grid_queue.cell_done(exp, tmp_path)
