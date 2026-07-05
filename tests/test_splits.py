from __future__ import annotations

import numpy as np
import pytest

from src.data.splits import (
    SPLIT_NAMES,
    build_lsssdd_split,
    build_splits,
    compute_channel_stats,
    scene_pool,
    scene_split_map,
)


def _records(n_train=40, n_eval=5, seed=7):
    rng = np.random.default_rng(seed)
    records = []
    for index in range(n_train):
        records.append(
            {
                "scene_id": f"{index:016x}t",
                "center_lon": float(rng.uniform(-30, 30)),
                "center_lat": float(rng.uniform(-40, 60)),
                "has_shoreline": bool(index % 3 == 0),
            }
        )
    for index in range(n_eval):
        records.append(
            {
                "scene_id": f"{index:016x}v",
                "center_lon": float(rng.uniform(-30, 30)),
                "center_lat": float(rng.uniform(-40, 60)),
            }
        )
    return records


FRACTIONS = {"train_frac": 0.75, "dev_frac": 0.15, "test_frac": 0.10}


def test_scene_pool_suffix_convention():
    assert scene_pool("05bc615a9b0e1159t") == "train_pool"
    assert scene_pool("590dd08f71056cacv") == "eval_final"
    with pytest.raises(ValueError):
        scene_pool("someweirdid")


def test_build_splits_disjoint_counts_and_eval_final():
    records = _records()
    splits = build_splits(records, fractions=FRACTIONS, seed=0)

    assert set(splits) == set(SPLIT_NAMES)
    all_ids = [scene for name in SPLIT_NAMES for scene in splits[name]]
    assert len(all_ids) == len(set(all_ids)) == len(records)

    assert len(splits["eval_final"]) == 5
    assert all(scene.endswith("v") for scene in splits["eval_final"])
    assert all(
        not scene.endswith("v")
        for name in ("train", "dev", "test")
        for scene in splits[name]
    )

    n_pool = 40
    assert len(splits["train"]) == pytest.approx(0.75 * n_pool, abs=3)
    assert len(splits["dev"]) == pytest.approx(0.15 * n_pool, abs=3)
    assert len(splits["test"]) == pytest.approx(0.10 * n_pool, abs=3)
    assert len(splits["dev"]) >= 1 and len(splits["test"]) >= 1


def test_build_splits_is_deterministic_in_seed():
    records = _records()
    assert build_splits(records, fractions=FRACTIONS, seed=0) == build_splits(
        records, fractions=FRACTIONS, seed=0
    )
    assert build_splits(records, fractions=FRACTIONS, seed=0) != build_splits(
        records, fractions=FRACTIONS, seed=1
    )


def test_build_splits_rejects_bad_fractions():
    with pytest.raises(ValueError):
        build_splits(
            _records(),
            fractions={"train_frac": 0.9, "dev_frac": 0.2, "test_frac": 0.1},
            seed=0,
        )


def test_build_splits_tiny_pool_keeps_dev_and_test_nonempty():
    splits = build_splits(_records(n_train=5, n_eval=2), fractions=FRACTIONS, seed=0)
    assert len(splits["train"]) >= 1
    assert len(splits["dev"]) >= 1
    assert len(splits["test"]) >= 1
    assert len(splits["eval_final"]) == 2


def test_lsssdd_split_fixed_seeded_disjoint():
    names = [f"{scene:02d}_{i}_{j}.jpg" for scene in range(1, 16) for i in range(4) for j in range(10)]

    first = build_lsssdd_split(names, train_frac=0.9, seed=0)
    second = build_lsssdd_split(names, train_frac=0.9, seed=0)
    assert first == second  # fixed + seeded

    train, val = set(first["train"]), set(first["val"])
    assert not train & val
    assert train | val == set(names)
    assert len(first["val"]) == pytest.approx(0.1 * len(names), abs=1)


def test_compute_channel_stats_matches_known_values(tmp_path):
    rng = np.random.default_rng(0)
    vh = rng.normal(-25.0, 2.0, size=(2, 32, 32))
    vv = rng.normal(-15.0, 3.0, size=(2, 32, 32))
    paths = []
    for index in range(2):
        chip = np.stack([vh[index], vv[index]]).astype(np.float16)
        chip[0, 0, 0] = np.nan  # nodata pixel must be excluded
        path = tmp_path / f"scene{index:012d}00t_r0_c{index}.npy"
        np.save(path, chip)
        paths.append(path)

    stats = compute_channel_stats(paths)

    assert stats["channels"]["VH"]["mean"] == pytest.approx(-25.0, abs=0.3)
    assert stats["channels"]["VV"]["mean"] == pytest.approx(-15.0, abs=0.3)
    assert stats["channels"]["VH"]["std"] == pytest.approx(2.0, abs=0.3)
    assert stats["channels"]["VV"]["std"] == pytest.approx(3.0, abs=0.3)
    assert stats["n_pixels"][0] == 2 * 32 * 32 - 2  # one NaN per chip excluded
    assert stats["scenes"] == sorted({p.name.split("_r")[0] for p in paths})


def test_scene_split_map_rejects_duplicate_membership():
    with pytest.raises(ValueError):
        scene_split_map({"train": ["a"], "dev": ["a"]})
    assert scene_split_map({"train": ["a"], "dev": ["b"]}) == {
        "a": "train",
        "b": "dev",
    }
