"""Hardware-neutral contracts for the frozen 32-cell experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from src.runtime.io import sha256_file


EXPECTED_PRECISION = "32-true"
MICRO_BATCH = 16
GRADIENT_ACCUMULATION = 1
EFFECTIVE_BATCH = 16
FROZEN_PATHS = (
    "configs/detector.yaml",
    "src/eval/scorer.py",
    "data/splits.json",
    "data/stats.json",
)
FAMILY_ORDER = ("cnn", "vit")
FRACTION_ORDER = (1.0, 0.5, 0.25, 0.1)


@dataclass(frozen=True)
class Cell:
    """One core training cell derived from the active arm manifest."""

    init: str
    short: str
    track: str
    fraction: float
    seed: int = 0

    @property
    def exp_id(self) -> str:
        return f"{self.short}-f{int(round(self.fraction * 100))}-s{self.seed}"


def load_cells(repo: str | Path) -> list[Cell]:
    """Derive and validate the exact seed-0, 32-cell core matrix."""

    root = Path(repo)
    config = yaml.safe_load((root / "configs/arms.yaml").read_text(encoding="utf-8"))
    arms: Mapping[str, Mapping[str, str]] = config["arms"]
    if tuple(config["seeds"]["core"]) != (0,):
        raise RuntimeError("the core matrix requires the frozen seed-0-only recipe")
    if set(float(value) for value in config["label_fracs"]) != {
        0.1,
        0.25,
        0.5,
        1.0,
    }:
        raise RuntimeError("the core matrix requires exactly f10/f25/f50/f100")

    cells = [
        Cell(
            init=init,
            short=str(meta["short"]),
            track=str(meta["track"]),
            fraction=fraction,
        )
        for fraction in FRACTION_ORDER
        for track in FAMILY_ORDER
        for init, meta in arms.items()
        if str(meta["track"]) == track
    ]
    if len(cells) != 32 or len({cell.exp_id for cell in cells}) != 32:
        raise RuntimeError("active arms.yaml did not resolve to 32 unique core cells")
    return cells


def frozen_hashes(repo: str | Path) -> dict[str, str]:
    """Hash every immutable scientific contract."""

    root = Path(repo)
    return {relative: sha256_file(root / relative) for relative in FROZEN_PATHS}


def verify_expected_hashes(
    root: str | Path, expected: Mapping[str, str]
) -> dict[str, str]:
    """Hash explicitly expected files and fail on the first mismatch."""

    base = Path(root)
    actual: dict[str, str] = {}
    for relative, wanted in expected.items():
        path = base / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required regular file is absent: {path}")
        got = sha256_file(path)
        if got != wanted:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: expected {wanted}, got {got}"
            )
        actual[relative] = got
    return actual
