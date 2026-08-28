"""TRAIN/DEV-only evidence collection for the BES2 root-cause diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import socket
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.h100.lightning_contract import assert_launch_process_contract
from src.analysis.bes2_contract import (
    BES2ContractError,
    canonical_json,
    read_regular_json,
    sha256_file,
)
from src.data.datasets import (
    FineTuneDataset,
    _channel_stats,
    nested_fraction_scenes,
)
from src.data.transforms import normalize
from src.eval.ground_truth import ground_truth_from_labels
from src.eval.infer_scene import infer_scene
from src.eval.scorer import Metrics, PredictionPoint, score_dataset
from src.eval.threshold import apply_threshold, select_f1_threshold
from src.models.bes2_diagnostic import (
    DIAGNOSTIC_VARIANTS,
    state_identity,
    state_tensor_hashes,
)

SAMPLE_MANIFEST_SCHEMA = 1
SAMPLE_SEED = 0
TRAIN_SAMPLES_PER_CLASS = 32
DEV_WINDOWS_PER_SCENE = 8
PIXEL_SUBSAMPLE_STRIDE = 16
EXPECTED_DEV_SCENES = 8
EXPECTED_TRAIN_SCENES = 111
FINAL_FORBIDDEN_TOKENS = frozenset(
    {"test", "eval_final", "validation.csv", "final_eval"}
)


def _path_has_forbidden_token(path: Path) -> bool:
    return any(part.lower() in FINAL_FORBIDDEN_TOKENS for part in path.parts)


def _resolve_within(root: Path, path: Path, description: str) -> Path:
    if path.is_symlink():
        raise BES2ContractError(f"{description} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise BES2ContractError(
            f"{description} is absent or outside the diagnostic data view: {path}"
        ) from exc
    if _path_has_forbidden_token(resolved):
        raise BES2ContractError(
            f"{description} contains a forbidden held-out token: {resolved}"
        )
    return resolved


def validate_h100_ready(path: str | Path) -> dict[str, object]:
    ready_path = Path(path)
    payload = read_regular_json(ready_path, "canonical H100 readiness")
    strict = payload.get("strict_fp32")
    venv = payload.get("venv")
    staged = venv.get("staged_data_view") if isinstance(venv, Mapping) else None
    if (
        payload.get("schema") != 2
        or payload.get("status") != "ready"
        or not isinstance(strict, Mapping)
        or set(strict) != {
            "cuda_matmul_fp32_precision",
            "cudnn_conv_fp32_precision",
            "cudnn_rnn_fp32_precision",
        }
        or set(strict.values()) != {"ieee"}
        or not isinstance(venv, Mapping)
        or not isinstance(staged, Mapping)
        or set(staged) != {"path", "sha256", "receipt"}
    ):
        raise BES2ContractError("canonical H100 readiness contract is invalid")
    return {
        "path": str(ready_path.absolute()),
        "sha256": sha256_file(ready_path),
        "acceptance_uuid": payload.get("acceptance_uuid"),
        "source": payload.get("source"),
        "strict_fp32": dict(strict),
        "hardware": payload.get("hardware"),
        "venv": {
            key: venv.get(key)
            for key in (
                "path",
                "sha256",
                "venv_build_sha256",
                "base_python",
                "wheelhouse",
            )
        },
        "accepted_data_view": dict(staged),
        "base_payload": payload.get("base_payload"),
        "runtime_amendment": payload.get("runtime_amendment"),
    }


def validate_data_scope(
    *,
    repo: str | Path,
    data_view_root: str | Path,
    data_config: Mapping[str, object],
) -> dict[str, object]:
    root = Path(data_view_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise BES2ContractError(
            "diagnostic data-view root must be an absolute non-symlink directory"
        )
    root = root.resolve(strict=True)
    paths = data_config.get("paths")
    if not isinstance(paths, Mapping):
        raise BES2ContractError("diagnostic data config lacks paths")

    chips_path = Path(str(paths["chips"]))
    raw_path = Path(str(paths["raw_xview3"]))
    if not chips_path.is_absolute():
        chips_path = root / chips_path
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    chips = _resolve_within(root, chips_path, "TRAIN chips")
    raw = _resolve_within(root, raw_path, "DEV raster root")
    labels = _resolve_within(
        root,
        raw / "labels" / "train.csv",
        "TRAIN+DEV8 label CSV",
    )
    repo_root = Path(repo).resolve()
    splits_path = Path(str(paths["splits"]))
    stats_path = Path(str(paths["stats"]))
    if not splits_path.is_absolute():
        splits_path = repo_root / splits_path
    if not stats_path.is_absolute():
        stats_path = repo_root / stats_path
    if splits_path.resolve() != (repo_root / "data/splits.json").resolve():
        raise BES2ContractError("diagnostics require the frozen splits.json")
    if stats_path.resolve() != (repo_root / "data/stats.json").resolve():
        raise BES2ContractError("diagnostics require the frozen stats.json")

    splits = read_regular_json(splits_path, "frozen splits").get("splits")
    if not isinstance(splits, Mapping):
        raise BES2ContractError("frozen split mapping is absent")
    train = sorted(map(str, splits.get("train", ())))
    dev8 = sorted(map(str, splits.get("dev", ())))[:EXPECTED_DEV_SCENES]
    test = set(map(str, splits.get("test", ())))
    final = set(map(str, splits.get("eval_final", ())))
    if len(train) != EXPECTED_TRAIN_SCENES or len(dev8) != EXPECTED_DEV_SCENES:
        raise BES2ContractError("diagnostic TRAIN/DEV8 scene counts are invalid")
    if set(train) & (set(dev8) | test | final) or set(dev8) & (test | final):
        raise BES2ContractError("diagnostic TRAIN/DEV8 scope overlaps held-out scenes")

    chip_dirs = sorted(path.name for path in chips.iterdir() if path.is_dir())
    missing_train = sorted(set(train) - set(chip_dirs))
    unexpected_chip_dirs = sorted(set(chip_dirs) - set(train))
    if missing_train or unexpected_chip_dirs:
        raise BES2ContractError(
            "diagnostic chip scope is invalid: "
            f"missing_train={missing_train[:3]}, "
            f"unexpected={unexpected_chip_dirs[:3]}"
        )

    grd = _resolve_within(root, raw / "GRD", "DEV GRD root")
    raster_dirs = {path.name for path in grd.iterdir() if path.is_dir()}
    missing_dev = sorted(set(dev8) - raster_dirs)
    unexpected_rasters = sorted(raster_dirs - set(dev8))
    if missing_dev or unexpected_rasters:
        raise BES2ContractError(
            "diagnostic raster scope is invalid: "
            f"missing_dev8={missing_dev[:3]}, "
            f"unexpected={unexpected_rasters[:3]}"
        )

    label_scene_ids: set[str] = set()
    with labels.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "scene_id" not in (reader.fieldnames or []):
            raise BES2ContractError("diagnostic label CSV lacks scene_id")
        for row in reader:
            label_scene_ids.add(str(row["scene_id"]))
    authorized_labels = set(train) | set(dev8)
    if label_scene_ids != authorized_labels:
        raise BES2ContractError(
            "diagnostic labels are not the exact TRAIN+DEV8 scope"
        )

    return {
        "root": str(root),
        "chips_root": str(chips),
        "raw_root": str(raw),
        "labels_csv": str(labels),
        "splits_path": str(splits_path.resolve()),
        "splits_sha256": sha256_file(splits_path),
        "stats_path": str(stats_path.resolve()),
        "stats_sha256": sha256_file(stats_path),
        "train_scene_ids": train,
        "dev_scene_ids": dev8,
        "label_scene_count": len(label_scene_ids),
    }


def _stable_order(items: Iterable[tuple[str, Path]]) -> list[tuple[str, Path]]:
    return sorted(
        items,
        key=lambda item: hashlib.sha256(
            f"{SAMPLE_SEED}\0{item[0]}\0{item[1].name}".encode("utf-8")
        ).hexdigest(),
    )


def _foreground(sidecar: Path) -> bool:
    payload = read_regular_json(sidecar, "chip label sidecar")
    labels = payload.get("labels")
    if not isinstance(labels, list):
        raise BES2ContractError(f"chip sidecar labels are invalid: {sidecar}")
    return any(
        bool(label.get("is_vessel"))
        and str(label.get("confidence") or "").upper() in {"HIGH", "MEDIUM"}
        for label in labels
        if isinstance(label, Mapping)
    )


def _dev_origins(height: int, width: int, tile: int) -> list[tuple[int, int]]:
    def axis(length: int) -> list[int]:
        maximum = max(0, length - tile)
        return sorted(
            {
                int(round(value))
                for value in np.linspace(0, maximum, num=5)
            }
        )

    return [(row, col) for row in axis(height) for col in axis(width)]


def build_sample_manifest(
    *,
    scope: Mapping[str, object],
    output: str | Path,
    crop_px: int = 512,
) -> dict[str, object]:
    """Freeze one deterministic TRAIN/DEV8 image sample for all variants."""

    import rasterio
    from rasterio.windows import Window

    chips_root = Path(str(scope["chips_root"]))
    raw_root = Path(str(scope["raw_root"]))
    data_root = Path(str(scope["root"]))
    foreground: list[tuple[str, Path]] = []
    background: list[tuple[str, Path]] = []
    sample_train_scenes = nested_fraction_scenes(
        list(map(str, scope["train_scene_ids"])),
        0.5,
        frac_seed=0,
    )
    for scene_id in sample_train_scenes:
        scene_dir = chips_root / str(scene_id)
        for chip in sorted(scene_dir.glob("*.npy")):
            sidecar = chip.with_suffix(".json")
            target = foreground if _foreground(sidecar) else background
            target.append((str(scene_id), chip))

    selected_train = [
        ("foreground", item)
        for item in _stable_order(foreground)[:TRAIN_SAMPLES_PER_CLASS]
    ] + [
        ("background", item)
        for item in _stable_order(background)[:TRAIN_SAMPLES_PER_CLASS]
    ]
    if len(selected_train) != 2 * TRAIN_SAMPLES_PER_CLASS:
        raise BES2ContractError("insufficient TRAIN foreground/background samples")

    entries: list[dict[str, object]] = []
    for sample_class, (scene_id, chip) in selected_train:
        sidecar = chip.with_suffix(".json")
        array = np.load(chip, mmap_mode="r")
        if array.ndim != 3 or array.shape[0] != 2:
            raise BES2ContractError(f"invalid TRAIN chip shape: {chip}")
        row = max(0, (array.shape[-2] - crop_px) // 2)
        col = max(0, (array.shape[-1] - crop_px) // 2)
        entries.append(
            {
                "split": "train",
                "kind": "chip",
                "class": sample_class,
                "scene_id": scene_id,
                "path": chip.resolve().relative_to(data_root).as_posix(),
                "sidecar": sidecar.resolve().relative_to(data_root).as_posix(),
                "path_sha256": sha256_file(chip),
                "sidecar_sha256": sha256_file(sidecar),
                "row": row,
                "col": col,
                "height": crop_px,
                "width": crop_px,
            }
        )

    for scene_id in scope["dev_scene_ids"]:
        scene_dir = raw_root / "GRD" / str(scene_id)
        vh_path = scene_dir / "VH_dB.tif"
        vv_path = scene_dir / "VV_dB.tif"
        with rasterio.open(vh_path) as vh_ds, rasterio.open(vv_path) as vv_ds:
            if (vh_ds.height, vh_ds.width) != (vv_ds.height, vv_ds.width):
                raise BES2ContractError(f"DEV raster geometry mismatch: {scene_id}")
            selected = []
            for row, col in _dev_origins(vh_ds.height, vh_ds.width, crop_px):
                vh = vh_ds.read(
                    1, window=Window(col, row, crop_px, crop_px)
                ).astype(np.float32)
                vv = vv_ds.read(
                    1, window=Window(col, row, crop_px, crop_px)
                ).astype(np.float32)
                if vh.shape != (crop_px, crop_px) or vv.shape != vh.shape:
                    continue
                if ((vh == -32768.0) | ~np.isfinite(vh)).mean() > 0.98:
                    continue
                digest = hashlib.sha256()
                digest.update(vh.tobytes())
                digest.update(vv.tobytes())
                selected.append((row, col, digest.hexdigest()))
            if len(selected) < DEV_WINDOWS_PER_SCENE:
                raise BES2ContractError(
                    f"DEV scene {scene_id} lacks {DEV_WINDOWS_PER_SCENE} valid windows"
                )
            selected = _stable_order(
                [
                    (
                        str(scene_id),
                        Path(f"r{row}-c{col}-{digest}"),
                    )
                    for row, col, digest in selected
                ]
            )[:DEV_WINDOWS_PER_SCENE]
            for _, encoded in selected:
                pieces = encoded.name.split("-")
                row = int(pieces[0][1:])
                col = int(pieces[1][1:])
                digest = pieces[2]
                entries.append(
                    {
                        "split": "dev",
                        "kind": "raster-window",
                        "scene_id": str(scene_id),
                        "vh_path": vh_path.resolve().relative_to(data_root).as_posix(),
                        "vv_path": vv_path.resolve().relative_to(data_root).as_posix(),
                        "row": row,
                        "col": col,
                        "height": crop_px,
                        "width": crop_px,
                        "window_sha256": digest,
                    }
                )

    manifest = {
        "sample_manifest_schema": SAMPLE_MANIFEST_SCHEMA,
        "purpose": "bes2-shared-train-dev-input-audit",
        "seed": SAMPLE_SEED,
        "selection": {
            "train_per_class": TRAIN_SAMPLES_PER_CLASS,
            "train_fraction": 0.5,
            "train_fraction_seed": 0,
            "train_fraction_scene_ids_sha256": hashlib.sha256(
                canonical_json(sample_train_scenes)
            ).hexdigest(),
            "dev_windows_per_scene": DEV_WINDOWS_PER_SCENE,
            "crop_px": crop_px,
            "pixel_subsample_stride": PIXEL_SUBSAMPLE_STRIDE,
        },
        "scope": {
            "train_scene_count": len(scope["train_scene_ids"]),
            "dev_scene_ids": list(scope["dev_scene_ids"]),
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    from src.analysis.bes2_contract import write_new_immutable

    write_new_immutable(output, manifest)
    return manifest


def _load_manifest_image(
    entry: Mapping[str, object],
    *,
    data_root: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    kind = entry.get("kind")
    row = int(entry["row"])
    col = int(entry["col"])
    height = int(entry["height"])
    width = int(entry["width"])
    if kind == "chip":
        path = data_root / str(entry["path"])
        if sha256_file(path) != entry.get("path_sha256"):
            raise BES2ContractError(f"sample chip changed: {path}")
        source = np.load(path).astype(np.float32)
        window = source[:, row : row + height, col : col + width]
    elif kind == "raster-window":
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(data_root / str(entry["vh_path"])) as vh_ds:
            vh = vh_ds.read(1, window=Window(col, row, width, height))
        with rasterio.open(data_root / str(entry["vv_path"])) as vv_ds:
            vv = vv_ds.read(1, window=Window(col, row, width, height))
        digest = hashlib.sha256()
        digest.update(vh.astype(np.float32).tobytes())
        digest.update(vv.astype(np.float32).tobytes())
        if digest.hexdigest() != entry.get("window_sha256"):
            raise BES2ContractError("sample DEV raster window changed")
        window = np.stack([vh, vv]).astype(np.float32)
    else:
        raise BES2ContractError(f"unknown sample kind: {kind!r}")
    if window.shape != (2, height, width):
        raise BES2ContractError(f"sample image shape is invalid: {window.shape}")

    invalid = (
        ~np.isfinite(window[0])
        | ~np.isfinite(window[1])
        | (window[0] == -32768.0)
        | (window[1] == -32768.0)
    )
    raw = np.concatenate([window, window[0:1] - window[1:2]], axis=0)
    raw[:, invalid] = np.nan
    normalized = normalize(raw.copy(), mean, std)
    if not np.isfinite(normalized).all():
        raise BES2ContractError("normalized diagnostic sample is non-finite")
    return raw, normalized


def load_sample_images(
    *,
    manifest: Mapping[str, object],
    scope: Mapping[str, object],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    stats = read_regular_json(scope["stats_path"], "frozen statistics")
    mean, std = _channel_stats(stats)
    raw_images: list[np.ndarray] = []
    normalized_images: list[np.ndarray] = []
    for entry in manifest["entries"]:
        raw, normalized = _load_manifest_image(
            entry,
            data_root=Path(str(scope["root"])),
            mean=mean,
            std=std,
        )
        raw_images.append(raw)
        normalized_images.append(normalized)
    return raw_images, normalized_images


def _covariance_evidence(values: np.ndarray) -> dict[str, object]:
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 3:
        raise BES2ContractError("input covariance sample is invalid")
    covariance = np.cov(values, rowvar=False, ddof=1)
    correlation = np.corrcoef(values, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = max(float(eigenvalues[-1]), 1.0) * 1.0e-10
    mean = values.mean(axis=0)
    centered = values - mean
    coefficients = np.linalg.pinv(
        covariance[:2, :2], rcond=1.0e-12
    ) @ covariance[:2, 2]
    centered_residual = (
        centered[:, 2] - centered[:, :2] @ coefficients
    )
    raw_formula_residual = values[:, 2] - (values[:, 0] - values[:, 1])
    return {
        "sample_count": int(values.shape[0]),
        "mean": mean.tolist(),
        "variance": values.var(axis=0, ddof=1).tolist(),
        "covariance": covariance.tolist(),
        "correlation": correlation.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "effective_matrix_rank": int((eigenvalues > tolerance).sum()),
        "centered_linear_redundancy": {
            "third_from_first_two_coefficients": coefficients.tolist(),
            "max_abs_residual": float(
                np.max(np.abs(centered_residual))
            ),
            "rmse": float(np.sqrt(np.mean(centered_residual**2))),
        },
        "raw_vh_minus_vv_formula": {
            "max_abs_residual": float(
                np.max(np.abs(raw_formula_residual))
            ),
            "rmse": float(np.sqrt(np.mean(raw_formula_residual**2))),
        },
    }


def input_covariance_statistics(
    raw_images: Sequence[np.ndarray],
    normalized_images: Sequence[np.ndarray],
) -> dict[str, object]:
    if len(raw_images) != len(normalized_images) or not raw_images:
        raise BES2ContractError("input image sample is empty or misaligned")
    raw_rows: list[np.ndarray] = []
    normalized_rows: list[np.ndarray] = []
    step = PIXEL_SUBSAMPLE_STRIDE
    for raw, normalized in zip(raw_images, normalized_images, strict=True):
        raw_values = raw[:, ::step, ::step].reshape(3, -1).T
        normalized_values = normalized[:, ::step, ::step].reshape(3, -1).T
        finite = np.isfinite(raw_values).all(axis=1)
        raw_rows.append(raw_values[finite])
        normalized_rows.append(normalized_values[finite])
    return {
        "channels": ["VH", "VV", "VH_minus_VV"],
        "pixel_subsample_stride": step,
        "raw_db": _covariance_evidence(np.concatenate(raw_rows)),
        "normalized": _covariance_evidence(np.concatenate(normalized_rows)),
    }


def _array_distribution(values: np.ndarray) -> dict[str, object]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(flat).all() or flat.size == 0:
        raise BES2ContractError("diagnostic numeric distribution is invalid")
    return {
        "count": int(flat.size),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "mean": float(flat.mean()),
        "standard_deviation": float(flat.std()),
        "quantiles": {
            str(level): float(np.quantile(flat, level))
            for level in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
    }


def equivalent_two_channel_stem(
    weight: torch.Tensor,
    covariance: Sequence[Sequence[float]],
    means: Sequence[float] | None = None,
) -> dict[str, object]:
    """Collapse a covariance-exact redundant third channel onto the first two."""

    source = weight.detach().cpu().double().numpy()
    matrix = np.asarray(covariance, dtype=np.float64)
    mean = (
        np.zeros(3, dtype=np.float64)
        if means is None
        else np.asarray(means, dtype=np.float64)
    )
    if source.ndim != 4 or source.shape[1] != 3:
        raise BES2ContractError(
            f"stem audit requires OIHW with three inputs, got {source.shape}"
        )
    if (
        matrix.shape != (3, 3)
        or mean.shape != (3,)
        or not np.isfinite(source).all()
        or not np.isfinite(matrix).all()
        or not np.isfinite(mean).all()
        or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-10)
    ):
        raise BES2ContractError("stem audit covariance is invalid")

    independent_covariance = matrix[:2, :2]
    cross_covariance = matrix[:2, 2]
    coefficients = np.linalg.pinv(
        independent_covariance, rcond=1.0e-12
    ) @ cross_covariance
    residual_variance = float(
        matrix[2, 2] - cross_covariance @ coefficients
    )
    residual_tolerance = 1.0e-8 * max(1.0, abs(float(matrix[2, 2])))
    if abs(residual_variance) > residual_tolerance:
        raise BES2ContractError(
            "stem audit third channel is not covariance-redundant"
        )
    effective = np.stack(
        [
            source[:, 0] + coefficients[0] * source[:, 2],
            source[:, 1] + coefficients[1] * source[:, 2],
        ],
        axis=1,
    )
    affine_intercept = float(mean[2] - coefficients @ mean[:2])
    effective_bias_shift = (
        affine_intercept * source[:, 2].sum(axis=(1, 2))
    )
    source_flat = source.transpose(1, 0, 2, 3).reshape(3, -1)
    effective_flat = effective.transpose(1, 0, 2, 3).reshape(2, -1)
    source_gram = source_flat @ source_flat.T
    effective_gram = effective_flat @ effective_flat.T
    source_eigenvalues = np.linalg.eigvalsh(source_gram)
    effective_eigenvalues = np.linalg.eigvalsh(effective_gram)

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return (
            float(np.dot(left.reshape(-1), right.reshape(-1)) / denominator)
            if denominator > 0.0
            else 0.0
        )

    def cancellation(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) + np.linalg.norm(right))
        return (
            float(np.linalg.norm(left + right) / denominator)
            if denominator > 0.0
            else 0.0
        )

    three_channel_variance = np.einsum(
        "oihw,ij,ojhw->o", source, matrix, source
    )
    two_channel_variance = np.einsum(
        "oihw,ij,ojhw->o",
        effective,
        independent_covariance,
        effective,
    )
    if (
        (three_channel_variance < -1.0e-8).any()
        or (two_channel_variance < -1.0e-8).any()
    ):
        raise BES2ContractError("stem audit predicted a negative variance")
    three_channel_variance = np.maximum(three_channel_variance, 0.0)
    two_channel_variance = np.maximum(two_channel_variance, 0.0)
    source_norms = np.linalg.norm(source_flat, axis=1)
    effective_norms = np.linalg.norm(effective_flat, axis=1)
    digest = hashlib.sha256()
    digest.update(str(effective.dtype).encode("ascii"))
    digest.update(str(tuple(effective.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(effective).tobytes())
    minimum = float(source_eigenvalues.min())
    return {
        "relation": (
            "centered x2 = a*x0 + b*x1; "
            "K0_effective = K0 + a*K2; K1_effective = K1 + b*K2"
        ),
        "covariance_relation": {
            "third_from_first_two_coefficients": coefficients.tolist(),
            "third_channel_affine_intercept": affine_intercept,
            "third_channel_residual_variance": residual_variance,
            "residual_variance_tolerance": residual_tolerance,
            "raw_vh_vv_difference_identity": bool(
                np.allclose(
                    coefficients,
                    np.asarray([1.0, -1.0]),
                    rtol=0.0,
                    atol=1.0e-6,
                )
            ),
        },
        "effective_bias_shift": _array_distribution(effective_bias_shift),
        "source_shape": list(source.shape),
        "effective_shape": list(effective.shape),
        "effective_tensor_sha256": digest.hexdigest(),
        "source_channel_norms": source_norms.tolist(),
        "effective_channel_norms": effective_norms.tolist(),
        "source_channel_gram": source_gram.tolist(),
        "source_channel_gram_eigenvalues": source_eigenvalues.tolist(),
        "source_channel_gram_condition_number": (
            float(source_eigenvalues.max() / minimum)
            if minimum > 0.0
            else None
        ),
        "effective_channel_gram": effective_gram.tolist(),
        "effective_channel_gram_eigenvalues": effective_eigenvalues.tolist(),
        "cancellation": {
            "first_channel_combination_ratio": cancellation(
                source[:, 0], coefficients[0] * source[:, 2]
            ),
            "second_channel_combination_ratio": cancellation(
                source[:, 1], coefficients[1] * source[:, 2]
            ),
            "cosine_first_vs_weighted_third": cosine(
                source[:, 0], coefficients[0] * source[:, 2]
            ),
            "cosine_second_vs_weighted_third": cosine(
                source[:, 1], coefficients[1] * source[:, 2]
            ),
        },
        "predicted_output_variance": {
            "three_channel": _array_distribution(three_channel_variance),
            "two_channel_projection": _array_distribution(
                two_channel_variance
            ),
            "total_projection_ratio": (
                float(two_channel_variance.sum() / three_channel_variance.sum())
                if three_channel_variance.sum() > 0.0
                else 0.0
            ),
            "per_output_absolute_difference": _array_distribution(
                np.abs(two_channel_variance - three_channel_variance)
            ),
        },
    }


class _ActivationAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum: torch.Tensor | None = None
        self.sum_sq: torch.Tensor | None = None
        self.nonfinite = 0
        self.total = 0
        self.pooled: list[torch.Tensor] = []

    def add(self, value: torch.Tensor) -> None:
        detached = value.detach().float()
        self.total += detached.numel()
        finite = torch.isfinite(detached)
        self.nonfinite += int((~finite).sum().item())
        if not finite.all():
            return
        if detached.ndim == 2:
            detached = detached[:, :, None, None]
        if detached.ndim != 4:
            raise BES2ContractError(
                f"activation hook expected NCHW, got {tuple(detached.shape)}"
            )
        reduce_dims = (0, 2, 3)
        channel_sum = detached.sum(dim=reduce_dims).double().cpu()
        channel_sum_sq = detached.square().sum(dim=reduce_dims).double().cpu()
        per_channel_count = detached.shape[0] * detached.shape[2] * detached.shape[3]
        self.count += per_channel_count
        self.sum = channel_sum if self.sum is None else self.sum + channel_sum
        self.sum_sq = (
            channel_sum_sq
            if self.sum_sq is None
            else self.sum_sq + channel_sum_sq
        )
        self.pooled.append(detached.mean(dim=(2, 3)).cpu())

    def finish(self) -> dict[str, object]:
        if (
            self.nonfinite
            or self.count <= 0
            or self.sum is None
            or self.sum_sq is None
            or not self.pooled
        ):
            raise BES2ContractError(
                f"activation sample is non-finite or empty: {self.nonfinite}/{self.total}"
            )
        mean = self.sum / self.count
        variance = torch.clamp(self.sum_sq / self.count - mean.square(), min=0)
        rows = torch.cat(self.pooled, dim=0).double()
        centered = rows - rows.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        energy = singular.square()
        total_energy = float(energy.sum())
        if total_energy > 0.0:
            probabilities = energy / total_energy
            entropy_rank = float(
                torch.exp(
                    -(probabilities * torch.log(probabilities.clamp_min(1.0e-30))).sum()
                )
            )
            participation = float(
                energy.sum().square() / energy.square().sum().clamp_min(1.0e-30)
            )
        else:
            entropy_rank = 0.0
            participation = 0.0
        return {
            "sample_count": int(rows.shape[0]),
            "channel_count": int(rows.shape[1]),
            "value_count_per_channel": self.count,
            "global_mean": float(mean.mean()),
            "global_variance": float(variance.mean()),
            "per_channel_mean": mean.tolist(),
            "per_channel_variance": variance.tolist(),
            "effective_rank_entropy": entropy_rank,
            "effective_rank_participation": participation,
            "singular_values": singular.tolist(),
            "nonfinite_count": self.nonfinite,
            "total_value_count": self.total,
        }


def activation_statistics(
    model: torch.nn.Module,
    normalized_images: Sequence[np.ndarray],
    *,
    device: str | torch.device,
    batch_size: int = 8,
) -> dict[str, object]:
    device = torch.device(device)
    model = model.to(device).eval()
    backbone = model.backbone.model
    layers: dict[str, torch.nn.Module] = {
        "post_first_convolution": backbone.stem[0],
        "post_stem": backbone.stem,
        **{
            f"post_stage_{index}": stage
            for index, stage in enumerate(backbone.stages)
        },
    }
    accumulators = {name: _ActivationAccumulator() for name in layers}
    pre_stem = _ActivationAccumulator()
    handles = [
        layer.register_forward_hook(
            lambda _module, _inputs, output, name=name: accumulators[name].add(
                output
            )
        )
        for name, layer in layers.items()
    ]
    try:
        with torch.no_grad():
            for start in range(0, len(normalized_images), batch_size):
                batch = torch.from_numpy(
                    np.stack(normalized_images[start : start + batch_size])
                ).to(device)
                pre_stem.add(batch)
                output = model.backbone(batch)
                if not torch.isfinite(output).all():
                    raise BES2ContractError("backbone activation output is non-finite")
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "pre_stem": pre_stem.finish(),
        **{name: accumulator.finish() for name, accumulator in accumulators.items()},
    }


def batch16_forward_backward_probe(
    model: torch.nn.Module,
    dataset: FineTuneDataset,
    indices: Sequence[int],
    *,
    device: str | torch.device,
) -> dict[str, object]:
    if len(indices) != 16:
        raise BES2ContractError("batch probe requires exactly 16 TRAIN samples")
    device = torch.device(device)
    samples = [dataset[index] for index in indices]
    batch = {
        key: torch.stack([sample[key] for sample in samples]).to(device)
        for key in ("image", "heatmap", "mask")
    }
    model = model.to(device).train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    loss = model.training_step(batch, 0)
    if not torch.isfinite(loss):
        raise BES2ContractError("batch-16 diagnostic loss is non-finite")
    loss.backward()
    nonfinite_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    result = {
        "batch_size": 16,
        "loss": float(loss.detach().cpu()),
        "nonfinite_gradient_names": nonfinite_gradients,
        "peak_memory_allocated_bytes": peak,
    }
    model.zero_grad(set_to_none=True)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if nonfinite_gradients:
        raise BES2ContractError(
            "batch-16 diagnostic gradients are non-finite: "
            + ", ".join(nonfinite_gradients[:8])
        )
    return result


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _layer_group(name: str) -> str:
    if name.startswith("backbone.model.stem"):
        return "stem"
    for index in range(4):
        if name.startswith(f"backbone.model.stages.{index}"):
            return f"stage_{index}"
    if name.startswith("head."):
        return "detector_head"
    return "other"


def layer_drift(
    initial_model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("state_dict"), Mapping
    ):
        raise BES2ContractError("diagnostic Lightning checkpoint is invalid")
    best_state = checkpoint["state_dict"]
    initial_state = initial_model.state_dict()
    if set(best_state) != set(initial_state):
        missing = sorted(set(initial_state) - set(best_state))
        unexpected = sorted(set(best_state) - set(initial_state))
        raise BES2ContractError(
            f"checkpoint/model state mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )

    parameter_names = {
        id(parameter): name for name, parameter in initial_model.named_parameters()
    }
    lr_scale: dict[str, float] = {}
    for group in initial_model._layer_decay_param_groups():
        scale = float(group.get("lr_scale", 1.0))
        for parameter in group["params"]:
            name = parameter_names.get(id(parameter))
            if name is None:
                raise BES2ContractError("layer-decay group contains unknown parameter")
            lr_scale[name] = scale

    tensors: dict[str, dict[str, object]] = {}
    aggregate: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "tensor_count": 0,
            "parameter_count": 0,
            "initial_norm_sq": 0.0,
            "best_norm_sq": 0.0,
            "update_norm_sq": 0.0,
        }
    )
    for name, initial in initial_state.items():
        best = best_state[name]
        if not initial.is_floating_point():
            continue
        initial_f = initial.detach().float().reshape(-1)
        best_f = best.detach().float().reshape(-1)
        delta = best_f - initial_f
        initial_norm = float(torch.linalg.vector_norm(initial_f))
        best_norm = float(torch.linalg.vector_norm(best_f))
        update_norm = float(torch.linalg.vector_norm(delta))
        denominator = initial_norm * best_norm
        cosine = (
            float(torch.dot(initial_f, best_f)) / denominator
            if denominator > 0.0
            else None
        )
        group_name = _layer_group(name)
        tensors[name] = {
            "group": group_name,
            "shape": list(initial.shape),
            "parameter_count": initial.numel(),
            "initial_sha256": _tensor_hash(initial),
            "best_sha256": _tensor_hash(best),
            "initial_norm": initial_norm,
            "best_norm": best_norm,
            "update_norm": update_norm,
            "relative_update_norm": (
                update_norm / initial_norm if initial_norm > 0.0 else None
            ),
            "cosine_similarity": cosine,
            "lr_scale": lr_scale.get(name),
        }
        item = aggregate[group_name]
        item["tensor_count"] += 1
        item["parameter_count"] += initial.numel()
        item["initial_norm_sq"] += initial_norm**2
        item["best_norm_sq"] += best_norm**2
        item["update_norm_sq"] += update_norm**2

    grouped = {}
    for group_name, item in sorted(aggregate.items()):
        initial_norm = math.sqrt(float(item.pop("initial_norm_sq")))
        best_norm = math.sqrt(float(item.pop("best_norm_sq")))
        update_norm = math.sqrt(float(item.pop("update_norm_sq")))
        grouped[group_name] = {
            **item,
            "initial_norm": initial_norm,
            "best_norm": best_norm,
            "update_norm": update_norm,
            "relative_update_norm": (
                update_norm / initial_norm if initial_norm > 0.0 else None
            ),
        }
    first_conv = "backbone.model.stem.0.weight"
    if first_conv not in tensors:
        raise BES2ContractError("first-convolution drift evidence is absent")
    return {
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "initial_state_sha256": state_identity(state_tensor_hashes(initial_model)),
        "tensor_count": len(tensors),
        "tensors": tensors,
        "groups": grouped,
        "first_convolution": tensors[first_conv],
    }


def _metrics_payload(metrics: Metrics) -> dict[str, object]:
    return {
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "ignored_predictions": metrics.ignored_predictions,
    }


def _prediction_distribution(
    predictions: Mapping[str, Sequence[PredictionPoint]],
) -> dict[str, object]:
    scores = sorted(
        [
            float(point.score)
            for points in predictions.values()
            for point in points
        ],
        reverse=True,
    )
    array = np.asarray(scores, dtype=np.float64)
    quantiles = {}
    if scores:
        quantiles = {
            str(level): float(np.quantile(array, level))
            for level in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
        }
    return {
        "candidate_count": len(scores),
        "scores_descending": scores,
        "mean": float(array.mean()) if scores else None,
        "standard_deviation": float(array.std()) if scores else None,
        "quantiles": quantiles,
        "per_scene_candidate_count": {
            scene_id: len(points)
            for scene_id, points in sorted(predictions.items())
        },
    }


def _match_distance_distribution(selected) -> dict[str, object]:
    by_outcome: dict[str, list[float]] = defaultdict(list)
    per_scene: dict[str, dict[str, object]] = {}
    for scene_id, result in sorted(selected.scene_results.items()):
        scene_outcomes: dict[str, list[float]] = defaultdict(list)
        for match in result.matches:
            if match.distance_m is None:
                continue
            distance = float(match.distance_m)
            if not math.isfinite(distance) or distance < 0.0:
                raise BES2ContractError("frozen scorer returned invalid distance")
            by_outcome[match.outcome].append(distance)
            scene_outcomes[match.outcome].append(distance)
        per_scene[scene_id] = {
            outcome: _array_distribution(np.asarray(values))
            for outcome, values in sorted(scene_outcomes.items())
        }
    return {
        "source": "frozen-scorer-selected-threshold-matches",
        "by_outcome": {
            outcome: _array_distribution(np.asarray(values))
            for outcome, values in sorted(by_outcome.items())
        },
        "per_scene": per_scene,
    }


def collect_dev_evidence(
    model: torch.nn.Module,
    *,
    data_config: Mapping[str, object],
    detector_config: Mapping[str, object],
    best_dev: Mapping[str, object],
    device: str | torch.device,
) -> dict[str, object]:
    import pandas as pd

    paths = data_config["paths"]
    splits = read_regular_json(paths["splits"], "frozen splits")["splits"]
    scene_ids = sorted(splits["dev"])[:EXPECTED_DEV_SCENES]
    if len(scene_ids) != EXPECTED_DEV_SCENES:
        raise BES2ContractError("fixed DEV8 scope is invalid")
    labels_path = Path(paths["raw_xview3"]) / "labels" / "train.csv"
    if _path_has_forbidden_token(labels_path):
        raise BES2ContractError("DEV evidence attempted a held-out label path")
    labels = pd.read_csv(labels_path)
    stats = read_regular_json(paths["stats"], "frozen statistics")

    ground_truth: dict[str, list] = {}
    predictions: dict[str, list[PredictionPoint]] = {}
    for scene_id in scene_ids:
        rows = labels[labels["scene_id"] == scene_id].to_dict(orient="records")
        ground_truth[scene_id] = ground_truth_from_labels(rows)
        predictions[scene_id] = infer_scene(
            model,
            Path(paths["raw_xview3"]) / "GRD" / scene_id,
            stats=stats,
            tau=float(detector_config["decode"]["candidate_floor"]),
            d_nms_m=float(detector_config["decode"]["d_nms_m"]),
            tile_px=int(detector_config["eval"]["tile_px"]),
            tile_stride_px=int(detector_config["eval"]["tile_stride_px"]),
            batch_size=int(detector_config["eval"]["infer_batch"]),
            device=device,
            precision=str(detector_config["schedule"]["precision"]),
        )

    unique_scores = sorted(
        {
            float(point.score)
            for points in predictions.values()
            for point in points
        },
        reverse=True,
    )
    curve = []
    for threshold in unique_scores:
        result = score_dataset(
            ground_truth,
            apply_threshold(predictions, threshold),
        )
        curve.append(
            {
                "threshold": threshold,
                **_metrics_payload(result.aggregate),
            }
        )
    selected_threshold = select_f1_threshold(ground_truth, predictions)
    selected = score_dataset(
        ground_truth,
        apply_threshold(predictions, selected_threshold),
    )
    selected_operating_point = {
        "epoch": int(best_dev["epoch"]),
        **_metrics_payload(selected.aggregate),
        "threshold": selected_threshold,
        "n_candidates": sum(map(len, predictions.values())),
    }
    if selected_operating_point != dict(best_dev):
        raise BES2ContractError(
            "replayed best checkpoint does not reproduce its bound DEV operating point"
        )

    per_scene = {
        scene_id: _metrics_payload(result.aggregate)
        for scene_id, result in sorted(selected.scene_results.items())
    }
    leave_one_out = {}
    for omitted in scene_ids:
        retained_gt = {
            key: value for key, value in ground_truth.items() if key != omitted
        }
        retained_predictions = {
            key: value for key, value in predictions.items() if key != omitted
        }
        threshold = select_f1_threshold(retained_gt, retained_predictions)
        retained_score = score_dataset(
            retained_gt,
            apply_threshold(retained_predictions, threshold),
        )
        omitted_score = score_dataset(
            {omitted: ground_truth[omitted]},
            apply_threshold({omitted: predictions[omitted]}, threshold),
        )
        leave_one_out[omitted] = {
            "threshold": threshold,
            "retained_seven": _metrics_payload(retained_score.aggregate),
            "omitted_scene": _metrics_payload(omitted_score.aggregate),
        }

    return {
        "scene_ids": scene_ids,
        "candidate_floor": float(detector_config["decode"]["candidate_floor"]),
        "selected_operating_point": selected_operating_point,
        "candidate_threshold_curve": curve,
        "score_distribution": _prediction_distribution(predictions),
        "matched_localization_distance_m": _match_distance_distribution(selected),
        "per_scene_at_selected_threshold": per_scene,
        "leave_one_dev_scene_out": leave_one_out,
    }


def runtime_provenance() -> dict[str, object]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {
            "name": properties.name,
            "compute_capability": [
                properties.major,
                properties.minor,
            ],
            "total_memory_bytes": properties.total_memory,
        }
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "slurm": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_RESTART_COUNT",
                "SLURM_CLUSTER_NAME",
                "SLURM_JOB_NODELIST",
            )
        },
        "strict_fp32": assert_launch_process_contract(),
    }


def manifest_train_indices(
    manifest: Mapping[str, object],
    dataset: FineTuneDataset,
) -> list[int]:
    by_path = {
        path.resolve(): index for index, path in enumerate(dataset.chip_paths)
    }
    indices = []
    for entry in manifest["entries"]:
        if entry.get("kind") != "chip":
            continue
        # The fixed covariance sample spans all 111 TRAIN scenes, while the
        # audit's forward/backward probe intentionally uses the f50 subset.
        # Skip fixed samples outside that subset, but reject ambiguous suffix
        # matches and require a complete batch from the retained entries.
        candidates = [
            path
            for path in by_path
            if path.as_posix().endswith(str(entry["path"]))
        ]
        if not candidates:
            continue
        if len(candidates) != 1:
            raise BES2ContractError(
                f"could not bind manifest chip to FineTuneDataset: {entry['path']}"
            )
        indices.append(by_path[candidates[0]])
    if len(indices) < 16:
        raise BES2ContractError("sample manifest lacks 16 TRAIN dataset entries")
    return indices
