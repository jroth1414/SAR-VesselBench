"""Arm R3 — LocateAnything-3B zero-shot probe (DEVPLAN P4.6). OFF the curves.

Zero-shot open-vocabulary grounding on ~200 labeled DEV chips with the
plan's prompts {"ship", "vessel", "boat"}: each chip is rendered exactly as
the study input (uint8 [VH, VV, VH-VV], frozen-stats scaling), the VLM is
asked to locate each prompt, returned box centers become scored candidate
points (score = 1/rank — the generative decoder emits no calibrated
confidence), and per-prompt results go through the SACRED scorer with a
P2.2b-style threshold sweep. Reported separately; DROPPABLE (license:
custom NVIDIA research license — recorded in data/weights/locateanything).

The checkpoint retains its published bf16 dtype. A V100 smoke gate passed
before the active fresh rerun; do not change the reference precision recipe.
Coordinate parsing is regex-based over the generated text (the model's own
README parses its output the same way); ``--smoke N`` probes N chips and
prints raw generations for eyeball verification before the full run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.references.runtime_provenance import (
    add_runtime_provenance_arguments,
    begin_reference_execution,
    finish_reference_execution,
    load_runtime_inputs,
    publish_reference_result,
    result_directory,
)

PROMPTS = ("ship", "vessel", "boat")
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
MODEL_PAYLOAD_FILE_COUNT = 39
MODEL_PAYLOAD_BYTES = 7_667_993_387
MODEL_PAYLOAD_SHA256 = (
    "4b097656ce640f7b6cc446f873af4e8e3a489c8ed27fae5ed67337f48c8318f9"
)
RESULT_SCHEMA = 2
EXPECTED_CHIP_SIZE = 800
SAMPLE_POLICY = "sorted-foreground-dev-chips-even-stride"
SAMPLE_ALGORITHM_VERSION = "floor-index-v1"
EXPECTED_ELIGIBLE_CANDIDATES = 1525
EXPECTED_GSD_M = 10.0
SCENE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# The model card's canonical detection template (README):
#   prompt = f"Locate all the instances that matches the following description: {cats}."
QUESTION = "Locate all the instances that matches the following description: {prompt}."
# The model emits Qwen-style grounding tokens: <box><x1><y1><x2><y2></box>
# with coordinates normalized to 0-1000 (or <box>None</box> for no hits).
BOX_GROUP = re.compile(r"<box>((?:<\d+>)+)</box>")
COORD = re.compile(r"<(\d+)>")


def parse_box_centers(
    text: str, *, image_w: float = 800.0, image_h: float = 800.0
) -> list[tuple[float, float]]:
    """Generated grounding text -> (col, row) box centers in image pixels.

    Parses the model's literal output — including degenerate whole-image
    boxes — with no filtering: R3 is a zero-shot reference and is scored
    as-is through the SACRED scorer.
    """

    centers = []
    for match in BOX_GROUP.finditer(text):
        nums = [int(c) for c in COORD.findall(match.group(1))]
        for i in range(0, len(nums) - 3, 4):
            x1, y1, x2, y2 = nums[i : i + 4]
            centers.append(
                (
                    ((x1 + x2) / 2.0) / 1000.0 * image_w,
                    ((y1 + y2) / 2.0) / 1000.0 * image_h,
                )
            )
    return centers


def chip_scene_metadata(
    sidecar: Mapping[str, object],
    *,
    expected_scene_id: str | None = None,
) -> tuple[str, int, int, int, float]:
    """Validate one chip sidecar and return its scene coordinate contract."""

    if not isinstance(sidecar, Mapping):
        raise RuntimeError("R3 chip sidecar must be a mapping")
    scene_id = sidecar.get("scene_id")
    if not isinstance(scene_id, str) or not SCENE_ID.fullmatch(scene_id):
        raise RuntimeError("R3 chip sidecar scene_id is missing or unsafe")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise RuntimeError(
            "R3 chip sidecar scene_id does not match its chip-directory scene"
        )

    origins = {}
    for field in ("row0", "col0"):
        value = sidecar.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"R3 chip sidecar {field} must be a non-negative integer"
            )
        origins[field] = value

    chip_size = sidecar.get("chip_size")
    if (
        isinstance(chip_size, bool)
        or not isinstance(chip_size, int)
        or chip_size != EXPECTED_CHIP_SIZE
    ):
        raise RuntimeError(f"R3 chip sidecar chip_size must equal {EXPECTED_CHIP_SIZE}")

    gsd = sidecar.get("gsd_m")
    if isinstance(gsd, bool) or not isinstance(gsd, (int, float)):
        raise RuntimeError("R3 chip sidecar gsd_m must be a finite number")
    gsd_m = float(gsd)
    if not math.isfinite(gsd_m) or not math.isclose(
        gsd_m, EXPECTED_GSD_M, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            f"R3 chip sidecar gsd_m must equal {EXPECTED_GSD_M}"
        )
    return (
        scene_id,
        origins["row0"],
        origins["col0"],
        chip_size,
        gsd_m,
    )


class SceneShoreDistanceCache:
    """Map chip-local centers through one canonical scene bathymetry at a time."""

    def __init__(self, raw_grd_root: str | Path) -> None:
        self.raw_grd_root = Path(raw_grd_root)
        self._scene_id: str | None = None
        self._transform = None
        self._shore = None

    def _load_scene(self, scene_id: str):
        import rasterio

        from src.eval.infer_scene import ShoreDistance

        scene_dir = self.raw_grd_root / scene_id
        vh_path = scene_dir / "VH_dB.tif"
        bathy_path = scene_dir / "bathymetry.tif"
        if vh_path.is_symlink() or not vh_path.is_file():
            raise RuntimeError(f"R3 scene VH raster is missing or linked: {vh_path}")
        if bathy_path.is_symlink() or not bathy_path.is_file():
            raise RuntimeError(
                f"R3 scene bathymetry raster is missing or linked: {bathy_path}"
            )
        with rasterio.open(vh_path) as dataset:
            transform = dataset.transform
            x_gsd, y_gsd = map(abs, dataset.res)
        if not (
            math.isclose(x_gsd, EXPECTED_GSD_M, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(y_gsd, EXPECTED_GSD_M, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise RuntimeError(
                f"R3 scene raster GSD must equal {EXPECTED_GSD_M} m: {vh_path}"
            )
        shore = ShoreDistance(scene_dir)
        if not shore.available:
            raise RuntimeError(f"R3 scene bathymetry is unavailable: {bathy_path}")
        return transform, shore

    def _scene_context(self, scene_id: str):
        if scene_id != self._scene_id:
            # Drop the prior distance array before loading the next scene.
            self._scene_id = None
            self._transform = None
            self._shore = None
            transform, shore = self._load_scene(scene_id)
            self._scene_id = scene_id
            self._transform = transform
            self._shore = shore
        return self._transform, self._shore

    def predictions_for_centers(
        self,
        centers: Sequence[tuple[float, float]],
        *,
        sidecar: Mapping[str, object],
        expected_scene_id: str,
    ):
        """Build chip-local scorer points with canonical scene shore distances."""

        from src.eval.scorer import PredictionPoint

        scene_id, row0, col0, _chip_size, gsd_m = chip_scene_metadata(
            sidecar, expected_scene_id=expected_scene_id
        )
        transform, shore = self._scene_context(scene_id)
        predictions = []
        for rank, (col, row) in enumerate(centers):
            if (
                isinstance(col, bool)
                or isinstance(row, bool)
                or not isinstance(col, (int, float))
                or not isinstance(row, (int, float))
                or not math.isfinite(float(col))
                or not math.isfinite(float(row))
            ):
                raise RuntimeError("R3 generated box center must be finite")
            col_px, row_px = float(col), float(row)
            x_geo, y_geo = transform * (col0 + col_px, row0 + row_px)
            distance_km = shore.lookup_km(float(x_geo), float(y_geo))
            if (
                distance_km is None
                or not math.isfinite(distance_km)
                or distance_km < 0.0
            ):
                raise RuntimeError(
                    "R3 canonical prediction shore distance is unavailable"
                )
            predictions.append(
                PredictionPoint(
                    x_m=col_px * gsd_m,
                    y_m=row_px * gsd_m,
                    score=1.0 / (rank + 1),
                    distance_from_shore_km=distance_km,
                )
            )
        return predictions



def chip_image(chip_path: Path, mean: np.ndarray, std: np.ndarray):
    from PIL import Image

    from src.references.yolo26_ref import _chip_to_uint8

    chips = np.load(chip_path).astype(np.float32)
    return Image.fromarray(_chip_to_uint8(chips, mean, std))


def chip_ground_truth_from_labels(labels: Sequence[Mapping[str, object]]):
    """Convert chip-local labels through the one shared evaluation contract."""

    from src.eval.ground_truth import ground_truth_from_labels

    adapted = []
    for label in labels:
        row = dict(label)
        row["detect_scene_column"] = label["chip_col"]
        row["detect_scene_row"] = label["chip_row"]
        adapted.append(row)
    return ground_truth_from_labels(adapted)


def chip_label_counts(labels: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Count every category, evaluating all rows so malformed data fails closed."""

    from src.eval.ground_truth import classify_label

    counts = {"positive": 0, "background": 0, "ignore": 0}
    for label in labels:
        counts[classify_label(label)] += 1
    return counts


def _sha256_file(path: Path, chunk_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _metrics_payload(metrics) -> dict[str, int | float]:
    return {
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "ignored_predictions": metrics.ignored_predictions,
    }


def select_evenly_strided(candidates: Sequence[object], n: int) -> list[object]:
    """Select exactly n deterministic positions spanning sorted candidates."""

    if n <= 0:
        raise ValueError("sample size must be positive")
    if len(candidates) < n:
        raise RuntimeError(
            f"R3 requires {n} eligible chips but found only {len(candidates)}"
        )
    indices = [(index * len(candidates)) // n for index in range(n)]
    if len(set(indices)) != n:
        raise AssertionError("floor-index sampling produced duplicate positions")
    return [candidates[index] for index in indices]


def _manifest_sha256(entries: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_model_payload(
    root: Path,
    *,
    expected_file_count: int = MODEL_PAYLOAD_FILE_COUNT,
    expected_bytes: int = MODEL_PAYLOAD_BYTES,
    expected_sha256: str = MODEL_PAYLOAD_SHA256,
) -> str:
    """Hash every non-cache model byte and reject missing, extra, or linked files."""

    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("LocateAnything payload must be a regular directory")
    entries = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"LocateAnything payload contains a symlink: {relative}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": relative.as_posix(),
                "size": size,
                "sha256": _sha256_file(path),
            }
        )
    digest = _manifest_sha256(entries)
    if (
        len(entries) != expected_file_count
        or total_bytes != expected_bytes
        or digest != expected_sha256
    ):
        raise RuntimeError(
            "LocateAnything payload identity mismatch: "
            f"files={len(entries)}, bytes={total_bytes}, sha256={digest}"
        )
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    import yaml
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    from src.eval.scorer import score_dataset
    from src.eval.threshold import apply_threshold, select_f1_threshold

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--weights", default="data/weights/locateanything")
    parser.add_argument("--n-chips", type=int, default=200)
    parser.add_argument("--smoke", type=int, default=0, help="probe N chips, print raw generations")
    parser.add_argument("--device", default="cuda")
    add_runtime_provenance_arguments(parser)
    args = parser.parse_args(argv)
    if args.n_chips <= 0:
        parser.error("--n-chips must be positive")
    if args.smoke < 0:
        parser.error("--smoke must be nonnegative")
    if not args.smoke and args.n_chips != 200:
        parser.error("a full corrected R3 run requires --n-chips 200")
    repo = Path(__file__).resolve().parents[2]
    try:
        runtime_inputs = load_runtime_inputs(
            args,
            repo=repo,
            required=not bool(args.smoke),
        )
        corrected_result_dir = (
            result_directory(args, "locateanything-zs")
            if not args.smoke
            else None
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    weights_dir = Path(args.weights)
    model_payload_sha256 = validate_model_payload(weights_dir)
    license_path = weights_dir / "LICENSE"
    source_note = weights_dir / "SOURCE.note"
    if not license_path.is_file() or not source_note.is_file():
        raise SystemExit(
            "provenance gate: LocateAnything LICENSE and SOURCE.note are required"
        )

    config = yaml.safe_load(Path(args.config).read_text())
    raw_grd_root = Path(config["paths"]["raw_xview3"]) / "GRD"
    splits = json.loads(Path(config["paths"]["splits"]).read_text())["splits"]
    stats = json.loads(Path(config["paths"]["stats"]).read_text())
    vh_s, vv_s = stats["channels"]["VH"], stats["channels"]["VV"]
    mean = np.array([vh_s["mean"], vv_s["mean"], vh_s["mean"] - vv_s["mean"]], dtype=np.float32)
    std = np.array(
        [vh_s["std"], vv_s["std"], float(np.hypot(vh_s["std"], vv_s["std"]))],
        dtype=np.float32,
    )

    # ~200 labeled dev chips, deterministic pick: sorted foreground chips,
    # evenly strided across all dev scenes.
    chips_root = Path(config["paths"]["chips"])
    labeled = []
    for scene_id in sorted(splits["dev"]):
        for sidecar in sorted((chips_root / scene_id).glob("*.json")):
            payload = json.loads(sidecar.read_text())
            chip_scene_metadata(payload, expected_scene_id=scene_id)
            counts = chip_label_counts(payload["labels"])
            if counts["positive"]:
                chip_path = sidecar.with_suffix(".npy")
                labeled.append(
                    (
                        chip_path,
                        payload,
                        chip_path.relative_to(chips_root).as_posix(),
                        _sha256_file(sidecar),
                    )
                )
    if len(labeled) != EXPECTED_ELIGIBLE_CANDIDATES:
        raise RuntimeError(
            "R3 eligible-candidate count gate failed: "
            f"{len(labeled)} != {EXPECTED_ELIGIBLE_CANDIDATES}"
        )
    sample = select_evenly_strided(labeled, args.n_chips)
    if args.smoke:
        sample = sample[: args.smoke]
    elif len(sample) != 200:
        raise RuntimeError("full corrected R3 sample must contain exactly 200 chips")
    if not args.smoke:
        candidate_scenes = {Path(item[2]).parts[0] for item in labeled}
        sample_scenes = {Path(item[2]).parts[0] for item in sample}
        if sample_scenes != candidate_scenes:
            raise RuntimeError("R3 floor-index sample does not cover every eligible scene")
    print(f"{len(sample)} dev chips selected (of {len(labeled)} labeled)")

    execution = None
    if not args.smoke:
        assert runtime_inputs is not None
        execution = begin_reference_execution(
            runtime_inputs,
            reference_precision="bfloat16",
            device=args.device,
            torch_module=torch,
        )

    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(str(weights_dir), trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            str(weights_dir), torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to(args.device)
        .eval()
    )

    @torch.no_grad()
    def ask(image, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": QUESTION.format(prompt=prompt)},
                ],
            }
        ]
        text = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = processor.process_vision_info(messages)
        inputs = processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(args.device)
        output = model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=tokenizer,  # explicit param of the custom generate
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,  # the model's custom generate asserts this
        )
        # the custom generate returns the DECODED response string directly
        # (or a tuple when verbose) — not token ids
        return output if isinstance(output, str) else output[0]

    shore_cache = SceneShoreDistanceCache(raw_grd_root)
    results = {}
    for prompt in PROMPTS:
        gt_by_chip, pred_by_chip = {}, {}
        for index, (
            chip_path,
            sidecar,
            relative,
            _sidecar_sha256,
        ) in enumerate(sample):
            image = chip_image(chip_path, mean, std)
            generation = ask(image, prompt)
            if args.smoke:
                print(f"--- {chip_path.name} [{prompt}]:\n{generation[-400:]}")
            _, _, _, chip_size, _ = chip_scene_metadata(
                sidecar, expected_scene_id=Path(relative).parts[0]
            )
            centers = parse_box_centers(
                generation, image_w=float(chip_size), image_h=float(chip_size)
            )
            chip_id = Path(relative).with_suffix("").as_posix()
            if chip_id in gt_by_chip:
                raise RuntimeError(f"duplicate sampled chip identity: {chip_id}")
            gt_by_chip[chip_id] = chip_ground_truth_from_labels(sidecar["labels"])
            pred_by_chip[chip_id] = shore_cache.predictions_for_centers(
                centers,
                sidecar=sidecar,
                expected_scene_id=Path(relative).parts[0],
            )
        threshold = select_f1_threshold(gt_by_chip, pred_by_chip)
        scored = score_dataset(gt_by_chip, apply_threshold(pred_by_chip, threshold))
        results[prompt] = {
            "f1": scored.aggregate.f1,
            "precision": scored.aggregate.precision,
            "recall": scored.aggregate.recall,
            "threshold": threshold,
            "aggregate": _metrics_payload(scored.aggregate),
            "slices": {
                name: _metrics_payload(metrics)
                for name, metrics in sorted(scored.slices.items())
            },
            "per_chip": {
                chip_id: {
                    "aggregate": _metrics_payload(result.aggregate),
                    "slices": {
                        name: _metrics_payload(metrics)
                        for name, metrics in sorted(result.slices.items())
                    },
                }
                for chip_id, result in sorted(scored.scene_results.items())
            },
        }
        print(f"[{prompt}] {results[prompt]}")

    if not args.smoke:
        assert execution is not None and corrected_result_dir is not None
        best = max(results, key=lambda p: results[p]["f1"])
        sample_manifest = [
            {"chip": relative, "sidecar_sha256": sidecar_sha256}
            for _chip_path, _payload, relative, sidecar_sha256 in sample
        ]
        candidate_manifest = [
            {"chip": relative, "sidecar_sha256": sidecar_sha256}
            for _chip_path, _payload, relative, sidecar_sha256 in labeled
        ]
        candidate_sha256 = _manifest_sha256(candidate_manifest)
        sample_sha256 = _manifest_sha256(sample_manifest)
        sample_counts = {"positive": 0, "background": 0, "ignore": 0}
        for _chip_path, sidecar, _relative, _sidecar_sha256 in sample:
            for category, count in chip_label_counts(sidecar["labels"]).items():
                sample_counts[category] += count
        provenance = finish_reference_execution(
            execution,
            device=args.device,
            torch_module=torch,
        )
        if validate_model_payload(weights_dir) != model_payload_sha256:
            raise RuntimeError("LocateAnything payload changed during corrected R3")
        payload = {
            "result_schema": RESULT_SCHEMA,
            "exp_id": "locateanything-zs",
            "reference": "R3",
            "source_git_sha": provenance["git_sha"],
            "scored_at": provenance["finished_utc"],
            "model": {
                "id": "nvidia/LocateAnything-3B",
                "revision": MODEL_REVISION,
                "payload_sha256": model_payload_sha256,
                "source_note_sha256": _sha256_file(source_note),
                "license_sha256": _sha256_file(license_path),
            },
            "precision": "bfloat16",
            "eval_contract_disposition": "full-rerun-under-corrected-contract",
            "legacy_result_reused": False,
            "ground_truth_contract": {
                "version": 2,
                "positive": "is_vessel=true and confidence in {HIGH,MEDIUM}",
                "background": "is_vessel=false and confidence in {HIGH,MEDIUM}",
                "ignore": "confidence=LOW",
                "sample_counts": sample_counts,
            },
            "sample": {
                "policy": SAMPLE_POLICY,
                "algorithm_version": SAMPLE_ALGORITHM_VERSION,
                "candidate_count": len(candidate_manifest),
                "candidate_sha256": candidate_sha256,
                "candidates": candidate_manifest,
                "n_chips": len(sample),
                "sha256": sample_sha256,
                "entries": sample_manifest,
            },
            "per_prompt": results,
            "best_prompt": best,
        }
        out, provenance_path = publish_reference_result(
            corrected_result_dir,
            metrics=payload,
            provenance=provenance,
        )
        print(f"-> {out}")
        print(f"runtime provenance -> {provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
