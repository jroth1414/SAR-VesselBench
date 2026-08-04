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
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROMPTS = ("ship", "vessel", "boat")
MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
RESULT_SCHEMA = 2
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


def _git_sha() -> str:
    repo = Path(__file__).resolve().parents[2]
    value = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError("reference source did not resolve to a full Git SHA")
    return value


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


def main(argv: Sequence[str] | None = None) -> int:
    import torch
    import yaml
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    from scripts.h100.contracts import atomic_write_json
    from src.eval.scorer import PredictionPoint, score_dataset
    from src.eval.threshold import apply_threshold, select_f1_threshold

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--weights", default="data/weights/locateanything")
    parser.add_argument("--n-chips", type=int, default=200)
    parser.add_argument("--smoke", type=int, default=0, help="probe N chips, print raw generations")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.n_chips <= 0:
        parser.error("--n-chips must be positive")
    if args.smoke < 0:
        parser.error("--smoke must be nonnegative")

    weights_dir = Path(args.weights)
    license_path = weights_dir / "LICENSE"
    source_note = weights_dir / "SOURCE.note"
    if not license_path.is_file() or not source_note.is_file():
        raise SystemExit(
            "provenance gate: LocateAnything LICENSE and SOURCE.note are required"
        )

    config = yaml.safe_load(Path(args.config).read_text())
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
    step = max(1, len(labeled) // args.n_chips)
    sample = labeled[::step][: args.n_chips]
    if args.smoke:
        sample = sample[: args.smoke]
    print(f"{len(sample)} dev chips selected (of {len(labeled)} labeled)")

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
            centers = parse_box_centers(generation)
            chip_id = Path(relative).with_suffix("").as_posix()
            if chip_id in gt_by_chip:
                raise RuntimeError(f"duplicate sampled chip identity: {chip_id}")
            gt_by_chip[chip_id] = chip_ground_truth_from_labels(sidecar["labels"])
            pred_by_chip[chip_id] = [
                PredictionPoint(
                    x_m=col * 10.0,
                    y_m=row * 10.0,
                    score=1.0 / (rank + 1),
                    distance_from_shore_km=9999.99,
                )
                for rank, (col, row) in enumerate(centers)
            ]
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
        out = Path("runs/locateanything-zs/final_metrics.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        best = max(results, key=lambda p: results[p]["f1"])
        sample_manifest = [
            {"chip": relative, "sidecar_sha256": sidecar_sha256}
            for _chip_path, _payload, relative, sidecar_sha256 in sample
        ]
        sample_sha256 = hashlib.sha256(
            json.dumps(
                sample_manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        sample_counts = {"positive": 0, "background": 0, "ignore": 0}
        for _chip_path, sidecar, _relative, _sidecar_sha256 in sample:
            for category, count in chip_label_counts(sidecar["labels"]).items():
                sample_counts[category] += count
        payload = {
            "result_schema": RESULT_SCHEMA,
            "exp_id": "locateanything-zs",
            "reference": "R3",
            "source_git_sha": _git_sha(),
            "scored_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "model": {
                "id": "nvidia/LocateAnything-3B",
                "revision": MODEL_REVISION,
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
                "policy": "sorted-foreground-dev-chips-even-stride",
                "n_chips": len(sample),
                "sha256": sample_sha256,
                "entries": sample_manifest,
            },
            "per_prompt": results,
            "best_prompt": best,
        }
        atomic_write_json(out, payload)
        print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
