"""Judy H100 BigEarthNet-S2 audit, paired run, and causal summary CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import yaml

from src.analysis.bes2_contract import (
    BES2ContractError,
    DIAGNOSTIC_METRICS_SCHEMA,
    DIAGNOSTIC_READY_SCHEMA,
    H100_CAMPAIGN_GIT_SHA,
    RANDOM_FROZEN_DEV_F1,
    S2_FROZEN_DEV_F1,
    assert_no_final_consumption,
    canonical_json,
    hash_binding,
    payload_sha256,
    read_regular_json,
    sha256_file,
    summarize_payload,
    utc_now,
    validate_diagnostic_metrics,
    validate_forecast_hours,
    validate_training_metrics,
    verify_training_path_identity,
    write_new_immutable,
)
from scripts.h100.data_staging import (
    EXPECTED_WEIGHT_DIRS,
    TRAINING_VIEW_CONTRACT,
)
from src.analysis.bes2_evidence import (
    activation_statistics,
    batch16_forward_backward_probe,
    build_sample_manifest,
    collect_dev_evidence,
    input_covariance_statistics,
    layer_drift,
    load_sample_images,
    manifest_train_indices,
    runtime_provenance,
    validate_data_scope,
    validate_h100_ready,
)
from src.models.bes2_diagnostic import (
    DIAGNOSTIC_VARIANTS,
    SOURCE_CHECKPOINT_SHA256,
)
from src.train.datamodule import FineTuneDataModule

AUDIT_SCHEMA = 1
TEST_RECEIPT_SCHEMA = 1
DATA_VIEW_RECEIPT_SCHEMA = 1
READY_FILENAME = "BES2_DIAGNOSTIC_READY.json"
AUDIT_FILENAME = "BES2_AUDIT.json"
SAMPLE_FILENAME = "BES2_SAMPLE_MANIFEST.json"
ROOT_CAUSE_FILENAME = "BES2_ROOT_CAUSE.json"
REPORT_FILENAME = "BES2_ROOT_CAUSE.md"


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BES2ContractError("could not resolve diagnostic source SHA") from exc


def _load_yaml(path: Path, description: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise BES2ContractError(
            f"{description} must be a regular non-symlink file: {path}"
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BES2ContractError(f"{description} must be a YAML mapping")
    return payload


def _resolve_config_path(repo: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo / path


def _scope_identity(scope: Mapping[str, object]) -> dict[str, object]:
    required = {
        "splits_sha256",
        "stats_sha256",
        "train_scene_ids",
        "dev_scene_ids",
        "label_scene_count",
    }
    if not required <= set(scope):
        raise BES2ContractError("diagnostic data scope identity is incomplete")
    identity = {
        "contract": TRAINING_VIEW_CONTRACT,
        "splits_sha256": scope["splits_sha256"],
        "stats_sha256": scope["stats_sha256"],
        "train_scene_ids": list(scope["train_scene_ids"]),
        "dev_scene_ids": list(scope["dev_scene_ids"]),
        "label_scene_count": scope["label_scene_count"],
    }
    identity["sha256"] = payload_sha256(identity)
    return identity


def _build_module(
    *,
    variant: str,
    detector: Mapping[str, object],
    weights_root: Path,
):
    import lightning as L

    from src.train.lit_modules import HeatmapLitModule

    L.seed_everything(0, workers=True)
    return HeatmapLitModule(
        init_name="bigearthnet_s2",
        lr=float(detector["optimizer"]["lr"]),
        layer_decay=float(detector["optimizer"]["layer_decay"]),
        weight_decay=float(detector["optimizer"]["weight_decay"]),
        epochs=50,
        warmup_epochs=int(detector["schedule"]["warmup_epochs"]),
        head_channels=int(detector["head"]["channels"]),
        diagnostic_variant=variant,
        weights_root=weights_root,
    )


def _load_best_state(model, checkpoint_path: Path) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise BES2ContractError("diagnostic checkpoint state_dict is absent")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise BES2ContractError("diagnostic checkpoint did not load strictly")


def _validate_test_receipt(path: Path, *, source_sha: str) -> dict:
    payload = read_regular_json(path, "diagnostic complete-test receipt")
    if (
        set(payload)
        != {
            "schema",
            "status",
            "purpose",
            "source_git_sha",
            "command",
            "exit_code",
            "passed",
            "duration_seconds",
        }
        or payload.get("schema") != TEST_RECEIPT_SCHEMA
        or payload.get("status") != "passed"
        or payload.get("purpose") != "bes2-diagnostic-complete-tests"
        or payload.get("source_git_sha") != source_sha
        or payload.get("exit_code") != 0
        or not isinstance(payload.get("command"), list)
        or not payload.get("command")
        or type(payload.get("passed")) is not int
        or int(payload["passed"]) <= 0
        or isinstance(payload.get("duration_seconds"), bool)
        or not isinstance(payload.get("duration_seconds"), (int, float))
        or not math.isfinite(float(payload["duration_seconds"]))
        or float(payload["duration_seconds"]) <= 0.0
    ):
        raise BES2ContractError("diagnostic complete-test receipt is invalid")
    return payload


def _validate_data_view_receipt(
    path: Path,
    *,
    source_sha: str,
    h100_ready_sha256: str,
    data_config_path: Path,
) -> dict:
    payload = read_regular_json(path, "diagnostic data-view receipt")
    expected = {
        "schema",
        "status",
        "purpose",
        "source_git_sha",
        "h100_ready_sha256",
        "data_config_sha256",
        "authorized_splits",
        "forbidden_splits",
        "staging_receipt_sha256",
        "staging_receipt",
    }
    staging = payload.get("staging_receipt")
    if (
        set(payload) != expected
        or payload.get("schema") != DATA_VIEW_RECEIPT_SCHEMA
        or payload.get("status") != "ready"
        or payload.get("purpose") != "bes2-diagnostic-train-dev-view"
        or payload.get("source_git_sha") != source_sha
        or payload.get("h100_ready_sha256") != h100_ready_sha256
        or payload.get("authorized_splits") != ["train", "dev8"]
        or payload.get("forbidden_splits") != ["test", "eval_final"]
        or payload.get("data_config_sha256") != sha256_file(data_config_path)
        or not isinstance(staging, Mapping)
        or payload.get("staging_receipt_sha256")
        != payload_sha256(staging)
    ):
        raise BES2ContractError("diagnostic data-view receipt is invalid")
    selected = staging.get("selected_artifacts")
    if (
        staging.get("schema") != 1
        or staging.get("status") != "ready"
        or staging.get("phase") != "train"
        or staging.get("purpose") != "campaign"
        or staging.get("contract") != TRAINING_VIEW_CONTRACT
        or staging.get("git_sha") != H100_CAMPAIGN_GIT_SHA
        or staging.get("training_cohort") is not None
        or staging.get("weights") != list(EXPECTED_WEIGHT_DIRS)
        or not isinstance(selected, list)
        or any(
            not isinstance(item, Mapping)
            or item.get("kind")
            not in {"chip_scene", "raster_scene", "core_weight"}
            for item in selected
        )
    ):
        raise BES2ContractError(
            "diagnostic staging receipt is not the TRAIN/DEV campaign view"
        )
    return payload


def _diagnostic_dataset(
    *,
    repo: Path,
    data_config: Mapping[str, object],
    detector: Mapping[str, object],
) -> FineTuneDataModule:
    # Kept behind the shared DataModule so the probe cannot silently construct
    # a different dataset recipe. setup() exposes the shared FineTuneDataset.
    module = FineTuneDataModule(
        chips_root=data_config["paths"]["chips"],
        splits_path=_resolve_config_path(
            repo, data_config["paths"]["splits"]
        ),
        stats_path=_resolve_config_path(
            repo, data_config["paths"]["stats"]
        ),
        label_frac=1.0,
        frac_seed=int(data_config["seed"]),
        batch_size=16,
        num_workers=0,
        crop=int(detector["input"]["crop_px"]),
        fg_frac=float(detector["sampler"]["fg_frac"]),
        seed=0,
    )
    module.setup("fit")
    return module


def _value_sensitive_post_stem(
    *,
    current_model,
    weights_root: Path,
) -> dict[str, object]:
    import lightning as L

    from src.models.init_loaders import build_init

    loaded = current_model.backbone.model.state_dict()
    L.seed_everything(0, workers=True)
    fresh = build_init(
        "cnn_random",
        load_weights=False,
        weights_root=weights_root,
    ).model.state_dict()
    keys = [
        key
        for key in sorted(loaded)
        if key != "stem.0.weight" and loaded[key].is_floating_point()
    ][:16]
    l2 = sum(
        float(torch.linalg.vector_norm(loaded[key].float() - fresh[key].float()))
        for key in keys
    )
    if len(keys) != 16 or not math.isfinite(l2) or l2 <= 1.0:
        raise BES2ContractError(
            "value-sensitive S2 post-stem load does not differ from seed-0 random"
        )
    return {"probe_keys": keys, "l2_vs_seed0_random": l2}


def _audit(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    weights_root = (
        args.weights_root
        if args.weights_root.is_absolute()
        else repo / args.weights_root
    ).resolve(strict=True)
    if not weights_root.is_dir():
        raise BES2ContractError("diagnostic weights root is not a directory")
    args.weights_root = weights_root
    source_sha = _git_sha(repo)
    if source_sha != args.expected_git_sha:
        raise BES2ContractError(
            f"diagnostic checkout SHA mismatch: {source_sha} != "
            f"{args.expected_git_sha}"
        )
    root = args.diagnostic_root
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise BES2ContractError(
            "diagnostic root must be an existing absolute non-symlink directory"
        )
    root = root.resolve(strict=True)
    control = root / ".control"
    control.mkdir(mode=0o700, exist_ok=True)
    ready_path = control / READY_FILENAME
    if ready_path.exists() or ready_path.is_symlink():
        payload = validate_readiness(
            ready_path,
            expected_source_sha=source_sha,
            diagnostic_root=root,
        )
        print(json.dumps(payload, indent=1))
        return 0
    occupied = [
        path.name
        for variant in DIAGNOSTIC_VARIANTS
        for stage in ("probe", "full")
        if (
            path := root / f"bes2-{variant}-{stage}-f100-s0"
        ).exists()
    ]
    if occupied:
        raise BES2ContractError(
            "diagnostic readiness requires empty run namespaces: "
            + ", ".join(occupied)
        )

    forecast = validate_forecast_hours(args.forecast_gpu_hours)
    final_gate = assert_no_final_consumption(args.original_runs_root)
    training_identity = verify_training_path_identity(
        repo, require_dev_ref=False
    )
    h100 = validate_h100_ready(args.h100_ready)
    if h100["sha256"] != args.h100_ready_sha256:
        raise BES2ContractError("canonical H100_READY SHA-256 mismatch")
    strict_runtime = runtime_provenance()
    gpu = strict_runtime.get("gpu")
    if not isinstance(gpu, Mapping) or "H100" not in str(gpu.get("name")):
        raise BES2ContractError("diagnostic audit requires one NVIDIA H100")

    detector_path = repo / "configs/detector.yaml"
    detector = _load_yaml(detector_path, "frozen detector config")
    data_config_path = args.data_config.resolve()
    data_config = _load_yaml(data_config_path, "diagnostic data config")
    view_receipt = _validate_data_view_receipt(
        args.data_view_receipt,
        source_sha=source_sha,
        h100_ready_sha256=h100["sha256"],
        data_config_path=data_config_path,
    )
    scope = validate_data_scope(
        repo=repo,
        data_view_root=args.data_view_root,
        data_config=data_config,
    )
    try:
        args.weights_root.relative_to(args.data_view_root.resolve(strict=True))
    except ValueError as exc:
        raise BES2ContractError(
            "diagnostic weights root is outside the TRAIN/DEV data view"
        ) from exc
    staging_receipt = view_receipt["staging_receipt"]
    if (
        staging_receipt.get("chips") != scope["train_scene_ids"]
        or staging_receipt.get("rasters") != scope["dev_scene_ids"]
    ):
        raise BES2ContractError(
            "diagnostic staging receipt differs from the physical TRAIN/DEV view"
        )
    weight_entries = list(args.weights_root.iterdir())
    if (
        sorted(
            item.name
            for item in weight_entries
            if item.is_dir() and not item.is_symlink()
        )
        != list(EXPECTED_WEIGHT_DIRS)
        or any(not item.is_dir() or item.is_symlink() for item in weight_entries)
    ):
        raise BES2ContractError(
            "diagnostic weight inventory is not the exact six checkpoint directories"
        )
    scope_identity = _scope_identity(scope)
    test_receipt = _validate_test_receipt(
        args.test_receipt,
        source_sha=source_sha,
    )

    sample_path = control / SAMPLE_FILENAME
    if sample_path.exists() and not sample_path.is_symlink():
        sample = read_regular_json(sample_path, "diagnostic sample manifest")
    else:
        sample = build_sample_manifest(
            scope=scope,
            output=sample_path,
            crop_px=int(detector["input"]["crop_px"]),
        )
    raw_images, normalized_images = load_sample_images(
        manifest=sample,
        scope=scope,
    )
    covariance = input_covariance_statistics(raw_images, normalized_images)
    data_module = _diagnostic_dataset(
        repo=repo,
        data_config=data_config,
        detector=detector,
    )
    train_indices = manifest_train_indices(sample, data_module.train_set)[:16]

    initializations: dict[str, object] = {}
    initial_activations: dict[str, object] = {}
    batch_probes: dict[str, object] = {}
    value_sensitive = None
    for variant in DIAGNOSTIC_VARIANTS:
        model = _build_module(
            variant=variant,
            detector=detector,
            weights_root=args.weights_root,
        )
        initializations[variant] = model.diagnostic_initialization
        if variant == "current_replay":
            value_sensitive = _value_sensitive_post_stem(
                current_model=model,
                weights_root=args.weights_root,
            )
        initial_activations[variant] = activation_statistics(
            model,
            normalized_images,
            device=args.device,
            batch_size=args.activation_batch_size,
        )
        del model

        probe_model = _build_module(
            variant=variant,
            detector=detector,
            weights_root=args.weights_root,
        )
        batch_probes[variant] = batch16_forward_backward_probe(
            probe_model,
            data_module.train_set,
            train_indices,
            device=args.device,
        )
        del probe_model

    replay_init = initializations["current_replay"]
    reset_init = initializations["first_conv_reset"]
    if (
        replay_init["head_state_sha256"] != reset_init["head_state_sha256"]
        or replay_init["parameter_count"] != reset_init["parameter_count"]
        or replay_init["changed_from_production_replay"] != []
        or reset_init["changed_from_production_replay"] != ["stem.0.weight"]
    ):
        raise BES2ContractError(
            "paired diagnostic initialization parity contract failed"
        )

    audit = {
        "audit_schema": AUDIT_SCHEMA,
        "status": "passed",
        "purpose": "bes2-checkpoint-channel-and-runtime-audit",
        "created_utc": utc_now(),
        "source_git_sha": source_sha,
        "training_path_identity": training_identity,
        "h100_acceptance": h100,
        "strict_runtime": strict_runtime,
        "data_view": {
            "receipt": hash_binding(args.data_view_receipt),
            "validated_receipt": view_receipt,
            "scope": scope,
            "scope_identity": scope_identity,
        },
        "test_suite": {
            "binding": hash_binding(args.test_receipt),
            "receipt": test_receipt,
        },
        "sample_manifest": {
            "binding": hash_binding(sample_path),
            "manifest_sha256": payload_sha256(sample),
        },
        "input_statistics": covariance,
        "initializations": initializations,
        "value_sensitive_s2_load": value_sensitive,
        "initial_activation_statistics": initial_activations,
        "batch16_forward_backward": batch_probes,
        "original_comparator": {
            "scope": "owner-frozen-numerical-comparators-only",
            "bigearthnet_s2_best_dev_f1": S2_FROZEN_DEV_F1,
            "random_best_dev_f1": RANDOM_FROZEN_DEV_F1,
            "original_campaign_artifacts_read": False,
        },
        "final_access_gate": final_gate,
        "forecast_gpu_hours": forecast,
        "bigearthnet_s1": (
            "out-of-scope; shared-helper use remains an unresolved compliance issue"
        ),
    }
    audit_path = control / AUDIT_FILENAME
    write_new_immutable(audit_path, audit)
    ready = {
        "diagnostic_ready_schema": DIAGNOSTIC_READY_SCHEMA,
        "status": "ready",
        "purpose": "bes2-root-cause-train-dev-only",
        "created_utc": utc_now(),
        "source": {
            "git_sha": source_sha,
            "training_path_identity": training_identity,
        },
        "diagnostic_root": str(root),
        "h100_acceptance": h100,
        "strict_fp32": strict_runtime["strict_fp32"],
        "data_view": {
            "contract": TRAINING_VIEW_CONTRACT,
            "receipt_sha256": sha256_file(args.data_view_receipt),
            "staging_receipt_sha256": view_receipt["staging_receipt_sha256"],
            "data_config_sha256": sha256_file(data_config_path),
            "scope_identity": scope_identity,
        },
        "checkpoint": replay_init["source_checkpoint"],
        "sample_manifest": hash_binding(sample_path),
        "audit": hash_binding(audit_path),
        "test_suite": hash_binding(args.test_receipt),
        "recipe": {
            "variants": list(DIAGNOSTIC_VARIANTS),
            "stages": {"probe_epochs": 5, "full_max_epochs": 50},
            "schedule_horizon_epochs": 50,
            "fraction": 1.0,
            "seed": 0,
            "train_scene_count": 111,
            "dev_scene_count": 8,
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "layer_decay": 0.65,
            "precision": "32-true",
            "devices_per_run": 1,
            "ddp": False,
        },
        "forecast_gpu_hours": forecast,
        "final_access_gate": final_gate,
        "canonical_h100_ready_unchanged": {
            "path": str(Path(args.h100_ready).absolute()),
            "sha256": h100["sha256"],
        },
    }
    write_new_immutable(ready_path, ready)
    print(json.dumps(ready, indent=1))
    return 0


def validate_readiness(
    path: Path,
    *,
    expected_source_sha: str,
    diagnostic_root: Path,
) -> dict[str, object]:
    payload = read_regular_json(path, "BES2 diagnostic readiness")
    expected = {
        "diagnostic_ready_schema",
        "status",
        "purpose",
        "created_utc",
        "source",
        "diagnostic_root",
        "h100_acceptance",
        "strict_fp32",
        "data_view",
        "checkpoint",
        "sample_manifest",
        "audit",
        "test_suite",
        "recipe",
        "forecast_gpu_hours",
        "final_access_gate",
        "canonical_h100_ready_unchanged",
    }
    source = payload.get("source")
    recipe = payload.get("recipe")
    strict_fp32 = payload.get("strict_fp32")
    h100_acceptance = payload.get("h100_acceptance")
    checkpoint = payload.get("checkpoint")
    canonical_h100 = payload.get("canonical_h100_ready_unchanged")
    final_gate = payload.get("final_access_gate")
    expected_strict = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }
    expected_recipe = {
        "variants": list(DIAGNOSTIC_VARIANTS),
        "stages": {"probe_epochs": 5, "full_max_epochs": 50},
        "schedule_horizon_epochs": 50,
        "fraction": 1.0,
        "seed": 0,
        "train_scene_count": 111,
        "dev_scene_count": 8,
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "layer_decay": 0.65,
        "precision": "32-true",
        "devices_per_run": 1,
        "ddp": False,
    }
    if (
        set(payload) != expected
        or payload.get("diagnostic_ready_schema") != DIAGNOSTIC_READY_SCHEMA
        or payload.get("status") != "ready"
        or payload.get("purpose") != "bes2-root-cause-train-dev-only"
        or not isinstance(source, Mapping)
        or set(source) != {"git_sha", "training_path_identity"}
        or source.get("git_sha") != expected_source_sha
        or not isinstance(source.get("training_path_identity"), Mapping)
        or source["training_path_identity"].get("status") != "byte-identical"
        or source["training_path_identity"].get("h100_campaign_git_sha")
        != H100_CAMPAIGN_GIT_SHA
        or not isinstance(strict_fp32, Mapping)
        or dict(strict_fp32) != expected_strict
        or not isinstance(h100_acceptance, Mapping)
    ):
        raise BES2ContractError("BES2 diagnostic readiness identity is invalid")
    if (
        h100_acceptance.get("strict_fp32") != expected_strict
        or not isinstance(checkpoint, Mapping)
        or dict(checkpoint)
        != {
            "relative_path": "bigearthnet_s2/model.safetensors",
            "sha256": SOURCE_CHECKPOINT_SHA256,
        }
        or not isinstance(canonical_h100, Mapping)
        or set(canonical_h100) != {"path", "sha256"}
        or not isinstance(canonical_h100.get("path"), str)
        or not Path(str(canonical_h100["path"])).is_absolute()
        or canonical_h100.get("sha256") != h100_acceptance.get("sha256")
        or sha256_file(str(canonical_h100["path"]))
        != canonical_h100.get("sha256")
        or not isinstance(final_gate, Mapping)
        or final_gate.get("status") != "unconsumed"
        or Path(str(payload.get("diagnostic_root"))).resolve()
        != diagnostic_root.resolve()
        or not isinstance(recipe, Mapping)
        or dict(recipe) != expected_recipe
    ):
        raise BES2ContractError("BES2 diagnostic readiness identity is invalid")
    data_view = payload.get("data_view")
    if not isinstance(data_view, Mapping) or set(data_view) != {
        "contract",
        "receipt_sha256",
        "staging_receipt_sha256",
        "data_config_sha256",
        "scope_identity",
    }:
        raise BES2ContractError("BES2 diagnostic data-view identity is invalid")
    scope_identity = data_view.get("scope_identity")
    if (
        data_view.get("contract") != TRAINING_VIEW_CONTRACT
        or not isinstance(scope_identity, Mapping)
        or any(
            not isinstance(data_view.get(name), str)
            or len(str(data_view.get(name))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(data_view.get(name))
            )
            for name in (
                "receipt_sha256",
                "staging_receipt_sha256",
                "data_config_sha256",
            )
        )
        or _scope_identity(scope_identity) != dict(scope_identity)
    ):
        raise BES2ContractError("BES2 diagnostic data-view identity is invalid")
    for name in ("sample_manifest", "audit", "test_suite"):
        binding = payload.get(name)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or sha256_file(binding["path"]) != binding["sha256"]
        ):
            raise BES2ContractError(
                f"BES2 diagnostic readiness {name} binding drifted"
            )
    validate_forecast_hours(payload.get("forecast_gpu_hours"))
    return payload


def _probe_prerequisites(
    root: Path,
    *,
    readiness_sha256: str,
) -> None:
    for variant in DIAGNOSTIC_VARIANTS:
        path = (
            root
            / f"bes2-{variant}-probe-f100-s0"
            / "diagnostic_metrics.json"
        )
        metrics = validate_diagnostic_metrics(
            path,
            expected_variant=variant,
            expected_stage="probe",
        )
        binding = metrics.get("readiness")
        if (
            not isinstance(binding, Mapping)
            or binding.get("sha256") != readiness_sha256
        ):
            raise BES2ContractError(
                f"{variant} probe is not bound to current readiness"
            )


def _run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    repo = args.repo.resolve()
    source_sha = _git_sha(repo)
    root = args.diagnostic_root.resolve()
    ready_path = root / ".control" / READY_FILENAME
    ready = validate_readiness(
        ready_path,
        expected_source_sha=source_sha,
        diagnostic_root=root,
    )
    ready_sha256 = sha256_file(ready_path)
    assert_no_final_consumption(args.original_runs_root)
    data_config_path = args.data_config.resolve(strict=True)
    data_config = _load_yaml(data_config_path, "diagnostic data config")
    data_identity = ready["data_view"]
    h100_acceptance = ready["h100_acceptance"]
    if not isinstance(h100_acceptance, Mapping):
        raise BES2ContractError("BES2 readiness lacks H100 acceptance identity")
    view_receipt = _validate_data_view_receipt(
        args.data_view_receipt,
        source_sha=source_sha,
        h100_ready_sha256=str(h100_acceptance.get("sha256")),
        data_config_path=data_config_path,
    )
    if (
        sha256_file(args.data_view_receipt)
        != data_identity["receipt_sha256"]
        or view_receipt["staging_receipt_sha256"]
        != data_identity["staging_receipt_sha256"]
        or sha256_file(data_config_path)
        != data_identity["data_config_sha256"]
    ):
        raise BES2ContractError(
            "current allocation data view differs from BES2 readiness"
        )
    scope = validate_data_scope(
        repo=repo,
        data_view_root=args.data_view_root,
        data_config=data_config,
    )
    if _scope_identity(scope) != data_identity["scope_identity"]:
        raise BES2ContractError(
            "current allocation TRAIN/DEV scope differs from BES2 readiness"
        )
    staging_receipt = view_receipt["staging_receipt"]
    if (
        staging_receipt.get("chips") != scope["train_scene_ids"]
        or staging_receipt.get("rasters") != scope["dev_scene_ids"]
    ):
        raise BES2ContractError(
            "current allocation staging receipt differs from its physical view"
        )
    requested_weights = (
        args.weights_root
        if args.weights_root.is_absolute()
        else repo / args.weights_root
    ).resolve(strict=True)
    try:
        requested_weights.relative_to(args.data_view_root.resolve(strict=True))
    except ValueError as exc:
        raise BES2ContractError(
            "run weights root is outside the TRAIN/DEV data view"
        ) from exc
    weight_entries = list(requested_weights.iterdir())
    if (
        sorted(
            item.name
            for item in weight_entries
            if item.is_dir() and not item.is_symlink()
        )
        != list(EXPECTED_WEIGHT_DIRS)
        or any(not item.is_dir() or item.is_symlink() for item in weight_entries)
    ):
        raise BES2ContractError("run weight inventory is not the exact six directories")
    checkpoint_binding = ready["checkpoint"]
    checkpoint_source = requested_weights / str(
        checkpoint_binding["relative_path"]
    )
    if sha256_file(checkpoint_source) != checkpoint_binding["sha256"]:
        raise BES2ContractError("bound BigEarthNet-S2 checkpoint drifted")
    args.weights_root = requested_weights

    stage = args.stage
    variant = args.variant
    if stage == "full":
        _probe_prerequisites(root, readiness_sha256=ready_sha256)
    run_dir = root / f"bes2-{variant}-{stage}-f100-s0"
    output_path = run_dir / "diagnostic_metrics.json"
    if output_path.exists() and not output_path.is_symlink():
        payload = validate_diagnostic_metrics(
            output_path,
            expected_variant=variant,
            expected_stage=stage,
        )
        print(json.dumps(payload, indent=1))
        return 0

    detector_path = repo / "configs/detector.yaml"
    detector = _load_yaml(detector_path, "frozen detector config")
    training_marker = run_dir / "training_metrics.json"
    if not training_marker.exists():
        from src.train.finetune import main as finetune_main

        result = finetune_main(
            [
                "--init",
                "bigearthnet_s2",
                "--label_frac",
                "1.0",
                "--seed",
                "0",
                "--git-sha",
                source_sha,
                "--data-config",
                str(data_config_path),
                "--detector-config",
                str(detector_path),
                "--weights-root",
                str(args.weights_root),
                "--workers",
                str(args.workers),
                "--diagnostic-variant",
                variant,
                "--diagnostic-stage",
                stage,
                "--diagnostic-root",
                str(root),
            ]
        )
        if result != 0:
            raise BES2ContractError(
                f"shared fine-tune entrypoint returned {result}"
            )

    training, checkpoint = validate_training_metrics(
        training_marker,
        expected_variant=variant,
        expected_stage=stage,
        candidate_floor=float(detector["decode"]["candidate_floor"]),
    )
    training_result = training["training_result"]
    if (
        training_result["git_sha"] != source_sha
        or training_result["detector_sha256"] != sha256_file(detector_path)
    ):
        raise BES2ContractError(
            "diagnostic training source/detector provenance mismatch"
        )
    sample_binding = ready["sample_manifest"]
    sample_path = Path(str(sample_binding["path"]))
    if sha256_file(sample_path) != sample_binding["sha256"]:
        raise BES2ContractError("diagnostic sample manifest drifted")
    sample = read_regular_json(sample_path, "diagnostic sample manifest")
    _raw_images, normalized_images = load_sample_images(
        manifest=sample,
        scope=scope,
    )

    initial = _build_module(
        variant=variant,
        detector=detector,
        weights_root=args.weights_root,
    )
    drift = layer_drift(initial, checkpoint)
    _load_best_state(initial, checkpoint)
    activations = activation_statistics(
        initial,
        normalized_images,
        device=args.device,
        batch_size=args.activation_batch_size,
    )
    dev_evidence = collect_dev_evidence(
        initial,
        data_config={
            **data_config,
            "paths": {
                **data_config["paths"],
                "chips": scope["chips_root"],
                "raw_xview3": scope["raw_root"],
                "splits": str(
                    _resolve_config_path(
                        repo, data_config["paths"]["splits"]
                    )
                ),
                "stats": str(
                    _resolve_config_path(
                        repo, data_config["paths"]["stats"]
                    )
                ),
            },
        },
        detector_config=detector,
        best_dev=training["training_result"]["best_dev"],
        device=args.device,
    )
    provenance = runtime_provenance()
    analysis_seconds = time.monotonic() - started
    training_hours = float(training["runtime"]["gpu_hours"])
    metrics = {
        "diagnostic_metrics_schema": DIAGNOSTIC_METRICS_SCHEMA,
        "status": "complete",
        "purpose": "bes2-root-cause",
        "variant": variant,
        "stage": stage,
        "created_utc": utc_now(),
        "readiness": {
            "path": str(ready_path),
            "sha256": ready_sha256,
        },
        "training": training,
        "dev_evidence": dev_evidence,
        "activation_statistics": activations,
        "layer_drift": drift,
        "runtime_provenance": {
            **provenance,
            "analysis_seconds": analysis_seconds,
            "training_gpu_hours": training_hours,
        },
        "gpu_hours": training_hours + analysis_seconds / 3600.0,
    }
    checkpoint.chmod(0o444)
    training_marker.chmod(0o444)
    initialization_path = run_dir / "initialization.json"
    if initialization_path.is_file() and not initialization_path.is_symlink():
        initialization_path.chmod(0o444)
    write_new_immutable(output_path, metrics)
    validate_diagnostic_metrics(
        output_path,
        expected_variant=variant,
        expected_stage=stage,
    )
    print(json.dumps(metrics, indent=1))
    return 0


def _range(values: Sequence[float]) -> str:
    if not values:
        return "n/a"
    return f"{min(values):.6f}–{max(values):.6f}"


def _markdown_report(payload: Mapping[str, object]) -> str:
    decision = payload["decision"]
    support = payload["supporting_evidence"]
    replay = support["current_replay"]
    reset = support["first_conv_reset"]
    replay_best = replay["best_dev"]
    reset_best = reset["best_dev"]
    recovery = decision.get("recovery")
    recovery_text = "not applicable" if recovery is None else f"{float(recovery):.6f}"

    def loo_ranges(item: Mapping[str, object]) -> tuple[str, str]:
        entries = item["dev_evidence"]["leave_one_dev_scene_out"].values()
        retained = [
            float(entry["retained_seven"]["f1"]) for entry in entries
        ]
        thresholds = [float(entry["threshold"]) for entry in entries]
        return _range(retained), _range(thresholds)

    replay_loo, replay_tau = loo_ranges(replay)
    reset_loo, reset_tau = loo_ranges(reset)
    replay_row = (
        f"| current_replay | {float(replay_best['f1']):.10f} | "
        f"{float(replay_best['precision']):.10f} | "
        f"{float(replay_best['recall']):.10f} | "
        f"{float(replay_best['threshold']):.10f} |"
    )
    reset_row = (
        f"| first_conv_reset | {float(reset_best['f1']):.10f} | "
        f"{float(reset_best['precision']):.10f} | "
        f"{float(reset_best['recall']):.10f} | "
        f"{float(reset_best['threshold']):.10f} |"
    )
    return f"""# BigEarthNet-S2 root-cause diagnostic

Status: **{payload['status']}**

Classification: **{decision['classification']}**

This is a single-seed TRAIN+fixed-DEV8 diagnostic. It makes no significance,
error-bar, or seed-variance claim. TEST and verified-final data were not used.

## Paired result

| Variant | Best DEV F1 | Precision | Recall | Threshold |
|---|---:|---:|---:|---:|
{replay_row}
{reset_row}

Frozen current-S2 comparator: {float(decision['frozen_s2_best_dev_f1']):.10f}

Frozen random comparator: {float(decision['frozen_random_best_dev_f1']):.10f}

Recovery: {recovery_text}

Replacement eligibility: **{decision['replacement_eligibility']}**

Required next step: {decision['required_next_step']}

## Sensitivity and mechanism evidence

- Current replay leave-one-DEV-scene-out retained-seven F1 range:
  {replay_loo}; threshold range: {replay_tau}.
- First-convolution reset leave-one-DEV-scene-out retained-seven F1 range:
  {reset_loo}; threshold range: {reset_tau}.
- Exact candidate-threshold curves, score distributions, per-scene metrics,
  activation statistics, and initialization-to-best layer drift are embedded
  in the bound per-run `diagnostic_metrics.json` artifacts.
- BigEarthNet-S1 was not modified or tested. Its use of the shared conversion
  helper remains a separate unresolved compliance issue.

## Reproduction

```bash
python -B -m src.analysis.bes2_root_cause summarize \
  --diagnostic-root {payload['runs']['current_replay']['path'].rsplit('/', 2)[0]}
```

The JSON decision and this report are content-bound to both full-run artifacts.
"""


def _write_new_text(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise BES2ContractError(f"immutable report already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            if not text.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial immutable report remains fail-closed for human inspection.
        raise


def _summary_from_runs(root: Path) -> dict[str, object]:
    replay_path = (
        root
        / "bes2-current_replay-full-f100-s0"
        / "diagnostic_metrics.json"
    )
    reset_path = (
        root
        / "bes2-first_conv_reset-full-f100-s0"
        / "diagnostic_metrics.json"
    )
    replay = validate_diagnostic_metrics(
        replay_path,
        expected_variant="current_replay",
    )
    reset = validate_diagnostic_metrics(
        reset_path,
        expected_variant="first_conv_reset",
    )
    if replay["readiness"]["sha256"] != reset["readiness"]["sha256"]:
        raise BES2ContractError("paired full runs bind different readiness receipts")
    return summarize_payload(
        replay,
        reset,
        replay_binding=hash_binding(replay_path),
        reset_binding=hash_binding(reset_path),
    )


def _validate_summary_outputs(root: Path) -> dict[str, object]:
    output = root / ".control" / ROOT_CAUSE_FILENAME
    report_path = root / ".control" / REPORT_FILENAME
    observed = read_regular_json(output, "BES2 root-cause result")
    expected = _summary_from_runs(root)
    expected["created_utc"] = observed.get("created_utc")
    if expected != observed:
        raise BES2ContractError(
            "existing BES2 root-cause JSON is not reproducible"
        )
    try:
        report = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BES2ContractError("existing BES2 report is unreadable") from exc
    if report != _markdown_report(observed):
        raise BES2ContractError(
            "existing BES2 Markdown report is not reproducible"
        )
    return observed


def _summarize(args: argparse.Namespace) -> int:
    root = args.diagnostic_root.resolve()
    output = root / ".control" / ROOT_CAUSE_FILENAME
    report = root / ".control" / REPORT_FILENAME
    if output.exists() or output.is_symlink() or report.exists() or report.is_symlink():
        if (
            output.is_file()
            and not output.is_symlink()
            and report.is_file()
            and not report.is_symlink()
        ):
            payload = _validate_summary_outputs(root)
            print(json.dumps(payload, indent=1))
            return 0
        raise BES2ContractError("root-cause JSON/report publication is incomplete")

    payload = _summary_from_runs(root)
    write_new_immutable(output, payload)
    _write_new_text(report, _markdown_report(payload))
    _validate_summary_outputs(root)
    print(json.dumps(payload, indent=1))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--repo", type=Path, required=True)
    audit.add_argument("--diagnostic-root", type=Path, required=True)
    audit.add_argument("--expected-git-sha", required=True)
    audit.add_argument("--h100-ready", type=Path, required=True)
    audit.add_argument("--h100-ready-sha256", required=True)
    audit.add_argument("--original-runs-root", type=Path, required=True)
    audit.add_argument("--data-view-root", type=Path, required=True)
    audit.add_argument("--data-view-receipt", type=Path, required=True)
    audit.add_argument("--data-config", type=Path, required=True)
    audit.add_argument("--test-receipt", type=Path, required=True)
    audit.add_argument("--weights-root", type=Path, default=Path("data/weights"))
    audit.add_argument("--forecast-gpu-hours", type=float, required=True)
    audit.add_argument("--device", default="cuda")
    audit.add_argument("--activation-batch-size", type=int, default=8)

    run = subparsers.add_parser("run")
    run.add_argument("--repo", type=Path, required=True)
    run.add_argument("--diagnostic-root", type=Path, required=True)
    run.add_argument("--original-runs-root", type=Path, required=True)
    run.add_argument("--variant", choices=DIAGNOSTIC_VARIANTS, required=True)
    run.add_argument("--stage", choices=("probe", "full"), required=True)
    run.add_argument("--data-view-root", type=Path, required=True)
    run.add_argument("--data-view-receipt", type=Path, required=True)
    run.add_argument("--data-config", type=Path, required=True)
    run.add_argument("--weights-root", type=Path, required=True)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--device", default="cuda")
    run.add_argument("--activation-batch-size", type=int, default=8)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--diagnostic-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        return _audit(args)
    if args.command == "run":
        return _run(args)
    if args.command == "summarize":
        return _summarize(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
