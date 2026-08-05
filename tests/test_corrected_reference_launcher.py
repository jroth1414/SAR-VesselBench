from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import run_corrected_references as launcher
from src.references import runtime_provenance, yolo26_ref


REFERENCE_SHA = "a" * 40
CORE_SHA = "b" * 40
REFERENCE_CAMPAIGN = "sprint7f-v100-references-20260804"
CORE_CAMPAIGN = "fresh34-v100-fp32-20260726"
ENVIRONMENT_SHA = "c" * 64


def _base_args(campaign_root: Path, lock: Path) -> list[str]:
    return [
        "--expected-git-sha",
        REFERENCE_SHA,
        "--reference-campaign-id",
        REFERENCE_CAMPAIGN,
        "--core-campaign-id",
        CORE_CAMPAIGN,
        "--core-git-sha",
        CORE_SHA,
        "--environment-sha256",
        ENVIRONMENT_SHA,
        "--environment-lock",
        str(lock),
        "--campaign-manifest",
        str(campaign_root / launcher.CAMPAIGN_MANIFEST_NAME),
    ]


def _repo_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "reference-checkout"
    (repo / "locks").mkdir(parents=True)
    (repo / "data").mkdir()
    lock = repo / "locks/env-v100node.txt"
    lock.write_text("torch==2.11.0+cu126\n", encoding="utf-8")
    (repo / "data/splits.json").write_text('{"splits": {}}\n', encoding="utf-8")
    (repo / "data/stats.json").write_text("{}\n", encoding="utf-8")

    payload = tmp_path / "payload"
    raw = payload / "raw-xview3"
    chips = payload / "chips"
    (raw / "labels").mkdir(parents=True)
    chips.mkdir(parents=True)
    (raw / "labels/train.csv").write_text("scene_id\n", encoding="utf-8")

    campaign_root = tmp_path / "reference-campaign"
    campaign_root.mkdir()
    (campaign_root / launcher.RESULTS_DIRECTORY_NAME).mkdir()
    data_config = campaign_root / "data-v100.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "raw_xview3": str(raw),
                    "chips": str(chips),
                    "splits": str(repo / "data/splits.json"),
                    "stats": str(repo / "data/stats.json"),
                }
            }
        ),
        encoding="utf-8",
    )
    return repo, lock, campaign_root, data_config


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_provenance, "_clean_git_sha", lambda _repo: REFERENCE_SHA)
    monkeypatch.setattr(
        runtime_provenance,
        "validate_runtime_environment",
        lambda _lock: ENVIRONMENT_SHA,
    )


def _create_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    repo, lock, campaign_root, data_config = _repo_fixture(tmp_path)
    _patch_runtime(monkeypatch)
    args = launcher.build_parser().parse_args(
        ["manifest", *_base_args(campaign_root, lock)]
    )
    manifest = launcher.create_campaign_manifest(args, repo=repo)
    return repo, lock, campaign_root, data_config


def test_normalized_environment_lock_sha256_is_order_and_name_canonical(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "environment.txt"
    lock.write_text("Z_pkg==2\na.pkg==1\n", encoding="utf-8")
    expected = hashlib.sha256(b"a-pkg==1\nz-pkg==2\n").hexdigest()
    assert runtime_provenance.normalized_environment_lock_sha256(lock) == expected


def test_manifest_is_immutable_exact_schema_and_binds_diagnostic_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lock, campaign_root, _data_config = _repo_fixture(tmp_path)
    _patch_runtime(monkeypatch)
    args = launcher.build_parser().parse_args(
        ["manifest", *_base_args(campaign_root, lock)]
    )
    manifest = launcher.create_campaign_manifest(args, repo=repo)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert set(payload) == runtime_provenance.CAMPAIGN_MANIFEST_KEYS
    assert payload == {
        "schema": 1,
        "campaign_role": "corrected-v100-references",
        "campaign_id": REFERENCE_CAMPAIGN,
        "core_campaign_id": CORE_CAMPAIGN,
        "core_git_sha": CORE_SHA,
        "git_sha": REFERENCE_SHA,
        "environment_sha256": ENVIRONMENT_SHA,
        "environment_lock_sha256": runtime_provenance.sha256_file(lock),
        "runtime_launcher_sha256": runtime_provenance.sha256_file(
            launcher.LAUNCHER_PATH
        ),
    }
    assert manifest.stat().st_mode & 0o777 == 0o444
    with pytest.raises(RuntimeError, match="already exists"):
        launcher.create_campaign_manifest(args, repo=repo)


def test_manifest_rejects_same_campaign_and_bad_core_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lock, campaign_root, _data_config = _repo_fixture(tmp_path)
    _patch_runtime(monkeypatch)
    parser = launcher.build_parser()

    args = parser.parse_args(["manifest", *_base_args(campaign_root, lock)])
    args.reference_campaign_id = CORE_CAMPAIGN
    with pytest.raises(RuntimeError, match="must differ"):
        launcher.create_campaign_manifest(args, repo=repo)

    args.reference_campaign_id = REFERENCE_CAMPAIGN
    args.core_git_sha = "bad"
    with pytest.raises(RuntimeError, match="core-git-sha"):
        launcher.create_campaign_manifest(args, repo=repo)


def test_external_data_config_requires_absolute_frozen_split_and_stats(
    tmp_path: Path,
) -> None:
    repo, _lock, _campaign_root, data_config = _repo_fixture(tmp_path)
    assert launcher._validate_data_config(data_config, repo) == data_config.resolve()

    payload = yaml.safe_load(data_config.read_text(encoding="utf-8"))
    payload["paths"]["splits"] = "data/splits.json"
    data_config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="paths.splits must be an absolute"):
        launcher._validate_data_config(data_config, repo)


def test_commands_are_score_only_explicit_and_carry_core_git_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lock, campaign_root, data_config = _create_manifest(tmp_path, monkeypatch)
    results = campaign_root / launcher.RESULTS_DIRECTORY_NAME
    manifest = campaign_root / launcher.CAMPAIGN_MANIFEST_NAME
    checkpoint = repo / "runs/yolo26-f100/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preserved-r2")
    monkeypatch.setattr(
        yolo26_ref,
        "EXPECTED_R2_BEST_SHA256",
        runtime_provenance.sha256_file(checkpoint),
    )
    parser = launcher.build_parser()
    r2 = parser.parse_args(
        [
            "r2",
            *_base_args(campaign_root, lock),
            "--data-config",
            str(data_config),
            "--results-root",
            str(results),
            "--gpu",
            "3",
            "--r2-weights",
            str(checkpoint),
        ]
    )
    command, exp_id = launcher.build_reference_command(
        r2,
        data_config=data_config,
        manifest=manifest,
        lock=lock,
        results_root=results,
        repo=repo,
    )
    assert exp_id == "yolo26-f100"
    assert command[1:5] == ["-m", "src.references.yolo26_ref", "score", "--config"]
    assert "train" not in command and "export" not in command
    assert command[command.index("--core-git-sha") + 1] == CORE_SHA
    assert command[command.index("--weights") + 1] == str(checkpoint.resolve())

    r3_weights = tmp_path / "locateanything"
    r3_weights.mkdir()
    (r3_weights / "SOURCE.note").write_text("source\n", encoding="utf-8")
    (r3_weights / "LICENSE").write_text("license\n", encoding="utf-8")
    r3 = parser.parse_args(
        [
            "r3",
            *_base_args(campaign_root, lock),
            "--data-config",
            str(data_config),
            "--results-root",
            str(results),
            "--gpu",
            "3",
            "--r3-weights",
            str(r3_weights),
        ]
    )
    command, exp_id = launcher.build_reference_command(
        r3,
        data_config=data_config,
        manifest=manifest,
        lock=lock,
        results_root=results,
        repo=repo,
    )
    assert exp_id == "locateanything-zs"
    assert command[1:3] == ["-m", "src.references.locateanything_zs"]
    assert command[command.index("--n-chips") + 1] == "200"
    assert "--smoke" not in command


def test_campaign_lock_prevents_concurrent_r2_r3(tmp_path: Path) -> None:
    with launcher._exclusive_campaign_lock(tmp_path, "r2"):
        with pytest.raises(RuntimeError, match="already owns"):
            with launcher._exclusive_campaign_lock(tmp_path, "r3"):
                raise AssertionError("unreachable")


def test_run_dispatches_one_gpu_and_validates_fresh_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, lock, campaign_root, data_config = _create_manifest(tmp_path, monkeypatch)
    results = campaign_root / launcher.RESULTS_DIRECTORY_NAME
    checkpoint = repo / "runs/yolo26-f100/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preserved-r2")
    monkeypatch.setattr(
        yolo26_ref,
        "EXPECTED_R2_BEST_SHA256",
        runtime_provenance.sha256_file(checkpoint),
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    args = launcher.build_parser().parse_args(
        [
            "r2",
            *_base_args(campaign_root, lock),
            "--data-config",
            str(data_config),
            "--results-root",
            str(results),
            "--gpu",
            "4",
            "--r2-weights",
            str(checkpoint),
        ]
    )
    calls: list[tuple[list[str], Path, dict[str, str], bool]] = []

    def fake_run(command, *, cwd, env, check):
        calls.append((command, cwd, env, check))
        result_dir = results / "yolo26-f100"
        result_dir.mkdir()
        (result_dir / "runtime_provenance.json").write_text(
            json.dumps(
                {"campaign_id": REFERENCE_CAMPAIGN, "git_sha": REFERENCE_SHA}
            ),
            encoding="utf-8",
        )
        (result_dir / "final_metrics.json").write_text(
            json.dumps({"source_git_sha": REFERENCE_SHA}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0)

    metrics, provenance = launcher.run_reference(
        args, repo=repo, runner=fake_run
    )
    assert metrics.name == "final_metrics.json"
    assert provenance.name == "runtime_provenance.json"
    assert len(calls) == 1
    command, cwd, env, check = calls[0]
    assert cwd == repo
    assert check is True
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert command.count("score") == 1


def test_make_reference_target_uses_only_standalone_launcher() -> None:
    source = (launcher.REPO / "Makefile").read_text(encoding="utf-8")
    recipe = source.split("references:", 1)[1].split("\n\n", 1)[0]
    assert "-m scripts.run_corrected_references" in recipe
    assert "src.references.yolo26_ref" not in recipe
    assert "src.references.locateanything_zs" not in recipe
