from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from scripts.h100 import build_venv
from scripts.h100.data_staging import training_labels_summary
from scripts.handoff import box
from scripts.handoff import runtime_amendment as amendment
from scripts.handoff.__main__ import _build_parser
from scripts.handoff.package import (
    EXPECTED_PYTHON as BASE_EXPECTED_PYTHON,
    PackageError,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _fixture_repo(root: Path) -> tuple[Path, str, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--initial-branch=fixture-runtime")
    _git(repo, "config", "user.name", "Repository Owner")
    _git(repo, "config", "user.email", "owner@example.invalid")
    files = {
        "locks/env-v100node.txt": "torch==2.11.0+cu126\n",
        "docs/CORRECTED_REFERENCES_RUNBOOK.md": "# Corrected references\n",
        "scripts/h100/acceptance.py": "ACCEPTANCE = 'strict-fp32'\n",
        "scripts/h100/build_venv.py": "RUNTIME = 'native-venv'\n",
        "scripts/h100/campaign.py": "CAMPAIGN = 'all-32-then-test'\n",
        "scripts/h100/cell.py": "CELL = 'phase-isolated'\n",
        "scripts/h100/contracts.py": "PRECISION = '32-true'\n",
        "scripts/h100/cutover.py": "CUTOVER = 'corrected-references'\n",
        "scripts/h100/data_staging.py": "STAGING = 'phase-isolated'\n",
        "scripts/h100/runtime_versions.py": (
            "EXPECTED_NATIVE_PYTHON_VERSION = '3.11.13'\n"
        ),
        "scripts/h100/lightning_contract.py": "CONTRACT = 'strict-fp32'\n",
        "scripts/h100/operator_cutover.py": "ISOLATION = 'v100-diagnostic'\n",
        "scripts/h100/reverse_results.py": "RESULTS = 'reverse-handoff'\n",
        "scripts/h100/wheelhouse.py": "WHEELHOUSE = 'verified-offline'\n",
        "scripts/handoff/control.py": "CONTROL = 'box-transfer-v1'\n",
        "scripts/handoff/results.py": "RESULTS = 'content-addressed'\n",
        "scripts/handoff/runtime_bootstrap.py": "BOOTSTRAP = 'hash-pinned'\n",
        "scripts/export_results.py": "EXPORT = 'h100-only'\n",
        "scripts/run_corrected_references.py": "REFERENCES = 'isolated'\n",
        "scripts/score_test_cohort.py": "BARRIER = 'all-32'\n",
        "scripts/score_test_split.py": "OUTPUT = 'test_metrics.json'\n",
        "src/analysis/curves.py": "GRID = 'complete-32'\n",
        "src/eval/final_eval.py": "FINAL = 'monotonicity-gated'\n",
        "src/eval/ground_truth.py": "GROUND_TRUTH = 'hm-positive-low-ignore'\n",
        "src/eval/ground_truth_audit.py": "AUDIT = 'exact-supports'\n",
        "src/eval/heldout_contract.py": "HELDOUT = 'cohort-bound'\n",
        "src/eval/result_contract.py": "RESULT_SCHEMA = 2\n",
        "src/references/locateanything_zs.py": "R3 = 'corrected-rerun'\n",
        "src/references/runtime_provenance.py": "PROVENANCE = 'fresh'\n",
        "src/references/yolo26_ref.py": "R2 = 'corrected-rescore'\n",
        "data/splits.json": json.dumps(
            {
                "splits": {
                    "train": ["train-scene"],
                    "dev": [f"dev-{index}" for index in range(8)],
                    "test": ["test-scene"],
                    "eval_final": [],
                }
            },
            sort_keys=True,
        )
        + "\n",
        "slurm/h100/V100_DIAGNOSTIC_ISOLATION.schema.json": "{}\n",
        "slurm/h100/campaign.sbatch": "#!/bin/bash\n# native venv campaign\n",
        "slurm/h100/smoke.sbatch": "#!/bin/bash\n# native venv smoke\n",
        "slurm/h100/submit.sh": "#!/bin/bash\n# native venv submit\n",
        "slurm/h100/shims/scontrol": "#!/bin/bash\n# child requeue defer shim\n",
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture ancestor")
    ancestor = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("runtime amendment fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--quiet", "-m", "fixture head")
    return repo, ancestor, _git(repo, "rev-parse", "HEAD")


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_runtime_package_is_deterministic_filtered_training_view_and_extracts(
    tmp_path: Path,
) -> None:
    repo, ancestor, head = _fixture_repo(tmp_path)
    base = amendment._expected_base_identity()
    output_a = tmp_path / "out-a"
    output_b = tmp_path / "out-b"
    output_a.mkdir()
    output_b.mkdir()
    labels = tmp_path / "train-dev8.csv"
    labels.write_text(
        "scene_id,value\n"
        + "train-scene,1\n"
        + "".join(f"dev-{index},1\n" for index in range(8)),
        encoding="utf-8",
    )
    label_summary = training_labels_summary(
        labels,
        splits_path=repo / "data/splits.json",
        production=False,
    )
    training_view = {
        "schema": 1,
        "contract": "train111-fixed-dev8-no-test-v1",
        "source_labels": {
            "sha256": "1" * 64,
            "bytes": 1,
            "archive_sha256": "2" * 64,
        },
        "splits_sha256": hashlib.sha256(
            (repo / "data/splits.json").read_bytes()
        ).hexdigest(),
        "labels": label_summary,
    }
    evaluation_ground_truth = {"fixture": True}
    assert {
        "scripts/h100/contracts.py",
        "scripts/h100/lightning_contract.py",
        "scripts/h100/runtime_versions.py",
        "scripts/h100/wheelhouse.py",
    } <= set(amendment._REQUIRED_RUNTIME_FILES)

    package_a = amendment._build_runtime_amendment(
        repo_root=repo,
        output_dir=output_a,
        maximum_physical_file_bytes=1024,
        base_identity=base,
        branch="fixture-runtime",
        required_ancestor=ancestor,
        production=False,
        training_labels_path=labels,
        training_view=training_view,
        evaluation_ground_truth=evaluation_ground_truth,
    )
    package_b = amendment._build_runtime_amendment(
        repo_root=repo,
        output_dir=output_b,
        maximum_physical_file_bytes=1024,
        base_identity=base,
        branch="fixture-runtime",
        required_ancestor=ancestor,
        production=False,
        training_labels_path=labels,
        training_view=training_view,
        evaluation_ground_truth=evaluation_ground_truth,
    )
    assert package_a.name == package_b.name
    assert _tree_hashes(package_a) == _tree_hashes(package_b)

    manifest = amendment._verify_runtime_amendment(
        package_a,
        base_identity=base,
        require_production=False,
    )
    assert manifest["source"]["git_commit"] == head
    assert manifest["base_payload"] == base
    assert manifest["contract"]["python"] == build_venv.EXPECTED_PYTHON_VERSION
    assert build_venv.EXPECTED_PYTHON_VERSION == "3.11.13"
    assert BASE_EXPECTED_PYTHON == "3.11.15"
    assert manifest["contract"]["runtime"] == "native-venv"
    assert manifest["contract"]["precision"] == "32-true"
    assert manifest["contract"]["tf32"] is False
    assert manifest["counts"] == {
        "git_bundles": 1,
        "training_label_artifacts": 1,
    }
    assert manifest["training_view"] == training_view
    assert manifest["evaluation_ground_truth"] == evaluation_ground_truth
    physical = {
        path.relative_to(package_a).as_posix()
        for path in package_a.rglob("*")
        if path.is_file()
    }
    assert physical <= {
        "manifest.json",
        "SHA256SUMS",
        "READY.json",
        *{
            part["path"]
            for artifact in manifest["artifacts"]
            for part in artifact["parts"]
        },
    }
    assert not any(
        token in path.lower()
        for path in physical
        for token in ("wheelhouse", "weights", "runs", ".venv", "jwt")
    )

    destination = tmp_path / "extracted"
    amendment._extract_runtime_amendment(
        package_a,
        destination,
        verifier=lambda root: amendment._verify_runtime_amendment(
            root,
            base_identity=base,
            require_production=False,
        ),
    )
    bundle = destination / amendment.RUNTIME_BUNDLE_PATH
    assert bundle.is_file()
    assert (destination / amendment.TRAINING_LABELS_ARTIFACT_PATH).read_bytes() == (
        labels.read_bytes()
    )
    assert (
        _git(tmp_path, "bundle", "list-heads", str(bundle))
        == f"{head} refs/heads/fixture-runtime"
    )
    receipt = json.loads(
        (destination / amendment.RUNTIME_EXTRACTED_RECEIPT).read_text()
    )
    assert receipt["package_id"] == manifest["package_id"]
    assert receipt["training_view"] == training_view

    first_part = package_a / manifest["artifacts"][0]["parts"][0]["path"]
    payload = bytearray(first_part.read_bytes())
    payload[0] ^= 1
    first_part.write_bytes(payload)
    with pytest.raises(PackageError, match="sha256"):
        amendment._verify_runtime_amendment(
            package_a,
            base_identity=base,
            require_production=False,
        )


def test_base_verifier_is_existing_verifier_plus_exact_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "base"
    root.mkdir()
    for name in ("READY.json", "manifest.json", "SHA256SUMS"):
        (root / name).write_text("{}\n", encoding="utf-8")
    calls: list[Path] = []

    def fake_verify(path: Path) -> dict[str, object]:
        calls.append(path)
        return {
            "package_id": amendment.BASE_PACKAGE_ID,
            "package_type": "h100-source-handoff",
            "source": {
                "git_commit": amendment.RUNTIME_REQUIRED_ANCESTOR,
                "environment_lock_sha256": amendment.BASE_ENVIRONMENT_LOCK_SHA256,
            },
            "artifacts": [
                {
                    "kind": "git_bundle",
                    "archive_sha256": amendment.BASE_REPO_BUNDLE_SHA256,
                }
            ],
        }

    monkeypatch.setattr(amendment, "verify_package", fake_verify)
    with pytest.raises(PackageError, match="immutable Sprint 7d"):
        amendment.verify_base_payload(root)
    assert calls == [root]


def test_box_callback_and_runtime_cli_surface(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    expected = {"package_id": "runtime-fixture"}
    assert box._verify_transfer_package(
        package,
        require_production=True,
        verifier=lambda path: expected,
    ) == expected

    parser = _build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {
        "build-runtime",
        "verify-runtime",
        "extract-runtime",
        "upload-runtime",
        "download-runtime",
    } <= set(choices)


def test_runtime_upload_refuses_wrong_folder_before_mutation(
    tmp_path: Path,
) -> None:
    local_ready = tmp_path / "READY.json"
    local_ready.write_bytes(b"runtime-ready")
    local = {"READY.json": local_ready}
    wrong = box.RemoteFile(
        path="READY.json",
        item_id="base-ready",
        size=len(b"base-ready"),
        sha1=hashlib.sha1(b"base-ready").hexdigest(),
        item=object(),
    )
    with pytest.raises(
        box.BoxTransferError,
        match="not empty or an exact package subset",
    ):
        box._require_matching_remote_subset({"READY.json": wrong}, local)

    unexpected = box.RemoteFile(
        path="data/chips/base.tar.zst",
        item_id="base-data",
        size=1,
        sha1=hashlib.sha1(b"x").hexdigest(),
        item=object(),
    )
    with pytest.raises(box.BoxTransferError, match="unexpected"):
        box._require_matching_remote_subset(
            {"data/chips/base.tar.zst": unexpected},
            local,
        )
    assert "require_matching_remote_subset=True" in inspect.getsource(
        box.upload_package_with_verifier
    )
