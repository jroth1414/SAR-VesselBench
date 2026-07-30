from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.handoff import box
from scripts.handoff import runtime_amendment as amendment
from scripts.handoff.__main__ import _build_parser
from scripts.handoff.package import PackageError


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
        "scripts/h100/build_venv.py": "RUNTIME = 'native-venv'\n",
        "slurm/h100/campaign.sbatch": "#!/bin/bash\n# native venv campaign\n",
        "slurm/h100/smoke.sbatch": "#!/bin/bash\n# native venv smoke\n",
        "slurm/h100/submit.sh": "#!/bin/bash\n# native venv submit\n",
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


def test_runtime_package_is_deterministic_code_only_and_extracts(
    tmp_path: Path,
) -> None:
    repo, ancestor, head = _fixture_repo(tmp_path)
    base = amendment._expected_base_identity()
    output_a = tmp_path / "out-a"
    output_b = tmp_path / "out-b"
    output_a.mkdir()
    output_b.mkdir()

    package_a = amendment._build_runtime_amendment(
        repo_root=repo,
        output_dir=output_a,
        maximum_physical_file_bytes=1024,
        base_identity=base,
        branch="fixture-runtime",
        required_ancestor=ancestor,
        production=False,
    )
    package_b = amendment._build_runtime_amendment(
        repo_root=repo,
        output_dir=output_b,
        maximum_physical_file_bytes=1024,
        base_identity=base,
        branch="fixture-runtime",
        required_ancestor=ancestor,
        production=False,
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
    assert manifest["contract"]["runtime"] == "native-venv"
    assert manifest["contract"]["precision"] == "32-true"
    assert manifest["contract"]["tf32"] is False
    assert manifest["counts"] == {"git_bundles": 1}
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
            for part in manifest["artifacts"][0]["parts"]
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
    assert (
        _git(tmp_path, "bundle", "list-heads", str(bundle))
        == f"{head} refs/heads/fixture-runtime"
    )
    receipt = json.loads(
        (destination / amendment.RUNTIME_EXTRACTED_RECEIPT).read_text()
    )
    assert receipt["package_id"] == manifest["package_id"]

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
