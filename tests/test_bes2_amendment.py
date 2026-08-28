from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.handoff import bes2_amendment as amendment
from scripts.handoff.__main__ import _build_parser
from scripts.handoff.box import upload_package_with_verifier
from scripts.handoff.package import PackageError, _canonical_json, _hash_file
from test_h100_handoff import _Client


def _run(*argv: str, cwd: Path) -> str:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _fixture_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, str, list[bool]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-q", "-b", amendment.BRANCH, cwd=repo)
    _run("git", "config", "user.name", "John Roth", cwd=repo)
    _run(
        "git",
        "config",
        "user.email",
        "jroth1414@users.noreply.github.com",
        cwd=repo,
    )
    (repo / "README.md").write_text("BES2 fixture\n", encoding="utf-8")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "fixture ancestor", cwd=repo)
    ancestor = _run("git", "rev-parse", "HEAD", cwd=repo)
    (repo / "diagnostic.py").write_text("VARIANTS = 2\n", encoding="utf-8")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "fixture head", cwd=repo)

    strict = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }
    ready = tmp_path / "H100_READY.json"
    ready.write_bytes(
        _canonical_json(
            {
                "schema": 2,
                "status": "ready",
                "acceptance_uuid": "00000000-0000-4000-8000-000000000000",
                "source": {"git_sha": "1" * 40},
                "strict_fp32": strict,
                "venv": {
                    "sha256": "2" * 64,
                    "venv_build_sha256": "3" * 64,
                    "base_python": {"sha256": "4" * 64},
                    "wheelhouse": {"sha256": "5" * 64},
                },
                "base_payload": {"identity_sha256": "6" * 64},
                "runtime_amendment": {"identity_sha256": "7" * 64},
            }
        )
    )
    weights = tmp_path / "weights"
    checkpoint = weights / amendment.S2_CHECKPOINT_RELATIVE
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture S2 checkpoint bytes")
    checkpoint_sha = _hash_file(checkpoint)

    monkeypatch.setattr(amendment, "REQUIRED_ANCESTOR", ancestor)
    monkeypatch.setattr(amendment, "S2_CHECKPOINT_SHA256", checkpoint_sha)

    def create_bundle(
        source_repo: Path,
        output: Path,
        branch: str,
        commit: str,
        required_ancestor: str,
        *,
        production: bool,
    ) -> None:
        assert branch == amendment.BRANCH
        assert commit == _run("git", "rev-parse", "HEAD", cwd=source_repo)
        assert required_ancestor == ancestor
        assert production is False
        output.parent.mkdir(parents=True, exist_ok=True)
        _run(
            "git",
            "bundle",
            "create",
            str(output),
            f"refs/heads/{branch}",
            cwd=source_repo,
        )

    verify_calls: list[bool] = []

    def verify_bundle(
        bundle: Path,
        *,
        branch: str,
        commit: str,
        required_ancestor: str,
        production: bool,
    ) -> dict[str, object]:
        assert (
            _run("git", "bundle", "list-heads", str(bundle), cwd=repo)
            == f"{commit} refs/heads/{branch}"
        )
        assert required_ancestor == ancestor
        verify_calls.append(production)
        return {"status": "verified"}

    monkeypatch.setattr(amendment, "_create_git_bundle", create_bundle)
    monkeypatch.setattr(
        amendment,
        "_verify_git_bundle_round_trip",
        verify_bundle,
    )
    return repo, ready, weights, _hash_file(ready), verify_calls


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str = "output",
) -> tuple[Path, Path, str, list[bool]]:
    repo, ready, weights, ready_sha, verify_calls = _fixture_source(
        tmp_path, monkeypatch
    )
    output = tmp_path / name
    output.mkdir()
    package = amendment.build_bes2_amendment(
        repo_root=repo,
        h100_ready_json=ready,
        weights_root=weights,
        output_dir=output,
        maximum_physical_file_bytes=8192,
    )
    return repo, package, ready_sha, verify_calls


def _tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_bes2_amendment_is_deterministic_closed_bootstrapped_and_ready_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, ready, weights, ready_sha, verify_calls = _fixture_source(
        tmp_path, monkeypatch
    )
    packages = []
    for name in ("first", "second"):
        output = tmp_path / name
        output.mkdir()
        packages.append(
            amendment.build_bes2_amendment(
                repo_root=repo,
                h100_ready_json=ready,
                weights_root=weights,
                output_dir=output,
                maximum_physical_file_bytes=8192,
            )
        )
    first, second = packages
    assert first.name == second.name
    assert _tree(first) == _tree(second)

    manifest = amendment.verify_bes2_amendment(
        first,
        expected_h100_ready_sha256=ready_sha,
    )
    assert manifest["contract"]["data_artifacts"] == 0
    assert manifest["contract"]["checkpoint_artifacts"] == 0
    assert manifest["contract"]["test_or_final_artifacts"] == 0
    assert manifest["contract"]["canonical_h100_ready_mutated"] is False
    assert manifest["contract"]["training_fractions"] == {
        "probe": [0.5],
        "full": [0.1, 0.5],
    }
    assert manifest["contract"]["fraction_analysis"] == (
        "per-fraction-no-pooling"
    )
    assert manifest["contract"]["diagnostic_run_count"] == {
        "probe": 2,
        "full": 4,
    }
    assert manifest["contract"]["maximum_approved_gpu_hours"] == 190.0
    assert manifest["source"]["required_ancestor"] == amendment.REQUIRED_ANCESTOR
    assert manifest["checkpoint"]["relative_path"] == amendment.S2_CHECKPOINT_RELATIVE
    physical = set(_tree(first))
    assert physical == {
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
        for token in ("test", "final", "weights", "checkpoint", "jwt", "token")
    )
    assert verify_calls and all(verify_calls)

    bootstrap = tmp_path / "pull-bes2.sh"
    receipt = amendment.generate_bes2_bootstrap(
        repo_root=repo,
        package_root=first,
        output=bootstrap,
        expected_h100_ready_sha256=ready_sha,
    )
    subprocess.run(["bash", "-n", str(bootstrap)], check=True)
    source = bootstrap.read_text(encoding="utf-8")
    assert stat.S_IMODE(bootstrap.stat().st_mode) == 0o700
    assert receipt["package_id"] == manifest["package_id"]
    assert manifest["source"]["git_commit"] in source
    assert "READY.json" in source
    assert ".partial" in source
    assert "scripts.handoff" not in source

    client = _Client(chunk_start_mode="none")
    upload_receipt = tmp_path / "upload-receipt.json"
    upload_package_with_verifier(
        client,
        "0",
        first,
        repo_root=repo,
        receipt_path=upload_receipt,
        verifier=amendment.prepare_bes2_verifier(ready_sha),
        chunked_threshold=1,
    )
    stored = [name for action, name in client.events if action == "stored"]
    assert stored[-1] == "READY.json"
    assert upload_receipt.is_file()

    choices = _build_parser()._subparsers._group_actions[0].choices
    assert {
        "build-bes2-amendment",
        "verify-bes2-amendment",
        "upload-bes2-amendment",
        "build-bes2-bootstrap",
    } <= set(choices)


@pytest.mark.parametrize("tamper", ["extra", "ready", "credential"])
def test_bes2_amendment_rejects_nonallowlisted_or_malformed_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _repo, package, ready_sha, _calls = _build(tmp_path, monkeypatch)
    if tamper == "extra":
        (package / "test_metrics.json").write_text("{}\n", encoding="utf-8")
        expected = "package tree"
    elif tamper == "credential":
        (package / "box-jwt.json").write_text("{}\n", encoding="utf-8")
        expected = "package tree"
    else:
        ready_path = package / "READY.json"
        payload = json.loads(ready_path.read_text(encoding="utf-8"))
        payload["extra"] = True
        ready_path.write_bytes(_canonical_json(payload))
        expected = "READY/package identity"
    with pytest.raises(PackageError, match=expected):
        amendment.verify_bes2_amendment(
            package,
            expected_h100_ready_sha256=ready_sha,
        )


def test_bes2_amendment_rejects_wrong_h100_binding_and_strict_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, package, _ready_sha, _calls = _build(tmp_path, monkeypatch)
    with pytest.raises(PackageError, match="H100 binding"):
        amendment.verify_bes2_amendment(
            package,
            expected_h100_ready_sha256="f" * 64,
        )

    ready = tmp_path / "bad-H100_READY.json"
    ready.write_bytes(
        _canonical_json(
            {
                "schema": 2,
                "status": "ready",
                "strict_fp32": {
                    "cuda_matmul_fp32_precision": "ieee",
                    "cudnn_conv_fp32_precision": "ieee",
                },
                "venv": {},
                "base_payload": {},
                "runtime_amendment": {},
            }
        )
    )
    with pytest.raises(PackageError, match="H100_READY identity"):
        amendment._h100_identity(ready)
