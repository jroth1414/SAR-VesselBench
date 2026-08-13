from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.handoff.final_eval_bootstrap import (
    _bootstrap_spec,
    _generate_final_eval_bootstrap,
)
from scripts.handoff.package import PackageError


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _hash(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    value.update(path.read_bytes())
    return value.hexdigest()


def _canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _part(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _hash(path),
        "sha1": _hash(path, "sha1"),
    }


def _split_file(
    source: Path,
    root: Path,
    relative: str,
    *,
    maximum: int,
) -> list[dict[str, object]]:
    payload = source.read_bytes()
    records = []
    for index, offset in enumerate(range(0, len(payload), maximum)):
        part = root / f"{relative}.part-{index:05d}"
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(payload[offset : offset + maximum])
        records.append(_part(part, root))
    return records


def _fixture_package(tmp_path: Path) -> tuple[Path, Path, dict[str, object], str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--initial-branch=fixture-final")
    _git(repo, "config", "user.name", "Repository Owner")
    _git(repo, "config", "user.email", "owner@example.invalid")
    (repo / "README.md").write_text("ancestor\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--quiet", "-m", "fixture ancestor")
    ancestor = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("exact amendment\n", encoding="utf-8")
    _git(repo, "commit", "--quiet", "-am", "fixture amendment")
    commit = _git(repo, "rev-parse", "HEAD")

    logical_bundle = tmp_path / "logical.bundle"
    _git(
        repo,
        "bundle",
        "create",
        str(logical_bundle),
        "refs/heads/fixture-final",
    )
    package = tmp_path / "package"
    package.mkdir()
    bundle_parts = _split_file(
        logical_bundle,
        package,
        "code/xview3-final-eval.bundle",
        maximum=97,
    )
    assert len(bundle_parts) > 1

    train = package / "data/final-inputs/training-view/train.csv"
    train.parent.mkdir(parents=True)
    train.write_bytes(b"scene_id,value\ntrain-scene,1\n")
    validation_source = tmp_path / "validation-source.bin"
    validation_source.write_bytes(b"\xff\x00opaque-human-labels-not-csv")
    validation_parts = _split_file(
        validation_source,
        package,
        "data/final-inputs/labels/validation.csv",
        maximum=7,
    )
    raster_source = tmp_path / "opaque-raster-archive.bin"
    raster_source.write_bytes(b"opaque-raster-archive-bytes" * 4)
    raster_parts = _split_file(
        raster_source,
        package,
        "data/final-inputs/rasters/final-0v.tar.gz",
        maximum=13,
    )

    def artifact(
        kind: str,
        name: str,
        logical: Path,
        parts: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "kind": kind,
            "name": name,
            "archive_bytes": logical.stat().st_size,
            "archive_sha256": _hash(logical),
            "archive_sha1": _hash(logical, "sha1"),
            "parts": parts,
        }

    artifacts = [
        artifact("git_bundle", "fixture-final", logical_bundle, bundle_parts),
        artifact("training_labels", "train-dev8.csv", train, [_part(train, package)]),
        artifact(
            "opaque_validation_labels",
            "validation.csv",
            validation_source,
            validation_parts,
        ),
        artifact(
            "final_scene_archive",
            "final-0v",
            raster_source,
            raster_parts,
        ),
    ]
    identity = "a" * 64
    package_id = f"xview3-h100-final-eval-{commit}-{identity}"
    manifest: dict[str, object] = {
        "package_type": "h100-final-eval-inputs",
        "package_id": package_id,
        "identity_sha256": identity,
        "source": {
            "branch": "fixture-final",
            "git_bundle_ref": "refs/heads/fixture-final",
            "git_commit": commit,
            "required_campaign_commit": ancestor,
            "git_bundle_sha256": _hash(logical_bundle),
        },
        "artifacts": artifacts,
    }
    _canonical(package / "manifest.json", manifest)
    payload_parts = [part for item in artifacts for part in item["parts"]]
    (package / "SHA256SUMS").write_text(
        "".join(f"{part['sha256']}  {part['path']}\n" for part in payload_parts),
        encoding="utf-8",
    )
    _canonical(
        package / "READY.json",
        {
            "status": "READY",
            "package_id": package_id,
            "git_commit": commit,
            "identity_sha256": identity,
        },
    )
    return repo, package, manifest, ancestor, commit


def _fake_boxsdk(tmp_path: Path) -> Path:
    root = tmp_path / "fake-sdk"
    package = root / "boxsdk"
    session = package / "session"
    session.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """
import hashlib
import os
from pathlib import Path

ROOT = Path(os.environ["FAKE_BOX_ROOT"])

def _entry(path, relative):
    if path.is_dir():
        return {"id": "d:" + relative, "name": path.name, "type": "folder"}
    value = hashlib.sha1(path.read_bytes()).hexdigest()
    return {"id": "f:" + relative, "name": path.name, "type": "file",
            "sha1": value, "size": path.stat().st_size}

class JWTAuth:
    @classmethod
    def from_settings_file(cls, _path, session=None):
        return cls()
    def authenticate_instance(self):
        return None

class _Folder:
    def __init__(self, identifier):
        self.relative = identifier[2:] if identifier.startswith("d:") else ""
    def get_items(self, limit, offset, fields):
        directory = ROOT / self.relative
        values = []
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(ROOT).as_posix()
            values.append(_entry(path, relative))
        return values[offset:offset + limit]

class _File:
    def __init__(self, identifier):
        self.relative = identifier[2:]
    def download_to(self, handle):
        handle.write((ROOT / self.relative).read_bytes())

class Client:
    def __init__(self, auth, session=None):
        pass
    def folder(self, identifier):
        return _Folder(identifier)
    def file(self, identifier):
        return _File(identifier)
""".lstrip(),
        encoding="utf-8",
    )
    (session / "__init__.py").write_text("", encoding="utf-8")
    (session / "session.py").write_text(
        """
class Session:
    def __init__(self, **kwargs):
        pass

class AuthorizedSession:
    def __init__(self, auth, **kwargs):
        pass
""".lstrip(),
        encoding="utf-8",
    )
    return root


def test_generated_bootstrap_is_complete_deterministic_and_executable(
    tmp_path: Path,
) -> None:
    repo, package, manifest, ancestor, commit = _fixture_package(tmp_path)
    output = tmp_path / "pull-final-eval.sh"
    result = _generate_final_eval_bootstrap(
        repo_root=repo,
        package_root=package,
        output=output,
        verifier=lambda _root: manifest,
        production=False,
    )
    subprocess.run(["bash", "-n", str(output)], check=True)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    script = output.read_text(encoding="utf-8")
    spec = _bootstrap_spec(package, manifest)
    assert result["remote_file_count"] == len(spec["files"])
    for relative, record in spec["files"].items():
        assert relative in script
        assert record["sha1"] in script
        assert record["sha256"] in script
    for pinned in (
        manifest["package_id"],
        commit,
        ancestor,
        spec["bundle_sha256"],
        spec["ready_sha256"],
        spec["manifest_sha256"],
        spec["sha256sums_sha256"],
    ):
        assert pinned in script
    assert len(spec["bundle_parts"]) > 1
    assert ".partial" in script
    assert "refusing to overwrite existing bootstrap" in script
    assert "merge-base" in script
    assert "fsck" in script
    assert "scripts.handoff" not in script
    assert "csv.reader" not in script
    assert "read_csv" not in script
    assert "rasterio" not in script
    assert "tarfile" not in script

    second = tmp_path / "pull-final-eval-second.sh"
    _generate_final_eval_bootstrap(
        repo_root=repo,
        package_root=package,
        output=second,
        verifier=lambda _root: manifest,
        production=False,
    )
    assert output.read_bytes() == second.read_bytes()
    with pytest.raises(PackageError, match="output exists"):
        _generate_final_eval_bootstrap(
            repo_root=repo,
            package_root=package,
            output=output,
            verifier=lambda _root: manifest,
            production=False,
        )

    fake_sdk = _fake_boxsdk(tmp_path)
    wrapper = tmp_path / "transfer-python"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "shift\n"
        f"export PYTHONPATH={fake_sdk}\n"
        f"exec {Path(os.sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    jwt = tmp_path / "jwt.json"
    jwt.write_text("{}\n", encoding="utf-8")
    jwt.chmod(0o600)
    target = tmp_path / "judy-handoff"
    target.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "TRANSFER_PYTHON": str(wrapper),
            "XVIEW3_TARGET_ROOT": str(target),
            "BOX_JWT_CONFIG": str(jwt),
            "BOX_FOLDER_ID": "938475029184750293",
            "FAKE_BOX_ROOT": str(package),
        }
    )
    completed = subprocess.run(
        ["bash", str(output)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "final-eval-bootstrap-complete"
    assert receipt["commit"] == commit
    downloaded = Path(receipt["package_root"])
    assert {
        path.relative_to(downloaded).as_posix(): _hash(path)
        for path in downloaded.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(package).as_posix(): _hash(path)
        for path in package.rglob("*")
        if path.is_file()
    }
    checkout = Path(receipt["checkout"])
    assert _git(checkout, "rev-parse", "HEAD") == commit
    assert _git(checkout, "branch", "--show-current") == "fixture-final"
    assert _git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert _git(checkout, "remote") == ""
    assert str(jwt) not in completed.stdout
    assert environment["BOX_FOLDER_ID"] not in completed.stdout


def test_production_generator_refuses_repository_output_and_extra_package_file(
    tmp_path: Path,
) -> None:
    repo, package, manifest, _ancestor, _commit = _fixture_package(tmp_path)
    with pytest.raises(PackageError, match="outside the repository"):
        _generate_final_eval_bootstrap(
            repo_root=repo,
            package_root=package,
            output=repo / "pull-final-eval.sh",
            verifier=lambda _root: manifest,
            production=True,
        )

    (package / "unexpected.bin").write_bytes(b"unexpected")
    with pytest.raises(PackageError, match="inventory differs"):
        _bootstrap_spec(package, manifest)
