from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.h100.data_staging import training_labels_summary
from scripts.handoff import final_eval_package as final_package
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


def _fixture_repo(root: Path, *, final_scenes: int = 2) -> tuple[Path, str, str]:
    repo = root / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet", "--initial-branch=fixture-final")
    _git(repo, "config", "user.name", "Repository Owner")
    _git(repo, "config", "user.email", "owner@example.invalid")
    splits = {
        "splits": {
            "train": ["train-scene"],
            "dev": [f"dev-{index}" for index in range(8)],
            "test": ["test-scene"],
            "eval_final": [f"final-{index}v" for index in range(final_scenes)],
        }
    }
    files = {
        ".gitignore": "data/raw/\n",
        "data/splits.json": json.dumps(splits, sort_keys=True) + "\n",
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture ancestor")
    ancestor = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("final package fixture\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--quiet", "-m", "fixture head")
    return repo, ancestor, _git(repo, "rev-parse", "HEAD")


def _scene_archive(
    path: Path,
    scene_id: str,
    *,
    include_vv: bool = True,
    unsafe_link: bool = False,
    bathymetry: bytes = b"opaque-bathymetry-raster-bytes",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w:gz") as archive:
        directory = tarfile.TarInfo(f"{scene_id}/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)
        members = {"VH_dB.tif": b"vh-opaque-raster-bytes"}
        if include_vv:
            members["VV_dB.tif"] = b"vv-opaque-raster-bytes"
        members["bathymetry.tif"] = bathymetry
        for name, content in members.items():
            info = tarfile.TarInfo(f"{scene_id}/{name}")
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
        if unsafe_link:
            link = tarfile.TarInfo(f"{scene_id}/unsafe-link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)


def _inputs(
    root: Path, repo: Path, *, final_scenes: int = 2
) -> tuple[Path, Path, Path]:
    labels = root / "train-dev8.csv"
    labels.write_text(
        "scene_id,value\n"
        "train-scene,1\n"
        + "".join(f"dev-{index},1\n" for index in range(8)),
        encoding="utf-8",
    )
    summary = training_labels_summary(
        labels,
        splits_path=repo / "data/splits.json",
        production=False,
    )
    assert summary["row_count"] == 9
    validation = root / "validation.csv"
    # Invalid UTF-8 and invalid CSV on purpose: package build, verification,
    # and staging must treat this artifact strictly as opaque bytes.
    validation.write_bytes(b"\xff\x00opaque-human-verified-label-bytes")
    archives = root / "archives"
    archives.mkdir()
    for index in range(final_scenes):
        scene = f"final-{index}v"
        _scene_archive(archives / f"{scene}.tar.gz", scene)
    return labels, validation, archives


def _build_fixture(
    *,
    repo: Path,
    ancestor: str,
    labels: Path,
    validation: Path,
    archives: Path,
    output: Path,
) -> Path:
    output.mkdir()
    return final_package._build_final_eval_package(
        repo_root=repo,
        training_labels_path=labels,
        validation_labels_path=validation,
        archive_dir=archives,
        output_dir=output,
        maximum_physical_file_bytes=1024,
        branch="fixture-final",
        required_ancestor=ancestor,
        production=False,
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_writable(root: Path) -> None:
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        current_path.chmod(0o755)
        for name in directories:
            (current_path / name).chmod(0o755)
        for name in files:
            (current_path / name).chmod(0o644)


def _valid_bathymetry_bytes(tmp_path: Path) -> bytes:
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin

    path = tmp_path / "valid-bathymetry.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        transform=from_origin(0.0, 1000.0, 500.0, 500.0),
    ) as dataset:
        dataset.write(np.full((1, 2, 2), -10.0, dtype=np.float32))
    return path.read_bytes()


def test_package_is_deterministic_narrow_and_validation_stays_opaque(
    tmp_path: Path,
) -> None:
    repo, ancestor, head = _fixture_repo(tmp_path)
    labels, validation, archives = _inputs(tmp_path, repo)
    package_a = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "out-a",
    )
    package_b = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "out-b",
    )
    assert package_a.name == package_b.name
    assert _tree_hashes(package_a) == _tree_hashes(package_b)

    manifest = final_package._verify_final_eval_package(
        package_a, require_production=False
    )
    assert manifest["source"]["git_commit"] == head
    assert manifest["contract"]["validation_access"] == (
        "opaque-byte-transfer-no-semantic-read-before-lock"
    )
    assert manifest["contract"]["downloaded_weights"] is False
    assert manifest["counts"] == {
        "git_bundles": 1,
        "training_label_artifacts": 1,
        "opaque_validation_label_artifacts": 1,
        "final_scene_archives": 2,
    }
    assert [record["scene_id"] for record in manifest["final_inputs"]["scenes"]] == [
        "final-0v",
        "final-1v",
    ]
    assert manifest["final_inputs"]["validation_labels"]["sha256"] == (
        hashlib.sha256(validation.read_bytes()).hexdigest()
    )
    for artifact in manifest["artifacts"][3:]:
        archive_bytes = b"".join(
            (package_a / part["path"]).read_bytes() for part in artifact["parts"]
        )
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            assert [member.name for member in archive.getmembers()] == [
                artifact["name"],
                f"{artifact['name']}/VH_dB.tif",
                f"{artifact['name']}/VV_dB.tif",
                f"{artifact['name']}/bathymetry.tif",
            ]
    kinds = {artifact["kind"] for artifact in manifest["artifacts"]}
    assert kinds == {
        "git_bundle",
        "training_labels",
        "opaque_validation_labels",
        "final_scene_archive",
    }
    physical = {
        path.relative_to(package_a).as_posix()
        for path in package_a.rglob("*")
        if path.is_file()
    }
    assert not any(
        token in path.lower()
        for path in physical
        for token in ("weights", "checkpoint", "runs", "wheelhouse", "jwt", ".venv")
    )


def test_stage_materializes_clean_checkout_vh_vv_and_immutable_receipt(
    tmp_path: Path,
) -> None:
    repo, ancestor, head = _fixture_repo(tmp_path)
    labels, validation, archives = _inputs(tmp_path, repo)
    package = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "out",
    )
    manifest = final_package._verify_final_eval_package(
        package, require_production=False
    )
    destination = tmp_path / "allocation-private-view"
    receipt = tmp_path / "FINAL_DATA_VIEW.json"
    final_package._stage_final_eval_package(
        package,
        destination,
        receipt,
        require_production=False,
        expected_package_id=str(manifest["package_id"]),
        expected_ready_sha256=hashlib.sha256(
            (package / "READY.json").read_bytes()
        ).hexdigest(),
        expected_manifest_sha256=hashlib.sha256(
            (package / "manifest.json").read_bytes()
        ).hexdigest(),
        expected_sha256sums_sha256=hashlib.sha256(
            (package / "SHA256SUMS").read_bytes()
        ).hexdigest(),
    )

    checkout = destination / "repo"
    assert _git(checkout, "rev-parse", "HEAD") == head
    assert _git(checkout, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert (checkout / "data/raw/xview3/labels/train.csv").read_bytes() == (
        labels.read_bytes()
    )
    assert (checkout / "data/raw/xview3/labels/validation.csv").read_bytes() == (
        validation.read_bytes()
    )
    for index in range(2):
        scene = checkout / f"data/raw/xview3/GRD/final-{index}v"
        assert (scene / "VH_dB.tif").read_bytes() == b"vh-opaque-raster-bytes"
        assert (scene / "VV_dB.tif").read_bytes() == b"vv-opaque-raster-bytes"
        assert (scene / "bathymetry.tif").read_bytes() == (
            b"opaque-bathymetry-raster-bytes"
        )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "status", "package", "source", "view"}
    assert payload["schema"] == 1
    assert payload["status"] == "final-eval-data-view-staged"
    assert payload["package"]["package_id"] == manifest["package_id"]
    assert payload["source"]["git_commit"] == head
    assert payload["view"]["validation_labels"] == {
        "path": "repo/data/raw/xview3/labels/validation.csv",
        "bytes": len(validation.read_bytes()),
        "sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        "access": "opaque-bytes-staged-not-semantically-read",
    }
    assert [scene["scene_id"] for scene in payload["view"]["scenes"]] == [
        "final-0v",
        "final-1v",
    ]
    assert all(
        [raster["path"].rsplit("/", 1)[-1] for raster in scene["rasters"]]
        == ["VH_dB.tif", "VV_dB.tif", "bathymetry.tif"]
        for scene in payload["view"]["scenes"]
    )
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o444
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    normalized, receipt_sha256 = final_package.validate_staged_view(
        receipt,
        expected_repo=checkout,
        expected_package_id=str(manifest["package_id"]),
        expected_evaluator_git_sha=head,
        expected_campaign_git_sha=ancestor,
        expected_splits_sha256=str(manifest["source"]["splits_sha256"]),
        production=False,
    )
    assert normalized == payload
    assert receipt_sha256 == hashlib.sha256(receipt.read_bytes()).hexdigest()

    with pytest.raises(PackageError, match="destination already exists"):
        final_package._stage_final_eval_package(
            package,
            destination,
            tmp_path / "second-receipt.json",
            require_production=False,
        )
    _make_writable(destination)


def test_stage_validation_failure_never_publishes_canonical_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, ancestor, _head = _fixture_repo(tmp_path)
    labels, validation, archives = _inputs(tmp_path, repo)
    package = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "out",
    )
    destination = tmp_path / "allocation-private-view"
    receipt = tmp_path / "FINAL_DATA_VIEW.json"

    def fail_validation(*_args: object, **_kwargs: object) -> tuple[dict, str]:
        raise PackageError("injected post-stage validation failure")

    monkeypatch.setattr(final_package, "validate_staged_view", fail_validation)
    with pytest.raises(PackageError, match="injected post-stage"):
        final_package._stage_final_eval_package(
            package,
            destination,
            receipt,
            require_production=False,
        )

    assert not os.path.lexists(receipt)
    assert not list(tmp_path.glob(".FINAL_DATA_VIEW.json.validated-*"))
    # The only residue is allocation-private scratch.  Its absence from the
    # canonical metadata namespace keeps a reviewed pre-lock retry possible.
    assert destination.is_dir()
    _make_writable(destination)


def test_public_production_stage_forwards_receipt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_stage(
        package_root: Path,
        destination: Path,
        receipt: Path,
        **kwargs: object,
    ) -> Path:
        observed.update(
            package_root=package_root,
            destination=destination,
            receipt=receipt,
            **kwargs,
        )
        return destination

    monkeypatch.setattr(final_package, "_stage_final_eval_package", fake_stage)
    package = tmp_path / "package"
    destination = tmp_path / "view"
    receipt = tmp_path / "FINAL_DATA_VIEW.json"
    assert final_package.stage_final_eval_package(
        package,
        destination,
        receipt,
        expected_package_id="package-id",
        expected_ready_sha256="1" * 64,
        expected_manifest_sha256="2" * 64,
        expected_sha256sums_sha256="3" * 64,
    ) == destination
    assert observed == {
        "package_root": package,
        "destination": destination,
        "receipt": receipt,
        "require_production": True,
        "expected_package_id": "package-id",
        "expected_ready_sha256": "1" * 64,
        "expected_manifest_sha256": "2" * 64,
        "expected_sha256sums_sha256": "3" * 64,
    }


def test_stage_post_link_fsync_failure_rolls_back_canonical_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, ancestor, _head = _fixture_repo(tmp_path)
    labels, validation, archives = _inputs(tmp_path, repo)
    package = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "out",
    )
    destination = tmp_path / "allocation-private-view"
    receipt = tmp_path / "FINAL_DATA_VIEW.json"
    original_link = os.link
    original_fsync = os.fsync
    state = {"linked": False, "raised": False}

    def observed_link(*args: object, **kwargs: object) -> None:
        original_link(*args, **kwargs)
        state["linked"] = True

    def fail_first_post_link_fsync(descriptor: int) -> None:
        if state["linked"] and not state["raised"]:
            state["raised"] = True
            raise OSError("injected receipt-directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "link", observed_link)
    monkeypatch.setattr(os, "fsync", fail_first_post_link_fsync)
    with pytest.raises(OSError, match="injected receipt-directory"):
        final_package._stage_final_eval_package(
            package,
            destination,
            receipt,
            require_production=False,
        )
    assert state == {"linked": True, "raised": True}
    assert not os.path.lexists(receipt)
    _make_writable(destination)


def test_stage_hidden_temp_cleanup_failure_keeps_committed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, ancestor, _head = _fixture_repo(tmp_path)
    labels, validation, archives = _inputs(tmp_path, repo)
    package = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "out",
    )
    destination = tmp_path / "allocation-private-view"
    receipt = tmp_path / "FINAL_DATA_VIEW.json"
    original_unlink = Path.unlink

    def fail_hidden_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(".FINAL_DATA_VIEW.json.validated-"):
            raise OSError("injected hidden-temp unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_hidden_unlink)
    assert final_package._stage_final_eval_package(
        package,
        destination,
        receipt,
        require_production=False,
    ) == destination
    assert receipt.is_file()
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o444
    assert list(tmp_path.glob(".FINAL_DATA_VIEW.json.validated-*"))
    _make_writable(destination)


def test_missing_extra_or_symlink_archive_fails_before_publication(
    tmp_path: Path,
) -> None:
    repo, ancestor, _head = _fixture_repo(tmp_path)
    labels, validation, archives = _inputs(tmp_path, repo)
    (archives / "final-1v.tar.gz").unlink()
    with pytest.raises(PackageError, match="archive inventory mismatch"):
        _build_fixture(
            repo=repo,
            ancestor=ancestor,
            labels=labels,
            validation=validation,
            archives=archives,
            output=tmp_path / "missing-out",
        )

    _scene_archive(archives / "final-1v.tar.gz", "final-1v")
    _scene_archive(archives / "extra-finalv.tar.gz", "extra-finalv")
    with pytest.raises(PackageError, match="archive inventory mismatch"):
        _build_fixture(
            repo=repo,
            ancestor=ancestor,
            labels=labels,
            validation=validation,
            archives=archives,
            output=tmp_path / "extra-out",
        )

    (archives / "extra-finalv.tar.gz").unlink()
    target = archives / "final-1v.tar.gz"
    target.unlink()
    target.symlink_to(archives / "final-0v.tar.gz")
    with pytest.raises(PackageError, match="symlink"):
        _build_fixture(
            repo=repo,
            ancestor=ancestor,
            labels=labels,
            validation=validation,
            archives=archives,
            output=tmp_path / "symlink-out",
        )


def test_verifier_detects_tamper_and_builder_rejects_unsafe_tar(
    tmp_path: Path,
) -> None:
    repo, ancestor, _head = _fixture_repo(tmp_path, final_scenes=1)
    labels, validation, archives = _inputs(tmp_path, repo, final_scenes=1)
    package = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "tamper-out",
    )
    manifest = final_package._verify_final_eval_package(
        package, require_production=False
    )
    scene_part = package / manifest["artifacts"][3]["parts"][0]["path"]
    payload = bytearray(scene_part.read_bytes())
    payload[0] ^= 1
    scene_part.write_bytes(payload)
    with pytest.raises(PackageError, match="sha256|identity|mismatch"):
        final_package._verify_final_eval_package(
            package, require_production=False
        )

    # Source validation rejects an original tar containing a symlink before
    # any package can be published.
    _scene_archive(
        archives / "final-0v.tar.gz",
        "final-0v",
        unsafe_link=True,
    )
    with pytest.raises(PackageError, match="links/devices"):
        _build_fixture(
            repo=repo,
            ancestor=ancestor,
            labels=labels,
            validation=validation,
            archives=archives,
            output=tmp_path / "unsafe-out",
        )


def test_cli_surface_and_partial_expected_identity_fail_closed() -> None:
    choices = final_package._parser()._subparsers._group_actions[0].choices
    assert {
        "build",
        "verify",
        "stage",
        "stage-training-labels",
        "upload",
        "download",
    } <= set(choices)
    with pytest.raises(PackageError, match="all three control hashes"):
        final_package._check_expected_controls(
            Path("."),
            {},
            expected_package_id="package",
            expected_ready_sha256=None,
            expected_manifest_sha256=None,
            expected_sha256sums_sha256=None,
        )


def test_training_only_stage_never_opens_validation_or_raster_payloads(
    tmp_path: Path,
) -> None:
    repo, ancestor, head = _fixture_repo(tmp_path)
    labels, validation, archives = _inputs(tmp_path, repo)
    package = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "out",
    )
    manifest = final_package._verify_final_eval_package(
        package, require_production=False
    )
    # Make every held-out payload unreadable. Control metadata remains
    # readable; the authorization-safe path must still succeed.
    for artifact in manifest["artifacts"][2:]:
        for part in artifact["parts"]:
            (package / part["path"]).chmod(0)
    output = tmp_path / "authorization-train-dev8.csv"
    result = final_package._stage_training_labels(
        package,
        output,
        require_production=False,
        expected_package_id=str(manifest["package_id"]),
        expected_ready_sha256=hashlib.sha256(
            (package / "READY.json").read_bytes()
        ).hexdigest(),
        expected_manifest_sha256=hashlib.sha256(
            (package / "manifest.json").read_bytes()
        ).hexdigest(),
        expected_sha256sums_sha256=hashlib.sha256(
            (package / "SHA256SUMS").read_bytes()
        ).hexdigest(),
    )
    assert output.read_bytes() == labels.read_bytes()
    assert result["evaluator_git_sha"] == head
    assert result["sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444


def test_staged_bathymetry_supports_shore_distance_and_unmatched_fp(
    tmp_path: Path,
) -> None:
    repo, ancestor, _head = _fixture_repo(tmp_path, final_scenes=1)
    labels, validation, archives = _inputs(tmp_path, repo, final_scenes=1)
    _scene_archive(
        archives / "final-0v.tar.gz",
        "final-0v",
        bathymetry=_valid_bathymetry_bytes(tmp_path),
    )
    package = _build_fixture(
        repo=repo,
        ancestor=ancestor,
        labels=labels,
        validation=validation,
        archives=archives,
        output=tmp_path / "out",
    )
    destination = tmp_path / "allocation-private-view"
    receipt = tmp_path / "FINAL_DATA_VIEW.json"
    final_package._stage_final_eval_package(
        package,
        destination,
        receipt,
        require_production=False,
    )

    from src.eval.infer_scene import ShoreDistance
    from src.eval.scorer import GroundTruthPoint, PredictionPoint, score_dataset

    scene_dir = destination / "repo/data/raw/xview3/GRD/final-0v"
    shore = ShoreDistance(scene_dir)
    assert shore.available
    prediction_distance = shore.lookup_km(0.0, 0.0)
    assert prediction_distance is not None
    result = score_dataset(
        {
            "final-0v": [
                GroundTruthPoint(
                    x_m=10_000.0,
                    y_m=10_000.0,
                    distance_from_shore_km=5.0,
                )
            ]
        },
        {
            "final-0v": [
                PredictionPoint(
                    x_m=0.0,
                    y_m=0.0,
                    score=0.9,
                    distance_from_shore_km=prediction_distance,
                )
            ]
        },
    )
    assert result.aggregate.fp == 1
    _make_writable(destination)
