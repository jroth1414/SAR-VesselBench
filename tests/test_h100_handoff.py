from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

import scripts.handoff.box as handoff_box
import scripts.handoff.package as handoff_package
from scripts.h100.contracts import cutover_acceptance_bindings, load_cells
from scripts.h100.wheelhouse import wheelhouse_identity
from scripts.handoff.box import (
    BoxTransferError,
    CHUNKED_UPLOAD_THRESHOLD,
    _download_fixture_package,
    _manifest_physical,
    _upload_fixture_package,
    _upload_one,
    download_package,
    preflight_box,
    upload_package,
)
from scripts.handoff.package import (
    BuildOptions,
    PackageError,
    REQUIRED_WEIGHT_DIRS,
    _scan_secret_content,
    _require_source_ancestor,
    _canonical_json,
    _create_tar_zst,
    _extract_fixture_package,
    _physical_record,
    _source_entries,
    _verify_fixture_package,
    build_package,
    extract_package,
    verify_package,
    validate_wheelhouse,
)
from scripts.handoff.results import (
    _result_package_id,
    _validate_attempts,
    build_results_package,
)


def _run(*argv: str, cwd: Path) -> str:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _write(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)


def _fixture_source(tmp_path: Path) -> tuple[BuildOptions, str]:
    repo = tmp_path / "repo"
    data = tmp_path / "payload-data"
    wheelhouse = tmp_path / "wheelhouse"
    package = tmp_path / "package-a"
    repo.mkdir()
    _run("git", "init", "-q", "-b", "sprint-7d-h100-fp32", cwd=repo)
    _run("git", "config", "user.name", "John Roth", cwd=repo)
    _run(
        "git",
        "config",
        "user.email",
        "jroth1414@users.noreply.github.com",
        cwd=repo,
    )
    split = {
        "splits": {
            "train": ["train-scene"],
            "dev": ["dev-scene"],
            "test": [],
            "eval_final": ["eval-scene"],
        }
    }
    _write(repo / "data/splits.json", json.dumps(split))
    _write(
        repo / "locks/env-v100node.txt",
        "# fixture exact lock\n"
        "torch==2.11.0+cu126\n",
    )
    definition = repo / "containers/h100-strict-fp32.def"
    _write(
        definition,
        "Bootstrap: docker\n"
        "From: python:3.11.15-slim-bookworm@sha256:"
        + "a" * 64
        + "\n",
    )
    _write(repo / "README.md", "fixture source\n")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "P7d: fixture", cwd=repo)
    commit = _run("git", "rev-parse", "HEAD", cwd=repo)

    for scene in ("train-scene", "dev-scene"):
        _write(data / "chips" / scene / "chip.bin", f"chip:{scene}".encode())
    _write(
        data / "raw/xview3/GRD/dev-scene/scene.tif",
        bytes(range(256)) * 8,
    )
    _write(
        data / "raw/xview3/labels/train.csv",
        "scene_id,x,y\ntrain-scene,1,2\ndev-scene,3,4\n",
    )
    for name, metadata in REQUIRED_WEIGHT_DIRS.items():
        root = data / "weights" / name
        _write(
            root / "SOURCE.note",
            f"hf://{metadata['model']}@{metadata['revision']}\n",
        )
        _write(root / "LICENSE.note", f"{metadata['license']}\n")
        checkpoint = {
            "satdino": "satdino-vit_base-16.pth",
            "sarmae": "SARMAE_vitb_checkpoint-last",
        }.get(name, "model.safetensors")
        _write(root / checkpoint, f"weights:{name}".encode())
        _write(root / ".cache/ignored.bin", b"cache")
    wheelhouse.mkdir()
    _write(
        wheelhouse / "torch-2.11.0+cu126-py3-none-any.whl",
        b"fixture wheel",
    )
    options = BuildOptions(
        repo_root=repo,
        data_root=data,
        package_root=package,
        wheelhouse=wheelhouse,
        apptainer_definition=definition,
        max_part_bytes=512,
        production=False,
    )
    return options, commit


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_fixture_build_is_deterministic_full_sha_and_extracts(tmp_path):
    options, commit = _fixture_source(tmp_path)
    first = build_package(options)
    second = build_package(
        BuildOptions(
            **{
                **options.__dict__,
                "package_root": tmp_path / "package-b",
            }
        )
    )
    assert _files(first) == _files(second)
    manifest = _verify_fixture_package(first)
    assert manifest["package_id"] == f"xview3-h100-fp32-{commit}"
    assert manifest["counts"] == {
        "chip_archives": 2,
        "raster_archives": 1,
        "label_archives": 1,
        "core_weight_archives": 6,
    }
    assert len(manifest["scenes"]["chips"]) == 2
    assert manifest["scenes"]["rasters"] == ["dev-scene"]
    assert {item["directory"] for item in manifest["weights"]} == set(
        REQUIRED_WEIGHT_DIRS
    )
    assert not any("eval" in name for name in _files(first))
    extracted = _extract_fixture_package(first, tmp_path / "extracted")
    assert (extracted / "data/raw/xview3/labels/train.csv").is_file()
    assert not (extracted / "data/raw/xview3/labels/validation.csv").exists()
    assert not any(".cache" in path.parts for path in extracted.rglob("*"))
    for name in REQUIRED_WEIGHT_DIRS:
        assert (extracted / f"data/weights/{name}/SOURCE.note").is_file()
        assert (extracted / f"data/weights/{name}/LICENSE.note").is_file()
    cloned = tmp_path / "clone"
    _run(
        "git",
        "clone",
        "-q",
        "-b",
        "sprint-7d-h100-fp32",
        str(extracted / "code/xview3.bundle"),
        str(cloned),
        cwd=tmp_path,
    )
    assert _run("git", "rev-parse", "HEAD", cwd=cloned) == commit


def test_inventory_and_secret_leakage_fail_closed(tmp_path):
    options, _ = _fixture_source(tmp_path)
    _write(options.data_root / "chips/extra-scene/chip.bin", b"extra")
    with pytest.raises(PackageError, match="inventory mismatch"):
        build_package(options)

    # The scanner's own source contains marker literals and must not self-fire.
    _scan_secret_content(Path(__file__).parents[1] / "scripts/handoff/package.py")
    jwt = tmp_path / "jwt.json"
    _write(
        jwt,
        '{"boxAppSettings":{"clientID":"x","clientSecret":"actual-secret"}}',
    )
    with pytest.raises(PackageError, match="credential material"):
        _scan_secret_content(jwt)
    pem = tmp_path / "innocent.txt"
    _write(
        pem,
        "-----BEGIN PRIVATE KEY-----\n"
        + "A" * 80
        + "\n-----END PRIVATE KEY-----\n",
    )
    with pytest.raises(PackageError, match="credential material"):
        _scan_secret_content(pem)
    large_log = tmp_path / "large.log"
    _write(
        large_log,
        b"x" * (5 * 1024 * 1024)
        + b"\nBOX_JWT_CONFIG=/outside/repo/real-jwt.json\n",
    )
    with pytest.raises(PackageError, match="credential material"):
        _scan_secret_content(large_log)
    dumped_box_config = tmp_path / "slurm-job.out"
    _write(
        dumped_box_config,
        '{"boxAppSettings":{"clientID":"x","clientSecret":"actual-secret"}}',
    )
    with pytest.raises(PackageError, match="credential material"):
        _scan_secret_content(dumped_box_config)
    encrypted_pem = tmp_path / "encrypted-key.log"
    _write(
        encrypted_pem,
        "-----BEGIN ENCRYPTED PRIVATE KEY-----\n"
        + "A" * 80
        + "\n-----END ENCRYPTED PRIVATE KEY-----\n",
    )
    with pytest.raises(PackageError, match="credential material"):
        _scan_secret_content(encrypted_pem)
    dotenv = tmp_path / ".env"
    _write(
        dotenv,
        '{"boxAppSettings":{"clientID":"x","clientSecret":"actual-secret"}}',
    )
    with pytest.raises(PackageError, match="credential material"):
        _scan_secret_content(dotenv)


def test_train_labels_require_core_coverage_and_exclude_eval_final(tmp_path):
    options, _ = _fixture_source(tmp_path)
    labels = options.data_root / "raw/xview3/labels/train.csv"
    labels.write_text(
        labels.read_text(encoding="utf-8") + "eval-scene,9,9\n",
        encoding="utf-8",
    )
    with pytest.raises(PackageError, match="eval_final scene leaked into train.csv"):
        build_package(options)

    labels.write_text("scene_id,x,y\ntrain-scene,1,2\n", encoding="utf-8")
    with pytest.raises(PackageError, match="missing core scenes"):
        build_package(options)


def test_wheelhouse_is_exact_and_contains_only_regular_wheels(tmp_path):
    options, _ = _fixture_source(tmp_path)
    lock = options.repo_root / "locks/env-v100node.txt"
    validate_wheelhouse(options.wheelhouse, lock)
    _write(
        options.wheelhouse / "unexpected-1.0-py3-none-any.whl",
        b"extra",
    )
    with pytest.raises(PackageError, match="outside the exact lock"):
        validate_wheelhouse(options.wheelhouse, lock)
    (options.wheelhouse / "unexpected-1.0-py3-none-any.whl").unlink()
    _write(options.wheelhouse / "README.txt", "not a wheel")
    with pytest.raises(PackageError, match="non-wheel"):
        validate_wheelhouse(options.wheelhouse, lock)


def test_wheelhouse_download_resolves_only_the_exact_lock(
    tmp_path, monkeypatch
):
    options, _ = _fixture_source(tmp_path)
    output = tmp_path / "fresh-wheelhouse"
    commands = []

    def fake_run(argv, *, cwd=None, capture=True):
        command = list(argv)
        commands.append(command)
        if command[-1] == "--version":
            return "Python 3.11.15"
        destination = Path(command[command.index("--dest") + 1])
        _write(
            destination / "torch-2.11.0+cu126-py3-none-any.whl",
            b"fixture wheel",
        )
        return ""

    monkeypatch.setattr(handoff_package, "_run", fake_run)
    monkeypatch.setattr(
        handoff_package, "_inside_repository_worktrees", lambda *_args: False
    )
    handoff_package.build_wheelhouse(
        repo_root=options.repo_root,
        output=output,
        python=tmp_path / "python3.11",
    )
    download = commands[1]
    assert "--only-binary=:all:" in download
    assert "--no-deps" in download
    assert [path.name for path in output.iterdir()] == [
        "torch-2.11.0+cu126-py3-none-any.whl"
    ]


def test_transfer_dependencies_are_exact_and_cli_import_is_lazy():
    repo = Path(__file__).parents[1]
    requirements = (
        repo / "requirements-transfer.txt"
    ).read_text(encoding="utf-8").splitlines()
    pins = {line for line in requirements if line and not line.startswith("#")}
    assert pins == {
        "boxsdk[jwt]==4.13.0",
        "packaging==26.2",
        "PyYAML==6.0.3",
        "pytest==9.1.1",
    }
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "yaml" or name.startswith("yaml."):
        raise ModuleNotFoundError("fixture blocks yaml")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import scripts.handoff.__main__
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_source_ancestry_helper_requires_reachable_commit(tmp_path):
    options, base = _fixture_source(tmp_path)
    repo = options.repo_root
    _write(repo / "second.txt", "second\n")
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "P7d: descendant", cwd=repo)
    head = _run("git", "rev-parse", "HEAD", cwd=repo)
    _require_source_ancestor(repo, head, base)
    with pytest.raises(PackageError, match="not descended"):
        _require_source_ancestor(repo, head, "0" * 40)


def test_source_root_symlink_is_rejected_before_resolution(tmp_path):
    target = tmp_path / "actual"
    _write(target / "file.txt", "payload")
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(PackageError, match="symlinks are not allowed"):
        _source_entries(link, Path("payload"))


def test_all_linked_worktrees_share_one_forbidden_boundary(tmp_path, monkeypatch):
    primary = tmp_path / "live"
    primary.mkdir()
    _run("git", "init", "-q", "-b", "dev", cwd=primary)
    _run("git", "config", "user.name", "John Roth", cwd=primary)
    _run(
        "git",
        "config",
        "user.email",
        "jroth1414@users.noreply.github.com",
        cwd=primary,
    )
    _write(primary / "README.md", "fixture\n")
    _run("git", "add", ".", cwd=primary)
    _run("git", "commit", "-q", "-m", "fixture", cwd=primary)
    linked = primary / ".venv-worktrees/sprint7d"
    linked.parent.mkdir()
    _run("git", "worktree", "add", "-q", "-b", "sprint-7d", str(linked), cwd=primary)

    jwt = primary / "box-jwt.json"
    _write(jwt, "{}")
    jwt.chmod(0o600)
    monkeypatch.setenv("BOX_JWT_CONFIG", str(jwt))
    monkeypatch.setenv("BOX_FOLDER_ID", "123")
    with pytest.raises(BoxTransferError, match="outside the repository"):
        handoff_box.credentials_from_environment(linked)
    assert handoff_package._inside_repository_worktrees(
        primary / "handoff-output", linked
    )


def test_box_jwt_path_rejects_symlink_before_resolution(tmp_path, monkeypatch):
    options, _ = _fixture_source(tmp_path)
    external = tmp_path / "external-box-jwt.json"
    _write(external, "{}")
    external.chmod(0o600)
    linked = options.repo_root / "box-jwt.json"
    linked.symlink_to(external)
    monkeypatch.setenv("BOX_JWT_CONFIG", str(linked))
    monkeypatch.setenv("BOX_FOLDER_ID", "123")

    with pytest.raises(BoxTransferError, match="no symlink components"):
        handoff_box.credentials_from_environment(options.repo_root)


def test_boxsdk_logging_is_silent_and_configuration_is_idempotent(
    monkeypatch, caplog
):
    sdk_logger = logging.getLogger("boxsdk")
    child_logger = logging.getLogger("boxsdk.network.default_network")
    monkeypatch.setattr(sdk_logger, "handlers", [])
    monkeypatch.setattr(sdk_logger, "level", logging.NOTSET)
    monkeypatch.setattr(sdk_logger, "propagate", True)
    monkeypatch.setattr(child_logger, "handlers", [])
    monkeypatch.setattr(child_logger, "level", logging.NOTSET)
    monkeypatch.setattr(child_logger, "propagate", True)

    handoff_box._silence_boxsdk_logging()
    handoff_box._silence_boxsdk_logging()

    assert sdk_logger.level == logging.CRITICAL + 1
    assert sdk_logger.propagate is False
    assert sdk_logger.handlers == [handoff_box._BOXSDK_NULL_HANDLER]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        child_logger.warning("sensitive request metadata")
    assert caplog.records == []


def test_box_client_applies_timeouts_to_auth_and_data_sessions(
    tmp_path, monkeypatch
):
    options, _ = _fixture_source(tmp_path)
    config = tmp_path / "box-jwt.json"
    _write(config, "{}")
    config.chmod(0o600)
    monkeypatch.setenv("BOX_JWT_CONFIG", str(config))
    monkeypatch.setenv("BOX_FOLDER_ID", "123")

    plain_sessions = []
    authorized_sessions = []
    settings_sessions = []
    authentication_calls = []
    client_sessions = []

    class Session:
        def __init__(self, *, default_network_request_kwargs):
            self.request_kwargs = dict(default_network_request_kwargs)
            plain_sessions.append(self)

    class AuthorizedSession:
        def __init__(self, auth, *, default_network_request_kwargs):
            self.auth = auth
            self.request_kwargs = dict(default_network_request_kwargs)
            authorized_sessions.append(self)

    class Auth:
        def authenticate_instance(self):
            authentication_calls.append(self)

    auth = Auth()

    class JWTAuth:
        @classmethod
        def from_settings_file(cls, path, *, session):
            assert path == str(config)
            settings_sessions.append(session)
            return auth

    class Client:
        def __init__(self, client_auth, *, session):
            self.auth = client_auth
            self.session = session
            client_sessions.append(session)

    boxsdk = ModuleType("boxsdk")
    setattr(boxsdk, "__path__", [])
    boxsdk.Client = Client
    boxsdk.JWTAuth = JWTAuth
    session_package = ModuleType("boxsdk.session")
    setattr(session_package, "__path__", [])
    session_module = ModuleType("boxsdk.session.session")
    session_module.AuthorizedSession = AuthorizedSession
    session_module.Session = Session
    monkeypatch.setitem(sys.modules, "boxsdk", boxsdk)
    monkeypatch.setitem(sys.modules, "boxsdk.session", session_package)
    monkeypatch.setitem(sys.modules, "boxsdk.session.session", session_module)
    sdk_logger = logging.getLogger("boxsdk")
    monkeypatch.setattr(sdk_logger, "handlers", [])
    monkeypatch.setattr(sdk_logger, "level", logging.NOTSET)
    monkeypatch.setattr(sdk_logger, "propagate", True)

    client, folder_id = handoff_box.client_from_environment(options.repo_root)

    expected = {"timeout": (30, 300)}
    assert handoff_box.BOX_REQUEST_TIMEOUT == (30, 300)
    assert [session.request_kwargs for session in plain_sessions] == [expected]
    assert settings_sessions == plain_sessions
    assert [session.request_kwargs for session in authorized_sessions] == [expected]
    assert authentication_calls == [auth]
    assert client.auth is auth
    assert client_sessions == authorized_sessions
    assert folder_id == "123"


def _rewrite_package_controls(
    package: Path,
    manifest: dict,
) -> None:
    (package / "manifest.json").write_bytes(_canonical_json(manifest))
    physical = {
        part["path"]: part["sha256"]
        for artifact in manifest["artifacts"]
        for part in artifact["parts"]
    }
    (package / "SHA256SUMS").write_text(
        "".join(
            f"{digest}  {relative}\n"
            for relative, digest in sorted(physical.items())
        )
    )
    ready = json.loads((package / "READY.json").read_text())
    ready["manifest"] = _physical_record(package / "manifest.json", package)
    ready["checksums"] = _physical_record(package / "SHA256SUMS", package)
    (package / "READY.json").write_bytes(_canonical_json(ready))


def _production_options(
    options: BuildOptions,
    base_commit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> BuildOptions:
    head = _run("git", "rev-parse", "HEAD", cwd=options.repo_root)
    monkeypatch.setattr(handoff_package, "SOURCE_BASE_COMMIT", base_commit)
    monkeypatch.setattr(handoff_package, "EXPECTED_CHIP_ARCHIVES", 2)
    monkeypatch.setattr(handoff_package, "EXPECTED_RASTER_ARCHIVES", 1)
    approved = {
        name: (
            checkpoint,
            hashlib.sha256(
                (options.data_root / "weights" / name / checkpoint).read_bytes()
            ).hexdigest(),
        )
        for name, (checkpoint, _digest) in handoff_package.REQUIRED_CHECKPOINTS.items()
    }
    monkeypatch.setattr(handoff_package, "REQUIRED_CHECKPOINTS", approved)
    return replace(
        options,
        package_root=options.package_root.parent / f"xview3-h100-fp32-{head}",
        max_part_bytes=10_000_000,
        production=True,
    )


def test_git_bundle_verifier_marks_temporary_bare_repo_safe(
    tmp_path, monkeypatch
):
    options, commit = _fixture_source(tmp_path)
    bundle = tmp_path / "source.bundle"
    _run(
        "git",
        "bundle",
        "create",
        str(bundle),
        "refs/heads/sprint-7d-h100-fp32",
        cwd=options.repo_root,
    )
    real_run = handoff_package._run
    verify_commands = []

    def checked_run(argv, **kwargs):
        command = list(argv)
        if "bundle" in command and "verify" in command:
            verifier = command[command.index("-C") + 1]
            assert f"safe.directory={verifier}" in command
            verify_commands.append(command)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(handoff_package, "_run", checked_run)
    verified = handoff_package._verify_git_bundle(
        bundle,
        "sprint-7d-h100-fp32",
        commit,
    )
    assert verify_commands
    assert verified["chips"] == ["dev-scene", "train-scene"]


def test_target_facing_verify_and_extract_reject_fixture_package(tmp_path):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    with pytest.raises(PackageError, match="requires a production package"):
        verify_package(package)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(PackageError, match="requires a production package"):
        extract_package(package, destination)
    assert not destination.exists()


def test_production_verifier_binds_bundle_split_and_definition(
    tmp_path, monkeypatch
):
    options, base = _fixture_source(tmp_path)
    production = _production_options(options, base, monkeypatch)
    package = build_package(production)
    manifest = verify_package(package)
    extracted = extract_package(package, tmp_path / "production-extracted")
    assert (
        extracted / "environment/Apptainer.def"
    ).read_bytes() == (
        options.repo_root / "containers/h100-strict-fp32.def"
    ).read_bytes()
    extraction_receipt = json.loads(
        (extracted / "HANDOFF_EXTRACTED.json").read_text(encoding="utf-8")
    )
    assert extraction_receipt["wheelhouse"] == wheelhouse_identity(
        extracted / "environment/wheelhouse"
    )

    manifest["scenes"]["chips"] = list(reversed(manifest["scenes"]["chips"]))
    _rewrite_package_controls(package, manifest)
    with pytest.raises(PackageError, match="verified Git bundle"):
        verify_package(package)


def test_production_builder_requires_exact_committed_definition(
    tmp_path, monkeypatch
):
    options, base = _fixture_source(tmp_path)
    production = _production_options(options, base, monkeypatch)
    alternate = tmp_path / "alternate.def"
    alternate.write_bytes(production.apptainer_definition.read_bytes())
    with pytest.raises(PackageError, match="must be exactly"):
        build_package(replace(production, apptainer_definition=alternate))


@pytest.mark.parametrize(
    ("historical_path", "content", "message"),
    [
        (
            "notes/innocent.json",
            '{"boxAppSettings":{"clientID":"x","clientSecret":"actual-secret"}}',
            "credential material",
        ),
        (
            "old/.venv-deleted/payload.bin",
            b"historical environment payload",
            "virtual-environment",
        ),
        (
            ".env",
            '{"boxAppSettings":{"clientID":"x","clientSecret":"actual-secret"}}',
            "credential material",
        ),
        (
            "old/__pycache__/module.pyc",
            b"historical cache payload",
            "denied payload path",
        ),
    ],
)
def test_production_history_scan_rejects_deleted_payloads(
    tmp_path, monkeypatch, historical_path, content, message
):
    options, base = _fixture_source(tmp_path)
    historical = options.repo_root / historical_path
    _write(historical, content)
    _run("git", "add", ".", cwd=options.repo_root)
    _run("git", "commit", "-q", "-m", "P7d: forbidden history fixture", cwd=options.repo_root)
    historical.unlink()
    _run("git", "add", "-u", cwd=options.repo_root)
    _run("git", "commit", "-q", "-m", "P7d: remove forbidden fixture", cwd=options.repo_root)
    production = _production_options(options, base, monkeypatch)
    with pytest.raises(PackageError, match=message):
        build_package(production)


@pytest.mark.parametrize(
    ("maximum", "message"),
    [
        (0, "positive integer"),
        (511, "physical limit"),
        (513, "deterministically sized"),
    ],
)
def test_verifier_enforces_physical_part_contract(tmp_path, maximum, message):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    manifest = json.loads((package / "manifest.json").read_text())
    assert any(len(item["parts"]) > 1 for item in manifest["artifacts"])
    manifest["contract"]["maximum_physical_file_bytes"] = maximum
    _rewrite_package_controls(package, manifest)
    with pytest.raises(PackageError, match=message):
        _verify_fixture_package(package)


def test_archive_split_uses_bounded_reads(tmp_path, monkeypatch):
    source = tmp_path / "large.tar.zst"
    source.write_bytes(b"abcdefghijklmnopqrstuvwxyz")
    observed_reads = []
    original_open = Path.open

    class TrackingReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, size=-1):
            observed_reads.append(size)
            return self.handle.read(size)

    def tracked_open(path, mode="r", *args, **kwargs):
        handle = original_open(path, mode, *args, **kwargs)
        if path == source and mode == "rb":
            return TrackingReader(handle)
        return handle

    monkeypatch.setattr(handoff_package, "SPLIT_COPY_BLOCK_BYTES", 4)
    monkeypatch.setattr(handoff_package, "_hash_file", lambda *_args: "0" * 64)
    monkeypatch.setattr(Path, "open", tracked_open)
    parts, size, _sha256, _sha1 = handoff_package._split_archive(source, 10)
    assert size == 26
    assert [part.stat().st_size for part in parts] == [10, 10, 6]
    assert observed_reads and max(observed_reads) <= 4


def test_verifier_rejects_package_control_and_physical_symlinks(tmp_path):
    root_case = tmp_path / "root"
    root_case.mkdir()
    root_options, _ = _fixture_source(root_case)
    real_root = build_package(root_options)
    linked_root = tmp_path / "linked-package"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(PackageError, match="symlink"):
        _verify_fixture_package(linked_root)

    control_case = tmp_path / "control"
    control_case.mkdir()
    control_options, _ = _fixture_source(control_case)
    control_package = build_package(control_options)
    manifest_path = control_package / "manifest.json"
    outside_manifest = tmp_path / "outside-manifest.json"
    manifest_path.replace(outside_manifest)
    manifest_path.symlink_to(outside_manifest)
    with pytest.raises(PackageError, match="symlink"):
        _verify_fixture_package(control_package)

    part_case = tmp_path / "part"
    part_case.mkdir()
    part_options, _ = _fixture_source(part_case)
    part_package = build_package(part_options)
    physical_parent = part_package / "data/chips"
    outside_parent = tmp_path / "outside-chips"
    physical_parent.replace(outside_parent)
    physical_parent.symlink_to(outside_parent, target_is_directory=True)
    with pytest.raises(PackageError, match="symlink"):
        _verify_fixture_package(part_package)


def test_weight_archives_use_exact_three_file_loader_allowlist(tmp_path):
    options, _ = _fixture_source(tmp_path)
    extra = options.data_root / "weights/satdino/pytorch_model.bin"
    _write(extra, b"unused checkpoint")
    package = build_package(
        replace(options, max_part_bytes=10_000_000)
    )
    extracted = _extract_fixture_package(package, tmp_path / "weight-extracted")
    weight_root = extracted / "data/weights/satdino"
    assert {path.name for path in weight_root.iterdir()} == {
        "SOURCE.note",
        "LICENSE.note",
        "satdino-vit_base-16.pth",
    }


def test_verifier_rejects_unknown_artifact_kind(tmp_path):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["artifacts"][0]["kind"] = "unknown_payload"
    _rewrite_package_controls(package, manifest)
    with pytest.raises(PackageError, match="artifact identity schema mismatch"):
        _verify_fixture_package(package)


def test_verifier_rejects_member_outside_declared_root(tmp_path):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    manifest = json.loads((package / "manifest.json").read_text())
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["kind"] == "chip_scene" and item["name"] == "train-scene"
    )
    assert len(artifact["parts"]) == 1
    archive = package / artifact["parts"][0]["path"]
    archive.unlink()
    rogue = tmp_path / "rogue.bin"
    _write(rogue, b"rogue")
    file_count, unpacked = _create_tar_zst(
        archive,
        [(rogue, Path("data/chips/sibling/rogue.bin"))],
    )
    part = _physical_record(archive, package)
    artifact.update(
        {
            "file_count": file_count,
            "unpacked_bytes": unpacked,
            "archive_bytes": part["bytes"],
            "archive_sha256": part["sha256"],
            "archive_sha1": part["sha1"],
            "parts": [part],
        }
    )
    _rewrite_package_controls(package, manifest)
    with pytest.raises(PackageError, match="escapes declared root"):
        _verify_fixture_package(package)


def test_verifier_rejects_declared_unpacked_size_overflow(tmp_path):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    manifest = json.loads((package / "manifest.json").read_text())
    artifact = next(
        item
        for item in manifest["artifacts"]
        if item["kind"] == "chip_scene" and item["name"] == "train-scene"
    )
    artifact["unpacked_bytes"] -= 1
    _rewrite_package_controls(package, manifest)
    with pytest.raises(PackageError, match="exceeds declared count/size"):
        _verify_fixture_package(package)


class _Item:
    def __init__(
        self,
        item_id: str,
        name: str,
        kind: str,
        *,
        parent: str | None = None,
        content: bytes = b"",
    ):
        self.id = item_id
        self.name = name
        self.object_type = kind
        self.type = kind
        self.parent = parent
        self.content = content
        self.size = len(content)
        self.sha1 = hashlib.sha1(content).hexdigest()


class _Uploader:
    def __init__(self, client, parent: str | None, path: str, name: str, item_id=None):
        self.client = client
        self.parent = parent
        self.path = Path(path)
        self.name = name
        self.item_id = item_id

    def start(self):
        self.client.events.append(("chunk-start", self.name))
        if self.client.chunk_start_mode == "raise":
            raise RuntimeError("fixture interruption")
        if self.client.chunk_start_mode == "none":
            return None
        return self.client._store(
            self.parent, self.name, self.path.read_bytes(), self.item_id
        )

    def resume(self):
        self.client.events.append(("chunk-resume", self.name))
        return self.client._store(
            self.parent, self.name, self.path.read_bytes(), self.item_id
        )


class _Folder:
    def __init__(self, client, item_id: str):
        self.client = client
        self.id = item_id

    def get(self, fields=None):
        return self.client.items[self.id]

    def get_items(self, **kwargs):
        if self.client._fail_probe(
            "list", ".xview3-handoff-preflight-list"
        ):
            raise RuntimeError("probe listing failed")
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 1000)
        children = [
            self.client.items[item_id]
            for item_id in self.client.children.get(self.id, [])
        ]
        return children[offset : offset + limit]

    def create_subfolder(self, name):
        item = self.client._new(name, "folder", self.id)
        self.client.events.append(("folder", name))
        return item

    def upload(self, path, name):
        self.client.events.append(("direct", name))
        stored = self.client._store(self.id, name, Path(path).read_bytes())
        if (
            self.client.fail_after_store_name == name
            or self.client._fail_probe("upload-response", name)
        ):
            raise RuntimeError("committed upload response was lost")
        return stored

    def get_chunked_uploader(self, path, file_name):
        return _Uploader(self.client, self.id, path, file_name)


class _File:
    def __init__(self, client, item_id: str):
        self.client = client
        self.id = item_id

    def get(self, fields=None):
        return self.client.items[self.id]

    def update_contents(self, path):
        item = self.client.items[self.id]
        if self.client._fail_probe("update", item.name):
            raise RuntimeError("probe update denied")
        lost_response = self.client._fail_probe("update-response", item.name)
        self.client.events.append(("update", item.name))
        stored = self.client._store(
            item.parent, item.name, Path(path).read_bytes(), self.id
        )
        if lost_response:
            raise RuntimeError("committed update response was lost")
        return stored

    def rename(self, name):
        item = self.client.items[self.id]
        if self.client._fail_probe("rename", item.name):
            raise RuntimeError("probe rename denied")
        lost_response = self.client._fail_probe("rename-response", item.name)
        previous = item.name
        item.name = name
        self.client.events.append(("rename", previous))
        if lost_response:
            raise RuntimeError("committed rename response was lost")
        return item

    def get_chunked_uploader(self, path):
        item = self.client.items[self.id]
        return _Uploader(
            self.client, item.parent, path, item.name, item_id=self.id
        )

    def download_to(self, handle):
        handle.write(self.client.items[self.id].content)

    def delete(self):
        item = self.client.items[self.id]
        if self.client._fail_probe("delete", item.name):
            raise RuntimeError("probe delete denied")
        lost_response = self.client._fail_probe("delete-response", item.name)
        item = self.client.items.pop(self.id)
        self.client.children[item.parent].remove(self.id)
        self.client.events.append(("delete", item.name))
        if lost_response:
            raise RuntimeError("committed delete response was lost")


class _Client:
    def __init__(
        self,
        *,
        chunk_start_mode="raise",
        permissions=None,
        fail_after_store_name=None,
        corrupt_after_ready=False,
        fail_probe_action=None,
    ):
        self.items = {"0": _Item("0", "root", "folder")}
        self.permissions = permissions or SimpleNamespace(
            can_upload=True,
            can_delete=True,
            can_rename=True,
        )
        self.items["0"].permissions = self.permissions
        self.items["0"].owned_by = SimpleNamespace(id="owner-1")
        self.children = {"0": []}
        self.events: list[tuple[str, str]] = []
        self.counter = 1
        self.chunk_start_mode = chunk_start_mode
        self.fail_after_store_name = fail_after_store_name
        self.corrupt_after_ready = corrupt_after_ready
        self.fail_probe_action = fail_probe_action

    def _fail_probe(self, action, name):
        is_probe = name.startswith(".xview3-handoff-preflight-")
        if (
            self.fail_probe_action == "delete-always"
            and action == "delete"
            and is_probe
        ):
            return True
        if self.fail_probe_action == action and is_probe:
            self.fail_probe_action = None
            return True
        return False

    def _new(self, name, kind, parent, content=b""):
        item_id = str(self.counter)
        self.counter += 1
        item = _Item(item_id, name, kind, parent=parent, content=content)
        self.items[item_id] = item
        self.children.setdefault(parent, []).append(item_id)
        if kind == "folder":
            self.children[item_id] = []
        return item

    def _store(self, parent, name, content, item_id=None):
        if item_id is None:
            item = self._new(name, "file", parent, content)
        else:
            item = self.items[item_id]
            item.content = content
            item.size = len(content)
            item.sha1 = hashlib.sha1(content).hexdigest()
        self.events.append(("stored", name))
        if name == "READY.json" and self.corrupt_after_ready:
            target = next(
                candidate
                for candidate in self.items.values()
                if candidate.object_type == "file"
                and candidate.name not in {"READY.json", "manifest.json", "SHA256SUMS"}
            )
            target.content += b"post-ready-corruption"
            target.size = len(target.content)
            target.sha1 = hashlib.sha1(target.content).hexdigest()
        return item

    def folder(self, item_id):
        return _Folder(self, str(item_id))

    def file(self, item_id):
        return _File(self, str(item_id))

    def user(self, user_id="me"):
        if user_id == "me":
            user = SimpleNamespace(id="service-account", max_upload_size=10 * 1024**3)
        elif user_id == "owner-1":
            user = SimpleNamespace(
                id="owner-1",
                space_amount=700 * 1024**3,
                space_used=0,
            )
        else:
            raise AssertionError(f"unexpected Box user query: {user_id}")
        return SimpleNamespace(get=lambda fields=None: user)


def test_box_mock_chunk_resume_ready_last_download_and_extract(tmp_path):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    client = _Client(chunk_start_mode="none")
    receipt = tmp_path / "upload-receipt.json"
    result = _upload_fixture_package(
        client,
        "0",
        package,
        repo_root=options.repo_root,
        receipt_path=receipt,
        chunked_threshold=1,
        minimum_free_bytes=500_000_000_000,
    )
    assert result["uploaded"] == len(_files(package))
    stored = [name for action, name in client.events if action == "stored"]
    assert stored[-1] == "READY.json"
    assert any(action == "chunk-resume" for action, _ in client.events)
    receipt_payload = json.loads(receipt.read_text())
    assert receipt_payload["package_id"].startswith("xview3-h100-fp32-")
    assert receipt_payload["remote_file_count"] == len(_files(package))
    assert "folder_id" not in receipt_payload
    second_receipt = tmp_path / "upload-receipt-second.json"
    events_before_second_upload = list(client.events)
    assert _upload_fixture_package(
        client,
        "0",
        package,
        repo_root=options.repo_root,
        receipt_path=second_receipt,
    )["uploaded"] == 0
    assert client.events == events_before_second_upload

    ready = json.loads((package / "READY.json").read_text())
    download_kwargs = {
        "repo_root": options.repo_root,
        "expected_ready_sha256": hashlib.sha256(
            (package / "READY.json").read_bytes()
        ).hexdigest(),
        "expected_manifest_sha256": ready["manifest"]["sha256"],
        "expected_sha256sums_sha256": ready["checksums"]["sha256"],
        "expected_package_id": ready["package_id"],
    }
    public_destination = tmp_path / "public-fixture-download"
    with pytest.raises(PackageError, match="requires a production package"):
        download_package(
            client,
            "0",
            public_destination,
            **download_kwargs,
        )
    assert not public_destination.exists()
    downloaded = _download_fixture_package(
        client,
        "0",
        tmp_path / "downloaded",
        **download_kwargs,
    )
    assert _files(downloaded) == _files(package)
    source_extract = _extract_fixture_package(package, tmp_path / "source-extract")
    receipt_extract = _extract_fixture_package(
        downloaded, tmp_path / "receipt-extract"
    )
    assert _files(source_extract) == _files(receipt_extract)
    assert not list(tmp_path.rglob("*.partial"))

    with pytest.raises(BoxTransferError, match="outside the repository"):
        _download_fixture_package(
            client,
            "0",
            options.repo_root / "forbidden-download",
            **download_kwargs,
        )
    with pytest.raises(BoxTransferError, match="outside the repository"):
        _upload_fixture_package(
            client,
            "0",
            package,
            repo_root=options.repo_root,
            receipt_path=options.repo_root / "forbidden-receipt.json",
        )

    corrupt = next(
        item
        for item in client.items.values()
        if item.object_type == "file"
        and item.name not in {"READY.json", "manifest.json", "SHA256SUMS"}
    )
    corrupt.content = b"x" + corrupt.content[1:]
    failed_destination = tmp_path / "failed-download"
    with pytest.raises(BoxTransferError, match="SHA-1 mismatch"):
        _download_fixture_package(
            client,
            "0",
            failed_destination,
            **download_kwargs,
        )
    assert not failed_destination.exists()
    assert not list(tmp_path.glob(".failed-download.downloading-*"))


def test_box_preflight_accepts_collaborated_root_and_probes_child():
    permissions = SimpleNamespace(
        can_upload=True,
        can_delete=False,
        can_rename=False,
    )
    client = _Client(permissions=permissions)
    preflight_box(client, "0", minimum_free_bytes=0)
    actions = [action for action, _name in client.events]
    assert actions == ["direct", "stored", "update", "stored", "rename", "delete"]
    assert client.children["0"] == []


def test_box_preflight_requires_root_child_upload_permission():
    permissions = SimpleNamespace(
        can_upload=False,
        can_delete=False,
        can_rename=False,
    )
    client = _Client(permissions=permissions)
    with pytest.raises(BoxTransferError, match="child upload"):
        preflight_box(client, "0", minimum_free_bytes=0)
    assert client.events == []


@pytest.mark.parametrize(
    "action",
    [
        "upload-response",
        "update",
        "update-response",
        "rename",
        "rename-response",
        "delete",
        "delete-response",
    ],
)
def test_box_preflight_child_probe_failure_is_cleaned_up(action):
    client = _Client(fail_probe_action=action)
    with pytest.raises(BoxTransferError, match="child-mutation probe failed"):
        preflight_box(client, "0", minimum_free_bytes=0)
    assert client.children["0"] == []


def test_box_preflight_sanitizes_initial_listing_failure():
    client = _Client(fail_probe_action="list")
    with pytest.raises(
        BoxTransferError, match="could not inspect destination"
    ):
        preflight_box(client, "0", minimum_free_bytes=0)
    assert client.events == []
    assert client.children["0"] == []


def test_box_preflight_fails_closed_when_cleanup_cannot_delete():
    client = _Client(fail_probe_action="delete-always")
    with pytest.raises(BoxTransferError, match="cleanup could not be verified"):
        preflight_box(client, "0", minimum_free_bytes=0)
    remaining = [client.items[item_id] for item_id in client.children["0"]]
    assert len(remaining) == 1
    assert remaining[0].name.startswith(".xview3-handoff-preflight-")


def test_box_preflight_refuses_to_probe_beside_ready():
    client = _Client()
    client._store("0", "READY.json", b"{}")
    events_before = list(client.events)
    with pytest.raises(BoxTransferError, match="READY.json"):
        preflight_box(client, "0", minimum_free_bytes=0)
    assert client.events == events_before
    assert [client.items[item_id].name for item_id in client.children["0"]] == [
        "READY.json"
    ]


def test_box_repair_invalidates_ready_before_child_probe(tmp_path):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    client = _Client()
    _upload_fixture_package(
        client,
        "0",
        package,
        repo_root=options.repo_root,
        receipt_path=tmp_path / "first-upload-receipt.json",
    )
    target = next(
        item
        for item in client.items.values()
        if item.object_type == "file"
        and item.name not in {"READY.json", "manifest.json", "SHA256SUMS"}
    )
    target.content += b"remote-corruption"
    target.size = len(target.content)
    target.sha1 = hashlib.sha1(target.content).hexdigest()

    event_offset = len(client.events)
    _upload_fixture_package(
        client,
        "0",
        package,
        repo_root=options.repo_root,
        receipt_path=tmp_path / "repair-upload-receipt.json",
    )
    repair_events = client.events[event_offset:]
    ready_delete = next(
        index
        for index, event in enumerate(repair_events)
        if event == ("delete", "READY.json")
    )
    probe_upload = next(
        index
        for index, (action, name) in enumerate(repair_events)
        if action == "direct" and name.startswith(".xview3-handoff-preflight-")
    )
    assert ready_delete < probe_upload


def test_box_preflight_uses_destination_owner_quota():
    client = _Client()

    def user(user_id="me"):
        if user_id == "me":
            value = SimpleNamespace(id="service-account", max_upload_size=10 * 1024**3)
        elif user_id == "owner-1":
            value = SimpleNamespace(
                id="owner-1",
                space_amount=600_000_000_000,
                space_used=50_000_000_000,
            )
        else:
            raise AssertionError(user_id)
        return SimpleNamespace(get=lambda fields=None: value)

    client.user = user
    result = preflight_box(client, "0")
    assert result.free_bytes == 550_000_000_000


def test_chunk_threshold_is_decimal_50_mb(monkeypatch):
    events = []

    class SizedPath:
        def __init__(self, size):
            self.size = size

        def stat(self):
            return SimpleNamespace(st_size=self.size)

        def __str__(self):
            return "/fixture/payload"

    class Uploader:
        def start(self):
            events.append("chunked")
            return SimpleNamespace(id="uploaded")

        def resume(self):
            raise AssertionError("resume not expected")

    class Folder:
        def upload(self, _path, _name):
            events.append("direct")
            return SimpleNamespace(id="uploaded")

        def get_chunked_uploader(self, _path, file_name):
            return Uploader()

    client = SimpleNamespace(folder=lambda _folder_id: Folder())
    monkeypatch.setattr(
        handoff_box,
        "_refresh_file",
        lambda _client, _item_id: SimpleNamespace(),
    )
    monkeypatch.setattr(
        handoff_box,
        "_remote_matches",
        lambda _remote, _local: True,
    )
    assert CHUNKED_UPLOAD_THRESHOLD == 50_000_000
    _upload_one(
        client,
        "0",
        SizedPath(50_000_000),
        "at-boundary",
        None,
        chunked_threshold=CHUNKED_UPLOAD_THRESHOLD,
    )
    _upload_one(
        client,
        "0",
        SizedPath(50_000_001),
        "above-boundary",
        None,
        chunked_threshold=CHUNKED_UPLOAD_THRESHOLD,
    )
    assert events == ["direct", "chunked"]


def test_chunk_resume_retries_exceptions_until_success(tmp_path, monkeypatch):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    events = []

    class Uploader:
        def start(self):
            events.append("start")
            raise RuntimeError("transient start failure")

        def resume(self):
            events.append("resume")
            if events.count("resume") < 3:
                raise RuntimeError("transient resume failure")
            return SimpleNamespace(id="uploaded")

    uploader = Uploader()
    folder = SimpleNamespace(
        get_chunked_uploader=lambda _path, file_name: uploader
    )
    client = SimpleNamespace(folder=lambda _folder_id: folder)
    monkeypatch.setattr(
        handoff_box,
        "_refresh_file",
        lambda _client, _item_id: SimpleNamespace(),
    )
    monkeypatch.setattr(
        handoff_box,
        "_remote_matches",
        lambda _remote, _local: True,
    )

    _upload_one(
        client,
        "0",
        payload,
        payload.name,
        None,
        chunked_threshold=0,
    )

    assert events == ["start", "resume", "resume", "resume"]


def test_chunk_resume_exceptions_exhaust_bounded_attempts(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    events = []

    class Uploader:
        def start(self):
            events.append("start")
            raise RuntimeError("private-session-id start failure")

        def resume(self):
            events.append("resume")
            raise RuntimeError("private-session-id resume failure")

    uploader = Uploader()
    folder = SimpleNamespace(
        get_chunked_uploader=lambda _path, file_name: uploader
    )
    client = SimpleNamespace(folder=lambda _folder_id: folder)

    with pytest.raises(BoxTransferError, match="Box upload failed") as error:
        _upload_one(
            client,
            "0",
            payload,
            payload.name,
            None,
            chunked_threshold=0,
        )

    assert events == ["start"] + [
        "resume"
    ] * handoff_box.MAX_CHUNK_RESUME_ATTEMPTS
    assert "private-session-id" not in str(error.value)


@pytest.mark.parametrize(
    "client",
    [
        _Client(fail_after_store_name="READY.json"),
        _Client(corrupt_after_ready=True),
    ],
)
def test_box_upload_failure_removes_any_remote_ready(tmp_path, client):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    with pytest.raises(BoxTransferError):
        _upload_fixture_package(
            client,
            "0",
            package,
            repo_root=options.repo_root,
            receipt_path=tmp_path / "must-not-exist-receipt.json",
        )
    assert all(item.name != "READY.json" for item in client.items.values())
    assert not (tmp_path / "must-not-exist-receipt.json").exists()


def test_box_upload_reverifies_local_package_before_receipt(tmp_path, monkeypatch):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    client = _Client()
    original_upload_one = handoff_box._upload_one
    mutated = False

    def mutate_then_upload(
        client_arg, parent_id, local, filename, existing, **kwargs
    ):
        nonlocal mutated
        if not mutated and filename not in {
            "READY.json",
            "manifest.json",
            "SHA256SUMS",
        }:
            local.write_bytes(
                local.read_bytes() + b"post-preflight-mutation"
            )
            mutated = True
        return original_upload_one(
            client_arg, parent_id, local, filename, existing, **kwargs
        )

    monkeypatch.setattr(handoff_box, "_upload_one", mutate_then_upload)
    receipt = tmp_path / "must-not-exist-receipt.json"
    with pytest.raises(PackageError):
        _upload_fixture_package(
            client,
            "0",
            package,
            repo_root=options.repo_root,
            receipt_path=receipt,
        )
    assert mutated
    assert all(item.name != "READY.json" for item in client.items.values())
    assert not receipt.exists()


def test_box_concurrent_final_remote_change_invalidates_ready(tmp_path, monkeypatch):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    client = _Client()
    _upload_fixture_package(
        client,
        "0",
        package,
        repo_root=options.repo_root,
        receipt_path=tmp_path / "first-receipt.json",
    )
    original_list_remote_files = handoff_box.list_remote_files
    calls = 0

    def mutate_before_final_listing(client_arg, folder_id):
        nonlocal calls
        calls += 1
        if calls == 2:
            target = next(
                item
                for item in client.items.values()
                if item.object_type == "file"
                and item.name not in {"READY.json", "manifest.json", "SHA256SUMS"}
            )
            target.content += b"concurrent-final-remote-mutation"
            target.size = len(target.content)
            target.sha1 = hashlib.sha1(target.content).hexdigest()
        return original_list_remote_files(client_arg, folder_id)

    monkeypatch.setattr(
        handoff_box,
        "list_remote_files",
        mutate_before_final_listing,
    )
    receipt = tmp_path / "must-not-exist-receipt.json"
    with pytest.raises(BoxTransferError, match="final Box tree"):
        _upload_fixture_package(
            client,
            "0",
            package,
            repo_root=options.repo_root,
            receipt_path=receipt,
        )
    assert calls >= 3
    assert all(item.name != "READY.json" for item in client.items.values())
    assert not receipt.exists()


def test_box_unexpected_remote_file_invalidates_ready(tmp_path):
    options, _ = _fixture_source(tmp_path)
    package = build_package(options)
    client = _Client()
    _upload_fixture_package(
        client,
        "0",
        package,
        repo_root=options.repo_root,
        receipt_path=tmp_path / "first-receipt.json",
    )
    client._store("0", "unexpected.txt", b"stale")
    with pytest.raises(BoxTransferError, match="unexpected package files"):
        _upload_fixture_package(
            client,
            "0",
            package,
            repo_root=options.repo_root,
            receipt_path=tmp_path / "must-not-exist-receipt.json",
        )
    assert all(item.name != "READY.json" for item in client.items.values())


def test_box_manifest_path_traversal_is_rejected():
    malicious = {
        "artifacts": [
            {"parts": [{"path": "/tmp/escape", "sha256": "0" * 64}]}
        ]
    }
    with pytest.raises(PackageError, match="unsafe package path"):
        _manifest_physical(malicious)


def _write_result_fixture(
    repo: Path,
    runs: Path,
    campaign_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, list[str]]:
    import csv
    import signal
    import threading
    import time
    import uuid

    from scripts.h100 import slurm_smoke
    from scripts.h100.acceptance import EXPECTED_FRACTION_WORKLOAD
    from scripts.h100.build_venv import RECEIPT_KIND
    from scripts.h100.campaign import hardware_class
    from scripts.h100.contracts import (
        frozen_hashes,
        sha256_file,
    )
    from scripts.h100.host_test_gate import HOST_COMMAND, HOST_TESTS
    from scripts.h100.lightning_contract import (
        CUDA_ACCELERATOR,
        PRECISION_PLUGIN,
        SINGLE_DEVICE_STRATEGY,
    )
    from scripts.h100.source_validation import expected_receipt

    arms = {
        "vit_random": {"short": "vitrand", "track": "vit", "role": "floor"},
        "satdino_b": {"short": "satdino", "track": "vit", "role": "optical"},
        "sarmae_b": {"short": "sarmae", "track": "vit", "role": "sar"},
        "vit_imagenet": {"short": "vitin1k", "track": "vit", "role": "imagenet"},
        "cnn_random": {"short": "cnnrand", "track": "cnn", "role": "floor"},
        "bigearthnet_s2": {"short": "beS2", "track": "cnn", "role": "optical"},
        "bigearthnet_s1": {"short": "beS1", "track": "cnn", "role": "sar"},
        "cnn_imagenet": {"short": "cnnin1k", "track": "cnn", "role": "imagenet"},
    }
    _write(
        repo / "configs/arms.yaml",
        yaml.safe_dump(
            {
                "arms": arms,
                "label_fracs": [0.1, 0.25, 0.5, 1.0],
                "seeds": {"core": [0], "reruns": [], "rerun_fracs": []},
            },
            sort_keys=False,
        ),
    )
    _write(repo / "configs/detector.yaml", "schedule:\n  precision: 32-true\n")
    _write(repo / "src/eval/scorer.py", "# frozen fixture scorer\n")
    _write(repo / "data/stats.json", "{}\n")
    _write(repo / "data/lsssdd_split.json", "{}\n")
    _run("git", "branch", "-m", "sprint-7e-judy-venv", cwd=repo)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-q", "-m", "P7e: result fixture", cwd=repo)
    git_sha = _run("git", "rev-parse", "HEAD", cwd=repo)
    detector = hashlib.sha256((repo / "configs/detector.yaml").read_bytes()).hexdigest()
    cells = load_cells(repo)
    ids = [cell.exp_id for cell in cells]
    strict_fp32 = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }
    runtime_contract = {
        "schema": 1,
        "status": "verified",
        "pre_trainer": {
            "schema": 1,
            "status": "verified",
            "stage": "pre-trainer",
            "precision": "32-true",
            "devices": 1,
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "strict_fp32": strict_fp32,
            "autocast": {"global": False, "cuda": False, "cpu": False},
            "process": {
                "WORLD_SIZE": "1",
                "SLURM_NTASKS": "1",
                "effective_world_size": 1,
            },
            "model": {
                "floating_parameter_count": 1,
                "floating_parameter_dtypes": ["torch.float32"],
            },
        },
        "resolved_trainer": {
            "accelerator": CUDA_ACCELERATOR,
            "precision_plugin": PRECISION_PLUGIN,
            "precision": "32-true",
            "gradient_scaler": None,
            "strategy": SINGLE_DEVICE_STRATEGY,
            "root_device_type": "cuda",
            "root_device_index": 0,
            "num_devices": 1,
            "world_size": 1,
            "device_ids": [0],
            "gradient_accumulation": 1,
        },
    }
    base_payload = {
        "package_id": (
            "xview3-h100-fp32-"
            "2726199efcebbebc89156e708b89df2a3415468a"
        ),
        "git_sha": "2726199efcebbebc89156e708b89df2a3415468a",
        "manifest_sha256": "c" * 64,
        "ready_sha256": "d" * 64,
        "sha256sums_sha256": "e" * 64,
        "repo_bundle_sha256": "f" * 64,
    }
    runtime_amendment = {
        "package_id": f"xview3-h100-runtime-{git_sha}-{'5' * 64}",
        "git_sha": git_sha,
        "manifest_sha256": "1" * 64,
        "ready_sha256": "2" * 64,
        "sha256sums_sha256": "3" * 64,
        "runtime_bundle_sha256": "4" * 64,
    }
    acceptance_uuid = str(uuid.uuid4())
    meta = campaign_path.parent

    base_python_path = meta / "python311/bin/python3.11"
    _write(base_python_path, b"fixture Python 3.11.13\n")
    base_python_runtime_sha256 = "6" * 64
    base_python = {
        "version": "3.11.13",
        "requested_path": str(base_python_path),
        "resolved_path": str(base_python_path.resolve()),
        "executable_sha256": sha256_file(base_python_path),
        "runtime": {
            "algorithm": "xview3-base-python-runtime-v1",
            "sha256": base_python_runtime_sha256,
        },
    }
    wheelhouse_path = meta / "base-extracted/environment/wheelhouse"
    _write(wheelhouse_path / "fixture.whl", b"fixture wheel")
    wheelhouse_identity = {
        "algorithm": "xview3-wheelhouse-tree-v1",
        "sha256": "7" * 64,
        "files": 1,
        "bytes": len(b"fixture wheel"),
    }
    extraction_receipt_payload = {
        "format_version": 1,
        "package_id": base_payload["package_id"],
        "manifest_sha256": base_payload["manifest_sha256"],
        "wheelhouse": wheelhouse_identity,
    }
    extraction_receipt_path = meta / "base-extracted/HANDOFF_EXTRACTED.json"
    _write(extraction_receipt_path, json.dumps(extraction_receipt_payload))
    extraction_receipt_sha256 = sha256_file(extraction_receipt_path)
    wheelhouse = {
        "identity": wheelhouse_identity,
        "artifacts": {"fixture.whl": {"sha256": "8" * 64, "bytes": 13}},
        "base_extraction": {
            "path": str(extraction_receipt_path),
            "sha256": extraction_receipt_sha256,
            "receipt": extraction_receipt_payload,
        },
        "reverified_after_build": True,
    }
    staged_base_extraction = {
        "path": "/scratch/payload/HANDOFF_EXTRACTED.json",
        "sha256": extraction_receipt_sha256,
        "receipt": extraction_receipt_payload,
    }
    venv_root = meta / "native-venv"
    _write(venv_root / "bin/python", b"fixture copied native Python\n")
    venv_sha256 = "b" * 64
    venv_build = {
        "schema": 1,
        "kind": RECEIPT_KIND,
        "status": "ready",
        "base_python": base_python,
        "wheelhouse": wheelhouse,
        "venv": {
            "path": str(venv_root),
            "tree": {"sha256": venv_sha256},
        },
    }
    venv_build_path = meta / "venv_build.json"
    _write(venv_build_path, json.dumps(venv_build))
    _write(Path(f"{venv_root}.build.json"), json.dumps(venv_build))
    venv_build_sha256 = sha256_file(venv_build_path)

    def fake_verify_native_venv(**kwargs):
        assert kwargs == {
            "repo": repo,
            "venv_root": venv_root,
            "base_python": base_python_path,
            "wheelhouse": wheelhouse_path,
            "base_extraction_receipt": extraction_receipt_path,
            "expected_venv_sha256": venv_sha256,
            "expected_receipt_sha256": venv_build_sha256,
            "expected_base_python_sha256": base_python["executable_sha256"],
            "expected_base_python_runtime_sha256": base_python_runtime_sha256,
            "expected_wheelhouse_sha256": wheelhouse_identity["sha256"],
            "expected_base_extraction_receipt_sha256": (
                extraction_receipt_sha256
            ),
            "expected_base_payload_package_id": base_payload["package_id"],
            "expected_base_payload_manifest_sha256": (
                base_payload["manifest_sha256"]
            ),
        }
        return venv_build

    monkeypatch.setattr(
        "scripts.handoff.results.verify_native_venv",
        fake_verify_native_venv,
    )

    def hardware(prefix: str) -> dict:
        return {
            "torch": "2.11.0+cu126",
            "cuda_build": "12.6",
            "driver_version": "590.1",
            "backend": strict_fp32,
            "child_probes": [
                {"runtime_contract": runtime_contract} for _ in range(8)
            ],
            "devices": [
                {
                    "name": "NVIDIA H100 80GB HBM3",
                    "uuid": f"GPU-{prefix}-{index}",
                    "compute_capability": [9, 0],
                    "total_memory_bytes": 80_000_000_000,
                }
                for index in range(8)
            ],
        }

    accepted_hardware = hardware("ACCEPT")
    allocation_zero = hardware("ALLOC0")
    allocation_one = hardware("ALLOC1")
    _write(meta / "h100_runtime.json", json.dumps(accepted_hardware))
    _write(meta / "h100_runtime-9001-r0.json", json.dumps(allocation_zero))
    _write(meta / "h100_runtime-9001-r1.json", json.dumps(allocation_one))

    smoke_bindings = slurm_smoke.make_bindings(
        git_sha=git_sha,
        detector_sha256=detector,
        venv_sha256=venv_sha256,
        venv_build_sha256=venv_build_sha256,
        base_python_sha256=base_python["executable_sha256"],
        base_python_runtime_sha256=base_python_runtime_sha256,
        wheelhouse_sha256=wheelhouse_identity["sha256"],
        base_extraction_receipt_sha256=extraction_receipt_sha256,
        base_payload=base_payload,
        runtime_amendment=runtime_amendment,
    )
    smoke_root = slurm_smoke.smoke_root(runs)
    signal_ready = smoke_root / slurm_smoke.SIGNAL_READY_NAME
    signal_errors: list[BaseException] = []

    def deliver_external_usr1() -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                pid = int(signal_ready.read_text().strip())
            except (FileNotFoundError, ValueError):
                time.sleep(0.005)
                continue
            os.kill(pid, signal.SIGUSR1)
            return
        signal_errors.append(TimeoutError("smoke PID file was not published"))

    signal_thread = threading.Thread(target=deliver_external_usr1)
    signal_thread.start()
    assert (
        slurm_smoke.run_allocation(
            runs_root=runs,
            bindings=smoke_bindings,
            job_id="smoke-1",
            restart_count=0,
            external_signal_ready=signal_ready,
            signal_timeout_seconds=5.0,
        )
        == slurm_smoke.HOST_REQUEUE_EXIT_CODE
    )
    signal_thread.join(timeout=5.0)
    assert not signal_thread.is_alive()
    if signal_errors:
        raise signal_errors[0]
    slurm_smoke.authorize_host_requeue(
        root=smoke_root,
        bindings=smoke_bindings,
        job_id="smoke-1",
        restart_count=0,
    )
    assert (
        slurm_smoke.run_allocation(
            runs_root=runs,
            bindings=smoke_bindings,
            job_id="smoke-1",
            restart_count=1,
        )
        == 0
    )
    smoke_ready_path = smoke_root / slurm_smoke.READY_NAME
    smoke_receipt = json.loads(smoke_ready_path.read_text())

    frozen = frozen_hashes(repo)
    source_validation_path = meta / "SOURCE_VALIDATED.json"
    source_validation = expected_receipt(
        git_sha=git_sha,
        frozen_sha256=frozen,
        base_payload=base_payload,
        runtime_amendment=runtime_amendment,
    )
    _write(source_validation_path, json.dumps(source_validation))
    source_validation_sha256 = sha256_file(source_validation_path)
    host_test_log_path = meta / "acceptance-logs/pytest-handoff-host.log"
    _write(host_test_log_path, "27 passed\n")
    host_test_receipt_path = meta / "HOST_HANDOFF_TESTS.json"
    host_test_receipt = {
        "schema": 1,
        "status": "passed",
        "slice": "host-handoff",
        "command": HOST_COMMAND,
        "source_validation_sha256": source_validation_sha256,
        "duration_seconds": 1.0,
        "log": {
            "path": str(host_test_log_path),
            "sha256": sha256_file(host_test_log_path),
        },
    }
    _write(host_test_receipt_path, json.dumps(host_test_receipt))
    host_test_receipt_sha256 = sha256_file(host_test_receipt_path)
    venv_test_log_path = meta / "acceptance-logs/pytest-venv-remaining.log"
    _write(venv_test_log_path, "remaining tests passed in native venv\n")
    test_suite_path = meta / "PYTEST_ACCEPTANCE.json"
    venv_test_command = [
        "-m",
        "pytest",
        "-q",
        *(f"--ignore={path}" for path in HOST_TESTS),
    ]
    test_suite = {
        "schema": 2,
        "status": "passed",
        "source_validation_sha256": source_validation_sha256,
        "coverage": {
            "host": HOST_TESTS,
            "venv": "all pytest collection except the two host-only files",
            "aggregate": "entire repository pytest suite",
        },
        "host_handoff": {
            "receipt_path": str(host_test_receipt_path),
            "receipt_sha256": host_test_receipt_sha256,
            "receipt": host_test_receipt,
        },
        "venv_remaining": {
            "command": venv_test_command,
            "duration_seconds": 1.0,
            "log": {
                "path": str(venv_test_log_path),
                "sha256": sha256_file(venv_test_log_path),
            },
        },
        "aggregate_duration_seconds": 2.0,
    }
    _write(test_suite_path, json.dumps(test_suite))
    test_suite_sha256 = sha256_file(test_suite_path)

    projection = {
        "fraction_workload": EXPECTED_FRACTION_WORKLOAD,
        "steps_per_epoch": {
            label: item["steps_per_epoch"]
            for label, item in EXPECTED_FRACTION_WORKLOAD.items()
        },
        "grid_steps_per_epoch": 8
        * sum(
            item["steps_per_epoch"]
            for item in EXPECTED_FRACTION_WORKLOAD.values()
        ),
        "steps_per_second": 2.0,
        "expected_gpu_hours": 8.0,
        "ceiling_gpu_hours": 10.0,
        "expected_wall_hours_ideal": 1.0,
        "ceiling_wall_hours_ideal": 1.25,
        "conservative_h100_wall_hours": 2.0,
        "remaining_v100_wall_hours": 3.0,
        "staging_seconds": 360.0,
        "staging_hours_per_allocation": 0.1,
        "allocation_wall_hours": 36.5,
        "signal_lead_hours": 0.25,
        "usable_training_hours_per_allocation": 36.15,
        "projected_allocation_count": 1,
        "training_wall_hours_before_staging": 1.9,
    }
    gates = {
        "pytest_seconds": 2.0,
        "hardware_probe_seconds": 1.0,
        "vit_gate_seconds": 1.0,
        "cnn_200step_seconds": 1.0,
    }
    ready = {
        "schema": 2,
        "status": "ready",
        "acceptance_uuid": acceptance_uuid,
        "created_utc": "2026-07-27T00:00:00+00:00",
        "source": {
            "git_sha": git_sha,
            "frozen_sha256": frozen,
        },
        "source_validation": {
            "path": str(source_validation_path),
            "sha256": source_validation_sha256,
            "receipt": source_validation,
        },
        "strict_fp32": strict_fp32,
        "hardware": accepted_hardware,
        "venv": {
            "path": str(venv_root),
            "sha256": venv_sha256,
            "venv_build_sha256": venv_build_sha256,
            "base_python": base_python,
            "wheelhouse": wheelhouse,
            "staged_base_extraction": staged_base_extraction,
        },
        "base_payload": base_payload,
        "runtime_amendment": runtime_amendment,
        "slurm_smoke": {
            "sha256": sha256_file(smoke_ready_path),
            "receipt": smoke_receipt,
        },
        "scratch_free_bytes": 600_000_000_000,
        "test_suite": {
            "path": str(test_suite_path),
            "sha256": test_suite_sha256,
            "receipt": test_suite,
        },
        "gates": gates,
        "projection": projection,
    }
    _write(meta / "H100_READY.json", json.dumps(ready))
    _write(meta / "throughput_projection.json", json.dumps(projection))
    cutover = {
        "schema": 2,
        "status": "cutover-ready",
        "created_utc": "2026-07-27T00:30:00+00:00",
        "h100_ready": ready,
        "acceptance": cutover_acceptance_bindings(ready),
        "cutover_forecast": {
            "conservative_h100_wall_hours": 2.0,
            "acceptance_remaining_v100_wall_hours": 3.0,
            "current_remaining_v100_wall_hours": 3.0,
        },
        "references": {
            "r2": {
                "metrics": {"exp_id": "yolo26-f100"},
                "metrics_sha256": "7" * 64,
                "provenance": {
                    "git_sha": "4" * 40,
                    "campaign_id": "fixture-v100",
                },
                "provenance_sha256": "8" * 64,
            },
            "r3": {
                "metrics": {"exp_id": "locateanything-zs"},
                "metrics_sha256": "9" * 64,
                "provenance": {
                    "git_sha": "4" * 40,
                    "campaign_id": "fixture-v100",
                },
                "provenance_sha256": "a" * 64,
            },
        },
        "v100_action": "none; this guard never stops or signals V100 processes",
    }
    cutover_path = meta / "CUTOVER_READY.json"
    _write(cutover_path, json.dumps(cutover))
    cutover_sha256 = sha256_file(cutover_path)
    archive_manifest_path = meta / "V100_CORE_ARCHIVE_MANIFEST.json"
    archive_manifest = {
        "schema": 1,
        "status": "v100-core-diagnostics-archived",
        "scope": "v100-core-diagnostics",
        "diagnostic_status": "non-reportable-diagnostic",
        "git_sha": "4" * 40,
        "campaign_id": "fixture-v100",
        "stopped_utc": "2026-07-27T01:00:00+00:00",
        "archived_utc": "2026-07-27T02:00:00+00:00",
        "file_count": 0,
        "total_bytes": 0,
        "empty_reason": "fixture has no V100 diagnostic payload",
    }
    _write(archive_manifest_path, json.dumps(archive_manifest))
    archive_manifest_sha256 = sha256_file(archive_manifest_path)
    archived_receipt_path = meta / "V100_CORE_ARCHIVED.json"
    archived_receipt = {
        "schema": 2,
        "status": "v100-core-archived",
        "created_utc": "2026-07-27T03:00:00+00:00",
        "attestation": "external-human-operator",
        "cutover_ready_sha256": cutover_sha256,
        "h100": {
            "acceptance_uuid": acceptance_uuid,
            "git_sha": git_sha,
            "venv_sha256": venv_sha256,
            "base_payload": base_payload,
            "runtime_amendment": runtime_amendment,
        },
        "v100": {
            "git_sha": "4" * 40,
            "campaign_id": "fixture-v100",
            "stopped_utc": "2026-07-27T01:00:00+00:00",
            "stop_mode": "graceful",
            "running_core_processes": 0,
            "diagnostic_status": "non-reportable-diagnostic",
        },
        "archive": {
            "manifest_path": str(archive_manifest_path),
            "manifest_sha256": archive_manifest_sha256,
        },
    }
    _write(archived_receipt_path, json.dumps(archived_receipt))
    archived_receipt_sha256 = sha256_file(archived_receipt_path)
    for name in ("vit-fp32.log", "cnn-200step-fp32.log"):
        _write(meta / "acceptance-logs" / name, f"{name}: passed\n")
    _write(meta / "slurm/campaign-9001.out", "campaign complete\n")

    accepted_class = hardware_class(accepted_hardware)
    allocation_class = hardware_class(allocation_one)
    markers: dict[str, dict] = {}
    cell_runtime: dict[str, dict] = {}
    for index, cell in enumerate(cells):
        score = 0.4 + cell.fraction * 0.1
        best_dev = score + 0.03
        marker = {
            "exp_id": cell.exp_id,
            "git_sha": git_sha,
            "detector_sha256": detector,
            "precision": "32-true",
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "h100_runtime_contract": runtime_contract,
            "best_dev_f1": best_dev,
            "last_dev": {"threshold": 0.25},
            "test_inference_precision": "32-true",
            "test_f1": score,
            "test_precision": score + 0.01,
            "test_recall": score - 0.01,
            "test_near_shore_f1": score - 0.02,
            "test_scored_at": "2026-07-27T00:00:00+00:00",
            "epochs_run": 2,
        }
        attempt_specs = (
            [(allocation_zero, 0, False), (allocation_one, 1, True)]
            if index == 0
            else [(allocation_one, index % 8, False)]
        )
        attempts = []
        for attempt_index, (allocation, gpu, resumed) in enumerate(
            attempt_specs, start=1
        ):
            device = allocation["devices"][gpu]
            attempt = {
                "attempt": attempt_index,
                "started_utc": f"2026-07-27T0{attempt_index}:00:00+00:00",
                "finished_utc": f"2026-07-27T0{attempt_index}:30:00+00:00",
                "slurm_job_id": "9001",
                "gpu_local_index": gpu,
                "gpu_uuid": device["uuid"],
                "gpu_name": device["name"],
                "gpu_total_memory_bytes": device["total_memory_bytes"],
                "compute_capability": device["compute_capability"],
                "driver_version": allocation_class["driver_version"],
                "torch": allocation_class["torch"],
                "cuda_build": allocation_class["cuda_build"],
                "resumed_from_last_ckpt": resumed,
                "active_seconds": 1800.0 if len(attempt_specs) == 2 else 3600.0,
            }
            if attempt_index == len(attempt_specs):
                attempt["exit_code"] = 0
            else:
                attempt["exit"] = "preempted"
            attempts.append(attempt)
        final_device = attempt_specs[-1][0]["devices"][attempt_specs[-1][1]]
        provenance = {
            "schema": 2,
            "campaign_id": "fixture-h100",
            "exp_id": cell.exp_id,
            "git_sha": git_sha,
            "detector_sha256": detector,
            "precision": "32-true",
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "venv_sha256": venv_sha256,
            "venv_build_sha256": venv_build_sha256,
            "base_python": base_python,
            "wheelhouse": wheelhouse,
            "base_payload": base_payload,
            "runtime_amendment": runtime_amendment,
            "acceptance_uuid": acceptance_uuid,
            "source_validation_sha256": source_validation_sha256,
            "cutover_ready_sha256": cutover_sha256,
            "v100_core_archived_sha256": archived_receipt_sha256,
            "strict_fp32": strict_fp32,
            "accepted_hardware_class": accepted_class,
            "slurm_job_id": "9001",
            "gpu_local_index": attempt_specs[-1][1],
            "gpu_name": final_device["name"],
            "gpu_uuid": final_device["uuid"],
            "gpu_total_memory_bytes": final_device["total_memory_bytes"],
            "compute_capability": final_device["compute_capability"],
            "driver_version": allocation_class["driver_version"],
            "torch": allocation_class["torch"],
            "cuda_build": allocation_class["cuda_build"],
            "started_utc": attempts[0]["started_utc"],
            "completed_utc": "2026-07-27T03:00:00+00:00",
            "attempts": attempts,
            "accumulated_active_seconds": 3600.0,
            "elapsed_hours": 1.0,
            "epochs_run": marker["epochs_run"],
            "best_dev_f1": marker["best_dev_f1"],
            "test_f1": marker["test_f1"],
            "test_scored_at": marker["test_scored_at"],
        }
        root = runs / cell.exp_id
        _write(root / "final_metrics.json", json.dumps(marker))
        _write(root / "runtime_provenance.json", json.dumps(provenance))
        _write(root / "config.yaml", "precision: 32-true\n")
        _write(root / "metrics/metrics.csv", "step,loss\n1,0.5\n")
        _write(root / "checkpoints/best.ckpt", b"best")
        _write(root / "checkpoints/last.ckpt", b"last")
        _write(runs / f"logs/h100/{cell.exp_id}.log", "complete\n")
        markers[cell.exp_id] = marker
        cell_runtime[cell.exp_id] = {
            "elapsed_hours": 1.0,
            "attempts": len(attempts),
            "epochs_run": marker["epochs_run"],
            "test_f1": marker["test_f1"],
        }

    grid = runs / "summary/grid.csv"
    grid.parent.mkdir(parents=True, exist_ok=True)
    grid_fields = (
        "exp_id",
        "init",
        "track",
        "role",
        "label_frac",
        "seed",
        "precision",
        "detector_sha256",
        "git_sha",
        "dev_f1",
        "dev_threshold",
        "test_f1",
        "epochs_run",
        "monotonicity_ok",
    )
    with grid.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=grid_fields)
        writer.writeheader()
        for init, arm in arms.items():
            for fraction in (0.1, 0.25, 0.5, 1.0):
                exp_id = f"{arm['short']}-f{int(fraction * 100)}-s0"
                marker = markers[exp_id]
                writer.writerow(
                    {
                        "exp_id": exp_id,
                        "init": init,
                        "track": arm["track"],
                        "role": arm["role"],
                        "label_frac": fraction,
                        "seed": 0,
                        "precision": "32-true",
                        "detector_sha256": detector,
                        "git_sha": git_sha,
                        "dev_f1": marker["best_dev_f1"],
                        "dev_threshold": marker["last_dev"]["threshold"],
                        "test_f1": marker["test_f1"],
                        "epochs_run": marker["epochs_run"],
                        "monotonicity_ok": True,
                    }
                )

    campaign = {
        "schema": 2,
        "campaign_id": "fixture-h100",
        "status": "complete",
        "git_sha": git_sha,
        "detector_sha256": detector,
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "venv_sha256": venv_sha256,
        "venv_build_sha256": venv_build_sha256,
        "base_python": base_python,
        "wheelhouse": wheelhouse,
        "base_payload": base_payload,
        "runtime_amendment": runtime_amendment,
        "acceptance_uuid": acceptance_uuid,
        "source_validation_sha256": source_validation_sha256,
        "cutover_ready_sha256": cutover_sha256,
        "v100_core_archived_sha256": archived_receipt_sha256,
        "archive_manifest_sha256": archive_manifest_sha256,
        "hardware": allocation_one,
        "accepted_hardware_class": accepted_class,
        "allocation_hardware_class": allocation_class,
        "strict_fp32": strict_fp32,
        "acceptance": {
            "uuid": acceptance_uuid,
            "created_utc": ready["created_utc"],
            "gates": gates,
            "frozen_sha256": ready["source"]["frozen_sha256"],
            "slurm_smoke": ready["slurm_smoke"],
            "scratch_free_bytes": ready["scratch_free_bytes"],
        },
        "operator_cutover": {
            "cutover_ready_sha256": cutover_sha256,
            "v100_core_archived_sha256": archived_receipt_sha256,
            "archive_manifest_sha256": archive_manifest_sha256,
            "v100_core_archived_path": str(archived_receipt_path),
            "archive_manifest_path": str(archive_manifest_path),
            "v100_core_archived": archived_receipt,
            "archive_manifest": archive_manifest,
        },
        "source_validation": {
            "path": str(source_validation_path),
            "sha256": source_validation_sha256,
            "receipt": source_validation,
        },
        "throughput_projection": projection,
        "slurm_job_id": "9001",
        "cell_order": ids,
        "complete": sorted(ids),
        "running": {},
        "events": [
            {
                "event": "grid_validated",
                "sha256": sha256_file(grid),
                "rows": 32,
            }
        ],
        "cell_runtime": cell_runtime,
        "updated_utc": "2026-07-27T00:00:00+00:00",
    }
    _write(campaign_path, json.dumps(campaign))
    return git_sha, ids


def test_reverse_attempt_accepts_uuid_reordered_across_requeue_inventories():
    backend = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }

    def hardware(order: list[int]) -> dict:
        return {
            "torch": "2.11.0+cu126",
            "cuda_build": "12.6",
            "driver_version": "590.1",
            "backend": backend,
            "devices": [
                {
                    "name": "NVIDIA H100 80GB HBM3",
                    "uuid": f"GPU-REQUEUE-{index}",
                    "compute_capability": [9, 0],
                    "total_memory_bytes": 80_000_000_000,
                }
                for index in order
            ],
        }

    first = hardware(list(range(8)))
    second = hardware([1, 2, 3, 4, 5, 6, 7, 0])
    device = second["devices"][7]
    provenance = {
        "attempts": [
            {
                "attempt": 1,
                "started_utc": "2026-07-27T00:00:00+00:00",
                "finished_utc": "2026-07-27T00:30:00+00:00",
                "slurm_job_id": "9001",
                "gpu_local_index": 7,
                "gpu_uuid": device["uuid"],
                "gpu_name": device["name"],
                "gpu_total_memory_bytes": device["total_memory_bytes"],
                "compute_capability": device["compute_capability"],
                "driver_version": "590.1",
                "torch": "2.11.0+cu126",
                "cuda_build": "12.6",
                "resumed_from_last_ckpt": True,
                "active_seconds": 1800.0,
                "exit_code": 0,
            }
        ],
        "gpu_uuid": device["uuid"],
    }
    allocation_records = [
        {"job_id": "9001", "restart": 0, "payload": first},
        {"job_id": "9001", "restart": 1, "payload": second},
    ]

    assert _validate_attempts(
        provenance,
        exp_id="vitin1k-f100-s0",
        allocation_records=allocation_records,
        accepted_hardware_class={
            "gpu_name": "NVIDIA H100 80GB HBM3",
            "total_memory_bytes": 80_000_000_000,
            "compute_capability": [9, 0],
            "driver_version": "590.1",
            "torch": "2.11.0+cu126",
            "cuda_build": "12.6",
        },
    ) == ["GPU-REQUEUE-0"]


def test_reverse_campaign_job_and_hardware_require_same_allocation(
    tmp_path, monkeypatch
):
    options, _ = _fixture_source(tmp_path)
    repo = options.repo_root
    runs = tmp_path / "runs"
    campaign_path = runs / ".h100/campaign_manifest.json"
    _git_sha, _ids = _write_result_fixture(
        repo, runs, campaign_path, monkeypatch
    )
    campaign = json.loads(campaign_path.read_text())

    unrelated_job_hardware = json.loads(
        (runs / ".h100/h100_runtime.json").read_text()
    )
    _write(
        runs / ".h100/h100_runtime-9002-r0.json",
        json.dumps(unrelated_job_hardware),
    )
    campaign["hardware"] = unrelated_job_hardware
    campaign_path.write_text(json.dumps(campaign))
    output = tmp_path / _result_package_id(campaign)

    with pytest.raises(PackageError, match="job/hardware pair"):
        build_results_package(
            repo=repo,
            runs_root=runs,
            campaign_manifest=campaign_path,
            output=output,
            max_part_bytes=1024,
        )


def test_reverse_results_requires_and_packages_exact_32_cells(
    tmp_path, monkeypatch
):
    options, _ = _fixture_source(tmp_path)
    repo = options.repo_root
    runs = tmp_path / "runs"
    campaign = runs / ".h100/campaign_manifest.json"
    git_sha, ids = _write_result_fixture(
        repo, runs, campaign, monkeypatch
    )
    campaign_payload = json.loads(campaign.read_text())
    output = tmp_path / _result_package_id(campaign_payload)
    built = build_results_package(
        repo=repo,
        runs_root=runs,
        campaign_manifest=campaign,
        output=output,
        max_part_bytes=1024,
    )
    manifest = verify_package(built)
    assert manifest["package_type"] == "h100-core-results"
    assert manifest["cells"] == ids
    assert manifest["counts"] == {
        "core_result_archives": 32,
        "provenance_archives": 1,
    }
    identity = manifest["source"]["result_identity"]
    assert identity["venv_sha256"] == campaign_payload["venv_sha256"]
    assert identity["venv_build_sha256"] == campaign_payload[
        "venv_build_sha256"
    ]
    assert identity["base_python"] == campaign_payload["base_python"]
    assert identity["base_payload"] == campaign_payload["base_payload"]
    assert identity["runtime_amendment"] == campaign_payload[
        "runtime_amendment"
    ]
    assert identity["acceptance_uuid"] == campaign_payload["acceptance"]["uuid"]
    assert identity["strict_fp32"] == campaign_payload["strict_fp32"]
    assert identity["accepted_hardware_class"] == campaign_payload[
        "accepted_hardware_class"
    ]
    assert identity["allocation_hardware_class"] == campaign_payload[
        "allocation_hardware_class"
    ]
    provenance_artifact = next(
        item
        for item in manifest["artifacts"]
        if item["kind"] == "campaign_provenance"
    )
    members = provenance_artifact["member_sha256"]
    mandatory = {
        "results/provenance/campaign_manifest.json",
        "results/provenance/H100_READY.json",
        "results/provenance/h100_runtime.json",
        "results/provenance/throughput_projection.json",
        "results/provenance/venv_build.json",
        "results/provenance/CUTOVER_READY.json",
        "results/provenance/SOURCE_VALIDATED.json",
        "results/provenance/HOST_HANDOFF_TESTS.json",
        "results/provenance/PYTEST_ACCEPTANCE.json",
        "results/provenance/V100_CORE_ARCHIVED.json",
        "results/provenance/V100_CORE_ARCHIVE_MANIFEST.json",
        "results/provenance/slurm-smoke/SLURM_SMOKE_READY.json",
        "results/provenance/slurm-smoke/SLURM_SMOKE_STATE.json",
        "results/provenance/summary/grid.csv",
        "results/provenance/acceptance-logs/pytest-handoff-host.log",
        "results/provenance/acceptance-logs/pytest-venv-remaining.log",
        "results/provenance/acceptance-logs/vit-fp32.log",
        "results/provenance/acceptance-logs/cnn-200step-fp32.log",
        "results/provenance/slurm/campaign-9001.out",
    }
    assert mandatory <= set(members)
    assert len(
        [
            name
            for name in members
            if name.startswith("results/provenance/allocations/h100_runtime-")
        ]
    ) == 2
    assert all(
        artifact.get("member_sha256")
        for artifact in manifest["artifacts"]
    )
    original_manifest = json.loads(json.dumps(manifest))

    tampered = json.loads(json.dumps(original_manifest))
    tampered["source"]["campaign_provenance_member_sha256"][
        "results/provenance/summary/grid.csv"
    ] = "0" * 64
    _rewrite_package_controls(built, tampered)
    with pytest.raises(PackageError, match="digest index is not archive-bound"):
        verify_package(built)

    tampered = json.loads(json.dumps(original_manifest))
    tampered["source"]["runtime_provenance_sha256"][ids[0]] = "0" * 64
    _rewrite_package_controls(built, tampered)
    with pytest.raises(PackageError, match="runtime-provenance digest"):
        verify_package(built)

    tampered = json.loads(json.dumps(original_manifest))
    core_artifact = next(
        item
        for item in tampered["artifacts"]
        if item["kind"] == "core_result" and item["name"] == ids[0]
    )
    core_artifact["member_sha256"].pop(f"results/core/{ids[0]}/log.txt")
    _rewrite_package_controls(built, tampered)
    with pytest.raises(PackageError, match="exact result allowlist"):
        verify_package(built)

    _rewrite_package_controls(built, original_manifest)
    built.rename(tmp_path / "prior-valid-results")
    (runs / ids[0] / "final_metrics.json").unlink()
    with pytest.raises((PackageError, RuntimeError), match="completion marker"):
        build_results_package(
            repo=repo,
            runs_root=runs,
            campaign_manifest=campaign,
            output=output,
            max_part_bytes=1024,
        )


def test_reverse_results_rejects_unbound_grid_gpu_and_inside_repo(
    tmp_path, monkeypatch
):
    import csv

    options, _ = _fixture_source(tmp_path)
    repo = options.repo_root
    runs = tmp_path / "runs"
    campaign = runs / ".h100/campaign_manifest.json"
    _git_sha, ids = _write_result_fixture(
        repo, runs, campaign, monkeypatch
    )
    campaign_payload = json.loads(campaign.read_text())
    package_id = _result_package_id(campaign_payload)
    output = tmp_path / package_id

    with pytest.raises(PackageError, match="outside the repository"):
        build_results_package(
            repo=repo,
            runs_root=runs,
            campaign_manifest=campaign,
            output=repo / package_id,
            max_part_bytes=1024,
        )

    grid = runs / "summary/grid.csv"
    original_grid = grid.read_bytes()
    with grid.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["test_f1"] = "0.999"
    with grid.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(PackageError, match="differs from final_metrics"):
        build_results_package(
            repo=repo,
            runs_root=runs,
            campaign_manifest=campaign,
            output=output,
            max_part_bytes=1024,
        )

    grid.write_bytes(original_grid)
    provenance_path = runs / ids[0] / "runtime_provenance.json"
    original_provenance = provenance_path.read_bytes()
    provenance = json.loads(original_provenance)
    provenance["attempts"][0]["gpu_uuid"] = "GPU-NOT-IN-ALLOCATION"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(PackageError, match="absent from its allocation"):
        build_results_package(
            repo=repo,
            runs_root=runs,
            campaign_manifest=campaign,
            output=output,
            max_part_bytes=1024,
        )

    provenance_path.write_bytes(original_provenance)
    log_root = runs / "logs/h100"
    outside_logs = tmp_path / "outside-h100-logs"
    log_root.rename(outside_logs)
    log_root.symlink_to(outside_logs, target_is_directory=True)
    with pytest.raises(PackageError, match="symlink"):
        build_results_package(
            repo=repo,
            runs_root=runs,
            campaign_manifest=campaign,
            output=output,
            max_part_bytes=1024,
        )
    log_root.unlink()
    outside_logs.rename(log_root)
    cutover = runs / ".h100/CUTOVER_READY.json"
    cutover.unlink()
    with pytest.raises(PackageError, match="invalid required result JSON"):
        build_results_package(
            repo=repo,
            runs_root=runs,
            campaign_manifest=campaign,
            output=output,
            max_part_bytes=1024,
        )
