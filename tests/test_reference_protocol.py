"""Scientific reference behavior and portable runtime provenance."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.references import locateanything_zs, yolo26_ref
from src.runtime import reference


def _symlinks_available() -> bool:
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "target"
        target.write_text("probe", encoding="utf-8")
        try:
            (Path(scratch) / "link").symlink_to(target)
        except OSError:
            return False
    return True


requires_symlinks = pytest.mark.skipif(
    not _symlinks_available(),
    reason="creating symlinks requires privilege on this platform",
)


GIT_SHA = "a" * 40


def _runtime_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Namespace, str]:
    runtime_lines = {"torch==2.11.0+cu126"}
    environment_sha256 = hashlib.sha256(
        ("\n".join(sorted(runtime_lines)) + "\n").encode("utf-8")
    ).hexdigest()
    lock = tmp_path / "environment.txt"
    lock.write_text("torch==2.11.0+cu126\n", encoding="utf-8")
    monkeypatch.setattr(reference, "_clean_git_sha", lambda _repo: GIT_SHA)
    monkeypatch.setattr(reference, "_runtime_environment_lines", lambda: runtime_lines)
    args = Namespace(
        expected_git_sha=GIT_SHA,
        environment_sha256=environment_sha256,
        environment_lock=lock,
        results_root=tmp_path / "results",
    )
    return args, environment_sha256


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronized: list[int] = []

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 2

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (9, 0)

    @staticmethod
    def get_arch_list() -> list[str]:
        return ["sm_70", "sm_90"]

    @staticmethod
    def get_device_name(index: int) -> str:
        return f"NVIDIA Example GPU {index}"

    def synchronize(self, index: int) -> None:
        self.synchronized.append(index)


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.11.0+cu126",
        version=SimpleNamespace(cuda="12.6"),
        cuda=_FakeCuda(),
    )


def test_environment_lock_identity_is_order_and_name_canonical(tmp_path: Path) -> None:
    lock = tmp_path / "environment.txt"
    lock.write_text("Z_pkg==2\na.pkg==1\n", encoding="utf-8")
    expected = hashlib.sha256(b"a-pkg==1\nz-pkg==2\n").hexdigest()
    assert reference.normalized_environment_lock_sha256(lock) == expected


def test_reference_runtime_records_generic_assigned_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, environment_sha256 = _runtime_fixture(tmp_path, monkeypatch)
    inputs = reference.load_runtime_inputs(args, repo=tmp_path, required=True)
    assert inputs is not None
    ticks = iter((1_000_000_000, 4_600_000_000_000))
    monkeypatch.setattr(reference.time, "monotonic_ns", lambda: next(ticks))
    torch = _fake_torch()

    execution = reference.begin_reference_execution(
        inputs,
        reference_precision="float32",
        device="cuda:1",
        torch_module=torch,
    )
    payload = reference.finish_reference_execution(
        execution,
        device="cuda:1",
        torch_module=torch,
    )

    assert payload["schema"] == reference.PROVENANCE_SCHEMA
    assert payload["environment_sha256"] == environment_sha256
    assert payload["reference_precision"] == "float32"
    assert payload["hardware"] == {
        "accelerator": "NVIDIA Example GPU 1",
        "compute_capability": "9.0",
        "device": "cuda:1",
        "device_index": 1,
        "torch_version": "2.11.0+cu126",
        "cuda_version": "12.6",
    }
    assert payload["elapsed_hours"] == payload["gpu_hours"] > 0
    assert torch.cuda.synchronized == [1]


def test_reference_runtime_fails_closed_but_smoke_needs_no_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _environment_sha256 = _runtime_fixture(tmp_path, monkeypatch)
    assert reference.load_runtime_inputs(Namespace(), repo=tmp_path, required=False) is None

    args.expected_git_sha = "bad"
    with pytest.raises(RuntimeError, match="expected-git-sha"):
        reference.load_runtime_inputs(args, repo=tmp_path, required=True)
    args.expected_git_sha = GIT_SHA
    args.environment_sha256 = None
    with pytest.raises(RuntimeError, match="environment-sha256"):
        reference.load_runtime_inputs(args, repo=tmp_path, required=True)
    with pytest.raises(RuntimeError, match="device must"):
        reference.probe_gpu_runtime(_fake_torch(), device="cpu")
    with pytest.raises(RuntimeError, match="unavailable"):
        reference.probe_gpu_runtime(_fake_torch(), device="cuda:2")


def test_reference_result_pair_is_fresh_atomic_and_metrics_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    real_atomic_write = reference.atomic_write_json

    def tracked_atomic_write(path: Path, payload: object) -> None:
        calls.append(path.name)
        real_atomic_write(path, payload)

    monkeypatch.setattr(reference, "atomic_write_json", tracked_atomic_write)
    provenance = {
        "schema": 1,
        "git_sha": GIT_SHA,
        "environment_sha256": "b" * 64,
        "environment_lock_sha256": "c" * 64,
        "hardware": {
            "accelerator": "NVIDIA Example GPU",
            "compute_capability": "9.0",
            "device": "cuda:0",
            "device_index": 0,
            "torch_version": "2.11.0+cu126",
            "cuda_version": "12.6",
        },
        "started_utc": "2026-01-01T00:00:00+00:00",
        "finished_utc": "2026-01-01T01:00:00+00:00",
        "elapsed_hours": 1.0,
        "gpu_hours": 1.0,
        "reference_precision": "float32",
    }
    result_dir = tmp_path / "references" / "yolo26-f100"

    metrics_path, provenance_path = reference.publish_reference_result(
        result_dir,
        metrics={"result_schema": 2},
        provenance=provenance,
    )

    assert calls == ["runtime_provenance.json", "final_metrics.json"]
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {"result_schema": 2}
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == provenance
    assert metrics_path.stat().st_mode & 0o777 == 0o444
    assert provenance_path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(RuntimeError, match="fresh result directory"):
        reference.publish_reference_result(
            result_dir,
            metrics={"result_schema": 2},
            provenance=provenance,
        )


def test_r2_uses_shared_classifier_and_exact_preserved_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [
        {
            "chip_col": 10.0,
            "chip_row": 20.0,
            "confidence": "medium",
            "is_vessel": "true",
            "vessel_length_m": 80.0,
        },
        {
            "chip_col": 30.0,
            "chip_row": 40.0,
            "confidence": "HIGH",
            "is_vessel": "false",
        },
        {
            "chip_col": 50.0,
            "chip_row": 60.0,
            "confidence": "LOW",
            "is_vessel": None,
        },
    ]
    assert len(yolo26_ref.yolo_label_lines(labels, 800)) == 1

    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "runs/yolo26-f100/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preserved-r2")
    monkeypatch.setattr(
        yolo26_ref,
        "EXPECTED_R2_BEST_SHA256",
        yolo26_ref._sha256_file(checkpoint),
    )
    selected, digest = yolo26_ref._known_r2_checkpoint(checkpoint)
    assert selected == checkpoint
    assert digest == yolo26_ref._sha256_file(checkpoint)
    alternate = tmp_path / "runs/yolo26-f100/weights/last.pt"
    alternate.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="preserved"):
        yolo26_ref._known_r2_checkpoint(alternate)


def test_r3_retains_low_ignore_labels_and_deterministic_sample() -> None:
    labels = [
        {
            "chip_col": 2.0,
            "chip_row": 3.0,
            "confidence": "HIGH",
            "is_vessel": True,
            "source": "manual",
            "distance_from_shore_km": 1.5,
        },
        {
            "chip_col": 4.0,
            "chip_row": 5.0,
            "confidence": "HIGH",
            "is_vessel": False,
            "source": "ais",
            "distance_from_shore_km": 8.0,
        },
        {
            "chip_col": 6.0,
            "chip_row": 7.0,
            "confidence": "LOW",
            "is_vessel": None,
            "source": "ais",
            "distance_from_shore_km": 9.0,
        },
    ]
    assert locateanything_zs.chip_label_counts(labels) == {
        "positive": 1,
        "background": 1,
        "ignore": 1,
    }
    points = locateanything_zs.chip_ground_truth_from_labels(labels)
    assert len(points) == 2
    assert points[1].confidence == "LOW" and points[1].is_low_confidence

    candidates = list(range(1525))
    selected = locateanything_zs.select_evenly_strided(candidates, 200)
    expected = [(index * 1525) // 200 for index in range(200)]
    assert selected == expected
    assert len(selected) == len(set(selected)) == 200
    assert locateanything_zs.SAMPLE_ALGORITHM_VERSION == "floor-index-v1"


@requires_symlinks
def test_r3_model_payload_identity_rejects_drift_and_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_bytes(b"config")
    nested = root / "code"
    nested.mkdir()
    (nested / "model.py").write_bytes(b"model")
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": locateanything_zs._sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    digest = locateanything_zs._manifest_sha256(entries)
    total_bytes = sum(entry["size"] for entry in entries)
    assert locateanything_zs.validate_model_payload(
        root,
        expected_file_count=2,
        expected_bytes=total_bytes,
        expected_sha256=digest,
    ) == digest

    (nested / "model.py").write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        locateanything_zs.validate_model_payload(
            root,
            expected_file_count=2,
            expected_bytes=total_bytes,
            expected_sha256=digest,
        )
    (nested / "alias.py").symlink_to(nested / "model.py")
    with pytest.raises(RuntimeError, match="symlink"):
        locateanything_zs.validate_model_payload(
            root,
            expected_file_count=3,
            expected_bytes=0,
            expected_sha256="0" * 64,
        )


def test_reference_entrypoints_bind_their_published_precision() -> None:
    r2_source = Path(yolo26_ref.__file__).read_text(encoding="utf-8")
    r3_source = Path(locateanything_zs.__file__).read_text(encoding="utf-8")
    assert 'reference_precision="float32"' in r2_source
    assert 'reference_precision="bfloat16"' in r3_source
    assert "required=not bool(args.smoke)" in r3_source
    assert "publish_reference_result(" in r2_source
    assert "publish_reference_result(" in r3_source
