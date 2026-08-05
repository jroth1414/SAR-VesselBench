from __future__ import annotations

import hashlib
import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "slurm/h100/submit.sh"
GIT_SHA = "1" * 40
SHA256 = "2" * 64


def _site_values(
    tmp_path: Path,
    *,
    h100_runs: Path,
) -> dict[str, str]:
    return {
        "H100_ACCOUNT": "geofam",
        "H100_PARTITION": "minor-use-case",
        "H100_RESERVATION": "geofam",
        "H100_PROJECT": "xview3",
        "H100_PROJECT_ROOT": str(REPO),
        "H100_JOB_LOG_DIR": str(tmp_path / "h100-job-logs"),
        "H100_TRANSFER_PYTHON": "/nonexistent/transfer-python",
        "H100_BASE_PACKAGE_ROOT": str(tmp_path / "base-package"),
        "H100_BASE_PACKAGE_ID": "base-package",
        "H100_BASE_GIT_SHA": GIT_SHA,
        "H100_BASE_REPO_BUNDLE_SHA256": SHA256,
        "H100_BASE_MANIFEST_SHA256": SHA256,
        "H100_BASE_READY_SHA256": SHA256,
        "H100_BASE_SHA256SUMS_SHA256": SHA256,
        "H100_WHEELHOUSE": str(tmp_path / "wheelhouse"),
        "H100_WHEELHOUSE_SHA256": SHA256,
        "H100_BASE_EXTRACTION_RECEIPT": str(tmp_path / "base-extraction.json"),
        "H100_BASE_EXTRACTION_RECEIPT_SHA256": SHA256,
        "H100_RUNTIME_PACKAGE_ROOT": str(tmp_path / "runtime-package"),
        "H100_RUNTIME_PACKAGE_ID": "runtime-package",
        "H100_RUNTIME_GIT_SHA": GIT_SHA,
        "H100_RUNTIME_BUNDLE": str(tmp_path / "runtime.bundle"),
        "H100_RUNTIME_BUNDLE_SHA256": SHA256,
        "H100_RUNTIME_MANIFEST_SHA256": SHA256,
        "H100_RUNTIME_READY_SHA256": SHA256,
        "H100_RUNTIME_SHA256SUMS_SHA256": SHA256,
        "H100_BASE_PYTHON": str(tmp_path / "python3.11"),
        "H100_BASE_PYTHON_LIB_DIR": str(tmp_path / "python-lib"),
        "H100_BASE_PYTHON_SHA256": SHA256,
        "H100_VENV_ROOT": str(tmp_path / "venv"),
        "H100_BASE_PYTHON_RUNTIME_SHA256": SHA256,
        "H100_VENV_SHA256": SHA256,
        "H100_VENV_BUILD_JSON": str(tmp_path / "venv.build.json"),
        "H100_VENV_BUILD_SHA256": SHA256,
        "H100_ENV_LOCK_SHA256": SHA256,
        "H100_RUNS_ROOT": str(h100_runs),
        "H100_CAMPAIGN_ID": "h100-campaign",
        "H100_REFERENCE_CAMPAIGN_ID": "v100-references",
        "H100_EXPECTED_GIT_SHA": GIT_SHA,
        "H100_EXPECTED_REFERENCE_GIT_SHA": GIT_SHA,
        "H100_V100_CORE_GIT_SHA": "48e10534a8c7baf0662acd548f52928da69f23c8",
        "H100_V100_CORE_CAMPAIGN_ID": "fresh34-v100-fp32-20260726",
        "H100_CUTOVER_READY": str(h100_runs / ".h100/CUTOVER_READY.json"),
        "H100_CUTOVER_READY_SHA256": SHA256,
        "H100_REFERENCES_PACKAGE_ROOT": str(tmp_path / "references-package"),
        "H100_REFERENCES_PACKAGE_ID": "references-package",
        "H100_REFERENCES_PRODUCER_GIT_SHA": GIT_SHA,
        "H100_REFERENCES_IDENTITY_SHA256": SHA256,
        "H100_REFERENCES_MANIFEST_SHA256": SHA256,
        "H100_REFERENCES_READY_SHA256": SHA256,
        "H100_REFERENCES_SHA256SUMS_SHA256": SHA256,
        "H100_DIAGNOSTIC_ISOLATION_PACKAGE_ROOT": str(
            tmp_path / "diagnostic-isolation-package"
        ),
        "H100_DIAGNOSTIC_ISOLATION_PACKAGE_ID": "diagnostic-isolation-package",
        "H100_DIAGNOSTIC_ISOLATION_PRODUCER_GIT_SHA": GIT_SHA,
        "H100_DIAGNOSTIC_ISOLATION_IDENTITY_SHA256": SHA256,
        "H100_DIAGNOSTIC_ISOLATION_MANIFEST_SHA256": SHA256,
        "H100_DIAGNOSTIC_ISOLATION_READY_SHA256": SHA256,
        "H100_DIAGNOSTIC_ISOLATION_SHA256SUMS_SHA256": SHA256,
        "H100_DETECTOR_SHA256": SHA256,
        "H100_SCORER_SHA256": SHA256,
        "H100_SPLITS_SHA256": SHA256,
        "H100_STATS_SHA256": SHA256,
        "H100_LSSSDD_SHA256": SHA256,
        "H100_REMAINING_V100_WALL_HOURS": "100",
        "H100_CURRENT_V100_DIAGNOSTIC_STATUS": (
            "continues-running-non-reportable-diagnostic"
        ),
        "H100_V100_CONTROL_PLANE": "box-transfer-v1",
    }


def _run_submit(
    tmp_path: Path,
    *,
    mode: str,
    h100_runs: Path,
    job_logs: Path | None = None,
    reference_runs: Path | None = None,
    extra_env: dict[str, str] | None = None,
    prepare_runtime_paths: bool = False,
    site_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    values = _site_values(
        tmp_path,
        h100_runs=h100_runs,
    )
    if reference_runs is not None:
        values["H100_REFERENCES_PACKAGE_ROOT"] = str(reference_runs)
    values.update(site_overrides or {})
    if job_logs is not None:
        values["H100_JOB_LOG_DIR"] = str(job_logs)
    if prepare_runtime_paths:
        Path(values["H100_WHEELHOUSE"]).mkdir(parents=True, exist_ok=True)
        Path(values["H100_BASE_EXTRACTION_RECEIPT"]).write_text("{}\n")
        python_lib = Path(values["H100_BASE_PYTHON_LIB_DIR"])
        python_lib.mkdir(parents=True, exist_ok=True)
        (python_lib / "libpython3.11.so.1.0").write_bytes(b"fixture libpython")
        base_python = Path(values["H100_BASE_PYTHON"])
        base_python.write_text("#!/bin/sh\nexit 0\n")
        base_python.chmod(0o755)
        venv_python = Path(values["H100_VENV_ROOT"]) / "bin/python"
        venv_python.parent.mkdir(parents=True, exist_ok=True)
        venv_python.write_text("#!/bin/sh\nexit 0\n")
        venv_python.chmod(0o755)
        Path(values["H100_VENV_BUILD_JSON"]).write_text("{}\n")
    site_env = tmp_path / f"site-{mode}.env"
    site_env.write_text(
        "".join(f"{name}={shlex.quote(value)}\n" for name, value in values.items())
    )
    return subprocess.run(
        [str(SUBMIT), mode],
        check=False,
        text=True,
        capture_output=True,
        env={
            "H100_SITE_ENV": str(site_env),
            "PATH": os.environ["PATH"],
            **(extra_env or {}),
        },
    )


@pytest.mark.parametrize("mode", ["smoke", "acceptance"])
@pytest.mark.parametrize("relationship", ["equal", "logs-child", "logs-parent"])
def test_submit_rejects_h100_write_root_overlap_before_any_write(
    tmp_path: Path, mode: str, relationship: str
):
    tree = tmp_path / "h100-output"
    references = tmp_path / "transferred-v100-references"
    references.mkdir()
    if relationship == "equal":
        h100_runs = job_logs = tree
    elif relationship == "logs-child":
        h100_runs = tree
        job_logs = h100_runs / "logs"
    else:
        job_logs = tree
        h100_runs = job_logs / "runs"
    result = _run_submit(
        tmp_path,
        mode=mode,
        h100_runs=h100_runs,
        job_logs=job_logs,
        reference_runs=references,
    )
    assert result.returncode == 2
    assert "H100_JOB_LOG_DIR overlaps H100 runs root" in result.stderr
    assert not (h100_runs / ".h100").exists()
    assert not job_logs.exists()


@pytest.mark.parametrize("mode", ["smoke", "acceptance"])
@pytest.mark.parametrize(
    ("write_name", "expected"),
    (
        ("runs", "H100_RUNS_ROOT overlaps H100_BASE_PACKAGE_ROOT"),
        ("logs", "H100_JOB_LOG_DIR overlaps H100_BASE_PACKAGE_ROOT"),
    ),
)
def test_submit_rejects_write_overlap_with_immutable_input_before_write(
    tmp_path: Path, mode: str, write_name: str, expected: str
):
    protected = tmp_path / "base-package"
    protected.mkdir()
    references = tmp_path / "transferred-v100-references"
    references.mkdir()
    h100_runs = (
        protected / "nested-runs"
        if write_name == "runs"
        else tmp_path / "h100-runs"
    )
    job_logs = (
        protected / "nested-logs"
        if write_name == "logs"
        else tmp_path / "h100-job-logs"
    )
    result = _run_submit(
        tmp_path,
        mode=mode,
        h100_runs=h100_runs,
        job_logs=job_logs,
        reference_runs=references,
    )
    assert result.returncode == 2
    assert expected in result.stderr
    assert not (h100_runs / ".h100").exists()
    assert not job_logs.exists()


@pytest.mark.parametrize(
    ("site_overrides", "expected"),
    (
        ({"H100_RUNS_ROOT": "relative/runs"}, "H100_RUNS_ROOT must be an absolute path"),
        ({"H100_RUNS_ROOT": "/"}, "H100_RUNS_ROOT must not resolve to /"),
        (
            {"H100_JOB_LOG_DIR": "relative/logs"},
            "H100_JOB_LOG_DIR must be an absolute path",
        ),
        ({"H100_JOB_LOG_DIR": "/"}, "H100_JOB_LOG_DIR must not resolve to /"),
    ),
)
def test_submit_rejects_unsafe_write_root_before_write(
    tmp_path: Path, site_overrides: dict[str, str], expected: str
):
    h100_runs = tmp_path / "h100-runs"
    result = _run_submit(
        tmp_path,
        mode="smoke",
        h100_runs=h100_runs,
        site_overrides=site_overrides,
    )
    assert result.returncode == 2
    assert expected in result.stderr
    assert not h100_runs.exists()


def test_submit_requires_box_control_plane_before_write(tmp_path: Path):
    h100_runs = tmp_path / "h100-runs"
    result = _run_submit(
        tmp_path,
        mode="smoke",
        h100_runs=h100_runs,
        site_overrides={"H100_V100_CONTROL_PLANE": "shared-filesystem"},
    )
    assert result.returncode == 2
    assert "H100_V100_CONTROL_PLANE must be box-transfer-v1" in result.stderr
    assert not h100_runs.exists()


def test_submit_write_root_guard_precedes_every_persistent_write():
    source = SUBMIT.read_text()
    guard = source.index('H100_JOB_LOG_DIR "$h100_job_log_root" "H100 runs root"')
    for marker in (
        "mkdir -p \"$H100_RUNS_ROOT/.h100/slurm\"",
        "mkdir -p \"$H100_JOB_LOG_DIR\"",
        "-m scripts.h100.cutover",
        "-m scripts.h100.operator_cutover",
        "sbatch \\",
    ):
        assert guard < source.index(marker)


def test_submit_rejects_unbound_python_library_path_before_write(tmp_path: Path):
    h100_runs = tmp_path / "h100-runs"
    result = _run_submit(
        tmp_path,
        mode="smoke",
        h100_runs=h100_runs,
        site_overrides={"H100_BASE_PYTHON_LIB_DIR": "relative/python-lib"},
    )
    assert result.returncode == 2
    assert (
        "H100_BASE_PYTHON_LIB_DIR must be an absolute existing directory"
        in result.stderr
    )
    assert not h100_runs.exists()


@pytest.mark.parametrize(
    ("mode", "missing_name"),
    (
        ("cutover-check", "H100_REFERENCES_PACKAGE_ROOT"),
        ("cutover-check", "H100_CAMPAIGN_ID"),
        ("campaign", "H100_DIAGNOSTIC_ISOLATION_PACKAGE_ROOT"),
    ),
)
def test_dynamic_control_modes_require_content_addressed_package_before_any_write(
    tmp_path: Path, mode: str, missing_name: str
):
    h100_runs = tmp_path / "h100-runs"
    job_logs = tmp_path / "h100-job-logs"
    result = _run_submit(
        tmp_path,
        mode=mode,
        h100_runs=h100_runs,
        job_logs=job_logs,
        site_overrides={missing_name: ""},
    )
    assert result.returncode == 2
    assert f"site.env is missing {missing_name}" in result.stderr
    assert not h100_runs.exists()
    assert not job_logs.exists()


@pytest.mark.parametrize("mode", ["smoke", "acceptance"])
def test_pre_cutover_modes_do_not_require_v100_paths(
    tmp_path: Path, mode: str
):
    fake_bin = tmp_path / f"fake-bin-{mode}"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/bin/sh\nexit 0\n")
    fake_sbatch.chmod(0o755)
    h100_runs = tmp_path / f"h100-runs-{mode}"
    result = _run_submit(
        tmp_path,
        mode=mode,
        h100_runs=h100_runs,
        prepare_runtime_paths=True,
        extra_env={"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )
    assert result.returncode == 0, result.stderr
    assert "H100_V100_RUNS_ROOT" not in (tmp_path / f"site-{mode}.env").read_text()


def test_submit_snapshots_sanitized_site_and_batches_reject_tampering(tmp_path: Path):
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    args_path = tmp_path / "sbatch.args"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$@\" > \"$FAKE_SBATCH_ARGS\"\n"
    )
    fake_sbatch.chmod(0o755)

    h100_runs = tmp_path / "h100-runs"
    result = _run_submit(
        tmp_path,
        mode="smoke",
        h100_runs=h100_runs,
        prepare_runtime_paths=True,
        site_overrides={
            "BOX_JWT_CONFIG": "/secret/jwt.json",
            "BOX_FOLDER_ID": "secret-folder-404384490657",
        },
        extra_env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "FAKE_SBATCH_ARGS": str(args_path),
            "LD_LIBRARY_PATH": "/inherited/loader/path",
        },
    )
    assert result.returncode == 0, result.stderr
    submitted_args = args_path.read_text().splitlines()
    assert submitted_args[-3] == "smoke"
    snapshot = Path(submitted_args[-2])
    digest = submitted_args[-1]
    assert snapshot.is_file() and not snapshot.is_symlink()
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o444
    snapshot_bytes = snapshot.read_bytes()
    assert hashlib.sha256(snapshot_bytes).hexdigest() == digest
    snapshot_text = snapshot_bytes.decode()
    assert "BOX_" not in snapshot_text
    assert "H100_V100_RUNS_ROOT" not in snapshot_text
    assert "H100_REFERENCES_PACKAGE_" not in snapshot_text
    assert "H100_DIAGNOSTIC_ISOLATION_PACKAGE_" not in snapshot_text
    assert "H100_V100_CONTROL_PLANE=box-transfer-v1" in snapshot_text
    assert "secret-folder-404384490657" not in snapshot_text

    assert "H100_BASE_PYTHON_LIB_DIR=" in snapshot_text
    assert "/inherited/loader/path" not in snapshot_text
    original_site = tmp_path / "site-smoke.env"
    original_site.write_text("H100_RUNS_ROOT=/now/different\n")
    assert snapshot.read_bytes() == snapshot_bytes

    tampered = tmp_path / "tampered-compute-site.env"
    tampered.write_bytes(snapshot_bytes + b"H100_RUNS_ROOT=/tampered\n")
    tampered.chmod(0o444)
    before = {path.relative_to(h100_runs) for path in h100_runs.rglob("*")}
    for batch, batch_mode in (
        (REPO / "slurm/h100/campaign.sbatch", "acceptance"),
        (REPO / "slurm/h100/smoke.sbatch", "smoke"),
    ):
        completed = subprocess.run(
            [
                str(batch),
                batch_mode,
                str(tampered),
                digest,
            ],
            cwd=REPO,
            check=False,
            text=True,
            capture_output=True,
            env={"PATH": os.environ["PATH"]},
        )
        assert completed.returncode == 2
        assert "compute-site snapshot SHA-256 mismatch" in completed.stderr
    after = {path.relative_to(h100_runs) for path in h100_runs.rglob("*")}
    assert after == before
    assert not (h100_runs / ".h100/SOURCE_VALIDATED.json").exists()
