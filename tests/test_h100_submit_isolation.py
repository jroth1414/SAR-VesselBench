from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "slurm/h100/submit.sh"
GIT_SHA = "1" * 40
SHA256 = "2" * 64


def _site_values(tmp_path: Path, *, h100_runs: Path, v100_runs: Path) -> dict[str, str]:
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
        "H100_RUNTIME_PACKAGE_ROOT": str(tmp_path / "runtime-package"),
        "H100_RUNTIME_PACKAGE_ID": "runtime-package",
        "H100_RUNTIME_GIT_SHA": GIT_SHA,
        "H100_RUNTIME_BUNDLE": str(tmp_path / "runtime.bundle"),
        "H100_RUNTIME_BUNDLE_SHA256": SHA256,
        "H100_RUNTIME_MANIFEST_SHA256": SHA256,
        "H100_RUNTIME_READY_SHA256": SHA256,
        "H100_RUNTIME_SHA256SUMS_SHA256": SHA256,
        "H100_BASE_PYTHON": "/nonexistent/python3.11",
        "H100_BASE_PYTHON_SHA256": SHA256,
        "H100_VENV_ROOT": str(tmp_path / "venv"),
        "H100_VENV_SHA256": SHA256,
        "H100_VENV_BUILD_JSON": str(tmp_path / "venv.build.json"),
        "H100_VENV_BUILD_SHA256": SHA256,
        "H100_ENV_LOCK_SHA256": SHA256,
        "H100_RUNS_ROOT": str(h100_runs),
        "H100_V100_RUNS_ROOT": str(v100_runs),
        "H100_REFERENCES_ROOT": str(tmp_path / "references"),
        "H100_CAMPAIGN_ID": "h100-campaign",
        "H100_REFERENCE_CAMPAIGN_ID": "v100-references",
        "H100_EXPECTED_GIT_SHA": GIT_SHA,
        "H100_EXPECTED_REFERENCE_GIT_SHA": GIT_SHA,
        "H100_CUTOVER_READY": str(h100_runs / ".h100/CUTOVER_READY.json"),
        "H100_CUTOVER_READY_SHA256": SHA256,
        "H100_V100_CORE_ARCHIVED": str(tmp_path / "V100_CORE_ARCHIVED.json"),
        "H100_V100_CORE_ARCHIVED_SHA256": SHA256,
        "H100_V100_ARCHIVE_MANIFEST": str(tmp_path / "v100-archive-manifest.json"),
        "H100_V100_ARCHIVE_MANIFEST_SHA256": SHA256,
        "H100_DETECTOR_SHA256": SHA256,
        "H100_SCORER_SHA256": SHA256,
        "H100_SPLITS_SHA256": SHA256,
        "H100_STATS_SHA256": SHA256,
        "H100_LSSSDD_SHA256": SHA256,
        "H100_REMAINING_V100_WALL_HOURS": "100",
    }


def _run_submit(
    tmp_path: Path,
    *,
    mode: str,
    h100_runs: Path,
    v100_runs: Path,
    job_logs: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    values = _site_values(tmp_path, h100_runs=h100_runs, v100_runs=v100_runs)
    if job_logs is not None:
        values["H100_JOB_LOG_DIR"] = str(job_logs)
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
        },
    )


@pytest.mark.parametrize("mode", ["smoke", "acceptance", "cutover-check", "campaign"])
@pytest.mark.parametrize("relationship", ["equal", "h100-child", "h100-parent"])
def test_submit_rejects_v100_runs_overlap_before_any_write(
    tmp_path: Path, mode: str, relationship: str
):
    tree = tmp_path / "campaigns"
    if relationship == "equal":
        h100_runs = v100_runs = tree / "v100"
    elif relationship == "h100-child":
        v100_runs = tree / "v100"
        h100_runs = v100_runs / "h100"
    else:
        h100_runs = tree
        v100_runs = h100_runs / "v100"
    v100_runs.mkdir(parents=True)
    result = _run_submit(
        tmp_path,
        mode=mode,
        h100_runs=h100_runs,
        v100_runs=v100_runs,
    )
    assert result.returncode == 2
    assert "H100_RUNS_ROOT overlaps live V100 runs root" in result.stderr
    assert not (h100_runs / ".h100").exists()
    assert not (tmp_path / "h100-job-logs").exists()


@pytest.mark.parametrize("mode", ["smoke", "acceptance", "cutover-check", "campaign"])
def test_submit_rejects_job_log_overlap_with_live_v100_before_write(
    tmp_path: Path, mode: str
):
    v100_runs = tmp_path / "v100-runs"
    v100_runs.mkdir()
    h100_runs = tmp_path / "h100-runs"
    result = _run_submit(
        tmp_path,
        mode=mode,
        h100_runs=h100_runs,
        v100_runs=v100_runs,
        job_logs=v100_runs / "slurm-logs",
    )
    assert result.returncode == 2
    assert "H100_JOB_LOG_DIR overlaps live V100 runs root" in result.stderr
    assert not h100_runs.exists()
    assert not (v100_runs / "slurm-logs").exists()


def test_submit_v100_guard_precedes_every_persistent_write():
    source = SUBMIT.read_text()
    guard = source.index("assert_disjoint_from_v100 H100_RUNS_ROOT")
    for marker in (
        "mkdir -p \"$H100_RUNS_ROOT/.h100/slurm\"",
        "mkdir -p \"$H100_JOB_LOG_DIR\"",
        "-m scripts.h100.cutover",
        "-m scripts.h100.operator_cutover",
        "sbatch \\",
    ):
        assert guard < source.index(marker)
