from __future__ import annotations

import stat
import subprocess
from pathlib import Path

from scripts.handoff import runtime_amendment


REPO = Path(__file__).resolve().parents[1]
SHIM = REPO / "slurm/h100/shims/scontrol"


def _child_environment(
    request_dir: Path,
    fake_bin: Path,
    real_call_marker: Path,
) -> dict[str, str]:
    return {
        "PATH": f"{SHIM.parent}:{fake_bin}:/usr/bin:/bin",
        "H100_REQUEUE_REQUEST_DIR": str(request_dir),
        "CUDA_VISIBLE_DEVICES": "3",
        "SLURM_JOB_ID": "4242",
        "REAL_SCONTROL_MARKER": str(real_call_marker),
    }


def _fake_real_scontrol(fake_bin: Path) -> None:
    path = fake_bin / "scontrol"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "touch \"$REAL_SCONTROL_MARKER\"\n"
        "exit 99\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_lightning_scontrol_call_is_deferred_to_checkpoint_barrier(
    tmp_path: Path,
) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    real_call_marker = tmp_path / "real-scontrol-called"
    _fake_real_scontrol(fake_bin)

    shim_stat = SHIM.stat()
    assert stat.S_ISREG(shim_stat.st_mode)
    assert not SHIM.is_symlink()
    # Git records only the executable bit. The checkout umask may therefore
    # materialize this tracked executable as 0755, 0750, 0770, and so on.
    assert stat.S_IMODE(shim_stat.st_mode) & stat.S_IXUSR
    completed = subprocess.run(
        ["scontrol", "requeue", "4242"],
        env=_child_environment(request_dir, fake_bin, real_call_marker),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert not real_call_marker.exists()
    assert (request_dir / "gpu-3.request").read_text(encoding="utf-8") == "4242\n"


def test_child_scontrol_fails_closed_for_wrong_job(tmp_path: Path) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    real_call_marker = tmp_path / "real-scontrol-called"
    _fake_real_scontrol(fake_bin)

    completed = subprocess.run(
        ["scontrol", "requeue", "9999"],
        env=_child_environment(request_dir, fake_bin, real_call_marker),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not real_call_marker.exists()
    assert list(request_dir.iterdir()) == []


def test_native_runtime_requires_shim_and_outer_batch_owns_real_scontrol() -> None:
    campaign = (REPO / "scripts/h100/campaign.py").read_text(encoding="utf-8")
    batch = (REPO / "slurm/h100/campaign.sbatch").read_text(encoding="utf-8")
    assert "slurm/h100/shims" in campaign
    assert "scontrol" not in campaign
    assert '"${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue' in batch
    assert "slurm/h100/shims/scontrol" in runtime_amendment._REQUIRED_RUNTIME_FILES
