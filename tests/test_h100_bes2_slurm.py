from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.handoff.bes2_amendment import BRANCH, REQUIRED_ANCESTOR


REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "slurm/h100/bes2_diagnostic.sbatch"
SUBMIT = REPO / "slurm/h100/submit_bes2_diagnostic.sh"


def test_bes2_slurm_scripts_are_tracked_executable() -> None:
    for path in (SBATCH, SUBMIT):
        relative = path.relative_to(REPO)
        record = subprocess.run(
            ["git", "ls-files", "--stage", "--", str(relative)],
            cwd=REPO,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        assert record.split(maxsplit=1)[0] == "100755"


def test_bes2_slurm_scripts_are_valid_shell() -> None:
    for path in (SBATCH, SUBMIT):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_bes2_batch_is_train_dev_only_sealed_and_requeue_safe() -> None:
    source = SBATCH.read_text(encoding="utf-8")

    assert "#SBATCH --gpus-per-node=2" in source
    assert "#SBATCH --signal=B:USR1@900" in source
    assert "#SBATCH --requeue" in source
    assert "500000000000" in source
    assert "-B -m scripts.h100.build_venv verify" in source
    assert "-m scripts.h100.data_staging" in source
    assert '--runs-root "$staging_guard"' in source
    assert "BES2_DATA_VIEW_READY.json" in source
    assert '--expected-git-sha "$H100_EXPECTED_GIT_SHA"' in source
    assert "-m scripts.h100.strict_fp32_probe" in source
    assert "--diagnostic" in source
    assert "--expected-gpus" in source
    assert "-m src.analysis.bes2_root_cause audit" in source
    assert "-m scripts.h100.bes2_diagnostic controller" in source
    assert '--fraction "$fraction"' in source
    assert 'if [[ "$mode" == "probe" ]]; then\n  run_controller_wave 0.5' in source
    full_waves = "else\n  run_controller_wave 0.1\n  run_controller_wave 0.5\nfi"
    assert full_waves in source
    assert "both finite f50 five-epoch probes" in source
    assert "preemption_requested=1" in source
    assert (
        "if [[ \"$preemption_requested\" -eq 1 ]]; then\n"
        "    request_outer_requeue"
    ) in source
    assert (
        "controller_pid=\"$!\"\n"
        "  if [[ \"$preemption_requested\" -eq 1 ]]; then\n"
        "    kill -TERM \"$controller_pid\""
    ) in source
    assert '"${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue' in source
    assert "remove_allocation_scratch" in source
    assert "--acceptance" not in source
    assert "score-test" not in source
    assert "TRAINING_COHORT" not in source
    assert "test_metrics.json" not in source
    assert "final_verified_metrics.json" in source
    assert "gpu get" not in source


def test_bes2_submit_is_clean_snapshot_and_fails_closed_on_final_access() -> None:
    source = SUBMIT.read_text(encoding="utf-8")

    assert "audit|probe|full" in source
    assert BRANCH in source
    assert REQUIRED_ANCESTOR in source
    assert "1a82d508fbeb9fdf6868a9637611e9018952fb43" in source
    assert "value > 0 && value <= 190" in source
    assert "verify-bes2-amendment" in source
    assert "assert_final_unconsumed" in source
    assert "initial BES2 diagnostic namespace must start empty" in source
    assert "initial BES2 job-log namespace must start empty" in source
    assert source.index("assert_final_unconsumed") < source.index(
        "mkdir -m 700 -p"
    )
    assert "FINAL_DATA_VIEW.json" in source
    assert "final_verified_metrics.json" in source
    assert "compute-site-" + "$" + "{snapshot_sha256}.env" in source
    assert "--export=NONE" in source
    assert "env -u BOX_JWT_CONFIG -u BOX_FOLDER_ID sbatch" in source
    snapshot_start = source.index("snapshot_names=(")
    snapshot = source[
        snapshot_start : source.index("if [[ -n", snapshot_start)
    ]
    assert "BOX_JWT_CONFIG" not in snapshot
    assert "gpus=1" in source
    assert "gpus=2" in source
    assert 'walltime="08:00:00"' in source
    assert 'walltime="12:00:00"' in source
    assert 'walltime="2-18:00:00"' in source
    assert "gpu get" not in source


def test_bes2_bundle_branch_matches_submit_and_compute_clone() -> None:
    submit = SUBMIT.read_text(encoding="utf-8")
    batch = SBATCH.read_text(encoding="utf-8")

    assert f'if [[ "$branch" != "{BRANCH}" ||' in submit
    assert (
        f"git clone --single-branch --branch {BRANCH} "
        '"$BES2_BUNDLE" "$repo"'
    ) in batch
