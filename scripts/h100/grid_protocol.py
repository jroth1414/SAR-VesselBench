"""Source and exact failure guards for a CPU-only grid finalizer."""
from __future__ import annotations

import os
import socket
from pathlib import Path

from scripts.h100.contracts import sha256_file
from scripts.h100.resume_recovery import read_json, regular
from scripts.h100.source_validation import git_output
from scripts.handoff.grid_contract import INVENTORY, POLICY, validate_grid_control

DIRECTORY = "grid-finalization"
SOURCE = "GRID_SOURCE_VALIDATED.json"
FAILED = "failed-campaign.json"
ATTEMPT = "GRID_ATTEMPT.json"
COMPLETION = "GRID_COMPLETE.json"
FAILURE = "GRID_FAILED.json"


def require(ok, message):
    if not ok:
        raise RuntimeError("grid finalization " + message)


def cpu_allocation():
    require(socket.gethostname().split(".")[0] == POLICY["node"]
        and os.environ.get("SLURM_JOB_ID", "").isdecimal()
        and os.environ.get("SLURM_CPUS_PER_TASK") == str(POLICY["cpus"])
        and os.environ.get("SLURM_RESTART_COUNT", "0") == "0"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and all(not os.environ.get(name) for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"))
        and all(os.environ.get(name, "") in ("", "0") for name in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE")),
        "requires the approved CPU-only dgx18 allocation without requeue")
    return {key: POLICY[key] for key in ("node", "gpus", "workers", "cpus", "memory", "time_limit", "requeue")}


def identity(receipt, digest):
    import re
    control = validate_grid_control(receipt.get("control"))
    package = receipt.get("package", {})
    require(set(receipt) == {"schema", "status", "git_sha", "control", "package"}
        and type(receipt["schema"]) is int and receipt["schema"] == 1
        and receipt["status"] == "grid-source-validated"
        and receipt["git_sha"] == control["source"]["git_commit"], "source receipt differs")
    from scripts.handoff.package import _canonical_json
    import hashlib
    require(set(package) == {"package_id", "manifest_sha256", "ready_sha256", "sha256sums_sha256", "bundle_sha256"}
        and package["package_id"] == f"xview3-h100-runtime-{receipt['git_sha']}-{hashlib.sha256(_canonical_json(control)).hexdigest()}"
        and package["bundle_sha256"] == control["source"]["git_bundle_sha256"]
        and all(re.fullmatch(r"[0-9a-f]{64}", str(v)) for k, v in package.items() if k != "package_id")
        and re.fullmatch(r"[0-9a-f]{64}", str(digest)), "package binding differs")
    return {"campaign_git_sha": INVENTORY["recipe_git_sha"], "git_sha": receipt["git_sha"],
        "grid_package_id": package["package_id"], "grid_manifest_sha256": package["manifest_sha256"],
        "source_receipt_sha256": digest}


def validate_failed(campaign, digest, *, bindings=None):
    inv = INVENTORY if bindings is None else bindings
    ids = set(inv["cells"])
    require(digest == inv["failed_manifest_sha256"] and campaign.get("schema") == 2
        and campaign.get("status") == "failed" and campaign.get("phase") == "score-test"
        and str(campaign.get("slurm_job_id")) == inv["job_id"]
        and campaign.get("running") == {}
        and campaign.get("fail_stop") == {"engaged": True, "failed": [], "allowed_to_finish": []}
        and campaign.get("git_sha") == inv["recipe_git_sha"]
        and campaign.get("h100_ready_sha256") == inv["h100_ready_sha256"]
        and campaign.get("training_cohort", {}).get("sha256") == inv["training_cohort_sha256"]
        and campaign.get("grid_execution") is None, "not the exact approved grid failure")
    for key in ("training_complete", "test_complete", "complete"):
        require(isinstance(campaign.get(key), list) and len(campaign[key]) == 32
            and set(campaign[key]) == ids, "completed matrix differs")
    for key in ("execution", "terminal_execution", "test_execution"):
        require(campaign.get(key) == inv[key], "historical execution differs")
    events = [e for e in campaign.get("events", []) if e.get("event") in ("grid_validation_failed", "grid_validated")]
    require(events == [inv["failed_grid_event"]], "failure event history differs")


def validate_checkout(repo, runs_root, receipt, execution):
    require(git_output(repo, "rev-parse", "HEAD") == execution["git_sha"]
        and git_output(repo, "branch", "--show-current") == receipt["control"]["source"]["branch"], "actual checkout differs")
    from scripts.handoff.grid_amendment import verify_grid_source
    verify_grid_source(repo, runtime_runs_root=runs_root)


def load_recipe(repo: Path, runs_root: Path):
    """Authorize only a claimed grid subprocess; old worker guards stay intact."""
    path = runs_root.resolve() / ".h100" / DIRECTORY / SOURCE
    name, digest = os.environ.get("H100_GRID_RECEIPT"), os.environ.get("H100_GRID_RECEIPT_SHA256")
    if not name and not digest:
        require(not os.path.lexists(path), "separate execution receipt is required")
        return None
    require(name and digest and Path(name) == path
        and sha256_file(regular(path, immutable=True)) == digest, "receipt must be canonical and hash-bound")
    receipt = read_json(path, immutable=True)
    execution = identity(receipt, digest)
    validate_checkout(repo, runs_root, receipt, execution)
    directory = path.parent
    require(not os.path.lexists(directory / FAILURE) and not os.path.lexists(directory / COMPLETION), "later failure or completion cannot be replayed")
    archived = directory / FAILED
    validate_failed(read_json(archived, immutable=True), sha256_file(archived))
    current = runs_root / ".h100/campaign_manifest.json"
    validate_failed(read_json(current), sha256_file(current))
    attempt = read_json(directory / ATTEMPT, immutable=True)
    from scripts.handoff.package import _canonical_json
    require(attempt.get("grid_execution") == execution
        and set(attempt) == {"schema", "status", "operation", "slurm_job_id", "grid_execution", "resources", "started_utc"}
        and type(attempt["schema"]) is int and attempt["schema"] == 1
        and attempt["status"] == "claimed"
        and _canonical_json(attempt["resources"]) == _canonical_json(cpu_allocation())
        and attempt.get("operation") == "grid-only"
        and str(attempt.get("slurm_job_id")) == os.environ.get("SLURM_JOB_ID")
        and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "unclaimed or GPU-visible collector")
    return execution["campaign_git_sha"]
