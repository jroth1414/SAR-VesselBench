"""Guarded TEST-only execution lineage and exact job-659990 fail-stop recovery."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
from pathlib import Path

from scripts.handoff.package import _canonical_json, _require_no_symlink_components
from scripts.handoff.test_recovery_contract import (
    CAMPAIGN_SOURCE, CELL, CELLS, POLICY, TERMINAL_SOURCE, validate_test_recovery_control,
)
from scripts.h100.contracts import sha256_file
from scripts.h100.source_validation import git_output
from scripts.h100.resume_recovery import regular, read_json, inherited_source
from scripts.h100.terminal_recovery import _write_bytes_once, write_once, inherited_execution

DIRECTORY = "test-recovery"
SOURCE = "TEST_SOURCE_VALIDATED.json"
ACTIVATION = "TEST_ACTIVATION.json"
FAILED = "failed-campaign.json"
PROVENANCE = "failed-cnnrand-provenance.json"
LAUNCH = "prior-TEST_LAUNCH.json"
NAMES = (SOURCE, ACTIVATION, FAILED, PROVENANCE, LAUNCH)
EVENT = "test_recovery_activated"


def require(condition, message):
    if not condition:
        raise RuntimeError("TEST recovery " + message)


def identity(payload, receipt_sha):
    control = validate_test_recovery_control(payload.get("control"))
    package = payload.get("package", {})
    require(set(payload) == {"schema", "status", "campaign_git_sha", "git_sha", "h100_ready_sha256",
        "terminal_execution", "previous_execution", "control", "package"}
        and payload["schema"] == 1 and payload["status"] == "test-source-validated"
        and payload["campaign_git_sha"] == CAMPAIGN_SOURCE
        and payload["git_sha"] == control["source"]["git_commit"]
        and payload["h100_ready_sha256"] == POLICY["h100_ready_sha256"]
        and payload["terminal_execution"].get("git_sha") == TERMINAL_SOURCE
        and payload["terminal_execution"].get("source_receipt_sha256") == POLICY["terminal_source_receipt_sha256"]
        and payload["previous_execution"].get("git_sha") == "3808f4ab5f8aea8e6b552f7f2b100fd47adc919f"
        and set(package) == {"package_id", "manifest_sha256", "ready_sha256", "sha256sums_sha256", "bundle_sha256"}
        and package["package_id"] == f"xview3-h100-runtime-{payload['git_sha']}-{hashlib.sha256(_canonical_json(control)).hexdigest()}"
        and package["bundle_sha256"] == control["source"]["git_bundle_sha256"], "invalid source/package receipt")
    import re
    require(all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for key, value in package.items() if key != "package_id")
        and re.fullmatch(r"[0-9a-f]{64}", receipt_sha), "invalid source hashes")
    return {"campaign_git_sha": CAMPAIGN_SOURCE, "git_sha": payload["git_sha"],
        "test_package_id": package["package_id"], "test_manifest_sha256": package["manifest_sha256"],
        "source_receipt_sha256": receipt_sha}


def inherited_terminal(runs_root):
    from scripts.h100.terminal_recovery import identity as terminal_identity
    path = runs_root / ".h100/terminal-recovery/TERMINAL_SOURCE_VALIDATED.json"
    digest = sha256_file(regular(path, immutable=True))
    require(digest == POLICY["terminal_source_receipt_sha256"], "terminal source bytes changed")
    payload = read_json(path)
    terminal = terminal_identity(payload, digest)
    require(payload["previous_execution"] == inherited_execution(runs_root), "inherited execution changed")
    return terminal


def validate_failed(prior, provenance, *, manifest_sha, provenance_sha, terminal, previous):
    require(manifest_sha == POLICY["failed_manifest_sha256"] and provenance_sha == POLICY["failed_provenance_sha256"],
        "failure bytes differ from approval")
    require(prior.get("git_sha") == CAMPAIGN_SOURCE and prior.get("execution") == previous
        and prior.get("terminal_execution") == terminal and prior.get("test_execution") is None
        and str(prior.get("slurm_job_id")) == POLICY["failed_job"]
        and prior.get("status") == "failed" and prior.get("phase") == "score-test"
        and prior.get("running") == {} and prior.get("test_complete") == []
        and set(prior.get("training_complete", [])) == set(CELLS)
        and prior.get("fail_stop") == {"engaged": True, "failed": [CELL], "allowed_to_finish": []}
        and prior.get("h100_ready_sha256") == POLICY["h100_ready_sha256"]
        and prior.get("training_cohort", {}).get("sha256") == POLICY["cohort_sha256"], "not the exact failed TEST campaign")
    events = [e for e in prior.get("events", []) if e.get("event") == "cell_failed"
        and e.get("exp_id") == CELL and e.get("phase") == "score-test"]
    require(len(events) == 1 and events[0].get("utc") == POLICY["failure_utc"]
        and events[0].get("exit_code") == 1, "recorded failure event differs")
    attempts = provenance.get("attempts", [])
    require(len(attempts) == POLICY["failed_attempt"] and provenance.get("execution") == previous
        and provenance.get("terminal_execution") == terminal and provenance.get("test_execution") is None,
        "failed cell source or attempt changed")
    attempt = attempts[-1]
    require(attempt.get("attempt") == POLICY["failed_attempt"] and attempt.get("phase") == "score-test"
        and str(attempt.get("slurm_job_id")) == POLICY["failed_job"]
        and attempt.get("execution") == previous and attempt.get("terminal_execution") == terminal
        and attempt.get("training_cohort_sha256") == POLICY["cohort_sha256"]
        and not any(k in attempt for k in ("finished_utc", "active_seconds", "exit", "exit_code")),
        "failed TEST attempt is not the approved open attempt")


def retained_inventory(runs_root):
    retained = {}
    for cell in CELLS:
        run = runs_root / cell
        marker = read_json(run / "final_metrics.json", immutable=True)
        checkpoint = marker["best_checkpoint"]
        names = {"final_metrics.json", "config.yaml", "checkpoints/best.ckpt", "checkpoints/last.ckpt",
            checkpoint["relative_path"]}
        if "terminal_recovery" in marker:
            names.add("terminal_recovery.json")
        files = {name: sha256_file(regular(run / name)) for name in sorted(names)}
        require(files[checkpoint["relative_path"]] == checkpoint["sha256"], "selected checkpoint bytes changed")
        if cell == CELL:
            require(checkpoint == {"relative_path": "checkpoints/best.ckpt", "sha256": POLICY["selected_checkpoint_sha256"],
                "epoch": POLICY["selected_checkpoint_epoch"]}, "cnnrand selected checkpoint differs")
        provenance = read_json(run / "runtime_provenance.json")
        retained[cell] = {"files": files, "prior_provenance": provenance}
    return retained


def validate_retained(runs_root, receipt):
    require(set(receipt.get("retained", {})) == set(CELLS), "retained cohort inventory differs")
    for cell, item in receipt["retained"].items():
        for name, digest in item["files"].items():
            require(sha256_file(regular(runs_root / cell / name)) == digest, "retained training bytes changed: " + cell + "/" + name)


def prepare(*, repo, package_root, expected_manifest_sha256, expected_git_sha, runs_root, publish):
    from scripts.handoff.test_recovery_amendment import verify_test_recovery_amendment, verify_test_recovery_source
    _require_no_symlink_components(repo, leaf="directory")
    _require_no_symlink_components(runs_root, leaf="directory")
    require(sha256_file(regular(package_root / "manifest.json")) == expected_manifest_sha256, "package manifest changed")
    manifest = verify_test_recovery_amendment(package_root)
    verify_test_recovery_source(repo)
    require(manifest["source"]["git_commit"] == expected_git_sha == git_output(repo, "rev-parse", "HEAD"), "actual source changed")
    inherited_source(runs_root)  # Includes final-consumption and original-namespace controls.
    terminal = inherited_terminal(runs_root)
    previous = inherited_execution(runs_root)
    require(sha256_file(regular(runs_root / ".h100/H100_READY.json", immutable=True)) == POLICY["h100_ready_sha256"], "readiness changed")
    require(sha256_file(regular(runs_root / ".h100/TRAINING_COHORT.json", immutable=True)) == POLICY["cohort_sha256"], "frozen cohort changed")
    if publish:
        require(socket.gethostname().split(".")[0] == POLICY["node"], "allocation must execute on dgx18")
    directory = runs_root / ".h100" / DIRECTORY
    prior = read_json(runs_root / ".h100/campaign_manifest.json")
    payload = {"schema": 1, "status": "test-source-validated", "campaign_git_sha": CAMPAIGN_SOURCE,
        "git_sha": expected_git_sha, "h100_ready_sha256": POLICY["h100_ready_sha256"],
        "terminal_execution": terminal, "previous_execution": previous, "control": manifest["amendment_identity"],
        "package": {"package_id": manifest["package_id"], "manifest_sha256": expected_manifest_sha256,
            "ready_sha256": sha256_file(regular(package_root / "READY.json")),
            "sha256sums_sha256": sha256_file(regular(package_root / "SHA256SUMS")),
            "bundle_sha256": manifest["source"]["git_bundle_sha256"]}}
    expected_identity = identity(payload, hashlib.sha256(_canonical_json(payload)).hexdigest())
    events = [e for e in prior.get("events", []) if e.get("event") == EVENT]
    if events:
        require(len(events) == 1 and prior.get("test_execution") == expected_identity, "active recovery cannot switch source")
        receipt = read_json(directory / ACTIVATION, immutable=True)
        require(receipt["test_execution"] == expected_identity and events[0]["activation_sha256"] == sha256_file(directory / ACTIVATION), "activation changed")
        validate_retained(runs_root, receipt)
    else:
        failed_raw = regular(runs_root / ".h100/campaign_manifest.json").read_bytes()
        provenance_raw = regular(runs_root / CELL / "runtime_provenance.json").read_bytes()
        validate_failed(prior, json.loads(provenance_raw), manifest_sha=hashlib.sha256(failed_raw).hexdigest(),
            provenance_sha=hashlib.sha256(provenance_raw).hexdigest(), terminal=terminal, previous=previous)
        require(not list(runs_root.glob("*/test_metrics.json")), "unexpected TEST result before activation")
        cohort = read_json(runs_root / ".h100/TRAINING_COHORT.json", immutable=True)
        require(cohort["cell_count"] == 32 and {c["exp_id"] for c in cohort["cells"]} == set(CELLS), "not the exact all32 cohort")
        launch_path = Path(os.environ.get("H100_PRIOR_TEST_LAUNCH", "/cluster-site-redacted/jroth/xview3-terminal-1ec7a68-handoff/test-launch-a8996efd/TEST_LAUNCH.json"))
        launch_raw = regular(launch_path, immutable=True).read_bytes()
        require(hashlib.sha256(launch_raw).hexdigest() == POLICY["prior_test_launch_sha256"], "prior TEST launch changed")
        retained = retained_inventory(runs_root)
        if publish:
            directory.mkdir(exist_ok=True)
            _require_no_symlink_components(directory, leaf="directory")
            for name, raw in ((FAILED, failed_raw), (PROVENANCE, provenance_raw), (LAUNCH, launch_raw)):
                _write_bytes_once(directory / name, raw)
            write_once(directory / ACTIVATION, {"schema": 1, "status": "approved-test-failure-verified",
                "test_execution": expected_identity, "terminal_execution": terminal, "previous_execution": previous,
                "failed_manifest_sha256": POLICY["failed_manifest_sha256"],
                "failed_provenance_sha256": POLICY["failed_provenance_sha256"], "cohort_sha256": POLICY["cohort_sha256"],
                "prior_test_launch_sha256": POLICY["prior_test_launch_sha256"], "retained": retained})
    if publish:
        require(socket.gethostname().split(".")[0] == POLICY["node"], "allocation must execute on dgx18")
        write_once(directory / SOURCE, payload)
    return payload


def load_test_execution(repo, runs_root, campaign_git_sha=None, *, require_activation=False):
    path = runs_root.resolve() / ".h100" / DIRECTORY / SOURCE
    name, digest = os.environ.get("H100_TEST_RECEIPT"), os.environ.get("H100_TEST_RECEIPT_SHA256")
    if not name and not digest:
        require(not os.path.lexists(path), "execution receipt is required")
        return None
    require(name and digest and Path(name) == path and sha256_file(regular(path, immutable=True)) == digest,
        "source receipt must be canonical and hash-bound")
    inherited_source(runs_root)
    payload = read_json(path)
    execution = identity(payload, digest)
    require(campaign_git_sha in (None, CAMPAIGN_SOURCE) and git_output(repo, "rev-parse", "HEAD") == execution["git_sha"], "actual source differs")
    from scripts.handoff.test_recovery_amendment import verify_test_recovery_source
    verify_test_recovery_source(repo, runtime_runs_root=runs_root)
    require(payload["terminal_execution"] == inherited_terminal(runs_root)
        and payload["previous_execution"] == inherited_execution(runs_root), "inherited lineage differs")
    require(sha256_file(regular(runs_root / ".h100/TRAINING_COHORT.json", immutable=True)) == POLICY["cohort_sha256"], "frozen cohort differs")
    receipt = read_json(path.parent / ACTIVATION, immutable=True)
    require(receipt["test_execution"] == execution, "activation source differs")
    for name, key in ((FAILED, "failed_manifest_sha256"), (PROVENANCE, "failed_provenance_sha256"), (LAUNCH, "prior_test_launch_sha256")):
        require(sha256_file(regular(path.parent / name, immutable=True)) == POLICY[key] == receipt[key], "failure archive differs")
    if require_activation:
        campaign = read_json(runs_root / ".h100/campaign_manifest.json")
        events = [e for e in campaign.get("events", []) if e.get("event") == EVENT]
        require(campaign.get("phase") == "score-test" and campaign.get("test_execution") == execution
            and len(events) == 1 and events[0]["activation_sha256"] == sha256_file(path.parent / ACTIVATION)
            and campaign.get("status") != "failed"
            and campaign.get("fail_stop") == {"engaged": False, "failed": [], "allowed_to_finish": []}, "worker requires activated TEST recovery")
    return execution


def load_terminal_context(repo, runs_root, campaign_git_sha=None, *, require_activation=False):
    execution = load_test_execution(repo, runs_root, campaign_git_sha, require_activation=require_activation)
    if execution is not None:
        return inherited_terminal(runs_root)
    from scripts.h100.terminal_recovery import load_terminal_execution
    return load_terminal_execution(repo, runs_root, campaign_git_sha)


def activate(controller, prior):
    execution = getattr(controller, "test_execution", None)
    if execution is None:
        return
    require(controller.phase == "score-test" and controller.cohort_sha256 == POLICY["cohort_sha256"]
        and not controller.running, "only frozen-cohort TEST is authorized")
    directory = controller.runs_root / ".h100" / DIRECTORY
    receipt = read_json(directory / ACTIVATION, immutable=True)
    require(receipt["test_execution"] == execution, "activation identity differs")
    validate_retained(controller.runs_root, receipt)
    events = [e for e in prior.get("events", []) if e.get("event") == EVENT]
    if events:
        require(len(events) == 1 and prior.get("test_execution") == execution
            and events[0]["activation_sha256"] == sha256_file(directory / ACTIVATION), "activated source changed")
        return  # Never clear a subsequent failure on requeue.
    require(sha256_file(regular(controller.manifest_path)) == POLICY["failed_manifest_sha256"], "failure changed before transition")
    validate_failed(prior, read_json(controller.runs_root / CELL / "runtime_provenance.json"),
        manifest_sha=sha256_file(controller.manifest_path), provenance_sha=sha256_file(controller.runs_root / CELL / "runtime_provenance.json"),
        terminal=controller.terminal_execution, previous=controller.execution)
    controller.events.append({"event": EVENT, "test_execution": execution,
        "activation_sha256": sha256_file(directory / ACTIVATION), "retry_ids": [CELL]})
    controller.failure_seen = False
    controller.failed_ids = set()
    controller.failure_allowed_ids = set()
    controller.write_manifest()


def close_approved_failure(provenance, *, runs_root, exp_id, execution):
    if exp_id != CELL or provenance.get("test_execution") is not None:
        return False
    directory = runs_root / ".h100" / DIRECTORY
    require(read_json(directory / ACTIVATION, immutable=True)["test_execution"] == execution
        and sha256_file(regular(directory / PROVENANCE, immutable=True)) == POLICY["failed_provenance_sha256"],
        "closure source or failed provenance differs")
    original = read_json(directory / PROVENANCE, immutable=True)
    require(provenance == original, "failed attempt changed before closure")
    from scripts.h100.campaign import reconcile_stale_open_attempt
    require(reconcile_stale_open_attempt(provenance, phase="score-test", recovered_utc=POLICY["failure_utc"]), "failed attempt was not open")
    provenance["attempts"][-1].pop("exit", None)
    provenance["attempts"][-1].update(exit_code=1, active_seconds_basis="started-to-recorded-cell-failure",
        recovered_from="owner-approved-test-failure")
    return True


def validate_provenance(provenance, cell, receipt, source_receipt):
    from scripts.handoff.test_recovery_results import validate_cell
    old = receipt["retained"][cell]["prior_provenance"]
    if provenance == old:
        return  # An as-yet pending cell; no new execution attempt exists.
    validate_cell(provenance, cell, {"source_receipt": source_receipt,
        "activation": receipt}, require_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "prepare"))
    for name in ("repo", "package-root", "runs-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    args = parser.parse_args()
    payload = prepare(repo=args.repo, package_root=args.package_root, expected_manifest_sha256=args.expected_manifest_sha256,
        expected_git_sha=args.expected_git_sha, runs_root=args.runs_root, publish=args.operation == "prepare")
    print(json.dumps({"status": payload["status"], "git_sha": payload["git_sha"]}, sort_keys=True))


if __name__ == "__main__":
    main()
