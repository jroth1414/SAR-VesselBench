"""Owner-approved, once-only checkpoint recovery with separate execution identity.

The inherited campaign git_sha stays bound to its immutable acceptance and
result schema. execution.git_sha always identifies the actual patched code.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
from pathlib import Path, PosixPath

from scripts.h100.contracts import atomic_write_json, sha256_file, verify_expected_hashes
from scripts.h100.replacement import assert_final_unconsumed
from scripts.h100.source_validation import git_output, validate_source_receipt
from scripts.handoff.package import _canonical_json, _require_no_symlink_components
from scripts.handoff.recovery_contract import (
    CAMPAIGN_SOURCE, POLICY, PREVIOUS_RECOVERY_SOURCE,
    PREVIOUS_RECOVERY_RECEIPT_SHA256, validate_recovery_control,
)

DIRECTORY = "recovery"
PREVIOUS_SOURCE_RECEIPT = "RECOVERY_SOURCE_VALIDATED.json"
SOURCE_RECEIPT = "RECOVERY_SOURCE_VALIDATED_V2.json"
ACTIVATION = "RECOVERY_ACTIVATION.json"
FAILED_MANIFEST = "failed-campaign.json"
EVENT = "resume_recovery_activated"


def regular(path: Path, *, immutable=False) -> Path:
    path = _require_no_symlink_components(path, leaf="file")
    if immutable and path.stat().st_mode & 0o222:
        raise RuntimeError(f"recovery evidence is writable: {path}")
    return path


def read_json(path: Path, *, immutable=False) -> dict:
    value = json.loads(regular(path, immutable=immutable).read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"recovery evidence must be an object: {path}")
    return value


def write_once(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        if read_json(path, immutable=True) != payload:
            raise RuntimeError(f"recovery immutable evidence changed: {path}")
        return
    atomic_write_json(path, payload)
    path.chmod(0o444)


def previous_source_binding() -> dict:
    return {"filename": PREVIOUS_SOURCE_RECEIPT,
            "sha256": PREVIOUS_RECOVERY_RECEIPT_SHA256,
            "git_sha": PREVIOUS_RECOVERY_SOURCE}


def validate_previous_source(runs_root: Path) -> None:
    """Keep the exact never-activated job-649961 receipt in place."""
    path = regular(runs_root / ".h100" / DIRECTORY / PREVIOUS_SOURCE_RECEIPT, immutable=True)
    if sha256_file(path) != PREVIOUS_RECOVERY_RECEIPT_SHA256:
        raise RuntimeError("previous recovery source receipt changed")
    payload = read_json(path)
    if payload.get("schema") != 1 or identity(payload, PREVIOUS_RECOVERY_RECEIPT_SHA256)["git_sha"] != PREVIOUS_RECOVERY_SOURCE:
        raise RuntimeError("previous recovery source identity changed")


def validate_runtime_checkout(repo: Path, runs_root: Path) -> None:
    """The launcher creates one non-ignored link after clean source validation."""
    link = repo / "runs"
    if os.path.lexists(link):
        canonical_runs = _require_no_symlink_components(runs_root, leaf="directory")
        if not link.is_symlink() or os.readlink(link) != str(canonical_runs):
            raise RuntimeError("recovery runs link differs from the canonical runs root")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty and not (dirty == "?? runs" and link.is_symlink()):
        raise RuntimeError(f"recovery execution checkout has changed:\n{dirty}")


def inherited_source(runs_root: Path) -> tuple[dict, str]:
    meta = runs_root / ".h100"
    ready_path = regular(meta / "H100_READY.json", immutable=True)
    if sha256_file(ready_path) != POLICY["h100_ready_sha256"]:
        raise RuntimeError("recovery H100 readiness differs from owner approval")
    ready = read_json(ready_path)
    ns = read_json(meta / "REPLACEMENT_NAMESPACE.json", immutable=True)
    if ns.get("git_sha") != CAMPAIGN_SOURCE or ns.get("runs_root") != str(runs_root):
        raise RuntimeError("recovery namespace differs from the approved campaign")
    if ns.get("runtime_manifest_sha256") != ready["runtime_amendment"]["manifest_sha256"]:
        raise RuntimeError("recovery inherited runtime binding mismatch")
    assert_final_unconsumed(runs_root)
    assert_final_unconsumed(Path(ns["original_runs_root"]))
    digest = ready["source_validation"]["sha256"]
    validate_source_receipt(
        meta / "SOURCE_VALIDATED.json", expected_sha256=digest,
        expected_git_sha=CAMPAIGN_SOURCE,
        expected_hashes=ready["source"]["frozen_sha256"],
        expected_base_payload=ready["base_payload"],
        expected_runtime_amendment=ready["runtime_amendment"],
    )
    return ready, digest


def prepare(*, repo: Path, package_root: Path, expected_manifest_sha256: str,
            expected_git_sha: str, runs_root: Path, publish: bool) -> dict:
    from scripts.handoff.recovery_amendment import verify_recovery_amendment, verify_recovery_source
    repo = _require_no_symlink_components(repo, leaf="directory")
    runs_root = _require_no_symlink_components(runs_root, leaf="directory")
    if sha256_file(regular(package_root / "manifest.json")) != expected_manifest_sha256:
        raise RuntimeError("recovery package manifest mismatch")
    manifest = verify_recovery_amendment(package_root)
    if manifest["source"]["git_commit"] != expected_git_sha:
        raise RuntimeError("recovery package execution source mismatch")
    if git_output(repo, "rev-parse", "HEAD") != expected_git_sha:
        raise RuntimeError("recovery checkout is not the actual execution source")
    if git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("recovery requires a clean execution checkout")
    verify_recovery_source(repo)
    ready, old_source_sha = inherited_source(runs_root)
    validate_previous_source(runs_root)
    prior_path = runs_root / ".h100/campaign_manifest.json"
    prior = read_json(prior_path)
    if not any(e.get("event") == EVENT for e in prior.get("events", [])):
        validate_initial_failure(prior, sha256_file(prior_path))
        if prior.get("execution") is not None:
            raise RuntimeError("new recovery source requires an unactivated original failure")
        for path in runs_root.glob("*/runtime_provenance.json"):
            validate_provenance_execution(read_json(path), None)
        cohort = runs_root / ".h100/TRAINING_COHORT.json"
        if cohort.exists() or cohort.is_symlink() or list(runs_root.glob("*/test_metrics.json")):
            raise RuntimeError("initial recovery found premature held-out state")
    elif prior.get("execution", {}).get("git_sha") != expected_git_sha:
        raise RuntimeError("active recovery cannot switch execution sources")
    frozen = verify_expected_hashes(repo, ready["source"]["frozen_sha256"])
    if publish and socket.gethostname().split(".")[0] != POLICY["node"]:
        raise RuntimeError("recovery allocation must execute on dgx18")
    payload = {
        "schema": 2, "status": "recovery-source-validated",
        "scope": "clean-recovery-checkout-before-data-links",
        "campaign_git_sha": CAMPAIGN_SOURCE, "git_sha": expected_git_sha,
        "h100_ready_sha256": POLICY["h100_ready_sha256"],
        "inherited_source_validation_sha256": old_source_sha,
        "previous_source_receipt": previous_source_binding(),
        "frozen_sha256": frozen,
        "control": manifest["amendment_identity"],
        "package": {
            "package_id": manifest["package_id"],
            "manifest_sha256": expected_manifest_sha256,
            "ready_sha256": sha256_file(regular(package_root / "READY.json")),
            "sha256sums_sha256": sha256_file(regular(package_root / "SHA256SUMS")),
            "bundle_sha256": manifest["source"]["git_bundle_sha256"],
        },
    }
    directory = runs_root / ".h100" / DIRECTORY
    current_path = directory / SOURCE_RECEIPT
    if os.path.lexists(current_path) and read_json(current_path, immutable=True) != payload:
        raise RuntimeError("recovery immutable evidence changed")
    activation_path = directory / ACTIVATION
    if not any(e.get("event") == EVENT for e in prior.get("events", [])) and os.path.lexists(activation_path):
        # A crash may publish activation before the campaign event. Resume only
        # the same already-published V2 source; activate() rechecks the full snapshot.
        if not current_path.exists():
            raise RuntimeError("activation exists without the same V2 source")
        activation = read_json(activation_path, immutable=True)
        if (activation.get("schema") != 1 or activation.get("status") != "approved-failures-verified"
                or activation.get("execution") != identity(payload, sha256_file(current_path))
                or activation.get("failed_manifest_sha256") != POLICY["failed_manifest_sha256"]):
            raise RuntimeError("interrupted activation differs from the same V2 source")
    if publish:
        directory.mkdir(exist_ok=True)
        _require_no_symlink_components(directory, leaf="directory")
        write_once(directory / SOURCE_RECEIPT, payload)
    return payload


def identity(payload: dict, receipt_sha: str) -> dict:
    control = validate_recovery_control(payload.get("control"))
    if (payload.get("schema") not in (1, 2) or payload.get("status") != "recovery-source-validated"
            or payload.get("campaign_git_sha") != CAMPAIGN_SOURCE
            or payload.get("git_sha") != control["source"]["git_commit"]
            or payload.get("h100_ready_sha256") != POLICY["h100_ready_sha256"]):
        raise RuntimeError("invalid recovery source receipt")
    if payload["schema"] == 2 and (payload.get("previous_source_receipt") != previous_source_binding()
            or payload["git_sha"] == PREVIOUS_RECOVERY_SOURCE):
        raise RuntimeError("invalid previous recovery source binding")
    package = payload.get("package", {})
    control_sha = hashlib.sha256(_canonical_json(control)).hexdigest()
    if (package.get("package_id") != f"xview3-h100-runtime-{payload['git_sha']}-{control_sha}"
            or package.get("bundle_sha256") != control["source"]["git_bundle_sha256"]):
        raise RuntimeError("recovery source/package identity mismatch")
    return {
        "campaign_git_sha": CAMPAIGN_SOURCE,
        "git_sha": payload["git_sha"],
        "recovery_package_id": payload["package"]["package_id"],
        "recovery_manifest_sha256": payload["package"]["manifest_sha256"],
        "source_receipt_sha256": receipt_sha,
    }


def load_execution(repo: Path, runs_root: Path, campaign_git_sha: str | None = None, *, require_test_activation=True) -> dict | None:
    """Validate actual code and receipt; never infer authorization from old git_sha."""
    from scripts.h100.test_recovery import load_test_execution
    if load_test_execution(repo, runs_root, campaign_git_sha, require_activation=require_test_activation) is not None:
        from scripts.h100.terminal_recovery import inherited_execution
        return inherited_execution(runs_root)
    if os.environ.get("H100_TERMINAL_RECEIPT") or os.environ.get("H100_TERMINAL_RECEIPT_SHA256"):
        from scripts.h100.terminal_recovery import inherited_execution, load_terminal_execution
        load_terminal_execution(repo, runs_root, campaign_git_sha)
        return inherited_execution(runs_root)
    if os.path.lexists(runs_root / ".h100/terminal-recovery/TERMINAL_SOURCE_VALIDATED.json"):
        raise RuntimeError("terminal recovery namespace requires its separate execution receipt")
    receipt_name = os.environ.get("H100_RECOVERY_RECEIPT")
    digest = os.environ.get("H100_RECOVERY_RECEIPT_SHA256")
    if not receipt_name and not digest:
        if any(os.path.lexists(runs_root / ".h100" / DIRECTORY / name)
               for name in (SOURCE_RECEIPT, PREVIOUS_SOURCE_RECEIPT)):
            raise RuntimeError("recovery execution receipt is required for this namespace")
        if campaign_git_sha == CAMPAIGN_SOURCE and git_output(repo, "rev-parse", "HEAD") != CAMPAIGN_SOURCE:
            raise RuntimeError("patched execution cannot use campaign identity without recovery receipt")
        return None
    expected = runs_root.resolve() / ".h100" / DIRECTORY / SOURCE_RECEIPT
    if not receipt_name or not digest or Path(receipt_name) != expected:
        raise RuntimeError("recovery receipt must be canonical and hash-bound")
    regular(expected, immutable=True)
    if sha256_file(expected) != digest:
        raise RuntimeError("recovery source receipt SHA-256 mismatch")
    payload = read_json(expected)
    if payload.get("schema") != 2:
        raise RuntimeError("recovery runtime requires its V2 source receipt")
    validate_previous_source(runs_root)
    result = identity(payload, digest)
    if git_output(repo, "rev-parse", "HEAD") != result["git_sha"]:
        raise RuntimeError("current execution HEAD differs from recovery identity")
    validate_runtime_checkout(repo, runs_root)
    return result


def validate_provenance_execution(provenance: dict, execution: dict | None) -> None:
    current = provenance.get("execution")
    attempts = provenance.get("attempts", [])
    if current is not None and current != execution:
        raise RuntimeError("existing cell execution identity differs from recovery")
    if execution is None:
        if any(attempt.get("execution") is not None for attempt in attempts):
            raise RuntimeError("cell attempt requires recovery execution identity")
        return
    seen_new = False
    for attempt in attempts:
        actual = attempt.get("execution")
        if actual is None:
            if seen_new or str(attempt.get("slurm_job_id")) != "644766" or attempt.get("phase") != "train":
                raise RuntimeError("unbound execution attempt in recovery namespace")
        else:
            if actual != execution:
                raise RuntimeError("cell attempt execution identity drifted")
            seen_new = True
    if seen_new and current != execution:
        raise RuntimeError("current cell provenance lost its execution identity")


def close_approved_failure(provenance: dict, *, runs_root: Path, exp_id: str,
                           execution: dict | None) -> bool:
    """Use the recorded failure time, rather than counting queue downtime as work."""
    if execution is None or exp_id not in POLICY["checkpoint_sha256"]:
        return False
    attempts = provenance.get("attempts", [])
    if not attempts or attempts[-1].get("execution") is not None:
        return False
    directory = runs_root / ".h100" / DIRECTORY
    receipt = read_json(directory / ACTIVATION, immutable=True)
    if receipt.get("execution") != execution:
        raise RuntimeError("failed-attempt closure lacks its recovery authorization")
    original = receipt["prior_failed_provenance"][exp_id]
    if provenance != original:
        raise RuntimeError("approved failed provenance changed before first retry")
    failed = read_json(directory / FAILED_MANIFEST, immutable=True)
    if sha256_file(directory / FAILED_MANIFEST) != POLICY["failed_manifest_sha256"]:
        raise RuntimeError("failed attempt event binding changed")
    event = [e for e in failed["events"] if e.get("event") == "cell_failed" and e.get("exp_id") == exp_id][-1]
    if event.get("exit_code") != 1:
        raise RuntimeError("approved checkpoint failure exit differs")
    from scripts.h100.campaign import reconcile_stale_open_attempt
    closed = reconcile_stale_open_attempt(provenance, phase="train", recovered_utc=event["utc"])
    if closed:
        attempts[-1].pop("exit", None)
        attempts[-1].update(exit_code=1, active_seconds_basis="started-to-recorded-cell-failure",
                            recovered_from="owner-approved-checkpoint-load-failure")
    return closed


def validate_initial_failure(prior: dict, digest: str) -> None:
    expected_failed = set(POLICY["checkpoint_sha256"])
    if (digest != POLICY["failed_manifest_sha256"]
            or prior.get("git_sha") != CAMPAIGN_SOURCE
            or prior.get("status") != "failed" or prior.get("phase") != "train"
            or prior.get("running") or prior.get("test_complete")
            or prior.get("training_cohort") is not None
            or set(prior.get("training_complete", [])) != set(POLICY["completed_cells"])
            or prior.get("fail_stop", {}).get("engaged") is not True
            or set(prior.get("fail_stop", {}).get("failed", [])) != expected_failed
            or prior.get("fail_stop", {}).get("allowed_to_finish")
            or prior.get("h100_ready_sha256") != POLICY["h100_ready_sha256"]):
        raise RuntimeError("campaign does not match the exact owner-approved failure")
    promoted = {e["exp_id"]: e.get("sha256") for e in prior.get("events", [])
                if e.get("event") == "checkpoint_promoted" and e.get("exp_id") in expected_failed}
    if promoted != POLICY["checkpoint_sha256"]:
        raise RuntimeError("approved checkpoint promotion events do not match")


def verify_checkpoint(path: Path, expected_sha256: str) -> dict:
    """Read each actual checkpoint on CPU before any worker can train."""
    import torch
    regular(path)
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"promoted checkpoint SHA-256 mismatch: {path}")
    unsupported = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if not unsupported <= {"pathlib.PosixPath"}:
        raise RuntimeError(f"checkpoint requires additional owner review: {sorted(unsupported)}")
    with torch.serialization.safe_globals([PosixPath]):
        value = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "optimizer_states", "lr_schedulers", "callbacks", "loops",
                "epoch", "global_step", "pytorch-lightning_version"}
    if not isinstance(value, dict) or not required <= set(value):
        raise RuntimeError("checkpoint lacks full Lightning training state")
    if not all(value[key] for key in ("state_dict", "optimizer_states", "lr_schedulers", "callbacks", "loops")):
        raise RuntimeError("checkpoint has empty required training state")
    if not all(isinstance(tensor, torch.Tensor) for tensor in value["state_dict"].values()):
        raise RuntimeError("checkpoint model state contains unexpected objects")
    if type(value["global_step"]) is not int or value["global_step"] <= 0:
        raise RuntimeError("checkpoint has no prior training progress")
    result = {"sha256": expected_sha256, "epoch": value["epoch"],
              "global_step": value["global_step"], "model_tensors": len(value["state_dict"]),
              "validation": "restricted-deserialization-with-full-training-state-present"}
    del value
    gc.collect()
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("checkpoint changed during recovery validation")
    return result


def activate(controller, prior: dict | None) -> None:
    """Called under campaign.lock, after inherited bindings pass, before launches."""
    execution = getattr(controller, "execution", None)
    if prior is None:
        if execution:
            raise RuntimeError("recovery cannot initialize a new campaign")
        return
    old_execution = prior.get("execution")
    if old_execution is not None and old_execution != execution:
        raise RuntimeError("recovery execution identity is missing or has drifted")
    if execution is None:
        return
    if controller.git_sha != CAMPAIGN_SOURCE or socket.gethostname().split(".")[0] != POLICY["node"]:
        raise RuntimeError("recovery campaign/node differs from owner approval")
    directory = controller.runs_root / ".h100" / DIRECTORY
    activation_path = directory / ACTIVATION
    activated = [e for e in prior.get("events", []) if e.get("event") == EVENT]
    if activated:
        receipt = read_json(activation_path, immutable=True)
        if (len(activated) != 1 or old_execution != execution
                or receipt.get("execution") != execution
                or activated[0].get("activation_sha256") != sha256_file(activation_path)
                or sha256_file(regular(directory / FAILED_MANIFEST, immutable=True)) != POLICY["failed_manifest_sha256"]):
            raise RuntimeError("recovery activation evidence drifted")
        # Deliberately retain restore_fail_stop's result: later failures are not retried.
        return
    raw = regular(controller.manifest_path).read_bytes()
    validate_initial_failure(prior, hashlib.sha256(raw).hexdigest())
    if json.loads(raw) != prior:
        raise RuntimeError("campaign changed during recovery validation")
    if controller.cohort_path.exists() or controller.phase != "train":
        raise RuntimeError("initial recovery cannot cross the held-out cohort gate")
    from scripts.h100.campaign import validate_training_completion
    by_id = {cell.exp_id: cell for cell in controller.cells}
    retained = {}
    for exp_id in POLICY["completed_cells"]:
        result = validate_training_completion(
            by_id[exp_id], runs_root=controller.runs_root, git_sha=CAMPAIGN_SOURCE,
            detector_sha256=controller.detector_sha256, candidate_floor=controller.candidate_floor,
        )
        run = controller.runs_root / exp_id
        retained[exp_id] = {
            "final_metrics_sha256": sha256_file(regular(run / "final_metrics.json", immutable=True)),
            "best_checkpoint_sha256": result["best_checkpoint"]["sha256"],
            "last_checkpoint_sha256": sha256_file(regular(run / "checkpoints/last.ckpt")),
            "config_sha256": sha256_file(regular(run / "config.yaml")),
            "runtime_provenance_sha256": sha256_file(regular(run / "runtime_provenance.json")),
            "runtime_provenance": read_json(run / "runtime_provenance.json"),
        }
    for exp_id in POLICY["checkpoint_sha256"]:
        final = controller.runs_root / exp_id / "final_metrics.json"
        if final.exists() or final.is_symlink():
            raise RuntimeError("approved failed cell unexpectedly has a completion")
    checkpoints = {exp_id: verify_checkpoint(controller.runs_root / exp_id / "checkpoints/last.ckpt", digest)
                   for exp_id, digest in sorted(POLICY["checkpoint_sha256"].items())}
    archive = directory / FAILED_MANIFEST
    if archive.exists():
        if regular(archive, immutable=True).read_bytes() != raw:
            raise RuntimeError("saved failure manifest differs from approved failure")
    else:
        # Preserve exact original bytes; this is not a regenerated failure record.
        with archive.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        archive.chmod(0o444)
    attempts = {exp_id: read_json(controller.runs_root / exp_id / "runtime_provenance.json")
                for exp_id in sorted(POLICY["checkpoint_sha256"])}
    receipt = {"schema": 1, "status": "approved-failures-verified", "execution": execution,
               "failed_manifest_sha256": POLICY["failed_manifest_sha256"],
               "retained_completions": retained, "verified_checkpoints": checkpoints,
               "prior_failed_provenance": attempts}
    write_once(activation_path, receipt)
    controller.events.append({"event": EVENT, "activation_sha256": sha256_file(activation_path),
                              "execution": execution, "retry_ids": sorted(checkpoints)})
    controller.training_complete_ids = set(POLICY["completed_cells"])
    controller.complete_ids = controller.training_complete_ids
    controller.failure_seen = False
    controller.failed_ids = set()
    controller.failure_allowed_ids = set()
    # A durable activation event precedes any worker launch. Crash before this
    # write rechecks the unchanged failed manifest; crash after cannot reset again.
    controller.write_manifest()


def record_config_execution(path: Path, execution: dict | None) -> None:
    if execution is None or not path.exists():
        return
    import yaml
    regular(path)
    payload = yaml.safe_load(path.read_text())
    if payload.get("git_sha") != CAMPAIGN_SOURCE:
        raise RuntimeError("resolved config campaign identity mismatch")
    if payload.get("execution_source") not in (None, execution):
        raise RuntimeError("resolved config execution identity drifted")
    payload["execution_source"] = execution
    path.write_text(yaml.safe_dump(payload), newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "prepare"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(repo=args.repo, package_root=args.package_root,
                      expected_manifest_sha256=args.expected_manifest_sha256,
                      expected_git_sha=args.expected_git_sha, runs_root=args.runs_root,
                      publish=args.operation == "prepare")
    print(json.dumps({"status": payload["status"], "git_sha": payload["git_sha"],
                      "campaign_git_sha": CAMPAIGN_SOURCE}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
