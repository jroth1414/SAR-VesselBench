"""One-time terminal DEV recovery; historical training and execution stay intact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
from pathlib import Path

from scripts.h100 import resume_recovery as previous
from scripts.h100.contracts import atomic_write_json, sha256_file, verify_expected_hashes
from scripts.h100.source_validation import git_output
from scripts.handoff.package import _canonical_json, _require_no_symlink_components
from scripts.handoff.terminal_contract import (
    CAMPAIGN_SOURCE, CELLS, POLICY, TERMINAL_ANCESTOR, validate_terminal_control,
)

DIRECTORY = "terminal-recovery"
SOURCE_RECEIPT = "TERMINAL_SOURCE_VALIDATED.json"
ACTIVATION = "TERMINAL_ACTIVATION.json"
FAILED_MANIFEST = "failed-campaign.json"
EVENT = "terminal_recovery_activated"
regular, read_json = previous.regular, previous.read_json

def _write_bytes_once(path: Path, raw: bytes) -> None:
    if os.path.lexists(path):
        if regular(path, immutable=True).read_bytes() != raw:
            raise RuntimeError(f"terminal immutable evidence changed: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=".terminal-publication-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fchmod(output.fileno(), 0o444)
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if regular(path, immutable=True).read_bytes() != raw:
                raise RuntimeError(f"terminal publication raced with different evidence: {path}")
    finally:
        temporary.unlink(missing_ok=True)

def write_once(path: Path, payload: dict) -> None:
    _write_bytes_once(path, _canonical_json(payload))


def inherited_execution(runs_root: Path) -> dict:
    previous.validate_previous_source(runs_root)
    directory = runs_root / ".h100" / previous.DIRECTORY
    source = regular(directory / previous.SOURCE_RECEIPT, immutable=True)
    activation = regular(directory / previous.ACTIVATION, immutable=True)
    if (sha256_file(source) != POLICY["previous_source_receipt_sha256"]
            or sha256_file(activation) != POLICY["previous_activation_sha256"]):
        raise RuntimeError("terminal recovery inherited execution receipts changed")
    result = previous.identity(read_json(source), sha256_file(source))
    if (result["git_sha"] != TERMINAL_ANCESTOR
            or read_json(activation).get("execution") != result):
        raise RuntimeError("terminal recovery inherited execution identity changed")
    return result


def identity(payload: dict, receipt_sha: str) -> dict:
    control = validate_terminal_control(payload.get("control"))
    if (payload.get("schema") != 1 or payload.get("status") != "terminal-source-validated"
            or payload.get("campaign_git_sha") != CAMPAIGN_SOURCE
            or payload.get("git_sha") != control["source"]["git_commit"]
            or payload.get("h100_ready_sha256") != POLICY["h100_ready_sha256"]
            or payload.get("previous_source_receipt_sha256") != POLICY["previous_source_receipt_sha256"]
            or payload.get("previous_activation_sha256") != POLICY["previous_activation_sha256"]
            or payload.get("previous_execution", {}).get("git_sha") != TERMINAL_ANCESTOR):
        raise RuntimeError("invalid terminal recovery source receipt")
    package = payload.get("package", {})
    control_sha = hashlib.sha256(_canonical_json(control)).hexdigest()
    if (package.get("package_id") != f"xview3-h100-runtime-{payload['git_sha']}-{control_sha}"
            or package.get("bundle_sha256") != control["source"]["git_bundle_sha256"]):
        raise RuntimeError("terminal recovery source/package identity mismatch")
    return {"campaign_git_sha": CAMPAIGN_SOURCE, "git_sha": payload["git_sha"],
            "terminal_package_id": package["package_id"],
            "terminal_manifest_sha256": package["manifest_sha256"],
            "source_receipt_sha256": receipt_sha}


def validate_initial_failure(prior: dict, digest: str, old_execution: dict) -> None:
    if (digest != POLICY["failed_manifest_sha256"] or prior.get("git_sha") != CAMPAIGN_SOURCE
            or prior.get("execution") != old_execution or prior.get("terminal_execution") is not None
            or prior.get("status") != "failed" or prior.get("phase") != "train"
            or prior.get("running") or prior.get("test_complete") or prior.get("training_cohort") is not None
            or set(prior.get("training_complete", [])) != set(POLICY["completed_cells"])
            or prior.get("fail_stop") != {"engaged": True, "failed": sorted(CELLS), "allowed_to_finish": []}
            or prior.get("h100_ready_sha256") != POLICY["h100_ready_sha256"]):
        raise RuntimeError("campaign is not the exact approved terminal failure")
    promoted = {e["exp_id"]: e.get("sha256") for e in prior.get("events", [])
                if e.get("event") == "checkpoint_promoted" and e.get("exp_id") in CELLS}
    if promoted != POLICY["checkpoint_sha256"]:
        raise RuntimeError("terminal input checkpoint promotion bindings differ")


def prepare(*, repo: Path, package_root: Path, expected_manifest_sha256: str,
            expected_git_sha: str, runs_root: Path, publish: bool) -> dict:
    from scripts.handoff.terminal_amendment import verify_terminal_amendment, verify_terminal_source
    repo = _require_no_symlink_components(repo, leaf="directory")
    runs_root = _require_no_symlink_components(runs_root, leaf="directory")
    if sha256_file(regular(package_root / "manifest.json")) != expected_manifest_sha256:
        raise RuntimeError("terminal package manifest mismatch")
    manifest = verify_terminal_amendment(package_root)
    if (manifest["source"]["git_commit"] != expected_git_sha
            or git_output(repo, "rev-parse", "HEAD") != expected_git_sha
            or git_output(repo, "status", "--porcelain", "--untracked-files=all")):
        raise RuntimeError("terminal recovery requires its exact clean execution checkout")
    verify_terminal_source(repo)
    ready, old_source_sha = previous.inherited_source(runs_root)
    old_execution = inherited_execution(runs_root)
    prior_path = runs_root / ".h100/campaign_manifest.json"
    prior = read_json(prior_path)
    activated = [e for e in prior.get("events", []) if e.get("event") == EVENT]
    if not activated:
        validate_initial_failure(prior, sha256_file(prior_path), old_execution)
        if (os.path.lexists(runs_root / ".h100/TRAINING_COHORT.json")
                or list(runs_root.glob("*/test_metrics.json"))):
            raise RuntimeError("terminal activation found premature held-out state")
    elif len(activated) != 1 or prior.get("terminal_execution", {}).get("git_sha") != expected_git_sha:
        raise RuntimeError("active terminal recovery cannot switch execution source")
    if publish and socket.gethostname().split(".")[0] != POLICY["node"]:
        raise RuntimeError("terminal recovery allocation must execute on dgx18")
    payload = {
        "schema": 1, "status": "terminal-source-validated", "campaign_git_sha": CAMPAIGN_SOURCE,
        "git_sha": expected_git_sha, "h100_ready_sha256": POLICY["h100_ready_sha256"],
        "inherited_source_validation_sha256": old_source_sha, "previous_execution": old_execution,
        "previous_source_receipt_sha256": POLICY["previous_source_receipt_sha256"],
        "previous_activation_sha256": POLICY["previous_activation_sha256"],
        "frozen_sha256": verify_expected_hashes(repo, ready["source"]["frozen_sha256"]),
        "control": manifest["amendment_identity"], "package": {
            "package_id": manifest["package_id"], "manifest_sha256": expected_manifest_sha256,
            "ready_sha256": sha256_file(regular(package_root / "READY.json")),
            "sha256sums_sha256": sha256_file(regular(package_root / "SHA256SUMS")),
            "bundle_sha256": manifest["source"]["git_bundle_sha256"],
        },
    }
    directory = runs_root / ".h100" / DIRECTORY
    source = directory / SOURCE_RECEIPT
    if os.path.lexists(source) and read_json(source, immutable=True) != payload:
        raise RuntimeError("terminal source receipt changed")
    if os.path.lexists(directory / ACTIVATION):
        if not source.exists():
            raise RuntimeError("terminal activation exists without the same source")
        activation = read_json(directory / ACTIVATION, immutable=True)
        if (activation.get("terminal_execution") != identity(payload, sha256_file(source))
                or activation.get("failed_manifest_sha256") != POLICY["failed_manifest_sha256"]):
            raise RuntimeError("terminal activation binding changed")
    if publish:
        directory.mkdir(exist_ok=True)
        _require_no_symlink_components(directory, leaf="directory")
        write_once(source, payload)
    return payload


def load_terminal_execution(repo: Path, runs_root: Path, campaign_git_sha: str | None = None) -> dict | None:
    name, digest = os.environ.get("H100_TERMINAL_RECEIPT"), os.environ.get("H100_TERMINAL_RECEIPT_SHA256")
    expected = runs_root.resolve() / ".h100" / DIRECTORY / SOURCE_RECEIPT
    if not name and not digest:
        if os.path.lexists(expected):
            raise RuntimeError("terminal recovery requires its execution receipt")
        return None
    if (not name or not digest or Path(name) != expected
            or sha256_file(regular(expected, immutable=True)) != digest):
        raise RuntimeError("terminal recovery receipt must be canonical and hash-bound")
    payload = read_json(expected)
    result = identity(payload, digest)
    if (campaign_git_sha not in (None, CAMPAIGN_SOURCE)
            or payload["previous_execution"] != inherited_execution(runs_root)
            or git_output(repo, "rev-parse", "HEAD") != result["git_sha"]):
        raise RuntimeError("terminal execution lineage or actual source differs")
    previous.validate_runtime_checkout(repo, runs_root)
    return result


def validate_worker_context(repo: Path, runs_root: Path, exp_id: str,
                            source_receipt: Path, expected_sha256: str) -> dict:
    if (Path(source_receipt) != runs_root.resolve() / ".h100" / DIRECTORY / SOURCE_RECEIPT
            or expected_sha256 != os.environ.get("H100_TERMINAL_RECEIPT_SHA256")
            or exp_id not in CELLS):
        raise RuntimeError("terminal worker is outside the approved context")
    execution = load_terminal_execution(repo, runs_root, CAMPAIGN_SOURCE)
    activation = read_json(runs_root / ".h100" / DIRECTORY / ACTIVATION, immutable=True)
    if (execution is None or activation.get("terminal_execution") != execution
            or activation.get("failed_manifest_sha256") != POLICY["failed_manifest_sha256"]
            or socket.gethostname().split(".")[0] != POLICY["node"]):
        raise RuntimeError("terminal worker lacks its activated execution identity")
    verified = activation.get("verified_checkpoints", {}).get(exp_id, {})
    validate_receipt_authorization(runs_root / exp_id, {
        "exp_id": exp_id, "terminal_execution": execution,
        "activation_sha256": sha256_file(runs_root / ".h100" / DIRECTORY / ACTIVATION),
        "input_checkpoint": {"sha256": POLICY["checkpoint_sha256"][exp_id]},
        "prior_best_checkpoint": {"sha256": verified.get("prior_best_checkpoint_sha256")},
    })
    for relative, wanted in (("checkpoints/last.ckpt", verified.get("sha256")),
                             ("checkpoints/best.ckpt", verified.get("prior_best_checkpoint_sha256"))):
        if sha256_file(regular(runs_root / exp_id / relative)) != wanted:
            raise RuntimeError("terminal worker original checkpoint changed")
    return execution


def validate_receipt_authorization(run_dir: Path, receipt: dict) -> None:
    """Validate portable immutable control lineage without GPU, node, or live HEAD."""
    runs_root = run_dir.parent
    exp_id = run_dir.name
    if exp_id not in CELLS or receipt.get("exp_id") != exp_id:
        raise RuntimeError("terminal receipt cell is not authorized")
    directory = runs_root / ".h100" / DIRECTORY
    source = regular(directory / SOURCE_RECEIPT, immutable=True)
    source_sha = sha256_file(source)
    payload = read_json(source)
    execution = identity(payload, source_sha)
    if receipt.get("terminal_execution") != execution:
        raise RuntimeError("terminal receipt source authorization differs")
    old_execution = inherited_execution(runs_root)
    if payload.get("previous_execution") != old_execution:
        raise RuntimeError("terminal receipt inherited source differs")
    if sha256_file(regular(runs_root / ".h100/H100_READY.json", immutable=True)) != POLICY["h100_ready_sha256"]:
        raise RuntimeError("terminal receipt readiness changed")
    inherited_source_path = regular(runs_root / ".h100/SOURCE_VALIDATED.json", immutable=True)
    if sha256_file(inherited_source_path) != payload.get("inherited_source_validation_sha256"):
        raise RuntimeError("terminal inherited source receipt changed")
    activation_path = regular(directory / ACTIVATION, immutable=True)
    activation = read_json(activation_path)
    if (sha256_file(activation_path) != receipt.get("activation_sha256")
            or activation.get("schema") != 1 or activation.get("status") != "approved-terminal-failures-verified"
            or activation.get("terminal_execution") != execution
            or activation.get("prior_execution") != old_execution
            or activation.get("failed_manifest_sha256") != POLICY["failed_manifest_sha256"]
            or sha256_file(regular(directory / FAILED_MANIFEST, immutable=True)) != POLICY["failed_manifest_sha256"]):
        raise RuntimeError("terminal receipt activation authorization differs")
    verified = activation.get("verified_checkpoints", {}).get(exp_id, {})
    if (verified.get("sha256") != POLICY["checkpoint_sha256"][exp_id]
            or verified.get("sha256") != receipt.get("input_checkpoint", {}).get("sha256")
            or verified.get("prior_best_checkpoint_sha256") != receipt.get("prior_best_checkpoint", {}).get("sha256")
            or verified.get("config_sha256") != sha256_file(regular(run_dir / "config.yaml"))):
        raise RuntimeError("terminal checkpoint/config authorization differs")


def validate_retained(runs_root: Path, execution: dict) -> None:
    activation = read_json(runs_root / ".h100" / DIRECTORY / ACTIVATION, immutable=True)
    retained = activation.get("retained_completions", {})
    if activation.get("terminal_execution") != execution or set(retained) != set(POLICY["completed_cells"]):
        raise RuntimeError("terminal retained inventory differs")
    names = {"final_metrics_sha256": "final_metrics.json", "config_sha256": "config.yaml",
             "best_checkpoint_sha256": "checkpoints/best.ckpt", "last_checkpoint_sha256": "checkpoints/last.ckpt"}
    for exp_id, expected in retained.items():
        for key, relative in names.items():
            if sha256_file(regular(runs_root / exp_id / relative, immutable=relative == "final_metrics.json")) != expected.get(key):
                raise RuntimeError(f"retained terminal artifact changed: {exp_id}/{relative}")
        original_attempts = expected.get("runtime_provenance", {}).get("attempts", [])
        actual = read_json(runs_root / exp_id / "runtime_provenance.json")
        if actual.get("attempts", [])[:len(original_attempts)] != original_attempts:
            raise RuntimeError(f"retained training history changed: {exp_id}")


def validate_provenance(provenance: dict, execution: dict) -> None:
    current = provenance.get("terminal_execution")
    if current not in (None, execution):
        raise RuntimeError("terminal provenance execution changed")
    seen = False
    for attempt in provenance.get("attempts", []):
        actual = attempt.get("terminal_execution")
        if actual is not None:
            if actual != execution:
                raise RuntimeError("terminal attempt execution changed")
            seen = True
        elif seen:
            raise RuntimeError("new attempt lost terminal execution binding")
    if seen and current != execution:
        raise RuntimeError("current provenance lost terminal identity")


def activate(controller, prior: dict | None) -> None:
    execution = getattr(controller, "terminal_execution", None)
    if execution is None:
        return
    if prior is None or socket.gethostname().split(".")[0] != POLICY["node"]:
        raise RuntimeError("terminal recovery cannot start a new campaign or change nodes")
    directory = controller.runs_root / ".h100" / DIRECTORY
    activation_path = directory / ACTIVATION
    events = [e for e in prior.get("events", []) if e.get("event") == EVENT]
    if events:
        receipt = read_json(activation_path, immutable=True)
        if (len(events) != 1 or prior.get("terminal_execution") != execution
                or receipt.get("terminal_execution") != execution
                or events[0].get("activation_sha256") != sha256_file(activation_path)
                or sha256_file(regular(directory / FAILED_MANIFEST, immutable=True)) != POLICY["failed_manifest_sha256"]):
            raise RuntimeError("terminal activation evidence drifted")
        return  # Later failure decisions remain fail-stop.
    raw = regular(controller.manifest_path).read_bytes()
    validate_initial_failure(prior, hashlib.sha256(raw).hexdigest(), controller.execution)
    if json.loads(raw) != prior or controller.phase != "train" or controller.cohort_path.exists():
        raise RuntimeError("terminal failure changed before activation")
    from scripts.h100.campaign import validate_training_completion
    retained = {}
    by_id = {cell.exp_id: cell for cell in controller.cells}
    for exp_id in POLICY["completed_cells"]:
        result = validate_training_completion(by_id[exp_id], runs_root=controller.runs_root,
            git_sha=CAMPAIGN_SOURCE, detector_sha256=controller.detector_sha256,
            candidate_floor=controller.candidate_floor)
        run = controller.runs_root / exp_id
        retained[exp_id] = {
            "final_metrics_sha256": sha256_file(regular(run / "final_metrics.json", immutable=True)),
            "best_checkpoint_sha256": result["best_checkpoint"]["sha256"],
            "last_checkpoint_sha256": sha256_file(regular(run / "checkpoints/last.ckpt")),
            "config_sha256": sha256_file(regular(run / "config.yaml")),
            "runtime_provenance_sha256": sha256_file(regular(run / "runtime_provenance.json")),
            "runtime_provenance": read_json(run / "runtime_provenance.json"),
        }
    checkpoints, attempts = {}, {}
    for exp_id in CELLS:
        run = controller.runs_root / exp_id
        if os.path.lexists(run / "final_metrics.json"):
            raise RuntimeError("terminal failed cell unexpectedly has completion evidence")
        checkpoints[exp_id] = previous.verify_checkpoint(run / "checkpoints/last.ckpt", POLICY["checkpoint_sha256"][exp_id])
        if (checkpoints[exp_id]["epoch"] != POLICY["epoch"]
                or checkpoints[exp_id]["global_step"] != POLICY["global_step"]):
            raise RuntimeError("terminal input training progress differs")
        checkpoints[exp_id]["prior_best_checkpoint_sha256"] = sha256_file(regular(run / "checkpoints/best.ckpt"))
        checkpoints[exp_id]["config_sha256"] = sha256_file(regular(run / "config.yaml"))
        attempts[exp_id] = read_json(run / "runtime_provenance.json")
    archive = directory / FAILED_MANIFEST
    _write_bytes_once(archive, raw)
    receipt = {"schema": 1, "status": "approved-terminal-failures-verified",
        "terminal_execution": execution, "prior_execution": controller.execution,
        "failed_manifest_sha256": POLICY["failed_manifest_sha256"],
        "retained_completions": retained, "verified_checkpoints": checkpoints,
        "prior_failed_provenance": attempts}
    write_once(activation_path, receipt)
    controller.events.append({"event": EVENT, "terminal_execution": execution,
        "activation_sha256": sha256_file(activation_path), "retry_ids": list(CELLS)})
    controller.training_complete_ids = set(POLICY["completed_cells"])
    controller.complete_ids = controller.training_complete_ids
    controller.failure_seen = False
    controller.failed_ids = set()
    controller.failure_allowed_ids = set()
    controller.write_manifest()


def close_approved_failure(provenance: dict, *, runs_root: Path, exp_id: str, execution: dict) -> bool:
    if exp_id not in CELLS or provenance.get("terminal_execution") is not None:
        return False
    directory = runs_root / ".h100" / DIRECTORY
    receipt = read_json(directory / ACTIVATION, immutable=True)
    if receipt.get("terminal_execution") != execution or provenance != receipt["prior_failed_provenance"][exp_id]:
        raise RuntimeError("terminal failed attempt changed before its approved closure")
    failed = read_json(directory / FAILED_MANIFEST, immutable=True)
    event = [e for e in failed["events"] if e.get("event") == "cell_failed" and e.get("exp_id") == exp_id][-1]
    from scripts.h100.campaign import reconcile_stale_open_attempt
    closed = reconcile_stale_open_attempt(provenance, phase="train", recovered_utc=event["utc"])
    if closed:
        provenance["attempts"][-1].pop("exit", None)
        provenance["attempts"][-1].update(exit_code=1,
            active_seconds_basis="started-to-recorded-cell-failure", recovered_from="owner-approved-terminal-failure")
    return closed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("check", "prepare"))
    for name in ("repo", "package-root", "runs-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    args = parser.parse_args()
    payload = prepare(repo=args.repo, package_root=args.package_root,
        expected_manifest_sha256=args.expected_manifest_sha256, expected_git_sha=args.expected_git_sha,
        runs_root=args.runs_root, publish=args.operation == "prepare")
    print(json.dumps({"status": payload["status"], "git_sha": payload["git_sha"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
