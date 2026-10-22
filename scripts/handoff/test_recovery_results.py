"""Portable archive-byte validation of the separate TEST recovery lineage."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
from .package import PackageError, _canonical_json, _hash_file, _require_no_symlink_components
from .test_recovery_contract import CAMPAIGN_SOURCE, CELL, CELLS, POLICY
from scripts.h100.test_recovery import SOURCE, ACTIVATION, FAILED, PROVENANCE, LAUNCH, NAMES, EVENT

PREFIX = "results/provenance/test-recovery/"


def require(ok, message):
    if not ok:
        raise PackageError("TEST recovery result " + message)


def sha(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def execution_identity(receipt):
    from scripts.h100.test_recovery import identity
    try:
        return identity(receipt, sha(receipt))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise PackageError("invalid TEST source receipt: " + str(exc)) from exc


def validate_export(recovery, source, members):
    names = {n for n in members if str(n).startswith(PREFIX)}
    if recovery is None:
        require(not names and source.get("test_execution") is None, "unbound evidence")
        return
    require(set(recovery) == {"source_receipt", "activation", "activation_sha256"}, "invalid evidence fields")
    receipt, activation = recovery["source_receipt"], recovery["activation"]
    execution = execution_identity(receipt)
    require(members.get("results/provenance/TRAINING_COHORT.json") == POLICY["cohort_sha256"]
        and source.get("training_cohort_sha256", source.get("training_cohort", {}).get("sha256")) == POLICY["cohort_sha256"],
        "actual frozen cohort archive bytes differ")
    require(source.get("test_execution") == execution and receipt["previous_execution"] == source.get("execution")
        and receipt["terminal_execution"] == source.get("terminal_execution")
        and source.get("git_commit") == CAMPAIGN_SOURCE
        and source.get("h100_ready_sha256") == POLICY["h100_ready_sha256"], "source history differs")
    require(set(activation) == {"schema", "status", "test_execution", "terminal_execution", "previous_execution",
        "failed_manifest_sha256", "failed_provenance_sha256", "cohort_sha256", "prior_test_launch_sha256", "retained"}
        and activation["schema"] == 1 and activation["status"] == "approved-test-failure-verified"
        and activation["test_execution"] == execution and activation["terminal_execution"] == receipt["terminal_execution"]
        and activation["previous_execution"] == receipt["previous_execution"]
        and set(activation["retained"]) == set(CELLS)
        and recovery["activation_sha256"] == sha(activation), "activation differs")
    expected = {PREFIX + SOURCE: execution["source_receipt_sha256"], PREFIX + ACTIVATION: recovery["activation_sha256"],
        PREFIX + FAILED: POLICY["failed_manifest_sha256"], PREFIX + PROVENANCE: POLICY["failed_provenance_sha256"],
        PREFIX + LAUNCH: POLICY["prior_test_launch_sha256"]}
    require(names == set(expected) and all(members[n] == d for n, d in expected.items()), "archive inventory/hashes differ")
    require(all(activation[key] == POLICY[key] for key in
        ("failed_manifest_sha256", "failed_provenance_sha256", "cohort_sha256", "prior_test_launch_sha256")), "approved bindings differ")
    for cell, item in activation["retained"].items():
        require(set(item) == {"files", "prior_provenance"}, "retained inventory fields differ")
        required = {"final_metrics.json", "config.yaml", "checkpoints/best.ckpt", "checkpoints/last.ckpt"}
        allowed = required | {"terminal_recovery.json", "checkpoints/terminal-best.ckpt"}
        require(required <= set(item["files"]) <= allowed, "retained file inventory differs")
    cnn = activation["retained"][CELL]
    require(cnn["files"]["checkpoints/best.ckpt"] == POLICY["selected_checkpoint_sha256"], "selected cnnrand checkpoint differs")


def validate_campaign(campaign, recovery):
    events = [e for e in campaign.get("events", []) if e.get("event") == EVENT]
    if recovery is None:
        require(campaign.get("test_execution") is None and not events, "campaign has unbound TEST source")
        return
    execution = execution_identity(recovery["source_receipt"])
    require(campaign.get("test_execution") == execution and campaign.get("phase") == "score-test"
        and campaign.get("training_cohort", {}).get("sha256") == POLICY["cohort_sha256"]
        and len(events) == 1 and events[0].get("test_execution") == execution
        and events[0].get("activation_sha256") == recovery["activation_sha256"]
        and events[0].get("retry_ids") == [CELL], "campaign transition differs")


def validate_cell(provenance, cell, recovery, *, require_complete=True):
    attempts = provenance.get("attempts", [])
    if recovery is None:
        require(provenance.get("test_execution") is None and all(a.get("test_execution") is None for a in attempts), "unbound attempt source")
        return
    execution = execution_identity(recovery["source_receipt"])
    old = recovery["activation"]["retained"][cell]["prior_provenance"]["attempts"]
    require(provenance.get("test_execution") == execution and len(attempts) > len(old), "missing new TEST attempt")
    for index, expected in enumerate(old):
        observed = dict(attempts[index])
        if cell == CELL and index == len(old) - 1:
            require(observed.get("finished_utc") == POLICY["failure_utc"] and observed.get("exit_code") == 1
                and observed.get("active_seconds_basis") == "started-to-recorded-cell-failure"
                and observed.get("recovered_from") == "owner-approved-test-failure", "failed attempt closure differs")
            seconds = (datetime.fromisoformat(POLICY["failure_utc"]) - datetime.fromisoformat(expected["started_utc"])).total_seconds()
            require(seconds >= 0 and observed.get("active_seconds") == seconds, "failed attempt duration includes queue time")
            for key in ("finished_utc", "exit_code", "active_seconds", "active_seconds_basis", "recovered_from"):
                observed.pop(key, None)
        require(observed == expected, "historical attempt changed")
    for attempt in attempts[len(old):]:
        require(attempt.get("test_execution") == execution and attempt.get("phase") == "score-test"
            and attempt.get("training_cohort_sha256") == POLICY["cohort_sha256"], "new attempt source/phase differs")
    if require_complete:
        require(attempts[-1].get("exit_code") == 0, "final TEST attempt failed")


def validate_retained_members(source, cell, members):
    recovery = source.get("test_recovery")
    if recovery is not None:
        root = "results/core/" + cell + "/"
        require(all(members.get(root + name) == digest for name, digest in recovery["activation"]["retained"][cell]["files"].items()),
            "retained training bytes changed in archive")


def validate_checkout(repo, runs_root, campaign):
    if campaign.get("grid_execution") is not None:
        from .grid_results import validate_checkout as validate_grid_checkout
        validate_grid_checkout(repo, runs_root, campaign)
        return
    from .test_recovery_amendment import verify_test_recovery_source
    from .package import _git_value
    path = _require_no_symlink_components(runs_root / ".h100/test-recovery" / SOURCE, leaf="file")
    require(not path.stat().st_mode & 0o222, "source is writable")
    receipt = json.loads(path.read_text())
    require(campaign.get("test_execution") == execution_identity(receipt)
        and _git_value(repo, "rev-parse", "HEAD") == receipt["git_sha"]
        and _git_value(repo, "branch", "--show-current") == receipt["control"]["source"]["branch"], "actual checkout differs")
    verify_test_recovery_source(repo)


def collect(repo, runs_root, campaign):
    directory = runs_root / ".h100/test-recovery"
    if campaign.get("test_execution") is None:
        require(not directory.exists(), "evidence exists without TEST identity")
        return None
    paths = {n: _require_no_symlink_components(directory / n, leaf="file") for n in NAMES}
    require(all(not p.stat().st_mode & 0o222 for p in paths.values()), "evidence is writable")
    recovery = {"source_receipt": json.loads(paths[SOURCE].read_text()), "activation": json.loads(paths[ACTIVATION].read_text()),
        "activation_sha256": _hash_file(paths[ACTIVATION])}
    source = {**campaign, "git_commit": campaign["git_sha"]}
    members = {PREFIX + n: _hash_file(p) for n, p in paths.items()}
    members["results/provenance/TRAINING_COHORT.json"] = _hash_file(runs_root / ".h100/TRAINING_COHORT.json")
    validate_export(recovery, source, members)
    validate_campaign(campaign, recovery)
    validate_checkout(repo, runs_root, campaign)
    from scripts.h100.test_recovery import validate_failed
    validate_failed(json.loads(paths[FAILED].read_text()), json.loads(paths[PROVENANCE].read_text()),
        manifest_sha=_hash_file(paths[FAILED]), provenance_sha=_hash_file(paths[PROVENANCE]),
        terminal=campaign["terminal_execution"], previous=campaign["execution"])
    return recovery


def content_validators(source, artifact):
    recovery = source.get("test_recovery")
    if artifact["kind"] == "core_result":
        return {f"results/core/{artifact['name']}/runtime_provenance.json":
            lambda raw: validate_cell(json.loads(raw), artifact["name"], recovery)}
    if artifact["kind"] != "campaign_provenance":
        return {}
    checks = {"results/provenance/campaign_manifest.json": lambda raw: validate_campaign(json.loads(raw), recovery)}
    if recovery is not None:
        activation = recovery["activation"]
        checks[PREFIX + SOURCE] = lambda raw: require(json.loads(raw) == recovery["source_receipt"], "source archive bytes differ")
        checks[PREFIX + ACTIVATION] = lambda raw: require(json.loads(raw) == activation, "activation archive bytes differ")
        def failed(raw):
            from scripts.h100.test_recovery import validate_failed
            validate_failed(json.loads(raw), activation["retained"][CELL]["prior_provenance"],
                manifest_sha=hashlib.sha256(raw).hexdigest(), provenance_sha=POLICY["failed_provenance_sha256"],
                terminal=activation["terminal_execution"], previous=activation["previous_execution"])
        checks[PREFIX + FAILED] = failed
        checks[PREFIX + PROVENANCE] = lambda raw: require(json.loads(raw) == activation["retained"][CELL]["prior_provenance"], "failed provenance archive differs")
    return checks
