"""Hash-bound owner-amendment contract regressions."""

from __future__ import annotations

import copy
import json
import stat

import pytest

from src.eval.final_authorization import (
    AUTHORIZED_OWNER,
    build_authorization,
    validate_authorization,
    validate_authorization_payload,
    write_authorization,
)
from src.eval.heldout_contract import HeldoutContractError


def _payload() -> dict[str, object]:
    selected = [f"cell-{index:02d}" for index in range(32)]
    return build_authorization(
        owner=AUTHORIZED_OWNER,
        created_utc="2026-08-13T05:00:00+00:00",
        campaign={
            "campaign_id": "campaign",
            "git_sha": "a" * 40,
            "runs_root": "/projects/geofam/jroth/runs",
            "manifest": {
                "relative_path": ".h100/campaign_manifest.json",
                "sha256": "b" * 64,
            },
            "terminal_event": {
                "event": "grid_validation_failed",
                "error": "grid.csv monotonicity STOP for beS1-f100-s0",
                "utc": "2026-08-13T04:34:00+00:00",
            },
        },
        evaluator_git_sha="c" * 40,
        cohort={
            "relative_path": ".h100/TRAINING_COHORT.json",
            "sha256": "d" * 64,
        },
        phase5_controls={
            "status": "reporting-controls-persisted",
            "h100_ready": {
                "relative_path": ".h100/H100_READY.json",
                "sha256": "a" * 64,
            },
            "cutover_ready": {
                "relative_path": ".h100/CUTOVER_READY.json",
                "sha256": "e" * 64,
            },
            "v100_diagnostic_isolation": {
                "relative_path": ".h100/V100_DIAGNOSTIC_ISOLATION.json",
                "sha256": "f" * 64,
            },
        },
        grid={
            "relative_path": "summary/grid.csv",
            "sha256": "1" * 64,
            "monotonicity_tolerance": 0.02,
            "monotonicity_ok": False,
            "violations": [
                {
                    "init": "beS1",
                    "from_fraction": 0.5,
                    "to_fraction": 1.0,
                    "from_test_f1": 0.8521,
                    "to_test_f1": 0.8221,
                    "drop": 0.03,
                }
            ],
        },
        test_results=[
            {
                "exp_id": exp_id,
                "relative_path": f"{exp_id}/test_metrics.json",
                "sha256": f"{index:064x}",
            }
            for index, exp_id in enumerate(selected, start=1)
        ],
        selected_cells=selected,
    )


def test_owner_authorization_is_exclusive_immutable_and_byte_bound(tmp_path):
    payload = _payload()
    path = tmp_path / "FINAL_EVAL_OWNER_AMENDMENT.json"
    digest = write_authorization(path, payload)

    observed, observed_digest = validate_authorization(path, expected=payload)
    assert observed == payload
    assert observed_digest == digest
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(HeldoutContractError, match="already exists"):
        write_authorization(path, payload)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["campaign"].update({"git_sha": "9" * 40}),
        lambda value: value["grid"].update({"sha256": "9" * 64}),
        lambda value: value["test_results"][7].update({"sha256": "9" * 64}),
        lambda value: value["selected_cells"].reverse(),
        lambda value: value["constraints"].update({"no_final_tuning": False}),
    ),
)
def test_owner_authorization_rejects_any_bound_evidence_drift(mutation):
    expected = _payload()
    observed = copy.deepcopy(expected)
    mutation(observed)
    with pytest.raises(HeldoutContractError):
        validate_authorization_payload(observed, expected=expected)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("drop", 0.019),
        ("drop", 0.031),
        ("from_fraction", 0.2),
        ("from_test_f1", float("nan")),
    ),
)
def test_owner_authorization_rejects_malformed_violation(field, value):
    payload = _payload()
    payload["grid"]["violations"][0][field] = value
    with pytest.raises(HeldoutContractError, match="monotonicity"):
        validate_authorization_payload(payload, expected=payload)


def test_owner_authorization_must_postdate_scientific_stop():
    payload = _payload()
    payload["created_utc"] = "2026-08-13T04:00:00+00:00"
    with pytest.raises(HeldoutContractError, match="predates"):
        validate_authorization_payload(payload, expected=payload)


def test_authorization_module_never_names_verified_final_resources():
    source = open(
        "src/eval/final_authorization.py", encoding="utf-8"
    ).read()
    assert "validation.csv" not in source
    assert "VH_dB.tif" not in source
    assert "VV_dB.tif" not in source
