"""Focused guards for the receipts-only Sprint 7f Git exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import export_results


def _write(path: Path, value: str, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    if immutable:
        path.chmod(0o444)


def _configure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, str]:
    runs = tmp_path / "runs"
    output = tmp_path / "results"
    reference_id = "yolo26-f100"
    monkeypatch.setattr(export_results, "RUNS", runs)
    monkeypatch.setattr(export_results, "OUT", output)
    return runs, output, reference_id


def _reference_pair(runs: Path, exp_id: str) -> None:
    _write(runs / exp_id / "final_metrics.json", '{"result_schema": 2}')
    _write(
        runs / exp_id / "runtime_provenance.json",
        '{"schema": 2, "campaign_id": "fixture-v100"}',
    )


def test_export_excludes_every_experiment_path_and_exports_only_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, output, reference_id = _configure(tmp_path, monkeypatch)
    _reference_pair(runs, reference_id)
    core_id = "vitin1k-f10-s0"
    _write(runs / core_id / "final_metrics.json", '{"result_schema": 2}')
    _write(runs / core_id / "test_metrics.json", '{"result_schema": 1}')
    _write(runs / "summary/grid.csv", "exp_id,test_f1\n")
    _write(runs / "summary/label_efficiency.png", "not-an-image")
    for name in export_results._META_FILES:
        _write(runs / ".h100" / name, '{"schema": 1}', immutable=True)

    assert export_results.main() == 0
    assert export_results.main() == 0
    assert not (output / reference_id).exists()
    assert not (output / core_id).exists()
    assert not (output / "summary").exists()
    receipt = json.loads(
        (output / export_results.OWNERSHIP_RECEIPT).read_text(encoding="utf-8")
    )
    assert receipt["schema"] == 2
    assert set(receipt["files"]) == {
        f".h100/{name}" for name in export_results._META_FILES
    }
    for name in export_results._META_FILES:
        assert (output / ".h100" / name).stat().st_mode & 0o222 == 0


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("TRAINING_COHORT.json", "TRAINING_COHORT"),
        ("V100_DIAGNOSTIC_ISOLATION.json", "DIAGNOSTIC_ISOLATION"),
    ],
)
def test_export_rejects_mutable_contract_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    match: str,
) -> None:
    runs, _output, _reference_id = _configure(tmp_path, monkeypatch)
    _write(runs / ".h100" / name, "{}")

    with pytest.raises(RuntimeError, match=match):
        export_results.main()


def test_export_prunes_only_unchanged_owned_stale_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runs, output, reference_id = _configure(tmp_path, monkeypatch)
    legacy_files = {}
    for name in export_results._REFERENCE_FILES:
        path = output / reference_id / name
        _write(path, "{}")
        legacy_files[f"{reference_id}/{name}"] = export_results._sha256(path)
    _write(
        output / export_results.OWNERSHIP_RECEIPT,
        json.dumps(
            {
                "schema": 1,
                "policy": "references-and-small-provenance-only-no-core",
                "files": legacy_files,
            }
        ),
    )
    historical = output / "historical-user-result.json"
    _write(historical, '{"keep": true}')
    assert export_results.main() == 0
    assert not (output / reference_id).exists()
    assert historical.read_text(encoding="utf-8") == '{"keep": true}'


def test_export_rejects_legacy_receipt_that_targets_a_core_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _runs, output, _reference_id = _configure(tmp_path, monkeypatch)
    core_relative = "vitin1k-f10-s0/runtime_provenance.json"
    core_path = output / core_relative
    _write(core_path, "{}")
    _write(
        output / export_results.OWNERSHIP_RECEIPT,
        json.dumps(
            {
                "schema": 1,
                "policy": "references-and-small-provenance-only-no-core",
                "files": {core_relative: export_results._sha256(core_path)},
            }
        ),
    )

    with pytest.raises(RuntimeError, match="outside the exporter-owned namespace"):
        export_results.main()
    assert core_path.is_file()


def test_export_refuses_modified_owned_or_unowned_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, output, _reference_id = _configure(tmp_path, monkeypatch)
    source = runs / ".h100/TRAINING_COHORT.json"
    _write(source, "{}", immutable=True)
    assert export_results.main() == 0
    destination = output / ".h100/TRAINING_COHORT.json"
    destination.chmod(0o644)
    destination.write_text('{"user_edit": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="was modified"):
        export_results.main()
    assert json.loads(destination.read_text()) == {"user_edit": True}

    destination.unlink()
    (output / export_results.OWNERSHIP_RECEIPT).unlink()
    _write(destination, '{"unowned": true}')
    with pytest.raises(RuntimeError, match="unowned export destination"):
        export_results.main()
    assert json.loads(destination.read_text()) == {"unowned": True}
