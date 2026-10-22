"""A 30+2 cohort validates recovery receipts before any held-out access."""
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.handoff.terminal_contract import CELLS, POLICY
from src.eval.heldout_contract import (
    HeldoutContractError, create_training_cohort, validate_training_cohort,
    validate_training_cohort_cell,
)
from src.eval.result_contract import sha256_file
from src.train.terminal_recovery import write_once
from test_result_contract import dev_result
from test_terminal_completion import fixture


def mixed_matrix(tmp_path, monkeypatch):
    cells = [SimpleNamespace(exp_id=name) for name in sorted(POLICY['completed_cells'] + list(CELLS))]
    for cell in cells:
        if cell.exp_id in CELLS:
            run, marker, _receipt = fixture(tmp_path, monkeypatch,
                improved=cell.exp_id == CELLS[0], cell_name=cell.exp_id)
        else:
            run = tmp_path / cell.exp_id
            checkpoint = run / 'checkpoints/best.ckpt'
            checkpoint.parent.mkdir(parents=True)
            torch.save({'epoch': 29}, checkpoint)
            best = dev_result(epoch=29)
            marker = {'result_schema': 2, 'exp_id': cell.exp_id,
                'git_sha': POLICY['campaign_git_sha'], 'detector_sha256': 'f' * 64,
                'precision': '32-true', 'micro_batch': 16, 'gradient_accumulation': 1,
                'effective_batch': 16, 'epochs_run': 50, 'best_dev_f1': best['f1'],
                'best_dev': best, 'last_dev': dev_result(epoch=44), 'train_loss': 1.25,
                'best_checkpoint': {'relative_path': 'checkpoints/best.ckpt',
                    'sha256': sha256_file(checkpoint), 'epoch': 29}}
        write_once(run / 'final_metrics.json', marker)
    (tmp_path / '.h100').mkdir()
    return cells


def cohort_args(runs, cells):
    return dict(cells=cells, runs_root=runs, git_sha=POLICY['campaign_git_sha'],
                detector_sha256='f' * 64, candidate_floor=.05)


def test_mixed_cohort_preserves_all_marker_and_checkpoint_bytes(tmp_path, monkeypatch):
    cells = mixed_matrix(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    path = tmp_path / '.h100/TRAINING_COHORT.json'
    args = cohort_args(tmp_path, cells)
    cohort = create_training_cohort(output=path, **args)
    validated, digest = validate_training_cohort(path=path, **args)
    assert validated == cohort and cohort['cell_count'] == 32
    assert all(p.read_bytes() == value for p, value in before.items())
    assert [record['exp_id'] for record in cohort['cells']] == [cell.exp_id for cell in cells]
    for cell in CELLS:
        _cohort, sha, record, marker = validate_training_cohort_cell(
            path=path, expected_sha256=digest, exp_id=cell, **args)
        assert sha == digest and marker['result_schema'] == 3
        assert record['best_checkpoint']['relative_path'].endswith(
            'terminal-best.ckpt' if cell == CELLS[0] else 'best.ckpt')
    assert not list(tmp_path.glob('*/test_metrics.json'))


@pytest.mark.parametrize('change', ['missing-proof', 'wrong-cell', 'invented-loss', 'test-before-freeze'])
def test_mixed_cohort_rejects_unapproved_terminal_or_premature_test(tmp_path, monkeypatch, change):
    cells = mixed_matrix(tmp_path, monkeypatch)
    marker_path = tmp_path / CELLS[0] / 'final_metrics.json'
    if change == 'test-before-freeze':
        write_once(tmp_path / CELLS[0] / 'test_metrics.json', {})
    else:
        marker = json.loads(marker_path.read_text())
        if change == 'missing-proof':
            del marker['terminal_recovery']
        elif change == 'wrong-cell':
            marker['exp_id'] = 'beS1-f10-s0'
        else:
            marker['train_loss'] = 0.0
        marker_path.chmod(0o644)
        marker_path.write_text(json.dumps(marker))
        marker_path.chmod(0o444)
    path = tmp_path / '.h100/TRAINING_COHORT.json'
    with pytest.raises(HeldoutContractError):
        create_training_cohort(output=path, **cohort_args(tmp_path, cells))
    assert not path.exists()
