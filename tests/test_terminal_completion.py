"""Terminal-only selection, immutable publication, and the narrow schema-3 exception."""
import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.handoff import terminal_contract as policy
from src.eval.result_contract import ResultContractError, sha256_file, validate_completion_payload
from src.eval.terminal_recovery import LOSS_REASON, select_terminal, validate_receipt_payload
from src.train.terminal_recovery import copy_once, write_once
from test_result_contract import dev_result


def fixture(tmp_path, monkeypatch, improved=True, cell_name="beS2-f10-s0"):
    run = tmp_path / cell_name
    (run / 'checkpoints').mkdir(parents=True)
    prior, last = dev_result(epoch=29), dev_result(epoch=44)
    terminal = dev_result(epoch=49, tp=4 if improved else 2)
    best = run / 'checkpoints/best.ckpt'
    torch.save({'epoch':29}, best)
    saved = {'epoch':49,'global_step':35050,
      'loops':{'fit_loop':{'epoch_progress':{x:dict(ready=50,started=50,processed=50,completed=49) for x in ('current','total')},
       'epoch_loop.batch_progress':{'current':dict(ready=701,started=701,processed=701,completed=701),'total':dict(ready=35050,started=35050,processed=35050,completed=35050),'is_last_batch':True}}},
      'callbacks':{'DevSceneEval':{'best':prior['f1'],'best_result':prior,'last_result':last},
       'ModelCheckpoint{fixture}':{'best_model_score':torch.tensor(prior['f1']),'best_model_path':str(best)}},
      'state_dict':{'weight':torch.ones(1)},'optimizer_states':[{'fixture':True}],'lr_schedulers':[{'last_epoch':49}],
      'hyper_parameters':{'weights_root':Path('/synthetic/weights')}}
    checkpoint = run / 'checkpoints/last.ckpt'
    torch.save(saved, checkpoint)
    digest = sha256_file(checkpoint)
    monkeypatch.setitem(policy.CHECKPOINT_SHA256, run.name, digest)
    from scripts.h100 import terminal_recovery as control
    monkeypatch.setattr(control, 'validate_receipt_authorization', lambda run, receipt: None)
    old = {'relative_path':'checkpoints/best.ckpt','sha256':sha256_file(best),'epoch':29}
    selected = {'relative_path':'checkpoints/terminal-best.ckpt','sha256':digest,'epoch':49} if improved else old
    if improved:copy_once(checkpoint,run/'checkpoints/terminal-best.ckpt',digest)
    execution={'campaign_git_sha':policy.POLICY['campaign_git_sha'],'git_sha':'a'*40,
      'terminal_package_id':'xview3-h100-runtime-'+'a'*40+'-'+'b'*64,
      'terminal_manifest_sha256':'c'*64,'source_receipt_sha256':'d'*64}
    receipt={'schema':1,'status':'terminal-dev-recovered','exp_id':run.name,
      'campaign_git_sha':policy.POLICY['campaign_git_sha'],
      'input_checkpoint':{'relative_path':'checkpoints/last.ckpt','sha256':digest,'epoch':49},
      'input_global_step':35050,'output_global_step':35050,'added_training_batches':0,
      'added_optimizer_steps':0,'added_scheduler_steps':0,'last_dev_before':last,
      'prior_best_dev':prior,'prior_best_checkpoint':old,'terminal_dev':terminal,
      'selected_dev':terminal if improved else prior,'selected_checkpoint':selected,
      'terminal_execution':execution,'activation_sha256':'e'*64,'loss_unavailable_reason':LOSS_REASON,
      'metrics_csv_absent':True,'terminal_model_unchanged':True,'runtime_seconds':1.0}
    marker={'result_schema':3,'exp_id':run.name,'git_sha':policy.POLICY['campaign_git_sha'],
      'detector_sha256':'f'*64,'precision':'32-true','micro_batch':16,'gradient_accumulation':1,
      'effective_batch':16,'epochs_run':50,'best_dev_f1':receipt['selected_dev']['f1'],
      'best_dev':receipt['selected_dev'],'best_checkpoint':selected,'last_dev':terminal,'train_loss':None}
    from test_h100_runtime import h100_runtime_contract
    marker['h100_runtime_contract']=h100_runtime_contract()
    write_once(run/'terminal_recovery.json',receipt)
    marker['terminal_recovery']={'relative_path':'terminal_recovery.json','sha256':sha256_file(run/'terminal_recovery.json')}
    return run, marker, receipt


@pytest.mark.parametrize('improved',[True,False])
def test_only_approved_terminal_record_allows_unavailable_loss(tmp_path,monkeypatch,improved):
    run, marker, receipt=fixture(tmp_path,monkeypatch,improved)
    before={p.name:sha256_file(p) for p in (run/'checkpoints').iterdir()}
    original_safe=set(torch.serialization.get_safe_globals())
    selected=validate_completion_payload(marker,run_dir=run,candidate_floor=.05)
    assert selected.name==('terminal-best.ckpt' if improved else 'best.ckpt')
    assert {p.name:sha256_file(p) for p in (run/'checkpoints').iterdir()}==before
    assert not (run/'metrics/metrics.csv').exists()
    assert set(torch.serialization.get_safe_globals())==original_safe


@pytest.mark.parametrize('field,value',[
 ('exp_id','beS2-f25-s0'),('output_global_step',35051),('added_optimizer_steps',1),
 ('added_training_batches',True),('metrics_csv_absent',False),('terminal_model_unchanged',False),
 ('loss_unavailable_reason','substitute-loss'),('runtime_seconds',float('nan')),
])
def test_terminal_receipt_rejects_scope_drift(tmp_path,monkeypatch,field,value):
    _run, marker, receipt=fixture(tmp_path,monkeypatch)
    receipt[field]=value
    with pytest.raises(ResultContractError):validate_receipt_payload(receipt,marker,candidate_floor=.05)


def test_receipt_cannot_hide_changed_original_checkpoint(tmp_path,monkeypatch):
    run, marker,_receipt=fixture(tmp_path,monkeypatch)
    torch.save({'epoch':49},run/'checkpoints/last.ckpt')
    with pytest.raises(ResultContractError,match='preserved checkpoint bytes'):
        validate_completion_payload(marker,run_dir=run,candidate_floor=.05)


def test_schema2_still_rejects_null_loss(tmp_path):
    from test_result_contract import valid_marker
    run,_checkpoint,marker=valid_marker(tmp_path)
    marker['train_loss']=None
    with pytest.raises(ResultContractError,match='finite number'):
        validate_completion_payload(marker,run_dir=run,candidate_floor=.05)


def test_selection_tie_keeps_old_checkpoint_and_rounding_drift_stops():
    prior=dev_result(epoch=29)
    assert not select_terminal(prior,{**prior,'epoch':49})
    with pytest.raises(ResultContractError,match='rounding'):
        select_terminal(prior,{**prior,'f1':prior['f1']+1e-12})


def test_publication_retries_exact_bytes_only(tmp_path):
    path=tmp_path/'receipt.json'
    write_once(path,{'a':1});before=path.read_bytes()
    write_once(path,{'a':1});assert path.read_bytes()==before
    with pytest.raises(ResultContractError):write_once(path,{'a':2})
    source=tmp_path/'source';source.write_bytes(b'checkpoint')
    destination=tmp_path/'selected';digest=sha256_file(source)
    copy_once(source,destination,digest);copy_once(source,destination,digest)
    assert source.read_bytes()==destination.read_bytes()


def test_signal_after_publication_can_retry_without_writable_evidence(tmp_path,monkeypatch):
    import os
    from src.train.terminal_recovery import Preempted
    target=tmp_path/'receipt.json'
    original=os.link
    def interrupt_after_link(source,destination):
        original(source,destination)
        raise Preempted()
    with monkeypatch.context() as patch:
        patch.setattr(os,'link',interrupt_after_link)
        with pytest.raises(Preempted):write_once(target,{'a':1})
    assert target.is_file() and not target.stat().st_mode & 0o222
    write_once(target,{'a':1})
    assert list(tmp_path.glob('.terminal-json-*'))==[]


def test_schema3_cannot_bypass_immutable_authorization(tmp_path,monkeypatch):
    from scripts.h100 import terminal_recovery as control
    real=control.validate_receipt_authorization
    run,marker,_receipt=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(control,'validate_receipt_authorization',real)
    with pytest.raises(ResultContractError,match='authorization'):
        validate_completion_payload(marker,run_dir=run,candidate_floor=.05)


def test_schema3_requires_runtime_evidence(tmp_path,monkeypatch):
    run,marker,_receipt=fixture(tmp_path,monkeypatch)
    del marker['h100_runtime_contract']
    with pytest.raises(ResultContractError):
        validate_completion_payload(marker,run_dir=run,candidate_floor=.05)


@pytest.mark.parametrize('change', ['unfinished-batch', 'unfinished-epoch', 'stale-dev', 'wrong-monitor', 'missing-optimizer'])
def test_saved_state_rejects_nonterminal_or_inconsistent_selection(tmp_path, monkeypatch, change):
    from src.eval.terminal_recovery import load_restricted, validate_saved_state
    run, _marker, _receipt = fixture(tmp_path, monkeypatch)
    saved = load_restricted(run / 'checkpoints/last.ckpt')
    if change == 'unfinished-batch':
        saved['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed'] = 700
    elif change == 'unfinished-epoch':
        saved['loops']['fit_loop']['epoch_progress']['current']['processed'] = 49
    elif change == 'stale-dev':
        saved['callbacks']['DevSceneEval']['last_result']['epoch'] = 39
    elif change == 'wrong-monitor':
        saved['callbacks']['ModelCheckpoint{fixture}']['best_model_score'] = torch.tensor(.9)
    else:
        saved['optimizer_states'] = []
    with pytest.raises(ResultContractError):
        validate_saved_state(saved, candidate_floor=.05)
