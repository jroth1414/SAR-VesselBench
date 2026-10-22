"""A real CPU Lightning interruption reproduces the skipped DEV and zero-step repair."""
from pathlib import Path

import lightning as L
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from scripts.handoff.terminal_contract import POLICY
from src.eval.result_contract import sha256_file
from src.eval.terminal_recovery import load_restricted, validate_saved_state
from src.train.checkpoint_loading import fit_with_path_metadata
from src.train.terminal_recovery import finish_dev
from test_checkpoint_resume import equal
from test_result_contract import dev_result


class Tiny(L.LightningModule):
    optimizer_constructions = 0
    training_batches = 0
    def __init__(self):
        super().__init__()
        self.layer=torch.nn.Linear(2,1)
        self.save_hyperparameters({'weights_root':Path('/synthetic/weights')})
    def training_step(self,batch,index):
        type(self).training_batches+=1
        x,y=batch
        loss=torch.nn.functional.mse_loss(self.layer(x),y)
        self.log('train_loss',loss)
        return loss
    def configure_optimizers(self):
        type(self).optimizer_constructions+=1
        optimizer=torch.optim.AdamW(self.parameters(),lr=.01)
        return {'optimizer':optimizer,'lr_scheduler':torch.optim.lr_scheduler.StepLR(optimizer,step_size=1)}


class Data(L.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.ones(4,2),torch.ones(4,1)),batch_size=2)


class DevSceneEval(L.Callback):
    def __init__(self,path,interrupt):
        self.path=path;self.interrupt=interrupt;self.calls=0
        self.best=None;self.best_result=None;self.last_result=None
    def state_dict(self):
        return dict(best=self.best,best_result=self.best_result,last_result=self.last_result)
    def load_state_dict(self,state):
        self.best=state['best'];self.best_result=state['best_result'];self.last_result=state['last_result']
    def on_train_epoch_end(self,trainer,module):
        self.calls+=1
        if trainer.current_epoch==1 and self.interrupt:
            trainer.save_checkpoint(self.path,weights_only=False)
            raise RuntimeError('interrupted-final-DEV')
        self.best_result=self.last_result=dev_result(epoch=trainer.current_epoch)
        self.best=self.best_result['f1']
        module.log('dev_f1',self.best)


def test_real_interrupted_terminal_resume_and_zero_training_repair(tmp_path,monkeypatch):
    # Scale only the policy's epoch/step boundary for a tiny CPU reproduction.
    monkeypatch.setitem(POLICY,'epoch',1)
    monkeypatch.setitem(POLICY,'global_step',4)
    monkeypatch.setitem(POLICY,'last_completed_dev_epoch',0)
    path=tmp_path/'interrupted.ckpt'
    def trainer(callback):
        return L.Trainer(accelerator='cpu',devices=1,precision='32-true',max_epochs=2,
          logger=CSVLogger(save_dir=str(tmp_path),name='',version='metrics'),
          callbacks=[callback,ModelCheckpoint(dirpath=tmp_path/'checkpoints',filename='best',monitor='dev_f1',mode='max',every_n_epochs=1,save_on_train_epoch_end=True)],
          enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=1)
    L.seed_everything(0,workers=True)
    first=trainer(DevSceneEval(path,True))
    with pytest.raises(RuntimeError,match='interrupted-final-DEV'):
        fit_with_path_metadata(first,Tiny(),datamodule=Data(),ckpt_path=None)
    digest=sha256_file(path)
    saved=load_restricted(path)
    validate_saved_state(saved,candidate_floor=.05)
    optimizer_before=copy_training_state(saved)
    csv_path=tmp_path/'metrics/metrics.csv'
    assert csv_path.is_file()
    callback=DevSceneEval(path,False)
    resumed=trainer(callback)
    fit_with_path_metadata(resumed,Tiny(),datamodule=Data(),ckpt_path=str(path))
    assert callback.calls==0 and resumed.global_step==4
    assert 'train_loss' not in resumed.callback_metrics
    assert not csv_path.exists()
    counts=(Tiny.optimizer_constructions,Tiny.training_batches)
    module=Tiny()
    def evaluate(model):
        model.eval()
        with torch.inference_mode():assert torch.isfinite(model.layer(torch.ones(1,2))).all()
        return dev_result(epoch=1,tp=4)
    terminal=finish_dev(module,saved,evaluate,candidate_floor=.05)
    assert terminal['tp']==4
    assert (Tiny.optimizer_constructions,Tiny.training_batches)==counts
    equal(module.state_dict(),saved['state_dict'])
    equal(copy_training_state(saved),optimizer_before)
    assert sha256_file(path)==digest


def copy_training_state(saved):
    import copy
    return copy.deepcopy({key:saved[key] for key in ('optimizer_states','lr_schedulers','loops','global_step')})
