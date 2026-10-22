"""CPU-only synthetic proof: reproduce the failed restore and test scoped Path loading."""
import hashlib
import json
import pickle
import pytest
from src.train.checkpoint_loading import fit_with_path_metadata
from pathlib import Path, PosixPath
import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

class TinyModule(L.LightningModule):
    def __init__(self, weights_root=Path('/synthetic/weights')):
        super().__init__()
        self.save_hyperparameters()
        self.layer=torch.nn.Linear(2,1)
    def training_step(self,batch,batch_idx):
        x,y=batch
        return torch.nn.functional.mse_loss(self.layer(x),y)
    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),lr=0.01)
        return {'optimizer':opt,'lr_scheduler':{'scheduler':torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=0.8),'interval':'step'}}

class TinyData(L.LightningDataModule):
    def __init__(self,chips_root=Path('/synthetic/chips')):
        super().__init__()
        self.save_hyperparameters()
    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.arange(16,dtype=torch.float32).reshape(8,2)/16,torch.ones(8,1)),batch_size=2,num_workers=0)

class Counter(L.Callback):
    def __init__(self): self.count=0
    def on_train_batch_end(self,*args,**kwargs): self.count+=1
    def state_dict(self): return {'count':self.count}
    def load_state_dict(self,state): self.count=state['count']

def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b,(a,b)

class VerifyRestore(L.Callback):
    def __init__(self,saved,counter): self.saved=saved; self.counter=counter; self.checked=False
    def on_train_start(self,trainer,module):
        assert trainer.global_step==self.saved['global_step']==2
        equal(module.state_dict(),self.saved['state_dict'])
        equal(trainer.optimizers[0].state_dict(),self.saved['optimizer_states'][0])
        equal(trainer.lr_scheduler_configs[0].scheduler.state_dict(),self.saved['lr_schedulers'][0])
        assert self.counter.count==2
        self.checked=True

def trainer(path,steps,callbacks):
    return L.Trainer(accelerator='cpu',devices=1,precision='32-true',max_epochs=2,max_steps=steps,
        default_root_dir=str(path),logger=False,enable_checkpointing=False,
        enable_model_summary=False,enable_progress_bar=False,callbacks=callbacks,num_sanity_val_steps=0)

def test_restricted_lightning_resume_restores_full_state(tmp_path):
    L.seed_everything(0, workers=True)
    root = tmp_path
    first = trainer(root, 2, [Counter()])
    fit_with_path_metadata(first, TinyModule(), datamodule=TinyData(), ckpt_path=None)
    checkpoint = root / "fixture.ckpt"
    first.save_checkpoint(checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint) == ["pathlib.PosixPath"]
    with pytest.raises(pickle.UnpicklingError, match="pathlib.PosixPath"):
        trainer(root, 3, [Counter()]).fit(TinyModule(), datamodule=TinyData(), ckpt_path=str(checkpoint))
    prior_safe = set(torch.serialization.get_safe_globals())
    with torch.serialization.safe_globals([PosixPath]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    counter = Counter()
    verify = VerifyRestore(saved, counter)
    resumed = trainer(root, 3, [counter, verify])
    fit_with_path_metadata(resumed, TinyModule(), datamodule=TinyData(), ckpt_path=str(checkpoint))
    assert verify.checked and resumed.global_step == 3 and counter.count == 3
    assert set(torch.serialization.get_safe_globals()) == prior_safe
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before


def test_restore_rejects_other_globals_and_cleans_scope(tmp_path):
    checkpoint = tmp_path / "unexpected.ckpt"
    torch.save({"unexpected": TinyData()}, checkpoint)
    prior_safe = set(torch.serialization.get_safe_globals())
    with pytest.raises(pickle.UnpicklingError, match="TinyData"):
        fit_with_path_metadata(trainer(tmp_path, 1, []), TinyModule(),
                               datamodule=TinyData(), ckpt_path=str(checkpoint))
    assert set(torch.serialization.get_safe_globals()) == prior_safe
