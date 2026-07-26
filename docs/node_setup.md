# Historical P100-node setup — superseded by V100 fp32 campaign

> **SUPERSEDED AGAIN (2026-07-26).** The active host exposes eight
> V100-SXM2-32GB cards and the core grid now uses shared `32-true`. Follow the
> README and DEVPLAN cold-start runbook. Everything below is retained only as
> evidence for the retired P100/5070 decisions; do not execute it.


At the time, the hardware was eight Tesla P100-PCIE-12GB GPUs (Pascal `sm_60`), not
the eight 32 GB V100s assumed by the historical handoff. GPU access is leased;
availability is dynamic. Order matters.

> **RESOLVED BY HARDWARE MOVE (2026-07-22).** The exact f10 commands below
> were measured and interrupted in epoch 0: projected 50-epoch training-only
> wall time was 20.50 h for ViT and 97.36 h for CNN. The owner moved all
> remaining unchanged one-GPU jobs to the RTX 5070 Ti. Do not reserve P100s or
> rerun the probe commands; that retired transfer decision is historical only.

## 1. Repo and environment compatibility gate

Work from the reviewed commit. `locks/env-p100node.txt` is the verified P100
freeze; the V100 lock is now active, but none of the commands below are:

```bash
python -m venv .venv-p100
.venv-p100/bin/pip install -r locks/env-p100node.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126
.venv-p100/bin/pip install --no-deps -e .
.venv-p100/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
```

The printed architecture list must include `sm_60`. A missing CUDA runtime,
missing `sm_60`, or failed import is a STOP; do not guess a replacement
training environment or silently change the frozen precision recipe.

## 2. Historical data and checkpoint snapshot

The transfer is already unpacked in the repository layout:

| What | Path | Required for |
|---|---|---|
| Chips + sidecars | `data/chips/` (150 scene dirs) | core training |
| Dev/test rasters | `data/raw/xview3/GRD/` (39 scene dirs) | dev/test scoring |
| Labels | `data/raw/xview3/labels/` | scoring |
| Downloaded weights | `data/weights/` | Arms 2, 3, 4, 6, 7, 8 |
| Finished runs | `runs/` | resumable skip markers/checkpoints |

LS-SSDD is no longer an active input or job. The frozen
`data/lsssdd_split.json` remains pinned historical evidence and must not be
deleted or regenerated. Eval-final rasters are absent and remain untouched.

The revised ImageNet files are:

- Arm 4, `vit_imagenet` / `vitin1k`:
  `data/weights/imagenet_vit_augreg_in1k/model.safetensors`, HF revision
  `458542882691a06a8b667c6fb5fe5c9573093a81`, SHA-256
  `678a1ce471be7da9822fe2508497a5bcf6da4c6802053151b232ba88a42c21a2`.
- Arm 8, `cnn_imagenet` / `cnnin1k`:
  `data/weights/imagenet_cnn_fcmae_ft_in1k/model.safetensors`, HF revision
  `7b29800e499fdc06de5b612970f3384dc8d29ca5`, SHA-256
  `ec152f1e375edc2b3dfac7a81155a449b4c5cbb7c5cf0b9494838f6c87518d73`.

The historical `vitsup-f10-s0` and `cnnsup-f10-s0` runs are superseded;
preserve them, but never treat them as revised Arm-4/8 results.

## 3. Reserve explicit GPUs and verify guards

```bash
gpu info
gpu get --lock 5  # only after BLOCKER-7 resolves; recheck available COUNT
gpu info
nvidia-smi -L
```

`gpu info` reports physical host IDs; attached GPUs are renumbered inside the
container. Use only the local indices printed by `nvidia-smi -L`. With five
attached cards they are 0–4:

```bash
CUDA_GPUS=(0 1 2 3 4)
CUDA_VISIBLE_DEVICES="${CUDA_GPUS[0]}" python scripts/gpu_sanity.py
sha256sum data/weights/imagenet_vit_augreg_in1k/model.safetensors
sha256sum data/weights/imagenet_cnn_fcmae_ft_in1k/model.safetensors
pytest tests/ -q
```

Sanity must identify a P100/`sm_60`, produce finite fp16 matmul and SDPA
outputs, and all tests must pass, including the six downloaded-weight cases.
Never substitute a physical host ID for a container-local CUDA index.

## 4. P100 micro-batch gate — passed 2026-07-22

Both exact checkpoints completed finite fp16-autocast forward/backward/AdamW
steps at micro-batch 8. Peak allocated/reserved memory was 3.796/4.078 GiB
for ViT and 10.193/10.523 GiB for ConvNeXt-V2. Real cells use micro-batch 8
plus accumulation 2, preserving effective batch 16.

## 5. Measured f10 commands — historical, do not rerun

These exact commands produced the throughput record. The path is retired and
the commands must not be rerun:

```bash
CUDA_VISIBLE_DEVICES="${CUDA_GPUS[0]}" \
  python -m src.train.finetune \
    --init vit_imagenet --label_frac 0.1 --seed 0 --micro-batch 8 &
PID_VIT=$!

CUDA_VISIBLE_DEVICES="${CUDA_GPUS[1]}" \
  python -m src.train.finetune \
    --init cnn_imagenet --label_frac 0.1 --seed 0 --micro-batch 8 &
PID_CNN=$!

wait "$PID_VIT" "$PID_CNN"
```

The future reportable IDs remain `vitin1k-f10-s0` and
`cnnin1k-f10-s0`. The probes produced no checkpoint or completion marker;
do not score them. After approved full runs finish, use
`scripts/score_test_split.py` without changing the frozen threshold protocol.

## 6. Historical remaining-matrix instructions

The active design has one seed: 32 core cells plus R2 YOLO26 and R3
LocateAnything = **34 total runs**. There are no LS-SSDD pretraining jobs,
seed reruns, or separate ImageNet R1. The command below is blocked until the
owner resolves BLOCKER-7; after approval, use only container-local IDs
explicitly reserved for the approved execution path:

```bash
python scripts/run_grid_node.py \
  --gpus "${CUDA_GPUS[@]}" --micro-batch 8
```

The queue skips valid completed cells and writes logs under
`runs/logs/node/`. Report one-seed curves as point estimates; do not make
variance or statistical-significance claims. Release leases after all queued
and scoring processes finish:

```bash
gpu release
```
