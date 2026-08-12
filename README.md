# xView3 label-efficient dark-vessel detection

This repository compares pretrained visual encoders for vessel detection in
Sentinel-1 synthetic-aperture radar (SAR). The detector and fine-tuning recipe
stay fixed. Encoder initialization changes within two architecture tracks:

- ViT-B/16: random, SatDINO optical, SARMAE SAR, and ImageNet-1K AugReg.
- ConvNeXt-V2-Base: random, BigEarthNet Sentinel-2 optical, BigEarthNet
  Sentinel-1 SAR, and ImageNet-1K FCMAE followed by supervised fine-tuning.

Each arm uses seed `0` and four nested scene budgets: 10%, 25%, 50%, and 100%.
The matrix contains 32 core runs.

## Evidence status

The committed H100 snapshot supplies every manuscript result table and figure.
The generator accepts metrics only from verified `DONE` runs and compares a
label fraction only after all eight arms at that fraction finish. It rejects
V100 diagnostic values and metrics from running cells.

When admitted, F1 values are corrected development-selection scores. The
current snapshot contains no 16-scene test metric, and the separate 50-scene
human-verified set remains sealed. This repository therefore makes no held-out
dark-vessel-performance claim.

```bash
python -m src.analysis.h100_results generate \
  --snapshot results/h100/h100_campaign_snapshot.json \
  --output-dir docs/results/generated
```

The command generates complete label-efficiency curves from 32 verified cells
or a deadline profile containing status and closed eight-arm cohorts.

## Study controls

Every core run uses one PyTorch Lightning path, fixed `[VH, VV, VH-VV]` input,
the same CenterNet-style point detector, scene-disjoint nested subsets, and an
effective batch of 16. Each process runs on one H100 under Lightning `32-true`.
CUDA matmul TF32 and cuDNN TF32 remain disabled.

CI protects the splits, backbone parity, checkpoint-loading contract, detector
configuration, training statistics, and scorer hash.

## Repository map

```text
configs/                 experiment and detector configuration
data/                    frozen split/statistics metadata only
docs/class_report/       class report source
docs/aipr2026/           supplemental AIPR manuscript source
docs/results/generated/  generated H100 tables, macros, and figures
results/h100/            sanitized H100 snapshot and public status log
locks/                   normalized H100 and CPU/test/paper locks
src/                     data, models, training, evaluation, and analysis
tests/                   unit, contract, and anti-drift tests
tools/                   report and submission utilities
```

The repository contains no imagery, labels, weights, checkpoints, or virtual
environment. The Canvas archive excludes the whole `data/` directory.

## Environment and tests

Python 3.11 is required. The H100 campaign used the normalized CUDA 12.6 lock
in `locks/env-h100-cu126.txt`. A separate exact CPU/test/paper-support lock is
`locks/env-cpu-test-paper.txt`. Its PyTorch packages come from the official CPU
wheel index. Install Tectonic 0.17.0 separately to build the manuscripts.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r locks/env-cpu-test-paper.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
python -m pytest -q
```

The H100 lock records the campaign environment rather than a portable CPU
installation recipe. Reproduce value-sensitive checkpoint and GPU tests in a
Python 3.11/CUDA 12.6 environment that matches it. Downloaded checkpoints
retain their upstream licenses; see `THIRD_PARTY_NOTICES.md`.

## Checkpoint sourcing

Download each non-random encoder from its immutable upstream revision. Never
use a moving model alias for a reportable run. The loader requires the
following layout under `data/weights/`:

| Initialization | Revision-pinned source | Required local checkpoint |
|---|---|---|
| SatDINO | `strakajk/satdino-vit_base-16@22b7a253` | `satdino/satdino-vit_base-16.pth` |
| SARMAE | `Wenquandan777/SARMAE@88a8d768` | `sarmae/SARMAE_vitb_checkpoint-last` |
| BigEarthNet S2 | `BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0@1afe6c91` | `bigearthnet_s2/model.safetensors` |
| BigEarthNet S1 | `BIFOLD-BigEarthNetv2-0/convnextv2_base-s1-v0.2.0@a0b43b44` | `bigearthnet_s1/model.safetensors` |
| ImageNet ViT | `timm/vit_base_patch16_224.augreg_in1k@458542882691a06a8b667c6fb5fe5c9573093a81` | `imagenet_vit_augreg_in1k/model.safetensors` |
| ImageNet CNN | `timm/convnextv2_base.fcmae_ft_in1k@7b29800e499fdc06de5b612970f3384dc8d29ca5` | `imagenet_cnn_fcmae_ft_in1k/model.safetensors` |

With the Hugging Face CLI, fetch a complete pinned snapshot before selecting
the artifact named above:

```bash
hf download REPOSITORY_ID --revision REVISION --local-dir /tmp/xview3-checkpoint
```

Create `LICENSE.note` beside each local checkpoint after reviewing its model
card and recording the repository, revision, source filename, retrieval date,
license, and file SHA-256. The loader refuses a directory without this note.
The two ImageNet artifacts have an additional byte-level lock:

```text
678a1ce471be7da9822fe2508497a5bcf6da4c6802053151b232ba88a42c21a2  imagenet_vit_augreg_in1k/model.safetensors
ec152f1e375edc2b3dfac7a81155a449b4c5cbb7c5cf0b9494838f6c87518d73  imagenet_cnn_fcmae_ft_in1k/model.safetensors
```

The committed key manifests in `tests/manifests/` define structural coverage.
Run `python -m pytest tests/test_fm_checkpoints_load.py -q` with all six files
present; its value-sensitive checks reject a silently random encoder. SARMAE
is gated and noncommercial, and all other artifacts retain their upstream
terms. See `THIRD_PARTY_NOTICES.md` before retrieval or use.

## Data preparation

Obtain xView3 imagery and labels from the xView3-SAR distribution described by
Paolo et al. Place each GRD scene under `data/raw/xview3/GRD/` and source labels
under `data/raw/xview3/labels/`. Keep human-verified evaluation labels outside
a training environment.

Review `configs/data.yaml`, then inspect the public preprocessing commands:

```bash
python -m src.data.download_sarfish --help
python -m src.data.chipper --help
python -m src.data.splits --help
python -m pytest tests/test_split_disjoint.py -q
```

The three committed JSON files under `data/` preserve frozen split and
aggregate-statistics metadata. Editing them changes the study.
`data/splits.json` also retains one legacy Windows source-label string from the
original split receipt. It is immutable provenance, not an executable path or
public setup default. The Canvas archive excludes the entire `data/` tree.

## Training and evaluation

One core cell uses the shared entry point:

```bash
python -m src.runtime.train \
  --init sarmae_b \
  --label_frac 0.10 \
  --seed 0
```

The launcher exports `NVIDIA_TF32_OVERRIDE=0` before Python can initialize
CUDA, then replaces itself with a fresh child process. The child's first
action applies and asserts IEEE matmul and cuDNN behavior. Only then does it
import the shared Lightning trainer. Direct invocation of
`src.train.finetune` without that verified startup marker is rejected.

An editable install provides the equivalent `xview3-train` command.

Run `python -m src.runtime.train --help` for the initialization names. A real
run also needs the recorded strict-FP32 runtime contract and six downloaded
checkpoint files. The final 50-scene evaluator has an explicit confirmation
gate and remains unused until the study freezes its cohort and analysis code.

## Papers and class archive

```bash
cd docs/class_report
tectonic -X compile --keep-intermediates final_report.tex
python ../../tools/check_report.py final_report.pdf --aux final_report.aux

cd ../aipr2026
tectonic -X compile --keep-intermediates paper.tex
```

Use Tectonic 0.17.0. The class report limits Introduction through Conclusion to five pages.
References and appendices start on later pages. Build the deterministic,
data-free Canvas archive from the repository root:

```bash
python tools/build_submission.py
```

The tool writes `dist/Roth_John_final_project.zip`, scans its allowlisted
payload, rejects private infrastructure and prohibited file types, and records
member hashes.

The full Git checkout runs 490 public tests plus six expected weight-dependent
skips. The data-free Canvas archive omits five test modules whose only inputs
are the frozen metadata under `data/`: three immutable-hash guards and the two
H100 snapshot-import suites. Those tests remain in Git and are required for a
release. Running `python -m pytest -q` after ZIP extraction exercises the
remaining package-safe scientific and runtime tests without reconstructing or
redistributing excluded metadata.

## Contributions

John Roth and Kyle Wagner jointly designed the study, implemented the data,
model, training, and evaluation code, ran the experiments, analyzed the
results, and wrote the class report.

## License

Original project code and documentation use the MIT License. Dataset files,
labels, pretrained weights, and third-party components remain under their
upstream terms. See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
