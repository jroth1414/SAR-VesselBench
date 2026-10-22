# Label-efficient dark-vessel detection in SAR

Code and result evidence for:

> John Roth, Kyle Wagner, and Liv d'Aliberti. *Label-Efficient Dark-Vessel
> Detection in SAR: Does SAR-Domain Pretraining Outperform Optical and ImageNet
> Transfer across ViT and CNN?* 54th Applied Imagery Pattern Recognition
> Workshop (AIPR 2026), Springer LNCS (to appear).

**Trained checkpoints:** [huggingface.co/roth1414/SAR-VesselBench](https://huggingface.co/roth1414/SAR-VesselBench)
(all 32 models reported in the paper).
**Poster:** [`docs/poster/poster_v2.pdf`](docs/poster/poster_v2.pdf).

## What the study does

We fine-tune one point-based vessel detector on Sentinel-1 SAR imagery from
[xView3-SAR](https://iuu.xview.us/) and change only how the image encoder is
initialized. Two backbone tracks, four initializations each:

| Track | Random | Optical | SAR | ImageNet |
|---|---|---|---|---|
| ViT-B/16 | `vit_random` | `satdino_b` (SatDINO) | `sarmae_b` (SARMAE) | `vit_imagenet` (AugReg) |
| ConvNeXt-V2-Base | `cnn_random` | `bigearthnet_s2` (BigEarthNet-v2 S2) | `bigearthnet_s1` (BigEarthNet-v2 S1) | `cnn_imagenet` (FCMAE + supervised) |

Each arm is trained on four nested subsets of the 111 training scenes
(12, 28, 56 and 111 scenes; `--label_frac` 0.1, 0.25, 0.5, 1.0) with seed 0:
32 models in total, all trained on H100 GPUs in strict FP32 (TF32 disabled).
Each model's checkpoint and threshold are chosen on 8 development scenes; the
model is then scored once on a 16-scene test split and once on the 50
human-verified scenes of the xView3-SAR validation partition.

## Repository layout

```text
src/models/      backbones, input adaptation (incl. the BigEarthNet-S2 stem), heatmap head
src/data/        downloads, scene splits, chipping, datasets and transforms
src/train/       fine-tuning entry point, loss, sampler, resume handling
src/eval/        frozen scorer, decoding, thresholds, scene inference, held-out contracts
src/analysis/    evidence validation, result macros, figures, bootstrap intervals
src/references/  external reference detectors (YOLO26, LocateAnything zero-shot)
scripts/         evidence staging, TEST scoring, label profiling, and the cluster
                 execution modules that the training and evaluation code imports
configs/         arms, data paths, detector recipe (frozen)
data/            fixed scene splits and normalization statistics (no imagery)
locks/           exact package locks; the H100 environment was built from env-v100node.txt
environment/     CPU lock used by continuous integration
results/h100/    evidence, generated numbers, code lineage, log excerpts
tests/           unit and contract tests (CI runs the science and evidence tests on Linux)
docs/poster/     the AIPR 2026 poster
```

## Setup

Python 3.11. Install PyTorch from the index that matches your CUDA version
first (the training runs used torch 2.11.0+cu126; exact versions of every
package are in `locks/env-v100node.txt`), then install this package:

```bash
pip install -r locks/env-v100node.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

For CPU-only tests and result generation, use `environment/env-cpu-test.txt`
with `--extra-index-url https://download.pytorch.org/whl/cpu`.

xView3-SAR is not redistributed here. Download it from
[iuu.xview.us](https://iuu.xview.us/) under its license (CC BY-NC-SA 4.0) and
place it as described in `configs/data.yaml`, then cut the training chips with
`python -m src.data.chipper` (see `--help`). Pretrained encoder weights go
under `data/weights/`; the loaders in `src/models/init_loaders.py` refuse to
load a checkpoint without its `LICENSE.note`.

## Reproducing the results

Train one model (one cell of the 8 × 4 grid) from a git checkout of this
repository, on a single GPU:

```bash
python -m src.train.finetune --init sarmae_b --label_frac 0.1 --seed 0
```

Every reported number comes from `results/h100/evidence/`. To validate the
evidence and regenerate the numbers file
(`results/h100/generated/heldout_results.tex`) on a CPU:

```bash
python -m src.analysis.heldout_results --output-dir results/h100/generated
```

The generator refuses to write anything unless every completion marker,
checkpoint hash, threshold and TEST/verified result binds to the frozen cohort
in `results/h100/evidence/TRAINING_COHORT.json`. Each
`final_verified_metrics.json` also stores the model's thresholded predictions
on the 50 verified scenes, so the verified-set scores can be re-checked against
the public xView3-SAR validation labels with the frozen scorer
(`rescore_verified` in `src/analysis/poster_coastal.py`).

## Code provenance

The training and evaluation code in this repository is the code the cluster
ran; the protected paths are unedited. At every stage the cluster pinned git
tree ids for those paths. `results/h100/provenance/CODE_LINEAGE.json` lists
them, and you can check this tree against them, for example:

```bash
git rev-parse HEAD:src/models   # 1fd2f60db4d3b95e5c51c77438d2b3727398d394
```

- `src/models`, `src/data`, `configs`, `data` and `locks` match every stage,
  from training launch through TEST scoring.
- All protected paths, including `src/train` and `src/eval`, match the
  terminal-recovery and TEST-scoring stages. At training launch, `src/train` and
  `src/eval` did not yet contain the later checkpoint-resume and recovery code;
  that code does not change the model, loss, data pipeline or optimizer.
- Cluster job tooling (scheduler scripts, environment builders, data transfer)
  is left out, except the modules under `scripts/` that the training and
  evaluation code imports.
- The code of the final verified-evaluation stage (commit `eabe977`) was not
  recovered. The verified scores can still be checked from the stored
  predictions, as described above.

Private cluster paths in the evidence and the log excerpts are replaced by
`/cluster-*-redacted`; `results/h100/evidence/REDACTIONS.json` records the
SHA-256 of every original file.

## License

Code: MIT (see `LICENSE`). Each checkpoint follows the license of its
pretrained encoder, and some are non-commercial (for example SARMAE,
CC BY-NC 4.0); see the Hugging Face model card. xView3-SAR data:
CC BY-NC-SA 4.0.

## Citation

See [`CITATION.cff`](CITATION.cff).
