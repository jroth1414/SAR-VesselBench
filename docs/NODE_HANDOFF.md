# Handoff to the P100-node agent

This server has eight Tesla P100-PCIE-12GB GPUs (Pascal `sm_60`), managed by
the `gpu` lease command. All five currently free P100s (host IDs 3--7) are
locked for this work, and this container exposes them as local CUDA devices
0--4. CUDA devices are renumbered from zero inside the container; host IDs
must never be passed as container CUDA indices. The earlier version of this
handoff assumed eight 32 GB V100s
(`sm_70`); those hardware, environment, batch-fit, and runtime claims are
**superseded**. The repo of record is
`github.com/jroth1414/JHU-xView3`, integration branch **`dev`**. Read
`AGENTS.md`, the DEVPLAN cold-start runbook, and the 2026-07-22 amendment in
`docs/decisions.md` before acting. Frozen study artifacts remain binding.

## Current study and data state

- Phases 0–3 remain DONE and tagged. All frozen artifacts are committed and
  SHA-256-pinned: scorer, splits.json (150 study scenes: 111/23/16 plus 50
  eval_final IDs), stats.json, lsssdd_split.json, and detector.yaml. Do not
  regenerate or edit them; a pin mismatch is a STOP.
- P3.6 passed, and Option B remains unchanged: plan-literal epochs, early
  stopping after four real dev evaluations, with a 50-epoch safety ceiling.
- On 2026-07-22 the owner superseded the random→LS-SSDD Arms 4/8 with core
  ImageNet controls. Historical `vitsup-f10-s0` and `cnnsup-f10-s0`
  artifacts remain evidence of the old design but are excluded from current
  curves and must not be used as skip markers for the new arms.
- The six other completed f10/seed-0 cells remain valid. New Arm-4/8 runs use
  init IDs `vit_imagenet` / `cnn_imagenet` and prefixes `vitin1k` /
  `cnnin1k`; the queue's skip-if-finished check discovers the exact
  remaining cells.
- The active matrix has one seed: 32 core cells plus R2 and R3 = 34 total.
  There is no seed-rerun tranche and no separate R1 or LS-SSDD training job.
- The transfer payload is already unpacked under this workspace: 150 chip
  scene directories, 39 dev/test raster scenes, labels, weights, and runs.
  The 50 eval-final rasters are absent by design and remain untouched.

## Your job

First pass the P100 environment, weight, and micro-batch gates below. Then run
only the remaining seed-0 core cells on GPUs explicitly leased to this job.
The runner has no implicit all-GPU default and no pretraining dependency.
Never pass an unreserved device ID. Do not launch seed 1/2 runs.

## Historical transfer record (superseded acquisition instructions)

The 2026-07-13 V100 handoff recorded a Box transfer of 150 per-scene chip
archives, 39 dev/test raster archives, runs, weights, labels, and a manifest.
That transfer has already been unpacked into this workspace and was verified
against its manifest. Its separate LS-SSDD archive and the instructions to run
two LS-SSDD pretrainings are historical only; the revised matrix does not read
LS-SSDD. Do not redownload, regenerate, or unpack data merely to follow the old
handoff. If fast node-local scratch is available, the unpacked chips and
dev/test rasters may be staged byte-for-byte, but frozen JSON files always
come from git and are never regenerated.

## New ImageNet checkpoint gate

Both revised weights must exist under the exact loader paths and match the
pinned bytes:

| Arm | Local file | HF revision | SHA-256 |
|---|---|---|---|
| 4, `vit_imagenet` | `data/weights/imagenet_vit_augreg_in1k/model.safetensors` | `458542882691a06a8b667c6fb5fe5c9573093a81` | `678a1ce471be7da9822fe2508497a5bcf6da4c6802053151b232ba88a42c21a2` |
| 8, `cnn_imagenet` | `data/weights/imagenet_cnn_fcmae_ft_in1k/model.safetensors` | `7b29800e499fdc06de5b612970f3384dc8d29ca5` | `ec152f1e375edc2b3dfac7a81155a449b4c5cbb7c5cf0b9494838f6c87518d73` |

The corresponding `LICENSE.note` and `SOURCE.note` files are mandatory.
The old `data/weights/imagenet_cnn/convnextv2_base_fcmae_ft_in22k_in1k.pt`
belongs to superseded optional R1 and is **not** Arm 8.

## Verified P100 execution and reservation state

1. Work from the current `dev` branch and a clean, reviewed commit. Use the
   verified P100 environment recorded in `locks/env-p100node.txt`; its PyTorch
   build has been checked on this node and reports `sm_60` in
   `torch.cuda.get_arch_list()`.
2. Inspect both the host lease state and the devices attached inside the
   container before selecting CUDA indices:

   ```bash
   gpu help
   gpu info
   gpu smi
   nvidia-smi -L
   ```

   `gpu info` reports physical/host IDs. `nvidia-smi -L` inside the attached
   container reports the only indices training commands may use, renumbered
   contiguously from zero. The locked physical devices are host IDs 3--7, but
   do **not** put `3 4 5 6 7` into `CUDA_VISIBLE_DEVICES`; this five-card
   attachment is local indices 0--4. Never use a device merely because it
   appears free in a host-level listing.

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/gpu_sanity.py
   ```

   Sanity must report a Tesla P100, `sm_60`, finite fp16 matmul, and a usable
   non-Flash SDPA path.
3. Verify files and guards before training:

   ```bash
   test "$(find data/chips -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 150
   test "$(find data/raw/xview3/GRD -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 39
   sha256sum data/weights/imagenet_vit_augreg_in1k/model.safetensors
   sha256sum data/weights/imagenet_cnn_fcmae_ft_in1k/model.safetensors
   pytest tests/ -q
   ```

   Every guard, including all six structural/value-sensitive downloaded-weight
   cases, must pass. A failure is a STOP; do not weaken a guard.
4. The P100 smoke gate has passed with micro-batch 8 and gradient accumulation
   2, preserving effective batch 16. The environment and batch-fit checks are
   therefore complete; repeat the smoke only if the environment or recipe
   changes.
5. Run and score `vitin1k-f10-s0` and `cnnin1k-f10-s0` first. Run these exact
   commands in two terminals; `0` and `1` are container-local CUDA indices:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m src.train.finetune \
     --init vit_imagenet --label_frac 0.1 --seed 0 --micro-batch 8

   CUDA_VISIBLE_DEVICES=1 python -m src.train.finetune \
     --init cnn_imagenet --label_frac 0.1 --seed 0 --micro-batch 8
   ```

   Do not substitute the historical `vitsup-f10-s0` / `cnnsup-f10-s0`
   markers.
6. Only after those two revised f10 cells complete, launch the resumable core
   queue using every container-local device actually attached. For the
   current five-card attachment, use:

   ```bash
   python scripts/run_grid_node.py --gpus 0 1 2 3 4 --micro-batch 8
   ```

   It runs seed 0 only and skips valid completed cells. Logs live in
   `runs/logs/node/`. When the queue and any scoring work are finished,
   return the leases with `gpu release`.

## Known gotchas you would otherwise rediscover

- Dev evals read whole scene rasters into RAM (~5 GB per scene per worker) —
  `infer_scene.py` is written for that; do not "optimize" it back to
  windowed reads (100× slower on striped GeoTIFFs).
- `data/*.json` freeze pins assume LF bytes; the repo's `.gitattributes`
  handles it — do not re-encode files.
- The former claim that recipe batch 16 fits directly referred to 32 GB
  V100s. On these 12 GB P100s, pass `--micro-batch 8`; the verified gradient
  accumulation of 2 keeps the effective recipe batch at 16. Do not change the
  effective batch.
- The `source` column in label CSVs is lowercase; `infer_scene.py`
  normalizes it for the frozen scorer. Dark-vessel GT exists ONLY in
  eval_final scenes — dark recall is expected to be 0-support on dev/test.

## Reporting back

After each wave, run `python -m src.analysis.curves` and export the summary
through the repository's results-export workflow; never commit `runs/` or
checkpoints. Watch `monotonicity_ok`: a False is a STOP (DEVPLAN §1b sanity
rule), not a thing to investigate quietly. With one seed, report curves as
point estimates and make no variance or statistical-significance claim.
Test-split scoring (`scripts/score_test_split.py`) may run on a reserved idle
GPU between waves; an unleased GPU is not "free" for this job.
