# Documented implementation decisions (per DEVPLAN P3.6 / ground rule 12)

Chronological log of judgment calls the plan left unspecified (or where a
cited mechanism did not cover the actual case), each bounded and applied
identically to every arm. This committed log lives here; `runs/` is gitignored and reserved for execution artifacts.

## sprint-1 (data)

- **xView3 rasters arrive already in dB** (`VH_dB.tif`, float32, nodata
  −32768). The P1.3 sketch's `log10(clip(raw,1,None))` assumed linear
  amplitude; applying it would double-log. Chips store the product's dB
  values as float16, NaN for nodata. (chipper.py docstring)
- **Scene stratification variables come from the label CSV** (mean label
  lat/lon per scene as the region-bin input; shoreline = any label < 5 km).
  Avoids extracting 554 rasters to read georeferences; coarse 6-bin k-means
  is insensitive to the difference. (splits.py)
- **LS-SSDD internal split is a seeded 90/10 over all 9,000 sub-images**,
  not the dataset's official 6,000/3,000 benchmark partition: LS-SSDD is a
  pretraining source only (never evaluated), and the val side exists only
  for early stopping. (DEVPLAN BLOCKER-5 note; lsssdd_split pin docstring)

## Grid budget (human decision, 2026-07-06)

- **Option B — plan-literal epochs with early stopping** (owner's call, per
  the professor's guidance: no prescribed stopping epoch; train until the
  validation signal stops improving). Interpretation: the frozen
  detector.yaml already encodes this — early stopping on dev F1 (the
  study's metric) with patience 4 dev evals; `epochs: 50` acts as a safety
  ceiling, not a stopping point. No amendment, no re-pin. Consequence
  accepted: grid cost is measured-not-fixed (~500–1,100 GPU-h depending on
  where stopping fires). The later P100 throughput gate rejected that node for
  the active grid, and the owner assigned the remaining unchanged one-GPU jobs
  to the validated RTX 5070 Ti. The then-current historical V100 forecast was retired.

## Threshold-transfer fragility (measured 2026-07-11)

- beS1-f10-s0's frozen dev threshold (0.817) collapsed on the test scenes:
  test F1 0.305 (P 0.96 / R 0.18) at the frozen operating point vs 0.839 at
  the test-optimal threshold (0.255) — the detector itself matches its dev
  quality; the confidence SCALE shifted between scene sets. Uniform across
  all 16 test scenes (global calibration offset, not per-scene failure).
  Plausibly linked to the earliest-stopping cell (epoch 25, least-settled
  confidences). The other seven f10 cells transferred within ~0.01-0.07.
- Protocol unchanged (P2.2b frozen-dev-threshold is the plan's own and the
  0.305 is the reportable number), but: (a) P7 analysis must include a
  threshold-sensitivity slice (frozen vs oracle gap per cell); (b) compare
  calibration behavior across fractions and arms; and (c) report both tiers
  where the gap is material. With seed 0 only, do not claim that this
  fragility is systematic or estimate its seed variance.

## Geographic revisit overlap (owner-raised, measured 2026-07-09)

- Scene-level splits do not imply geographic disjointness in xView3:
  measured over the frozen split, 22/23 dev scenes have a train scene
  centered within 32 km (Sentinel-1 revisits — same swath, different date),
  and even the official human-verified validation partition shares
  geography (42/50 scenes within 50 km of a train scene; max 172 km). This
  is inherent to the dataset (AOIs over fishing hot-spots) and to the
  organizers' own temporal train/validation boundary — the official
  challenge was evaluated under the same overlap.
- Consequences: (a) absolute F1 on dev/test is an IN-REGION estimate,
  inflated by background familiarity, and must be labeled as such in the
  writeup; (b) the controlled comparison is unaffected — identical
  familiarity for every arm; (c) no re-split: geo-disjoint splits are not
  achievable within xView3's train partition and would break comparability
  with official practice. Chip-level leakage remains impossible by
  construction (scene-level frozen splits + guards).
- Additional absolute-number context: dev/test positives are
  MEDIUM-confidence AIS vessels (LOW ignored, no dark vessels — those exist
  only in eval_final), and dev F1 is reported at the dev-selected threshold
  (its max by construction). Test F1 at the frozen threshold and the
  once-only eval_final numbers are the honest reporting tiers.

## Arms 4/8 capacity caveat (owner-raised, 2026-07-09)

- LS-SSDD cannot "saturate" an 86-89M backbone and is not expected to: ~6,000
  ship instances over only 15 distinct Sentinel-1 scenes, and the seeded
  90/10 split is sub-image-level, so val tiles share those same 15
  backgrounds. Measured: the ViT pretraining's val_loss fell monotonically
  to ~6e-10 over 50 epochs — near-perfect memorization of a homogeneous
  source. Arms 4/8 therefore measure SMALL, TASK-MATCHED, SUPERVISED
  transfer (their designed role), not FM-scale pretraining; a weak or
  below-floor result is a pre-registered finding (risk register), and the
  writeup must report source-data scale (9k images vs SARMAE's 1M) in the
  fairness table and state this caveat explicitly. Do NOT "fix" it by
  pooling more SAR datasets — that scope was deliberately cut (Appendix D).

## Grid execution fixes (2026-07-07)

- **Early-stopping patience is in EPOCHS, not dev evals**: Lightning's
  EarlyStopping checks every epoch and the logged dev_f1 persists between
  our every-5-epoch evals, so stale-value checks burned the patience within
  4 epochs — every first-wave cell stopped at epoch ~9 after ONE real eval.
  Patience is now `early_stop_patience × dev_every_epochs` (= 20 epochs = 4
  real evals, the intended option-B semantics). The four affected runs
  (satdino-f100 probe + three f10 cells) were deleted and rerun.
- **ConvNeXt cells on the 16 GB dev card use micro-batch 8 with gradient
  accumulation 2** (effective batch stays the recipe's 16): batch-16
  ConvNeXt @512 overflows VRAM into Windows shared memory (~20× slowdown,
  observed live). Hardware adaptation only — the optimizer sees identical
  effective batches; V100-node runs use the plain recipe. (Focal-loss
  normalization is per-micro-batch positives — a bounded numerical nuance
  of accumulation, identical across all CNN arms.)

## sprint-3 (detector)

- **BigEarthNet stem adaptation 2→3 / 10→3**: timm's `adapt_input_conv`
  implements Repeat-with-rescaling only *from* 3-channel sources, so
  `repeat_with_rescaling` (init_loaders.py) generalizes the identical
  tile-and-rescale math (cyclic tile to target count, scale by src/dst).
  Note for a possible human-approved ablation: the S1 checkpoint's stem
  order is (VV, VH) while our input is (VH, VV, VH−VV) — blind tiling maps
  dst-VH ← src-VV etc.; a semantics-aware channel permutation is NOT applied
  because it would be an invented scheme (ground rule 12).
- **SatDINO**: pos_embed layout is `[cls, patches…, gsd_register]`; the
  trailing GSD-register position and the `gsd_register` parameter are
  dropped when loading into the vanilla timm ViT.
- **SARMAE**: `sar_encoder.*` ENCODER only (decoder_*, mask_token,
  optical_encoder.*, sar_alignment_ffn.* dropped).
- **Third-channel normalization**: frozen stats.json holds per-pol (VH, VV)
  stats; VH−VV is normalized with derived mean (µVH−µVV) and std
  √(σVH²+σVV²). The independence approximation only scales one channel by a
  constant shared by every arm — cannot become a between-arm confound.
  (datasets.py)
- **Intensity jitter ±1 dB** == the plan's "±0.1 in log space" (chips are
  dB = 10·log10). Applied before normalization. (transforms.py)
- **LS-SSDD single-channel stand-in**: sub-images are 8-bit single-pol JPGs;
  the fixed channel rep becomes [x, x, 0] (VH=VV=x ⇒ VH−VV=0), normalized by
  per-dataset train-split mean/std cached to `data/lsssdd_stats.json`.
  (datasets.py)
- **Frozen-scorer adapters live in infer_scene.py**: real CSVs use lowercase
  `source` values → normalized to 'Manual'/'AIS'/'AIS/Manual' when building
  GroundTruthPoints (the frozen scorer's dark test is `== "Manual"`);
  prediction-side `distance_from_shore_km` (needed for near-shore FP
  counting) is derived from the scene's 500 m bathymetry (land = elevation
  ≥ 0, euclidean distance transform in km).
- **Whole-scene inference reads each raster into RAM once** and slices
  512/384 windows from memory — per-window reads against striped compressed
  GeoTIFFs were ~100× slower and made the every-5-epoch dev eval unusable.
  The plan's 512/384 tiling and global-NMS decode are unchanged.
- **P3.6 early-signal protocol** (dev-card budget, identical across all six
  runs so the within-track comparisons stay fair): 100% labels, 5 epochs,
  batch 8, 24,000 sampled chips/epoch, full 8-scene dev eval at the last
  epoch, seed 0. CLI overrides only — the frozen detector.yaml remains
  authoritative for the real grid on the node. Run ids carry a `-p36`
  suffix so they can never be confused with grid cells.
- **The P3.6 sweep caught a three-part optimizer bug (2026-07-05/06),
  all fixed with regression tests (`test_layer_decay`):**
  (1) timm's layer-decay grouping fails SILENTLY on the `features_only`
  wrapper's flattened names — the CNN track had lr_scale 1.0 everywhere;
  the backbone is now the plain timm model (`forward_features`, same
  stride-32 map, standard names).
  (2) ConvNeXt's native per-block ladder (~38 rungs → earliest layers at
  ~1e-7, frozen) is depth-mismatched vs ViT-B's 13 rungs; the CNN now uses
  the ConvNeXt official fine-tune convention (12 layer-ids mirroring ViT-B)
  so the shared layer_decay=0.65 means the same thing in both tracks.
  (3) timm's `lr_scale` is only applied by timm schedulers — with torch's
  LambdaLR the ladder sat inert and EVERY arm trained at base lr; the scale
  is now baked into each group's lr at optimizer construction.
  Consequence: all pre-fix short runs (the smoke, the first P3.6 sweep)
  trained without any layer decay — internally consistent across arms
  (identical wrong optimizer) but not the plan's recipe; the P3.6 sweep was
  rerun under the corrected optimizer. The first sweep's bigearthnet_s1
  fp16 divergence (NaN forward at epoch 3, decode correctly refused the
  non-finite heatmap) is expected to be resolved by proper layer decay; a
  finite-loss abort in the LightningModule now fails loudly instead of
  burning epochs. A GRN-fp16-overflow hypothesis was tested and DISPROVEN
  (autocast already runs `norm` in fp32) — no GRN patch is shipped.

## Arms 4/8 and seed-count amendment (human decision, 2026-07-22)

- **The original Arms 4/8 are superseded, not silently rewritten.** The
  random-init→LS-SSDD→xView3 design was actually run at f10/seed 0 as
  `vitsup-f10-s0` and `cnnsup-f10-s0`. Those outcomes and checkpoints remain
  historical diagnostics, but are excluded from the revised study tables and
  curves. Their LS-SSDD source training drove validation loss essentially to
  zero on a 9,000-tile, 15-scene corpus whose seeded train/validation tiles
  share scene backgrounds (see the pre-existing capacity caveat above). The
  owner therefore judged this stage to be source-corpus memorization rather
  than a defensible generic representation baseline. No LS-SSDD pretraining
  job remains in the active matrix.
- **Revised Arm 4 (run prefix `vitin1k`; init `vit_imagenet`)** is the
  headless encoder from `timm/vit_base_patch16_224.augreg_in1k`: supervised
  ImageNet-1K AugReg training from random initialization. The pinned HF
  distribution is `timm/vit_base_patch16_224.augreg_in1k`, revision
  `458542882691a06a8b667c6fb5fe5c9573093a81`, file `model.safetensors`,
  SHA-256
  `678a1ce471be7da9822fe2508497a5bcf6da4c6802053151b232ba88a42c21a2`
  (Apache-2.0).
- **Revised Arm 8 (run prefix `cnnin1k`; init `cnn_imagenet`)** is the
  headless encoder from `timm/convnextv2_base.fcmae_ft_in1k`: ImageNet-1K
  FCMAE self-supervision followed by supervised ImageNet-1K classification
  fine-tuning. The pinned HF distribution is
  `timm/convnextv2_base.fcmae_ft_in1k`, revision
  `7b29800e499fdc06de5b612970f3384dc8d29ca5`, file
  `model.safetensors`, SHA-256
  `ec152f1e375edc2b3dfac7a81155a449b4c5cbb7c5cf0b9494838f6c87518d73`
  (CC BY-NC 4.0).
- **Interpretation is bounded:** both replacements use the same generic
  source dataset (ImageNet-1K) and end at the same supervised ImageNet-1K
  classification objective, but their training histories are not matched:
  ViT uses supervised AugReg from scratch, whereas ConvNeXt-V2 uses
  FCMAE→supervised fine-tuning. They are the within-track generic-ImageNet
  controls; comparisons of their absolute cross-track scores cannot isolate
  architecture from pretraining history. Backbone architecture, fixed
  `[VH, VV, VH−VV]` input, detector, schedule, splits, and sacred scorer are
  unchanged.
- **One seed only.** The active study reports seed-0 point estimates; the
  former seed-1/2 rerun tranche is removed. Curves are descriptive and must
  not claim seed variance or statistical uncertainty. The active run count is
  32 core cells (8 arms × 4 label fractions × 1 seed) plus R2 YOLO26 and R3
  LocateAnything, for **34 total runs**. The former optional ImageNet R1 is
  absorbed into the two core ImageNet roles and is no longer a separate run.

## P100 f10 throughput tripwire and resolution (human decision, 2026-07-22)

- All five free physical P100s (host IDs 3–7) were locked to the container;
  the two revised f10 cells launched concurrently on container-local CUDA
  devices 0 and 1 with the frozen effective batch 16 (micro-batch 8,
  accumulation 2). Both exact checkpoints loaded fully and losses stayed
  finite.
- Stabilized epoch-0 throughput was **0.95 steps/s for ViT** and **0.20
  steps/s for ConvNeXt-V2**, with 1,402 training batches per f10 epoch. Before
  whole-scene dev evaluation, that projects to 20.50 and 97.36 training hours
  respectively at the 50-epoch ceiling (10.25 and 48.68 hours even at 25
  epochs).
- Comparable completed f10 runs on the development GPU took 3.06–3.49 hours
  for ViT and 6.43–6.48 hours for 50-epoch CNN cells. The P100 projections are
  therefore about **5.9× and 15.0× slower**, beyond DEVPLAN's mandatory >2×
  compute/time tripwire.
- Both trainers were interrupted gracefully during epoch 0. Their partial
  configuration/metric probes were moved under
  `runs/probes/2026-07-22-p100-throughput/`; the active run-ID paths are
  clear, and no checkpoint or `final_metrics.json` exists. The structured
  measurement is
  `results/throughput/p100_f10_probe_2026-07-22.json`.
- **RESOLVED by hardware move:** the owner selected the already-validated RTX
  5070 Ti for the 26 remaining core cells. The same one-GPU Trainer path is
  retained; DDP, global effective batch, detector, optimizer, schedule, splits,
  seed, and scorer are unchanged. Completed 5070 timings project roughly
  19–22 continuous compute days after the ignored data, prior run records, and
  both amended ImageNet checkpoint directories are transferred and verified.
  The P100 probes remain stopped provenance and must not be resumed without a
  new human decision.

## fp16 focal-loss overflow fix (human decision, 2026-07-23)

- **cnnin1k-f10-s0 died deterministically at epoch 18** (twice, same batch —
  optimizer step ~641/701) via the finite-loss abort, after a healthy and
  leading trajectory (dev F1 0.862/0.873/0.869 at epochs 4/9/14). Both partial
  runs and their checkpoints are archived under
  `runs/probes/2026-07-23-5070-gpu-contention-crash/`. (The first archived
  attempt there also records an unrelated 3-way GPU-contention OOM crash and a
  harness-kill: this cell took three environmental strikes before the real
  numeric defect surfaced.)
- **Root cause is in the loss, not the backbone**: under fp16 autocast the
  focal loss's `clamp(eps, 1 - eps)` upper bound is a NO-OP — `1 - 1e-6`
  rounds to exactly 1.0 in half precision (largest representable half below
  1.0 is `1 - 2^-11`), and fp16 `sigmoid(x) == 1.0` exactly for `x >~ 9`. A
  Gaussian-shoulder pixel adjacent to a confidently detected vessel is a
  penalty-reduced NEGATIVE with `target < 1`; when the sharpening detector
  drives its logit past ~9, `log(1 - prob) = log(0) = -inf`. The strongest
  CNN arm crossed first — this is a capability-triggered failure, not noise.
- **Evidence**: static — fp16 numerics shown above are exact; dynamic — a
  1,402-batch sweep with the archived epoch-9 checkpoint measured max logit
  5.15 (no saturation yet) and fp16-vs-fp32 loss agreement to ~1e-5 on every
  batch, i.e. the fix is numerically inert away from the saturation cliff.
- **Fix (owner-approved)**: `penalty_reduced_focal_loss` casts its inputs to
  fp32 at entry (losses.py), restoring the clamp and bounding the worst-case
  per-pixel penalty at `-log(1e-6)`. This implements the plan's own
  precision rule ("logit/softmax math in fp32" — DEVPLAN node adaptation
  rule 2), applied identically to every arm. Regression:
  `tests/test_focal_loss_fp16.py`.
- **Scope (owner decision)**: rerun `cnnin1k-f10-s0` only. The seven
  completed f10 cells stand — their training never entered the saturation
  regime (their losses were finite throughout, and the parity sweep bounds
  the counterfactual difference at input-rounding scale, well under run-level
  cuDNN nondeterminism). Do not silently rerun completed cells for this.

## V100 full-fp32 core-grid amendment (human decision, 2026-07-26)

- **Trigger:** the fresh V100 campaign (`fresh34-v100-20260724`, `dev` at
  `2fb24e08b903313fe097e5c559218a2201c06de1`) failed at
  `cnnin1k-f100-s0`, epoch 30, batch 2467, with a non-finite training loss.
  The Sprint 7b fp32 focal-loss island was already present. All tensors in
  the last saved checkpoint were finite, while its GradScaler had fallen to
  512, which is consistent with repeated AMP overflows before a NaN arose in
  the model forward. The exact failing crop/augmentation stream cannot be
  replayed because sampler and persistent-worker RNG states were not saved.
- **Diagnosis:** bounded fp32 focal arithmetic, finite generated targets, and
  finite checkpoint tensors make an AMP activation overflow the leading
  explanation. Full fp32 is estimated at 85–95% confidence to remove this
  numerical failure class; it is a mitigation with strong evidence, not a
  proof that every 50-epoch trajectory will remain finite.
- **Feasibility gate:** a fresh `cnnin1k-f100-s0` probe at batch 16 completed
  200 true-fp32 training steps with finite loss (`0.5623`) and no OOM.
  Training stabilized near 0.72 step/s; a separate full step measured
  30.420 GiB allocated and 30.516 GiB reserved on a 32 GiB V100. The probe's
  one-scene dev pass exposed a separate hard-coded fp16 inference autocast;
  this amendment therefore propagates the shared precision to dev, test, and
  final model forwards as well. The established float16 heatmap canvas stays
  unchanged because sigmoid scores are bounded and this is decode storage,
  not model-forward arithmetic.
- **Owner-approved recipe amendment:** set the shared core trainer and all
  core model-forward inference to Lightning `32-true`, re-pin
  `configs/detector.yaml`, and rerun all 32 seed-0 core cells from scratch.
  This explicitly supersedes the earlier one-cell-only rerun decision above.
  The detector SHA256 moves from the AMP pin
  `4fd1bfe88861cc676dd67b2092e379fbcf401dd9c1d42fb09e81a84b9cdbe2f8`
  to `c42ae65bf9045cc93f0d73ae1437b2f6a1300670cb49d8e93f83a39d58a62a12`.
  A one-cell precision exception is forbidden. R2 YOLO26 and R3
  LocateAnything remain external references under their published precision
  paths, but they are also rerun fresh for the requested 34-experiment
  campaign.
- **Budget override:** measured CNN training is about 2.42x slower than the
  live AMP rate. The central full-core projection is about 1,550 GPU-hours,
  explicitly overriding both the greater-than-2x STOP and the prior
  1,300-GPU-hour ceiling. One controller starts R2/R3 on GPUs 0/1 and
  core cells on GPUs 2–7, then returns each released reference GPU to the
  same core queue. This keeps the expected wall clock near eight days, with
  nine days reserved for the 50-epoch/evaluation tail; serialized references
  would instead require roughly 9–10.7 days.
- **Superseded artifacts:** the stopped AMP campaign (10 complete, 6
  interrupted, 1 failed, 17 unstarted) is retained under
  `runs/archive/superseded_amp_fresh34_20260726T211204Z/`; previously exported
  rows are under `results/superseded/2026-07-26-mixed-precision-grid/`. None of
  these core results may be mixed with or reported in the fp32 grid. Completion
  markers now carry and validate the git SHA, detector hash, precision,
  micro-batch, accumulation, and effective batch; exported results also retain
  runtime provenance. This prevents stale or recipe-mismatched results from
  being silently skipped on resume.
- **Independent sanity STOP remains:** the superseded ViT ImageNet AMP curve
  dropped from f10 dev F1 0.8858 to f50 0.8514, beyond the predeclared 0.02
  tolerance. The fp32 rerun replaces those observations but does not presume
  to resolve the scientific monotonicity check; the new grid must pass it.

## H100 strict-IEEE-FP32 uniform-core cutover amendment (human decision, 2026-07-26)

- **Decision and scope:** retain the Sprint-7c full-fp32 numerical fix but move the canonical 32-cell core grid, all at once, to one uniformly recorded H100 hardware/environment class to reduce wall clock. This is an execution-hardware amendment, not approval for BF16, TF32, FP16, DDP, a smaller batch, a shorter schedule, or a per-arm exception. `configs/detector.yaml` remains frozen at its Sprint-7c pin.
- **Branch/base:** the amendment lives on `sprint-7d-h100-fp32`, intentionally stacked on `sprint-7c-fp32-grid` commit `48e10534a8c7baf0662acd548f52928da69f23c8`; the integration target is `dev` and the merge order is Sprint 7c then Sprint 7d.
- **Numeric recipe:** every H100 core train and dev/test/final model forward uses Lightning `32-true` with CUDA-matmul TF32 and cuDNN TF32 explicitly disabled, no autocast or GradScaler, micro-batch 16, accumulation 1, effective batch 16, one process/GPU, and no DDP. The float16 bounded-score heatmap canvas remains decode storage only.
- **Uniform restart and reporting:** after cutover, all 32 seed-0 core cells restart from scratch on the accepted H100 class. V100 core checkpoints/results/markers then become diagnostic provenance only. Never combine V100 and H100 core cells in a curve, table, grid summary, resume decision, or completion namespace.
- **Cutover barrier:** the current V100 full-fp32 campaign continues uninterrupted until every H100 receipt/environment/test/load/numerical/memory/inference/throughput/launch-hygiene gate passes and fresh valid V100 R2/R3 completion markers exist.
  The measured approximately 474 GB uncompressed source plus wheelhouse, Box upload/receipt, target environment, and H100 execution remain pending until separately recorded; the Box preflight remains fail-closed unless at least 500 GB (500,000,000,000 bytes) is available, and a failed gate leaves the uniform V100 fallback running and triggers a STOP rather than partial migration.
- **References:** R2 YOLO26 and R3 LocateAnything remain on V100 under their independently pinned precision paths. They are excluded from the H100 handoff and controlled curves and remain separately reported canonical references after their fresh markers validate.
- **Forward core-only handoff:** build beneath a root addressed by the full code SHA; the manifest names one git bundle, 150 per-scene chip `.tar.zst` files, 39 frozen dev/test per-scene GRD `.tar.zst` files, the train CSV only, six separate exact core-weight-directory `.tar.zst` files, the exact offline wheelhouse, and a pinned Apptainer definition.
  Package control files are the manifest, `SHA256SUMS`, and `READY`. The Box destination is supplied only at runtime through `BOX_FOLDER_ID`; no folder ID, JWT, URL token, or credential belongs in the repo, docs, logs, or manifest.
- **Box transfer hardening (observed 2026-07-30):** the first real forward upload attempt exposed that Box SDK v4 request I/O had no timeout and its warning logger could emit raw request metadata; the attempt ended without an upload receipt and does not constitute a completed handoff. Before retry, both JWT-auth and authorized data sessions are bounded at 30-second connect / 300-second read timeouts, SDK logging is suppressed before authentication, and each chunk session handles exceptions across at most five resume attempts. A fresh invocation revalidates and content-skips committed SHA-1/size matches; `READY.json` remains last.
- **Forward exclusions:** no `validation.csv` or eval-final asset, `runs/`, virtual environment/cache, JWT/token, raw LS-SSDD, preprocessing manifest, YOLO/reference payload, LocateAnything, YOLO, superseded result/checkpoint, old R1/IN22K weight, or LS-SSDD-trained ImageNet-role weight.
  The package manifest itself is control metadata and is not the excluded preprocessing-manifest material.
- **H100 acceptance:** target receipt must validate every count/hash and safe archive member; restore the exact clean SHA from the git bundle; permit only the explicit access check/acquisition of the definition's digest-pinned OCI base, while installing every Python package offline with `--no-index` from the verified wheelhouse; record a uniform H100 inventory and environment/container hashes; run the complete guards and all six offline value-sensitive loads; prove TF32-off strict fp32; and pass representative ViT/CNN batch-16 train plus full-scene inference, including the worst-case 200-step CNN gate. From those measurements, project the complete 32-cell H100 finish on the eight-card allocation, including measured verify/clone/extract staging for every projected Slurm allocation, and at the same acceptance snapshot project the live remaining V100 core finish from its measured state and rates. `cutover-check` refreshes the remaining-V100 forecast and refuses to issue `CUTOVER_READY.json` if the conservative H100 wall clock no longer wins. The owner must accept the recorded comparison before cutover.
  No “few days” estimate is accepted evidence before that apples-to-apples throughput measurement. A greater-than-2x miss remains an additional compute STOP, not a substitute for the earlier-finish criterion.
- **Reverse handback:** return a content-addressed core-results bundle containing campaign/launch state and, for every core cell, resolved config, metrics, logs, provenance/completion marker, and best plus last checkpoints, together with its manifest, `SHA256SUMS`, and `READY`.
  Exclude source data, weights, environments/caches, secrets, R2/R3, eval-final material, superseded results, and V100 core diagnostics; verify the remote receipt before declaring handback complete.
- **Status discipline:** this decision authorizes implementation and fixture validation of transfer/runner tooling, but neither that tooling nor this documentation proves the real package was built/uploaded, the target accepted it, any H100 gate passed, cutover occurred, or any H100 run/result exists.
  Update the cold-start ledger only from recorded artifacts; until then, the H100 state is approved but gated and not launched.


## Judy H100 native-venv runtime amendment (human decision, 2026-07-30)

- **Decision:** use a native, sealed Python venv for the Judy H100 lane only.
  This supersedes the Sprint-7d Apptainer/SIF launch plan and the subsequent
  uncommitted Enroot/Pyxis adaptation discussion. It does not change the V100
  environment, references, scientific recipe, frozen detector, or strict-FP32
  cutover conditions.
- **Branch/base:** implementation lives on `sprint-7e-judy-venv`, stacked on
  clean Sprint-7d commit `2726199efcebbebc89156e708b89df2a3415468a`.
  The exact Sprint-7d base payload and history remain immutable; Sprint 7e is a
  separate runtime-only amendment and merges after Sprint 7d into `dev`.
- **Immutable base-payload evidence:** the verified package is
  `xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a`, 201 files and
  294,278,292,176 bytes. Its control SHA-256 values are READY
  `b0d6ee18f9ddbd0d604cbea06610dcdbae6a9eb6d1f5ff3ea3431bd9e2d55f81`,
  manifest `fccb0b505c89836a148afec709bb799f7af4908d955ea1b142e153154d830896`,
  and SHA256SUMS
  `21c83b2e3b1b9d67bf00b8abca3ce267a5efd9362c1206b8d29ab21ca3e2d396`.
  Its historical Apptainer definition remains byte-bound provenance only and
  is not executed on Judy.
- **Dual transfer contract:** Sprint 7e produces a small code-only package with
  one exact `sprint-7e-judy-venv` Git bundle and manifest/checksum/READY controls.
  Its content-addressed identity binds all base-payload controls. It contains
  no data, weights, wheelhouse, run, cache, venv, reference, eval-final, or
  secret material and uses a separate initially empty Box folder so a base
  payload transfer is never mutated or interrupted.
- **Native environment (owner-amended 2026-07-31):** use Judy's exact Python
  3.11.13 executable for one uniform H100 venv. Python 3.11.13 and 3.11.15
  share the CPython 3.11 ABI and no measurable model-training, accuracy, or
  strict-FP32-stability change is expected; that expectation never replaces
  the complete target acceptance gates. The historical Sprint-7d OCI and
  wheelhouse-resolution Python-3.11.15 metadata remain immutable base-payload
  provenance only. Build `python -m venv --copies` at the final persistent
  path; install entirely offline with `--no-index` from the verified
  wheelhouse. The wheelhouse intentionally excludes the venv bootstrap packages
  `pip`, `setuptools`, and `wheel`; dependency resolution must retain the
  interpreter-bundled bootstrap packages rather than use `--ignore-installed`.
  Judy's clean Python 3.11.13 venv supplies `setuptools==65.5.0`, satisfying
  torch's `setuptools<82` requirement. Require `pip check`, exact normalized
  freeze with only those bootstrap extras, safe relative links, bytecode
  removal, read-only tree modes, a deterministic tree digest, executable plus
  full base-Python-runtime hashes, and the canonical Sprint-7d
  extraction/wheelhouse identity. Snapshot Judy's canonical libpython
  directory and use it as the complete loader path under Slurm `--export=NONE`.
  Activation, relocation, inherited loader paths, mixed Python patches,
  user-site packages, and inherited Box credentials are forbidden.
- **Numerical and scheduling recipe unchanged:** torch remains
  `2.11.0+cu126`; every H100 core train and model-forward inference stays
  Lightning `32-true`, CUDA-matmul/cuDNN TF32 off, no autocast/GradScaler,
  batch 16, accumulation 1, one process/GPU, eight H100s, and no DDP. BF16,
  FP16, TF32, reduced batches, shortened schedules, and per-arm exceptions
  remain out of scope.
- **Fail-closed evidence:** source, Slurm smoke, H100 acceptance, aggregate
  tests, cutover, campaign, per-cell, external V100 diagnostic isolation, and
  reverse-result receipts move to native schema 2 and separately bind the Sprint-7e source,
  Sprint-7d base payload, Sprint-7e runtime amendment, venv tree/build/base
  Python, strict backend, and hardware. Legacy schema-1 SIF evidence cannot
  satisfy these gates. The Slurm smoke must prove external batch-shell to
  native-Python-child `SIGUSR1`, requeue, and checkpoint resume.
- **Superseded transfer:** runtime package
  `xview3-h100-runtime-ad6226a14e8af61b400a0de44482441890b05d83-127e9181c298c2b6b517f533ff5e234f282cd191cfa356fe78017be544b9de3b`
  was built and uploaded with `contract.python=3.11.15`. Retain it as immutable
  provenance, but it cannot qualify or launch the Judy 3.11.13 lane. A new
  content-addressed amendment from the current branch tip requires a separate,
  initially empty Box folder; the 294 GB Sprint-7d base is reused unchanged.
- **Status (superseded operational wording):** this amendment authorized
  replacement packaging and validation. Sprint 7f below supersedes any implied
  V100 fallback/stop requirement: V100 remains untouched only as a
  non-reportable diagnostic and cannot satisfy the corrected H100 campaign.

## Judy/V100 separate-filesystem control plane (historical Sprint 7e clarification, superseded below)

- **Topology:** Judy and the live V100 host have completely separate storage.
  No V100 runs path is or will be mounted on Judy. A dummy Judy directory is
  not an acceptable isolation surrogate.
- **Correction:** runtime commit
  `13110cec972f23b014e68043b9954ad577916031` incorrectly required
  `H100_V100_RUNS_ROOT` in every mode. That package remains immutable
  provenance but cannot submit even the one-GPU smoke honestly. Its scientific
  and environment recipe is unchanged; a new code-only Sprint-7e package must
  replace only the runtime-control layer.
- **Control plane:** every Judy submission requires the exact
  `H100_V100_CONTROL_PLANE=box-transfer-v1` literal in its immutable
  compute-site snapshot. Smoke and H100 acceptance consume Judy-local inputs
  only. The corrected reference campaign manifest plus final R2/R3
  metrics/provenance cross Box to Judy for `cutover-check`;
  `CUTOVER_READY.json` crosses Box to the V100 operator; and the human-authored
  diagnostic-isolation attestation crosses back to Judy before the campaign
  gate is validated. No stop/archive receipt is required. Campaign execution
  has no mounted V100/reference path dependency.
- **Write isolation:** before its first persistent write or Slurm submission,
  Judy canonicalizes `H100_RUNS_ROOT` and `H100_JOB_LOG_DIR` and rejects
  equality or ancestor overlap with each other or the immutable repo,
  base/runtime packages, wheelhouse, and sealed venv. Cutover-check
  additionally protects the local transferred-reference evidence root.
- **Verified login-node environment evidence; not compute-qualified:** Judy
  successfully built and independently verified Python 3.11.13 with torch
  `2.11.0+cu126` on the login-node runtime. The sealed venv tree is
  `101fd4571e3daaab97bf1d6d6f7a98fb6514e2b9bc54eab5d2b49e41ace4f8ca`;
  its build receipt is
  `c7051cb6d743923fcea64d618a59685620309e766cbc90e73179aee9fa8fc9be`.
  The verified base-Python executable is
  `dbbb197739dbd20dcae2eacc394b97e7d092a7512c60f54980075b7f037afd39`,
  base runtime `cce52efebc4dbd691ffbdc46fac68efb9971dc89bd63dae34113bfcebc0f4739`,
  wheelhouse `fb0106771ddb223ba2b93af3463f6accbc018ece18275975ef7770749fd11138`,
  and extraction receipt `eb52d4e45083f55af3ae2e14be7fc2c8c610d0567454495542574c8e431a2b5d`.
  The receipt is not source-SHA-bound, but later compute job 541358 proved its
  full base-Python runtime closure does not match `dgx18`. It therefore remains
  useful diagnostic evidence and is not valid for H100 compute qualification.
- **Historical status:** this was the Sprint-7e gap before Sprint 7f. No GPU or
  Slurm gate had run. The Sprint-7f decision and implementation below replace
  the code-only amendment and supply the content-addressed dynamic control
  protocol; all target acceptance and transfer gates still remain pending.

## Evaluation-contract correction and concurrent V100 diagnostic (human decision, 2026-08-04)

- **Approval and supersession:** the owner approved Sprint 7f on
  `sprint-7f-eval-contract`, stacked on accepted Sprint-7e commit
  `26bece168cd3b9b262ffec5939b836df21b352cd`. This decision supersedes only
  prior instructions to stop/archive V100 as an H100 launch precondition and
  the defective evaluation-caller/provenance behavior. The frozen detector,
  scorer, splits, stats, LS historical split, precision, batch, architecture,
  optimizer/schedule, seed, and verified-final lock remain unchanged.
- **Ground-truth defect and remedy:** the scorer caller admitted every
  HIGH/MEDIUM row as a positive, including non-vessels. The centralized
  converter now includes only explicitly true HIGH/MEDIUM vessels, excludes
  HIGH/MEDIUM non-vessels as background, retains LOW rows as ignores, preserves
  source/shore fields, and rejects ambiguous booleans. The frozen scorer SHA-256
  remains
  `85dec7ab083b531547fe81fea5aa02e4d828457ff157619afae29981cf49cd32`;
  this is an input-contract correction, not a scorer change.
- **Measured source evidence:** using only
  `data/raw/xview3/labels/train.csv` (SHA-256
  `42871b3ddf12d2a732d11d07897c21efc6c688c5d1a6c59a90839a5539e15415`)
  and the frozen split, the eight training-time dev scenes contain
  517 vessel HIGH/MEDIUM positives, 107 excluded non-vessel HIGH/MEDIUM rows,
  and 118 LOW ignores; full dev contains 1,479/804/441; test contains
  1,165/420/325. The 16 test scenes contain zero dark-vessel positives and only
  two valid near-shore vessel positives. No `validation.csv` or eval-final
  label was read for this audit.
- **Scientific consequence:** the faulty dev target set drove threshold,
  checkpoint selection, and early stopping. Historical checkpoints cannot
  reconstruct the correct best epoch uniformly, so every existing and future
  V100 core result is a non-reportable diagnostic. Corrected rescoring of
  selected V100 best/last checkpoints is permitted only in the gitignored
  diagnostic namespace and cannot repair reportability.
- **Owner-selected V100 disposition:** let the live V100 campaign continue for
  diagnostic value. Sprint 7f must not stop, signal, pause, reconfigure, or
  otherwise mutate its controller or children. Its markers cannot suppress,
  resume, or satisfy corrected H100 cells. A human-authored immutable
  `V100_DIAGNOSTIC_ISOLATION.json` binds separate namespaces and returns to
  Judy through Box before H100 launch. Later stop/archive work is optional and
  not a launch prerequisite.
- **Throughput cutover if the diagnostic finishes first:** preserve the
  original acceptance-time comparison, which requires a positive remaining
  V100 forecast and a strictly earlier conservative H100 finish. At the later
  `cutover-check`, positive remaining V100 time still requires H100 to be
  strictly earlier and binds
  `continues-running-non-reportable-diagnostic`. If the V100 work instead
  finishes naturally before cutover, exactly zero remaining hours are accepted
  only with explicit `complete-non-reportable-diagnostic` status. The
  diagnostic remains non-reportable and the corrected uniform H100 rerun
  remains scientifically mandatory in either case.
- **Checkpoint-bound completion:** reportable core training uses exact
  `result_schema == 2`. `best_dev` records the full operating point and is
  bound to the actual best checkpoint's safe relative path, SHA-256, and
  Lightning epoch. `last_dev` remains diagnostic only. Missing, legacy,
  non-finite, hash-drifted, or epoch-mismatched markers fail closed and are
  written atomically.
- **Held-out isolation:** all 32 corrected training markers must validate
  before one immutable `TRAINING_COHORT.json` is created. Canonical acceptance
  and training views contain only 111 TRAIN chips, fixed DEV8 rasters, six
  weights, and a source-built 13,911-row TRAIN+DEV8 CSV; acceptance alone also
  stages the offline wheelhouse. They contain no TEST chip, raster, or label
  row. Cohort freeze returns exit 75, and only a fresh allocation may validate
  the all-32 cohort before staging 16 TEST rasters, no chips, six weights, and
  an audited 15,079-row TRAIN+TEST CSV. That scoring phase writes separate
  immutable `test_metrics.json` files
  with the exact 16-scene/support contract and never mutates
  `final_metrics.json`. Final evaluation remains locked until Phase 5 and a
  separate explicit owner confirmation.
- **References:** R2 retains only its exact pinned best-weight payload
  (SHA-256
  `15520cb6cff9d4b01ed5c4a7e039fab763e8e5b0ca5b8e6bffd591ef0d7b8064`)
  and is rescored under the corrected shared converter; R3 requires a full
  corrected-contract rerun. Both emit schema-2 provenance and retain their
  independent precision recipes.
- **Box/Judy control:** Sprint 7f is a new content-addressed runtime amendment
  reusing the verified Sprint-7d payload; a fresh compute-built sealed Judy
  venv is required before acceptance. Besides its Git
  bundle/control metadata, its sole data artifact is the deterministic
  source-audited TRAIN+DEV8 CSV needed to keep TEST rows out of the pre-cohort
  view. Compute-time source validation hashes only pinned package controls and
  the base Git bundle; the phase stager hashes each selected archive, avoiding
  any pre-cohort read of TEST or combined-label archive bytes.
  Separately allowlisted JSON-only packages transfer corrected references
  V100→Judy, `CUTOVER_READY.json` Judy→V100, and diagnostic isolation
  V100→Judy. Every package uses manifest + `SHA256SUMS` + READY-last,
  verified download/rename semantics, and a distinct Box child folder. None
  contains a credential, run tree, checkpoint, or process-control command.
- **Isolation boundary:** this is a fail-closed canonical allocation-view
  contract, not host-global ACL isolation. The immutable Sprint-7d plaintext
  package is already accessible to the Judy Unix identity, so same-UID code
  outside the approved launcher is outside the guarantee. Production study
  code receives only the phase view and validates it before semantic access.
- **Status discipline:** source implementation and local tests do not claim a
  Judy transfer, H100 acceptance, Slurm smoke, reference completion, cohort
  freeze, held-out access, or H100 launch. The live V100 diagnostic continues
  independently while those gates remain pending.

## Judy Slurm scratch portability correction (operational correction, 2026-08-05)

- **Observed failure:** Judy Slurm smoke job 540200 reached `dgx09` and
  verified the pinned runtime bundle and training-view label member, then
  exited `FAILED (1:0)` after one second because Judy did not export
  `SLURM_TMPDIR`. The failure occurred before GPU discovery, the sealed native
  runtime child, signal/requeue exercise, or training. No
  `SLURM_SMOKE_READY.json` or canonical campaign state was published, and the
  live V100 diagnostic was neither read nor modified.
- **Site contract:** Slurm does not guarantee `SLURM_TMPDIR`; the Judy launcher
  therefore requires an explicit `H100_SCRATCH_ROOT` in ignored `site.env`.
  Submission and compute-side validation require one canonical, existing,
  non-symlink, writable/searchable root visible on every eligible node and
  disjoint from persistent runs/logs, source, packages, wheelhouse, and the
  sealed venv.
- **Allocation ownership and capacity:** each allocation creates exactly
  `H100_SCRATCH_ROOT/xview3-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT}` as a
  current-user-owned mode-`0700` directory and proves it writable. Campaign
  staging still requires at least 500,000,000,000 free bytes on the allocation
  scratch filesystem before extraction.
- **Cleanup and resume:** guarded exit cleanup may remove only that exact
  canonical, non-symlink allocation child. Requeue receives a fresh child and
  reconstructs source plus its phase-specific data view from verified
  immutable packages. Checkpoints, controller state, the frozen cohort, and
  metrics remain under persistent `H100_RUNS_ROOT`; scratch never confers
  resume authority.
- **Reservation interface:** an empty `H100_RESERVATION` means submit without
  `--reservation`; a nonempty value is passed exactly. This is a scheduling
  choice only and does not change hardware qualification or the training
  recipe.
- **Scientific scope:** this correction changes only Judy site portability and
  scratch lifecycle. It does not modify the frozen detector, scorer, splits,
  stats, evaluation contract, precision, batch, seed, model, or 32-cell
  campaign. A new content-addressed runtime amendment and a passing full smoke
  are required before H100 acceptance or training; V100 continues untouched as
  a non-reportable diagnostic.

## Judy compute-runtime provenance and verifier-entry correction (operational correction, 2026-08-05)

- **Observed entry-point behavior:** smoke 541320 and 32-GB probe 541333 were
  killed with exit 137 only when invoking the literal
  `scripts/h100/build_venv.py` path. Ladder 541341 imported the same module,
  walked the read-only venv, and completed its full 7.4-GB deterministic tree
  manifest. Module-form probe 541353 was not killed and reached the intended
  strict provenance check. This rules out the requested-memory limit and
  identifies the direct script path as incompatible with Judy endpoint
  controls.
- **Measured provenance difference:** job 541358 on `dgx18` found the same base
  Python executable SHA-256 as the login build,
  `dbbb197739dbd20dcae2eacc394b97e7d092a7512c60f54980075b7f037afd39`,
  but a different full runtime closure. The login receipt records
  `cce52efebc4dbd691ffbdc46fac68efb9971dc89bd63dae34113bfcebc0f4739`;
  `dgx18` measured
  `a4af214a34be0512d657350faed6aa76ab9f937d64e27858b043e0d147664733`.
  The observed inputs include the kernel identity and an additional mapped
  locale file on compute. The verifier correctly failed closed.
- **Decision:** use only `python -B -m scripts.h100.build_venv` for build and
  verification. Do not execute its literal file path, weaken the receipt, or
  normalize host-runtime inputs. Before building again, measure at least one
  additional eligible DGX node and require exact agreement with the accepted
  compute fingerprint. If compute nodes agree, build one fresh sealed venv
  inside a compute allocation at a new persistent path and update ignored
  `site.env` from that receipt. Preserve the login-built venv and receipt as
  diagnostic evidence; never overwrite them. If DGX fingerprints differ,
  STOP for an owner decision.
- **Scientific scope and status:** this changes only Judy runtime invocation
  and provenance qualification. It does not modify the detector, scorer,
  splits, stats, labels, evaluation contract, precision, batch, model,
  optimizer, schedule, seed, or campaign grid. No smoke, H100 acceptance, or
  H100 training is claimed; the V100 diagnostic remains untouched.

## H100 readiness receipt integration correction (operational correction, 2026-08-05)

- **Pre-acceptance audit finding:** before submitting the first eight-H100
  acceptance allocation, a producer/consumer audit found that phase-isolation
  commit `e4be3c3` correctly replaced the scratch
  `staged_base_extraction` record with `venv.staged_data_view`, but the cutover
  validator and synthetic fixtures still required the retired field. An
  otherwise successful acceptance would therefore have written
  `H100_READY.json` that cutover rejected. No H100 acceptance or training job
  had started, so no result or marker was invalidated.
- **Canonical contract:** acceptance now embeds the canonical
  `H100_DATA_VIEW.json` digest and receipt in `venv.staged_data_view`.
  Downstream validation requires the exact TRAIN/DEV8 acceptance purpose,
  source and dual-package identities, absent cohort, 111 TRAIN chip scenes,
  fixed DEV8 rasters, six weights, offline environment, and the 13,911-row
  TRAIN+DEV8 label contract. The allocation-private path is provenance only
  and need not survive guarded scratch cleanup. Never recreate a full
  `HANDOFF_EXTRACTED.json` beneath allocation scratch.
- **Persistent extraction identity:** `H100_READY.json` intentionally omits
  only the site-local path from
  `venv.wheelhouse.base_extraction`; package/manifest/wheelhouse identity
  remains embedded. Reverse-results verification recovers the full path from
  the already verified, persistent `venv_build.json`, requires its pathless
  identity to equal acceptance, and then revalidates the sealed venv and
  source bytes.
- **Additional fail-closed checks:** cutover now binds the immutable corrected
  evaluation-ground-truth file, its SHA-256, and its embedded receipt, and
  requires the exact three-key IEEE backend to equal the hardware probe
  backend. Regression tests cover receipt digest, phase/package,
  scene/label inventory, cutover consumption, and reverse-result packaging.
- **Scientific scope and status:** this changes no detector, scorer, frozen
  split/stat, label rule, precision, batch, model, optimizer, schedule, seed,
  or campaign cell. Repackage Sprint 7f and rerun smoke before acceptance; the
  V100 campaign continues untouched as a non-reportable diagnostic.
