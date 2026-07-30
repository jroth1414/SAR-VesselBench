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
- **Native environment:** owner-approved on 2026-07-30, require Judy's site-managed exact Python 3.11.13 executable; the historical Sprint-7d OCI Python-3.11.15 metadata remains immutable base-payload provenance only;
  build `python -m venv --copies` at the final persistent path; install entirely
  offline with `--no-index` from the verified wheelhouse; require `pip check`,
  exact normalized freeze, safe relative links, bytecode removal, read-only
  tree modes, a deterministic tree digest, and build/base-Python hashes. Invoke
  `venv/bin/python` directly under a clean environment. Activation, relocation,
  user-site packages, and inherited Box credentials are forbidden.
- **Numerical and scheduling recipe unchanged:** torch remains
  `2.11.0+cu126`; every H100 core train and model-forward inference stays
  Lightning `32-true`, CUDA-matmul/cuDNN TF32 off, no autocast/GradScaler,
  batch 16, accumulation 1, one process/GPU, eight H100s, and no DDP. BF16,
  FP16, TF32, reduced batches, shortened schedules, and per-arm exceptions
  remain out of scope.
- **Fail-closed evidence:** source, Slurm smoke, H100 acceptance, aggregate
  tests, cutover, campaign, per-cell, external V100 archive, and reverse-result
  receipts move to native schema 2 and separately bind the Sprint-7e source,
  Sprint-7d base payload, Sprint-7e runtime amendment, venv tree/build/base
  Python, strict backend, and hardware. Legacy schema-1 SIF evidence cannot
  satisfy these gates. The Slurm smoke must prove external batch-shell to
  native-Python-child `SIGUSR1`, requeue, and checkpoint resume.
- **Status:** this amendment authorizes code, packaging, and validation work.
  It does not claim the runtime amendment was uploaded, Judy built the venv,
  any H100 gate passed, V100 was stopped, cutover occurred, or H100 training
  started. V100 continues until all acceptance, throughput, R2/R3, and human
  operator gates pass.
