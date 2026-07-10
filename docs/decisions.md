# Documented implementation decisions (per DEVPLAN P3.6 / ground rule 12)

Chronological log of judgment calls the plan left unspecified (or where a
cited mechanism did not cover the actual case), each bounded and applied
identically to every arm. The DEVPLAN references `runs/decisions.md`; `runs/`
is gitignored, so the committed log lives here.

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
  where stopping fires); the dev card runs recipe-conform cells (batch 16
  verified at 7 GB) continuously, and the V100 node clears the tail when
  available.

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
