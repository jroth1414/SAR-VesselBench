# AIPR 2026 poster: presenter notes

Talking points and background for every block, bullet and figure on the
poster, in reading order. Each part has three layers:

- **Say:** the one or two sentences to say while pointing at it.
- **Background:** what you need to know to go one level deeper.
- **If asked:** likely questions with short, defensible answers.

All numbers match `generated/poster_numbers.json` and the committed evidence
(`results/h100/evidence`, replacement32 cohort). Part 7 has the full
per-cell table and every bootstrap interval.

### Poster v2: what changed (read first)

Version 2 keeps the layout and swaps text for pictures. Bullets that are no
longer printed are still described below; use them as talking points.

| Block | v2 change | Notes section |
|---|---|---|
| Motivation | Dark-vessel radar crop + one sentence (3 bullets removed) | 2, "Dark-vessel radar crop" |
| Data | Scene map added; 2 short bullets (study-scene bullet removed) | 2, "Scene map" |
| 8 arms × 4 budgets | Nested-budget waffle replaces the "only the initialization changes" sentence | 3, "Budget waffle" |
| Detector | 2 bullets (the FP32 / GPU-hours note is now a talking point) | 3 |
| Label-efficiency chart | ★ YOLO26 reference; LocateAnything-3B in the caption | 4, "Reference detectors" |
| Findings | 4 items; the win/tie/loss scoreboard replaces old findings 2 and 5 | 5, "Scoreboard" |
| Inside the detector | New block: the best cell's heatmaps on the two crops | 5, "Inside the detector" |
| Limits, Open questions | One line each (3 + 3); the longer v1 wording below is background | 5 |

---

## 0. Pitch (memorize this)

**30 seconds.** "Dark vessels don't broadcast AIS, so satellites have to
find them, and Sentinel-1 radar works at night and through cloud. Human
labels for radar are scarce, so pretraining should help. We asked whether
pretraining on radar beats pretraining on optical satellite images or on
ImageNet. We tested 8 initializations, 2 backbones and 4 label budgets in
one controlled detector. The answer is no: any pretraining helps a lot with
few labels, but radar pretraining wins in only 6 of 16 comparisons against
optical and 3 of 16 against ImageNet. And no detector works near shore,
because the training labels have almost no near-shore vessels."

**2-minute walk (v2).** Answer strip → dark-vessel crop → scene and map →
the 8 × 4 table and waffle → detector → label-efficiency chart (point at the
YOLO26 star) → scoreboard → near-shore chart, crops and heatmaps → Limits →
Open questions.

### Numbers to have ready

| What | Value |
|---|---|
| Cells (8 arms × 4 budgets) | 32, one seed each |
| Training budgets | 12 ⊂ 28 ⊂ 56 ⊂ 111 scenes (10, 25, 50, 100% of 111) |
| Splits | 111 train, 23 dev (8 used for selection), 16 test; 50 verified |
| Compute | 567.6 H100 GPU-hours, strict FP32 |
| Test F1 range | 0.657–0.888 (16 scenes, 1,165 vessels) |
| Verified F1 range | 0.409–0.574 (50 scenes, 8,642 vessels) |
| Mean F1 drop, test → verified | 0.306 |
| Gains over random at 12 scenes | +0.063 to +0.143 test, +0.048 to +0.111 verified; 12 of 12 intervals above zero |
| SAR vs optical | SAR wins 6, optical wins 2, 8 ties (of 16) |
| SAR vs ImageNet | SAR wins 3, ImageNet wins 7, 6 ties (of 16) |
| Best cell | ImageNet ConvNeXt, 111 scenes: 0.888 test, 0.574 verified |
| Near shore (≤ 2 km) | recall ≤ 0.021 in every cell; near-shore F1 0.015–0.041 |
| Labels within 2 km | 0.4% of training vessels (27 of 7,678) vs 31% of verified (2,686 of 8,642) |
| Dark-vessel recall | 0.136–0.228 (3,644 dark vessels) |
| Rerun variation | 28 unchanged cells retrained: test F1 moved 0.008 on average, up to 0.043 |
| YOLO26 reference (111 scenes) | 0.896 test F1 (test only; near-shore F1 0.000) |
| LocateAnything-3B zero-shot | 0.122 F1 on dev chips (best prompt "boat") |
| Best-cell heatmap peak | 0.987 offshore crop vs 0.001 harbor crop (threshold 0.896) |

---

## 1. Title, authors, answer strip

### Title

"Label-Efficient Dark-Vessel Detection in SAR: Does SAR-Domain Pretraining
Outperform Optical and ImageNet Transfer Across ViT and CNN?"

- **Say:** "The question is in the title, and the blue strip answers it."
- **Background:** "Label-efficient" means accuracy per labeled training
  scene. "Transfer" means initializing the detector's encoder from a
  pretrained checkpoint and fine-tuning everything on xView3.

### Answer strip

> **Answer: no.** SAR pretraining beats optical in 6 of 16 comparisons and
> ImageNet in 3 of 16 (ImageNet wins 7); its clearest edge is the CNN with
> 12–28 labeled scenes.

- **Say:** "A comparison is one backbone, one label budget and one test
  set: 2 × 4 × 2 = 16. SAR 'wins' only when the whole 95% interval of SAR F1
  minus the other F1 is above zero."
- **Background, SAR vs optical (16):**
  - SAR wins 6: CNN at 12 scenes (test +0.080, verified +0.045), CNN at 28
    (test +0.052, verified +0.014), ViT at 56 (test +0.028, verified +0.053).
  - Optical wins 2: ViT at 28 (verified −0.081), CNN at 111 (verified −0.073).
  - The other 8 are ties (the interval crosses zero).
- **Background, SAR vs ImageNet (16):**
  - SAR wins 3: ViT 111 test (+0.035), ViT 56 verified (+0.018), CNN 12
    verified (+0.037).
  - ImageNet wins 7: ViT 28 test (−0.076), CNN 28 test (−0.043), CNN 56 test
    (−0.026), ViT 12 verified (−0.029), ViT 28 verified (−0.075), CNN 56
    verified (−0.034), CNN 111 verified (−0.095).
  - 6 ties.
- **If asked "Why 'clearest edge' at CNN 12–28 when ViT 56 also wins?"** The
  CNN wins at two neighbouring budgets on both test sets, and the CNN pair
  (BigEarthNet-S1 vs BigEarthNet-S2) is the cleanest modality comparison:
  same architecture, same dataset family, same supervised task. The ViT win
  at 56 sits between an optical win at 28 and a tie at 111, so it doesn't
  form a trend.
- **If asked "Is 'no' too strong?"** "No" answers "does SAR pretraining
  outperform across ViT and CNN". It helps in some settings, but not
  consistently, not in both backbones, and it trails ImageNet more often
  than it beats it.

---

## 2. Column 1

### Motivation

**Dark-vessel radar crop (v2).** A 14 × 5.6 km VH window from verified scene
`9b89b9dcce7dc85ev`: yellow circles are vessels matched to AIS (12), orange
squares are dark vessels with no AIS match (7). Printed text: "Dark vessels
send no AIS, yet radar [2] still sees them. Human-checked labels are scarce:
which pretraining stretches them?"
- **Say:** "Every mark here is a real verified vessel. The yellow ones
  broadcast AIS; the orange ones don't, but the radar shows them just as
  clearly. That's the gap satellites fill."
- **Background:**
  - The window was picked by a fixed rule in code: among windows more than
    5 km offshore with at least 3 AIS and 3 dark vessels, the one with the
    most of both (HIGH/MEDIUM confidence only).
  - "Dark" means the label's source is manual only (no AIS correlation).
  - In two spots a circle and a square overlap: two separate labels sit
    within a few hundred metres (for example, vessels side by side).
- **If asked "Did the detector find these?"** That's a different figure
  ("What the best detector finds" and "Inside the detector"). This crop shows
  only the labels.

The three v1 bullets below are no longer printed; they are the background
for this picture.

**Bullet 1: dark vessels, Sentinel-1 SAR [2].**
- **Say:** "Dark vessels have no matching AIS report; radar sees them anyway."
- **Background:**
  - AIS (Automatic Identification System) is a ship-borne VHF transponder
    that broadcasts identity and position. Ships can switch it off, spoof it,
    or not carry it (small vessels).
  - A "dark vessel" is a detected vessel with no AIS match. It matters for
    illegal, unreported and unregulated (IUU) fishing, sanctions evasion and
    smuggling.
  - Sentinel-1 is an ESA C-band SAR mission. SAR is an active sensor, so it
    images day and night and through cloud. Metal hulls are strong radar
    reflectors and appear as bright points on dark sea.

**Bullet 2: vessels fill a few pixels; clutter.**
- **Say:** "A fishing boat is a handful of 10 m pixels in a 25,000-pixel-wide
  scene, and coasts, waves and wind create bright returns that look similar."
- **Background:** False-alarm sources include sea clutter at high wind,
  rocks, small islands, piers, offshore platforms, buoys, azimuth ambiguities
  ("ghosts") and range sidelobes. The scene on the poster is 29,658 × 21,656
  pixels.

**Bullet 3: human-verified labels are scarce.**
- **Say:** "Expert radar labeling is slow, so we ask which pretraining source
  stretches a small label budget furthest, and whether the answer changes
  with the backbone."
- **If asked "Why not just use the AIS-derived labels?"** We do for
  training, but they're incomplete: they miss vessels, especially near shore
  and dark ones (see the Data bullets and the near-shore chart).

### Data: xView3-SAR [1]

**Scene and chip figure** (scene `835f7629c3a3a9abt`).
- **Say:** "Left: one full Sentinel-1 scene, about 297 × 217 km. Right: one
  8 × 8 km chip from the orange box. The model trains on 512-pixel crops cut
  from chips like this."
- **Background:**
  - The imagery is Sentinel-1 IW-mode GRD: VV and VH polarizations in
    decibels at 10 m pixel spacing.
  - The scene's white corners are outside the radar swath (no data). Bright
    areas are land; dark areas are calm water.
  - Chips are 800 × 800 pixels; training samples random 512 × 512 crops
    (5.1 km) from them.
  - The overview is decimated 20× for display; the chip is full resolution.
- **If asked "Where is this?"** It's one of the study scenes; the chip shows a
  rocky coastline with small islands, the kind of place where clutter is
  worst.

**Scene map (v2).** Every study scene (111 train, 23 dev, 16 test) and the
50 verified scenes, colored by split, on Natural Earth coastlines; an inset
covers the Gulf of Guinea.
- **Say:** "Scenes cover the Bay of Biscay and Iberia, the North Sea,
  Iceland, the Adriatic and the Gulf of Guinea. Test scenes sit near training
  scenes, so this measures in-region performance."
- **Background:**
  - Each marker is the mean position of that scene's labels (xView3 gives
    every label a latitude and longitude); the scene footprints themselves
    are about 300 × 200 km.
  - The map projection is a simple latitude/longitude plot with a mild
    vertical stretch; it's a locator, not an area-true map.
- **If asked "Why does that matter?"** It's Limits 2: revisits put test
  scenes close to training scenes, so we can't claim transfer to new regions.

The v1 bullet below ("150 frozen study scenes") is no longer printed; the
map and the waffle carry it.

**Bullet: 150 frozen study scenes, 111/23/16; nested budgets.**
- **Say:** "We froze 150 scenes, split by scene, so no scene appears in two
  splits. The four training budgets are nested: every 12-scene scene is also
  in the 28, and so on."
- **Background:**
  - The split is stratified by coarse geographic region and shoreline
    presence (seed 0).
  - Nesting means a bigger budget only adds scenes, so differences between
    budgets come from more data, not different data.
  - Normalization statistics come from training scenes only.
- **If asked "Why only 150 of xView3's scenes?"** To fit 32 full fine-tuning
  runs in the compute budget and to keep every cell on identical data. The
  question is relative (which initialization), not maximum accuracy.

**Bullet: train, dev and test labels are machine-made.**
- **Say:** "Train, dev and test labels come from xView3's automated pipeline:
  a CFAR detector, a ConvNet filter and AIS matching. They miss vessels, so
  they aren't complete ground truth."
- **Background:**
  - CFAR (constant false alarm rate) is a classic radar detector: it flags
    pixels much brighter than their local background.
  - AIS matching links detections to AIS positions. The ConvNet filter
    removes false detections.
  - Each label also has a confidence (HIGH / MEDIUM / LOW). HIGH and MEDIUM
    vessels are positives, HIGH and MEDIUM non-vessels are background, and
    LOW labels define "ignore" regions that are neither rewarded nor
    penalized.
- **If asked "Then why trust test F1?"** It measures agreement with these
  labels, which is fine for comparing initializations under identical
  conditions. That's why we added the human-verified set: absolute
  performance there is much lower.

**Bullet: 50 near-shore, human-verified scenes, opened once.**
- **Say:** "We kept 50 scenes with human-reviewed labels sealed until all 32
  models and their test results were frozen, then scored each model on them
  exactly once."
- **Background:**
  - These are xView3 validation scenes, whose labels add review by
    professional analysts.
  - They contain 8,642 vessels: 3,644 dark (no AIS match) and 2,686 within
    2 km of shore.
  - "Opened once" means no model or threshold choice ever saw them.
    Evaluation hashes bind each result to the frozen cohort.
- **If asked "Why near-shore?"** The verified set is coastal-heavy (31% of
  vessels within 2 km, against 0.4% of training labels), which makes it a
  hard, realistic stress test. It also means its F1 drop mixes a harder scene
  mix with better labels (see Limits).

### What the best detector finds (two radar crops)

- **Say:** "This is the best model, ImageNet ConvNeXt at 111 scenes, on two
  8 × 8 km verified crops. Offshore it finds almost every vessel. In this
  harbor it finds none. Its verified F1 is 0.724 offshore and 0.026 within
  2 km of shore."
- **Background:**
  - Blue rings are found vessels (true positives, a prediction within 200 m).
    Orange squares are missed vessels. Pink × are false alarms.
  - The image is VH backscatter in dB, contrast-stretched (2nd–99.5th
    percentile).
  - Offshore crop: scene `4da9db72dea50504v`. Harbor crop: scene
    `5e9a2c1bcf179e9bv`.
- **If asked "Are these cherry-picked?"** Yes, deliberately, by a fixed rule
  in code. The offshore crop is the window with the highest hit rate among
  windows holding at least 4 vessels and none near shore. The harbor crop is
  the window with the most misses among windows where at least 80% of
  vessels are near shore. They illustrate the pattern; the recall chart in
  column 2 gives the full statistics. Over whole scenes, the offshore scene
  has 106 hits and 143 misses, and the harbor scene 112 and 119.
- **If asked "Why are the harbor vessels missed?"** They sit next to bright
  land clutter, and the model saw almost no near-shore vessels in training
  (1 within 1 km, 26 within 1–2 km). It learned not to fire near land.

---

## 3. Column 2

### 8 arms × 4 budgets = 32 cells (table)

- **Say:** "Two tracks, one per backbone. In each track the architecture is
  fixed and only the starting weights change: random, optical, SAR or
  ImageNet. We compare domains within an architecture, never across."

| Role | ViT-B/16 track | ConvNeXt-V2-Base track |
|---|---|---|
| Random | fresh initialization | fresh initialization |
| Optical | SatDINO [6]: self-supervised DINO on fMoW-RGB (satellite RGB) | BigEarthNet-S2 [8]: supervised 19-class land cover on Sentinel-2 (10 bands) |
| SAR | SARMAE [7]: masked autoencoder on SAR-1M | BigEarthNet-S1 [8]: supervised 19-class land cover on Sentinel-1 (VV, VH) |
| ImageNet | AugReg [5]: supervised ImageNet-1k | FCMAE [4]: self-supervised FCMAE, then supervised ImageNet-1k fine-tune |

- **Background:**
  - All checkpoints are released, revision-pinned weights. We keep the
    encoder and discard the source task heads (classifiers, MAE decoders).
  - Detector sizes: 89,996,801 trainable parameters (ViT) and 93,988,865
    (CNN).
  - The ImageNet arms use ImageNet-1k (about 1.28 M images), not ImageNet-21k.
  - BigEarthNet v2 (reBEN) has about 550 k paired Sentinel-1/Sentinel-2
    patches over Europe.
  - SARMAE's checkpoint holds SAR and optical branches; we load only the SAR
    encoder.
- **Input adaptation:**
  - The BigEarthNet-S1 stem takes 2 bands (VV, VH). We tile it to our 3
    channels and rescale the weights to keep activation magnitude
    ("repeat with rescaling").
  - BigEarthNet-S2 now keeps a seeded 3-channel first convolution and loads
    every other tensor. This is the fix behind Finding 3.
- **If asked "Why these checkpoints?"** They were the strongest public,
  license-compatible checkpoints that match each architecture and source
  domain. The CNN pair is the cleanest test of modality, since S1 and S2
  share architecture, dataset family and task. The ViT pair differs in
  objective too (DINO vs MAE), which is why we don't read a pure modality
  effect from it.
- **If asked "Why not pool the two tracks?"** Their architectures and
  checkpoint histories differ, so we judge generality by whether the sign
  and pattern of SAR minus optical agree across tracks. They don't.
- **If asked "Doesn't optical pretraining on RGB lose information?"** All
  arms see the same 3-channel radar input; pretraining only sets the starting
  weights. Fine-tuning updates the full encoder.

### Budget waffle (v2, under the arms table)

111 squares, one per training scene, in columns of four. The darkest block
is the 12-scene budget; each lighter block adds scenes up to 28, 56 and 111.
- **Say:** "Each square is a labeled training scene. The budgets are nested:
  the 28-scene set contains the 12, and so on, so a bigger budget only adds
  data."
- **Background:** It replaces the v1 sentence "Only the initialization
  changes inside a track; we compare domains within an architecture." Say
  that sentence aloud when you walk the table.

### One shared point detector (diagram and bullets)

Walk the diagram left to right, then along the bottom right to left.

1. **[VH, VV, VH − VV] 512² crop.** Two polarizations in dB plus their
   difference, which is the cross-pol ratio in dB. Ships tend to stand out
   more in the ratio than sea does.
2. **One encoder per track.** ViT-B/16 outputs patch features at stride 16
   (32 × 32 for a 512 crop). ConvNeXt-V2-B outputs stride 32, plus one
   learned upsampling stage.
3. **Shared 256-channel head.** It predicts a vessel-presence heatmap at
   stride 4 (128 × 128 cells of 40 m). Targets are Gaussians with σ = 2
   cells around each vessel.
4. **Heatmap peaks ≥ 0.05, 120 m NMS.** Local maxima above a low candidate
   floor become detections. Non-maximum suppression removes duplicates
   within 120 m. Full scenes are tiled in 512-pixel windows at stride 384.
5. **Threshold τ picked on dev.** Each run sweeps thresholds on 8 dev scenes
   and keeps the checkpoint and threshold with the best dev F1 (τ ranges
   0.59–0.94 across cells). That pair is stored with the checkpoint hash.
6. **F1, 200 m geo match.** Predictions are greedily matched to labels
   within 200 m in confidence order. Unmatched predictions are false
   positives; unmatched labels are false negatives. Predictions near LOW
   labels are ignored.

**Bullet: CenterNet-style heatmap [9], focal loss [10]; strict FP32, 567.6 H100 GPU-hours.**
- **Background:**
  - CenterNet ("objects as points") detects each object as a heatmap peak at
    its center.
  - The loss is CenterNet's penalty-reduced focal loss (α = 2, β = 4), which
    down-weights the overwhelming number of easy background pixels.
  - Strict FP32 means TF32 is disabled for matrix multiplies and cuDNN.
  - Cells ran one per GPU on an 8-GPU H100 node, from about 2 GPU-hours
    (early-stopped small runs) to about 51 (CNN at 111 scenes).

**Bullet: same optimizer, schedule, data order and seed (0) in every cell.**
- **Background:**
  - AdamW, learning rate 1e-4, weight decay 0.05, layer-wise decay 0.65,
    gradient clip 1.0.
  - 5 warm-up epochs, then cosine decay, at most 50 epochs, batch 16.
  - Sampling: half of the sampled chips contain vessels and half are
    background chips. In a chip with vessels, 70% of crops are placed within
    128 pixels of a vessel. Augmentation: flips, 90° rotations, ±1 dB
    intensity jitter.
  - Dev evaluation every 5 epochs; early-stopping patience is 4 evaluations
    (20 epochs).
- **If asked "Isn't one recipe unfair to some arms?"** Possibly, and we list
  it. One recipe is the controlled choice: any per-arm tuning would make the
  comparison depend on tuning effort. SARMAE's 28-scene dip (Limits) is the
  clearest sign that a different learning rate or patience could help some
  arms.

**Bullet: 8 dev scenes pick checkpoint and threshold; test and verified scenes scored once.**
- **Background:**
  - Dev F1 is optimistic because of this selection: it averages 0.078 above
    test F1 (range 0.029–0.142).
  - Test and verified scenes use the stored threshold with no retuning.
- **If asked "Why only 8 of the 23 dev scenes?"** For speed: dev runs every 5
  epochs in all 32 cells. The fixed first 8 (sorted) dev scenes hold 517
  vessels. It's listed as a limitation.

### Recall falls toward the coast (two-panel chart)

**Top panel, "Where the vessels are."** The share of vessels in each
distance-to-shore bin: gray is training labels (machine), navy is verified
labels (human).

| Distance to shore | 0–1 km | 1–2 km | 2–5 km | 5–10 km | > 10 km |
|---|---|---|---|---|---|
| Training vessels | 1 (0.0%) | 26 (0.3%) | 507 (6.6%) | 665 (8.7%) | 6,479 (84.4%) |
| Verified vessels | 2,058 (24%) | 628 (7%) | 1,001 (12%) | 776 (9%) | 4,179 (48%) |

**Bottom panel, "Where the detectors find them."** Recall per bin on the
50 verified scenes. The gray band spans all 32 cells; the green line is the
best cell.

| Recall | 0–1 km | 1–2 km | 2–5 km | 5–10 km | > 10 km |
|---|---|---|---|---|---|
| Best cell | 0.005 | 0.040 | 0.306 | 0.608 | 0.676 |
| All 32 cells (range) | 0.001–0.009 | 0.027–0.065 | 0.157–0.333 | 0.302–0.617 | 0.486–0.695 |

- **Say:** "Training labels put 84% of vessels more than 10 km offshore. The
  verified set has 31% within 2 km. Recall collapses toward the coast for
  every model: under 1% within 1 km."
- **Background:**
  - xView3 records distance to shore only up to 10 km; larger distances are
    coded 9999.99, hence the "> 10" bin.
  - Recall is computed by re-scoring each cell's stored predictions against
    the verified labels. This reproduces every cell's stored TP/FP/FN
    exactly, which validates the per-bin numbers.
- **If asked "Is it the labels or the clutter?"** Both, and we can't fully
  separate them with this design. The models predict almost nothing near
  shore (the best cell puts 34 of its 4,302 predictions within 2 km), which
  points to learned suppression from label-poor training data more than to
  confusion. Clutter is real too. A plausible reason the training labels
  lack near-shore vessels is that automated detection and AIS matching are
  hardest next to land, though scene selection may also contribute; we
  haven't separated the two.

---

## 4. Column 3

### Pretraining helps most when labels are scarce (2 × 2 chart)

- **What it shows:** F1 against the number of training scenes (12, 28, 56,
  111). Columns are ViT-B/16 and ConvNeXt-V2-Base. The top row is 16 test
  scenes; the bottom row is 50 verified scenes. Colors: gray dashed = random,
  orange = optical, blue = SAR, green = ImageNet.
- **Say:** "Every pretrained line starts well above random at 12 scenes. The
  gap closes as labels grow. Note the different y-axes: verified F1 is about
  0.3 lower."
- **Background, test F1 at 12 → 111 scenes:**
  - ViT: random 0.693 → 0.831; optical 0.806 → 0.857; SAR 0.804 → 0.886;
    ImageNet 0.813 → 0.851.
  - CNN: random 0.657 → 0.816; optical 0.720 → 0.840; SAR 0.800 → 0.822;
    ImageNet 0.792 → 0.888.
- **Background, verified F1 at 12 → 111:**
  - ViT: random 0.412 → 0.522; optical 0.486 → 0.562; SAR 0.493 → 0.559;
    ImageNet 0.523 → 0.558.
  - CNN: random 0.409 → 0.511; optical 0.457 → 0.552; SAR 0.501 → 0.478;
    ImageNet 0.464 → 0.574.
- **Talking point: 12 ≈ 111.** ViT-ImageNet with 12 scenes is statistically
  indistinguishable from ViT-random with 111: 0.813 vs 0.831 test, interval
  [−0.067, +0.018]; 0.523 vs 0.522 verified, interval [−0.016, +0.019]. Say
  "indistinguishable", not "equal". The report also notes that 28 pretrained
  scenes match the full-data random floor on test in both tracks.
- **Two dips to expect questions on:**
  - ViT-SAR drops from 12 to 28 scenes (test 0.804 → 0.772).
  - CNN-SAR drops from 56 to 111 (test 0.848 → 0.822; verified 0.528 →
    0.478).
- **If asked about either dip:** both runs picked their epoch-4 checkpoint,
  the first dev evaluation, right as warm-up ends, and early stopping ended
  them at epoch 25.
  - For ViT-SAR at 28 we traced it in detail. Training loss fell from 0.482
    to 0.034 while dev precision fell from 0.858 to 0.768, so the model
    overfit, and the early checkpoint's threshold transferred poorly.
  - CNN-SAR at 111 shows the same epoch-4 / epoch-25 pattern in its run
    record.
  - With more data (ViT-SAR at 56 and 111) the run trained to epoch 39.
  - The August cohort reproduced the ViT-SAR dip exactly.

### Reference detectors (v2: ★ on the chart, note in the caption)

**YOLO26 (★ in both test panels).** A standard supervised box detector
trained on all 111 scenes in the earlier V100 reference campaign, scored on
the same 16 test scenes with the same label rules and scorer: test F1 0.896
(1,062 found, 144 false alarms, 103 missed; near-shore F1 0.000). It sits
just right of the 111 tick so it doesn't hide data, and it was never run on
the verified set.
- **Say:** "A tuned off-the-shelf detector with all the labels lands where
  our best cell does. The study is about the low-label end, where the
  starting weights matter."
- **If asked "Why not include it in the comparison?"** It's a different
  detector family with its own recipe and only one budget, so it's context,
  not an arm.

**LocateAnything-3B zero-shot (caption).** NVIDIA's open-vocabulary
vision-language model, prompted with text and no training on SAR. Best
prompt "boat": F1 0.122 (precision 0.161, recall 0.098) on a sample of dev
chips holding 244 vessels; "ship" 0.047, "vessel" 0.016.
- **Say:** "General vision-language models don't transfer to radar out of
  the box."
- **If asked "Why only in the caption?"** It was scored on dev chips, not the
  test scenes, so plotting it beside the others would mislead.

### SAR vs. optical: no stable winner (difference chart)

- **How to read it:** each point is SAR F1 minus optical F1 at one budget.
  Above zero, SAR wins; below zero, optical wins. Filled circles on solid
  lines are test; open squares on dotted lines are verified. Bars are 95%
  intervals. A bar that doesn't cross zero is a win.
- **Say:** "In the CNN, SAR's lead is clear at 12 and 28 scenes and gone by
  111, where optical wins on verified. In the ViT there's no trend: optical
  wins at 28, SAR at 56."
- **Background, all 16 differences:**

| Budget | ViT test | ViT verified | CNN test | CNN verified |
|---|---|---|---|---|
| 12 | −0.002 [−0.046, +0.039] | +0.008 [−0.003, +0.019] | **+0.080** [+0.036, +0.121] | **+0.045** [+0.027, +0.062] |
| 28 | −0.066 [−0.169, +0.023] | **−0.081** [−0.129, −0.036] | **+0.052** [+0.028, +0.077] | **+0.014** [+0.001, +0.028] |
| 56 | **+0.028** [+0.011, +0.043] | **+0.053** [+0.041, +0.066] | +0.026 [−0.004, +0.064] | −0.006 [−0.016, +0.003] |
| 111 | +0.029 [−0.006, +0.072] | −0.003 [−0.011, +0.007] | −0.018 [−0.079, +0.056] | **−0.073** [−0.101, −0.046] |

- **How the intervals are made:** a paired scene bootstrap with 10,000
  resamples and seed 0. Each resample draws evaluation scenes with
  replacement; both arms are scored on the same draw, so the difference is
  paired. Micro-F1 is recomputed from summed TP/FP/FN. The interval is the
  2.5th–97.5th percentile.
- **If asked "Do intervals cover training randomness?"** No, only scene
  sampling. That's why the caption gives the rerun number: retraining 28
  unchanged cells moved test F1 by 0.008 on average and up to 0.043. Treat
  differences under about 0.04 as possibly run-to-run noise, even when the
  interval excludes zero.
- **If asked "Why is the CNN pair the 'cleanest' comparison?"**
  BigEarthNet-S1 and -S2 share architecture, dataset family (the same
  European patches) and supervised task, so the main difference is the
  sensor. Even there, the SAR advantage shrinks as labels grow.

### References

Ten references in two columns; `references.pdf` is the printable copy with
full titles and page ranges.

| # | Reference | Cited for |
|---|---|---|
| 1 | Paolo et al., xView3-SAR, NeurIPS 2022 | dataset, labels, dark-vessel task |
| 2 | Torres et al., GMES Sentinel-1 mission, RSE 2012 | the sensor |
| 3 | Dosovitskiy et al., ViT, ICLR 2021 | ViT-B/16 backbone |
| 4 | Woo et al., ConvNeXt V2, CVPR 2023 | CNN backbone and FCMAE |
| 5 | Steiner et al., How to train your ViT (AugReg), TMLR 2022 | ViT ImageNet checkpoint |
| 6 | Straka & Gruber, SatDINO, arXiv 2025 | ViT optical checkpoint |
| 7 | Liu et al., SARMAE, CVPR 2026 | ViT SAR checkpoint |
| 8 | Clasen et al., reBEN (BigEarthNet v2), IGARSS 2025 | CNN S1 and S2 checkpoints |
| 9 | Zhou et al., Objects as points (CenterNet), 2019 | heatmap detector |
| 10 | Lin et al., Focal loss, ICCV 2017 | loss |

---

## 5. Column 4

### Findings

**v2 printed findings:**
1. **Pretraining pays off with few labels:** +0.063 to +0.143 test F1 at 12
   scenes.
2. **SAR pretraining rarely wins:** the scoreboard (below).
3. **The input adapter matters:** a seeded 3-channel stem lifts
   BigEarthNet-S2 above random at every budget.
4. **No detector works near shore:** recall ≤ 0.021 within 2 km in every
   cell.

**Scoreboard (v2, finding 2).** Two 4 × 4 dot grids. Rows are budgets (12,
28, 56, 111 scenes); columns are ViT test, ViT verified, CNN test, CNN
verified. Blue = SAR wins, orange = optical wins, green = ImageNet wins,
hollow = tie. A win means the whole 95% interval of SAR F1 minus the rival's
F1 is on one side of zero.
- **Say:** "Each dot is one head-to-head. Against optical, SAR wins 6, loses
  2 and ties 8, and its wins cluster in the CNN at 12–28 scenes. Against
  ImageNet it wins 3 and loses 7."
- **Background:** It carries v1 findings 2 and 5 (below) and the answer
  strip. The difference chart in column 3 shows the same SAR-vs-optical
  intervals as numbers.

The v1 findings below are the background for each item.

**1. Pretraining pays off with few labels.** Gains at 12 scenes: +0.063 to
+0.143 test F1, +0.048 to +0.111 verified F1.
- **Background (12-scene gain over random, 95% interval):**

| Arm | Test | Verified |
|---|---|---|
| ViT optical | +0.114 [+0.069, +0.159] | +0.074 [+0.060, +0.089] |
| ViT SAR | +0.111 [+0.081, +0.141] | +0.081 [+0.068, +0.095] |
| ViT ImageNet | +0.120 [+0.067, +0.177] | +0.111 [+0.092, +0.130] |
| CNN optical | +0.063 [+0.020, +0.107] | +0.048 [+0.034, +0.063] |
| CNN SAR | +0.143 [+0.067, +0.208] | +0.093 [+0.066, +0.120] |
| CNN ImageNet | +0.135 [+0.099, +0.180] | +0.056 [+0.037, +0.074] |

- **If asked "Does the gain persist at 111?"** It shrinks, and some arms tie
  or trail random on verified (CNN-SAR at 111: 0.478 vs random 0.511). The
  value of pretraining is mostly in the low-label regime.

**2. SAR's edge over optical depends on backbone and budget.** CNN +0.080
[+0.036, +0.121] test F1 at 12 scenes, but −0.073 [−0.101, −0.046] verified
F1 at 111. ViT: the order flips between 28 and 56 scenes.
- See the SAR-vs-optical table above.
- **If asked "Why would SAR pretraining fade with more labels?"** With enough
  target labels, fine-tuning overwrites much of what the initialization
  gave. Also, neither SAR corpus is maritime-specific: BigEarthNet-S1 is
  land cover over Europe, so "same sensor" is not "same task".

**3. The input adapter matters.** With a seeded 3-channel stem instead of
band slicing, BigEarthNet-S2 beats random at every budget (+0.025 to +0.063
test F1).
- **Background:** The S2 checkpoint expects 10 bands. An earlier cohort kept
  3 of the 10 first-layer band kernels and scaled them by 10/3. That raised
  the first-layer weight norm 5.85-fold and left the arm at or below random.
  The fix keeps a seeded 3-channel first convolution and loads everything
  else. All 32 cells were then retrained under one code version
  (replacement32).
- **If asked "So the optical CNN result depends on a bug fix?"** Yes, and
  that's the point of the finding: cross-sensor transfer studies should
  report their adapter rule, because it can decide the result.

**4. No detector works near shore.** Recall ≤ 0.021 within 2 km in every
cell. Training labels put 0.4% of vessels there; the verified set, 31%.
- **Background:** Near-shore F1 is 0.015–0.041 across cells. Offshore
  verified F1 (excluding near-shore vessels and predictions) is 0.518–0.724.
- See the recall chart and crops for the story.

**5. ImageNet is the strongest default.** ImageNet beats SAR pretraining in 7
of 16 comparisons (SAR wins 3). The best cell on both test (0.888) and
verified F1 (0.574) is the ImageNet ConvNeXt with 111 scenes.
- **If asked "Why does ImageNet do so well?"**
  - It's a large, diverse, carefully labeled corpus. Supervised ImageNet
    features are object-centric, which suits "find a small bright object".
  - FCMAE adds long self-supervised pretraining before the supervised
    fine-tune.
  - Image counts are of the same order (ImageNet-1k is about 1.28 M images;
    SAR-1M is on the order of a million SAR images), so it isn't simply more
    images. Objective, labels and schedule differ too.
- **If asked "So should practitioners skip SAR pretraining?"** Our data says
  ImageNet is a strong, safe default. SAR pretraining gave the best CNN
  result at 12 scenes on verified (+0.037 over ImageNet) and the best ViT
  result at 111 on test. A matched comparison would settle it (Open
  questions).

### Inside the detector (v2)

The best cell's heatmap (ImageNet ConvNeXt, 111 scenes) on the same two
crops as "What the best detector finds", over the radar image. Color is
detector confidence (0–1); the blue tick on the color bar is its operating
threshold (0.896, shown as 0.90). Peak confidence: 0.987 offshore, 0.001 in
the harbor.
- **Say:** "This is what the network actually outputs. Offshore every vessel
  lights up above threshold. In the harbor the model isn't unsure, it's
  silent: its highest confidence anywhere in the window is 0.001."
- **Background:**
  - Made on the local GPU from the cell's `best.ckpt`, whose SHA-256 matches
    the evidence. The code repeats the scorer's own tiling (512-pixel tiles
    at stride 384, max-pasted), so these are the values the scorer saw.
  - Each peak covers a few 40 m cells, so the display widens it with a
    200 m maximum filter (the matching radius). The peak values come from
    the raw heatmap.
- **If asked "So is the coast the problem, or the labels?"** Confident
  silence points to learned suppression: training labels had 27 vessels
  within 2 km, so near land the model learned "no vessel here". A
  confused model would show medium confidence on clutter instead.

### Limits

**v2 printed limits:** "One seed per cell; 8 dev scenes for selection.",
"Machine-made test labels; in-region test scenes.", "Checkpoints also differ
in objective and data." The v1 versions below give the detail, and the
SARMAE overfitting story (v1 item 3) is now talk-only.

1. **One seed per cell; 8 dev scenes pick checkpoints and thresholds.**
   Single seed because 32 full fine-tunes cost 567.6 GPU-hours. The rerun
   evidence (≤ 0.043, mean 0.008) is our proxy for seed noise.
2. **Dev and test labels are machine-made.** Test F1 measures agreement with
   those labels, not with every vessel.
3. **Two SAR test curves drop by more than 0.02 as labels grow; ViT-SAR at 28
   overfit after warm-up and early stopping kept epoch 4.** This is the
   predeclared monotonicity gate: no curve may drop more than 0.02 as labels
   grow. It failed for ViT-SAR 12 → 28 (−0.033) and CNN-SAR 56 → 111
   (−0.026). We disclosed the failure, didn't retune, and ran the final
   evaluation as planned.
4. **Checkpoints differ in objective and corpus, not only domain.** SatDINO
   uses DINO and SARMAE uses MAE; the ImageNet arms are supervised. A
   "domain" effect is therefore confounded with objective and data.
5. **In-region test.** Sentinel-1 revisits place some test scenes near
   training scenes, so test F1 measures in-region generalization, not
   transfer to new regions. No scene crosses splits.

- **If asked "Did you correct for multiple comparisons?"** No. We report
  intervals and counts descriptively. The headline is a pattern (wins
  scattered, no consistent order), which is robust to that choice.

### Open questions & next steps

**v2 printed lines:** "Would joint SAR + optical foundation models win?",
"Does SAR trail ImageNet with matched pretraining?", "Next: near-shore
labels, shoreline input, seeds." The v1 detail follows; the "where optical
can't help" question (v1 item 3) is now talk-only.

1. **Geospatial foundation models pretrained jointly on SAR and optical:**
   do they beat these single-source checkpoints? Multi-sensor pretraining is
   the obvious next arm. The poster deliberately names no specific model.
2. **Matched SAR vs ImageNet:** does SAR still trail ImageNet when both use
   the same objective, architecture and number of pretraining images? This
   would separate domain from recipe (Limits 4).
3. **Where optical can't help:** does SAR pretraining help near shore and
   for small dark vessels? Dark-vessel recall is only 0.136–0.228 in every
   cell, so there's room to show a domain effect where it matters most.
4. **Next steps:**
   - **Near-shore training labels:** the root cause of finding 4.
   - **A shoreline input channel:** see below.
   - **Several seeds:** turns the rerun proxy into proper seed intervals.
   - **Matched pretraining objectives across sensors:** the same question as
     open question 2.

**Background on the shoreline channel (if asked).** Add a fourth per-pixel
input: distance to shore (capped, scaled) or a land/water mask, derived from
the `bathymetry.tif` or `owiMask.tif` layer that ships with each xView3
scene. It lets the model tell "bright spot on land" from "bright spot on
water next to land", so it can suppress land returns and use a more
sensitive threshold near shore.
- It only helps together with near-shore labels; alone it may teach "near
  shore means no vessel" faster.
- To keep the encoder comparison fair, add it at the head (late fusion), so
  pretrained stems stay untouched (compare finding 3).

---

## 6. Tough questions

- **"Isn't this just 'use ImageNet'?"** For this detector and these
  checkpoints, ImageNet is the most reliable default. The contribution is
  the controlled test: same detector, recipe, data, nested budgets, once-only
  verified evaluation, and honest intervals. It shows SAR-domain pretraining
  isn't automatically better.
- **"Your intervals are narrow on verified. Why?"** 50 scenes and 8,642
  vessels give tighter scene-sampling intervals than 16 test scenes with
  1,165 vessels. Training noise isn't in them (see the rerun number).
- **"Why F1 and not average precision?"** Operations need a single operating
  point. F1 at a threshold fixed on dev is what a deployed system would
  achieve, and it exposes threshold-transfer failures (like the SAR dips)
  that AP would hide.
- **"Why 200 m matching?"** It follows the xView3 evaluation convention for
  point detections at this resolution. Labels and detections are points, not
  boxes.
- **"Could test data leak into training?"** Splits are by scene, frozen and
  hashed. Training never loads test or verified labels. Revisits can put
  scenes of the same area in different splits; that's Limits 5.
- **"Why so few scenes when xView3 has many more?"** The study measures
  relative label efficiency under a fixed compute budget. All 32 cells train
  on identical nested subsets.
- **"Why ViT-B and ConvNeXt-V2-B?"** They are representative transformer and
  CNN backbones of similar size (about 90 M parameters as detectors) with
  public checkpoints in every role we needed.
- **"What would you do with more compute?"** Several seeds, the full dev set
  for selection, per-arm tuning as a sensitivity check, near-shore labels,
  and multi-sensor foundation models.
- **"Why is verified F1 0.3 lower than test?"** The verified set has human
  labels that include vessels the automated pipeline missed, and it's
  coastal-heavy. Precision stays at 0.639–0.909, while recall falls to
  0.286–0.434: the models miss vessels rather than invent them.
- **"What does 'dark' mean in your numbers?"** A verified vessel with no AIS
  match: 3,644 of 8,642. Recall on them is 0.136–0.228.

---

## 7. Reference tables

### Every cell (verified columns on the 50 scenes)

| Cell | Dev F1 | Test F1 | Verified F1 | Precision | Recall | Dark recall | Near-shore F1 | Threshold | Best / last epoch | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|
| vitrand-f10 | 0.792 | 0.693 | 0.412 | 0.706 | 0.291 | 0.150 | 0.018 | 0.81 | 14 / 35 | 3.1 |
| vitrand-f25 | 0.857 | 0.773 | 0.456 | 0.767 | 0.325 | 0.155 | 0.019 | 0.86 | 24 / 45 | 7.5 |
| vitrand-f50 | 0.893 | 0.809 | 0.501 | 0.806 | 0.363 | 0.174 | 0.023 | 0.85 | 34 / 50 | 14.9 |
| vitrand-f100 | 0.900 | 0.831 | 0.522 | 0.824 | 0.382 | 0.182 | 0.038 | 0.82 | 24 / 45 | 25.2 |
| satdino-f10 | 0.893 | 0.806 | 0.486 | 0.806 | 0.347 | 0.149 | 0.022 | 0.90 | 49 / 50 | 4.2 |
| satdino-f25 | 0.885 | 0.837 | 0.506 | 0.768 | 0.377 | 0.166 | 0.023 | 0.92 | 29 / 50 | 8.2 |
| satdino-f50 | 0.920 | 0.837 | 0.515 | 0.848 | 0.370 | 0.159 | 0.020 | 0.94 | 34 / 50 | 14.8 |
| satdino-f100 | 0.922 | 0.857 | 0.562 | 0.862 | 0.417 | 0.184 | 0.028 | 0.90 | 44 / 50 | 28.0 |
| sarmae-f10 | 0.886 | 0.804 | 0.493 | 0.747 | 0.368 | 0.189 | 0.031 | 0.59 | 4 / 25 | 2.2 |
| sarmae-f25 | 0.862 | 0.772 | 0.425 | 0.829 | 0.286 | 0.136 | 0.015 | 0.77 | 4 / 25 | 4.1 |
| sarmae-f50 | 0.918 | 0.865 | 0.568 | 0.858 | 0.425 | 0.197 | 0.027 | 0.88 | 39 / 50 | 14.9 |
| sarmae-f100 | 0.929 | 0.886 | 0.559 | 0.909 | 0.403 | 0.168 | 0.021 | 0.92 | 39 / 50 | 28.2 |
| vitin1k-f10 | 0.898 | 0.813 | 0.523 | 0.790 | 0.390 | 0.178 | 0.031 | 0.85 | 39 / 50 | 4.2 |
| vitin1k-f25 | 0.877 | 0.848 | 0.500 | 0.803 | 0.363 | 0.150 | 0.024 | 0.89 | 29 / 50 | 8.1 |
| vitin1k-f50 | 0.923 | 0.859 | 0.551 | 0.841 | 0.409 | 0.178 | 0.021 | 0.86 | 39 / 50 | 14.8 |
| vitin1k-f100 | 0.937 | 0.851 | 0.558 | 0.871 | 0.411 | 0.175 | 0.028 | 0.86 | 44 / 50 | 28.2 |
| cnnrand-f10 | 0.799 | 0.657 | 0.409 | 0.639 | 0.300 | 0.168 | 0.016 | 0.72 | 29 / 50 | 7.1 |
| cnnrand-f25 | 0.817 | 0.706 | 0.441 | 0.669 | 0.329 | 0.173 | 0.019 | 0.82 | 44 / 50 | 14.2 |
| cnnrand-f50 | 0.872 | 0.780 | 0.492 | 0.774 | 0.361 | 0.183 | 0.025 | 0.85 | 49 / 50 | 26.6 |
| cnnrand-f100 | 0.900 | 0.816 | 0.511 | 0.810 | 0.373 | 0.178 | 0.020 | 0.86 | 39 / 50 | 51.5 |
| beS2-f10 | 0.837 | 0.720 | 0.457 | 0.643 | 0.354 | 0.193 | 0.027 | 0.83 | 49 / 50 | 6.8 |
| beS2-f25 | 0.862 | 0.751 | 0.479 | 0.705 | 0.363 | 0.199 | 0.020 | 0.89 | 34 / 50 | 14.0 |
| beS2-f50 | 0.883 | 0.821 | 0.534 | 0.738 | 0.419 | 0.227 | 0.023 | 0.86 | 39 / 50 | 26.8 |
| beS2-f100 | 0.904 | 0.840 | 0.552 | 0.756 | 0.434 | 0.228 | 0.026 | 0.85 | 49 / 50 | 51.4 |
| beS1-f10 | 0.876 | 0.800 | 0.501 | 0.678 | 0.398 | 0.194 | 0.041 | 0.78 | 34 / 50 | 6.7 |
| beS1-f25 | 0.887 | 0.803 | 0.493 | 0.745 | 0.369 | 0.187 | 0.027 | 0.88 | 14 / 35 | 9.8 |
| beS1-f50 | 0.900 | 0.848 | 0.528 | 0.809 | 0.392 | 0.179 | 0.020 | 0.90 | 29 / 50 | 26.5 |
| beS1-f100 | 0.907 | 0.822 | 0.478 | 0.862 | 0.331 | 0.147 | 0.015 | 0.83 | 4 / 25 | 25.8 |
| cnnin1k-f10 | 0.889 | 0.792 | 0.464 | 0.837 | 0.321 | 0.142 | 0.040 | 0.86 | 4 / 25 | 3.4 |
| cnnin1k-f25 | 0.896 | 0.846 | 0.500 | 0.807 | 0.362 | 0.169 | 0.026 | 0.94 | 9 / 30 | 8.5 |
| cnnin1k-f50 | 0.927 | 0.873 | 0.562 | 0.859 | 0.418 | 0.180 | 0.027 | 0.89 | 34 / 50 | 26.5 |
| cnnin1k-f100 | 0.935 | 0.888 | 0.574 | 0.900 | 0.421 | 0.178 | 0.026 | 0.90 | 49 / 50 | 51.4 |

Cell names: `vitrand`/`cnnrand` = random, `satdino`/`beS2` = optical,
`sarmae`/`beS1` = SAR, `vitin1k`/`cnnin1k` = ImageNet; `f10`–`f100` = 12, 28,
56, 111 scenes. "Best / last epoch" is the selected checkpoint and the epoch
where training stopped (50 = ran to the end).

### SAR vs ImageNet, all 16 (95% intervals; bold = win)

| Budget | ViT test | ViT verified | CNN test | CNN verified |
|---|---|---|---|---|
| 12 | −0.009 [−0.064, +0.038] | **−0.029** [−0.046, −0.013] | +0.008 [−0.055, +0.061] | **+0.037** [+0.014, +0.060] |
| 28 | **−0.076** [−0.168, −0.006] | **−0.075** [−0.120, −0.034] | **−0.043** [−0.085, −0.009] | −0.007 [−0.018, +0.005] |
| 56 | +0.006 [−0.013, +0.025] | **+0.018** [+0.008, +0.027] | **−0.026** [−0.044, −0.005] | **−0.034** [−0.045, −0.023] |
| 111 | **+0.035** [+0.002, +0.079] | +0.000 [−0.009, +0.010] | −0.066 [−0.129, +0.007] | **−0.095** [−0.121, −0.069] |

### Label support

| Set | Scenes | Vessels (HIGH/MEDIUM) | Non-vessel background | LOW (ignored) |
|---|---|---|---|---|
| Train (all 111) | 111 | 7,678 | — | — |
| Dev used for selection | 8 | 517 | 107 | 118 |
| Dev (full) | 23 | 1,479 | 804 | 441 |
| Test | 16 | 1,165 | 420 | 325 |
| Verified | 50 | 8,642 (3,644 dark, 2,686 within 2 km) | — | — |

---

## 8. Glossary

- **AIS:** Automatic Identification System; ship transponder broadcasts.
- **Dark vessel:** a vessel with no matching AIS report.
- **SAR:** synthetic aperture radar; an active microwave imager.
- **Sentinel-1 IW GRD:** ESA C-band SAR, Interferometric Wide swath,
  Ground Range Detected product (10 m pixels, about 250 km swath).
- **VV / VH:** co-polarized (vertical send, vertical receive) and
  cross-polarized (vertical send, horizontal receive) backscatter.
- **dB:** decibels, 10·log10 of backscatter power; VH − VV in dB is the
  cross-pol ratio.
- **CFAR:** constant false alarm rate detector; thresholds each pixel
  against its local background statistics.
- **Precision / recall / F1:** fraction of predictions that are vessels;
  fraction of vessels found; their harmonic mean.
- **Micro-F1:** F1 from TP/FP/FN summed across all scenes.
- **Paired scene bootstrap:** resample scenes with replacement and score
  both models on the same resample; the spread of the difference gives the
  interval.
- **DINO / MAE / FCMAE:** self-supervised pretraining by self-distillation,
  masked-image reconstruction, and its fully convolutional variant for
  ConvNeXt.
- **AugReg:** supervised ViT training with strong augmentation and
  regularization.
- **CenterNet:** detects objects as heatmap peaks at their centers.
- **Focal loss:** a loss that down-weights easy examples, used for extreme
  foreground/background imbalance.
- **NMS:** non-maximum suppression; keeps the strongest of nearby detections.
- **Layer-wise learning-rate decay:** lower learning rates for earlier
  encoder layers during fine-tuning (factor 0.65 per layer here).
- **Early stopping (patience 4):** stop if dev F1 doesn't improve for 4
  evaluations (20 epochs).
- **Nested budgets:** each smaller training set is a subset of the larger
  ones.
