# Scientific Decision Record

This file records decisions that affect interpretation or reproducibility. Site
operations, transfer credentials, scheduler settings, and retired machine paths
belong outside the public repository.

## D1: Scene-level partitions

The study splits xView3 by scene rather than by chip. A scene identifier cannot
appear in more than one partition. The frozen split assigns 111 training, 23
development, and 16 test scenes, with 50 separate human-verified scenes reserved
for one final evaluation. All label fractions draw nested subsets from the fixed
training partition.

## D2: Train-only normalization

The project computes SAR normalization statistics once from the full training
scene set and reuses them for each fraction and architecture. Recomputing
statistics on a fraction would make input normalization another experimental
variable.

## D3: Point-native detector and frozen scorer

Every core arm uses one CenterNet-style heatmap detector, peak decoding, and
distance NMS. The scorer matches predictions to reference points in geodesic
space. Its near-shore false-positive correction is frozen and hash-pinned.
Threshold selection remains outside the scorer and uses development data.

## D4: Two architecture-matched tracks

The ViT track uses ViT-B/16 for random, optical, SAR, and ImageNet roles. The CNN
track uses ConvNeXt-V2-Base for the same roles. The project compares pretraining
sources within a track. Matched-role cross-track comparisons measure an
architecture difference and carry the stated checkpoint-history caveat.

## D5: Downloaded encoders only

The active study downloads all six pretrained encoders. It performs no new
pretraining. The ImageNet roles replaced the retired LS-SSDD supervised arms.
The frozen LS-SSDD split remains as historical provenance and never enters the
active training path.

## D6: One run seed

The core grid uses seed 0 only: eight arms by four label fractions, or 32 core
fine-tunes. The two external references produce 34 total reported experiments
when their corrected records pass. The paper cannot claim seed variance or show
empirical error bars.

## D7: Strict FP32 after non-finite FP16 loss

A uniform mixed-precision campaign produced a non-finite forward loss in the
ImageNet-initialized CNN despite the focal-loss computation already running in
an FP32 island. The owner approved a uniform restart of every core cell with
Lightning `32-true`. The project also disables TF32 so Ampere-or-newer hardware
does not substitute reduced-mantissa matrix multiplication inside the FP32
recipe. External references retain their published independent precision.

This amendment changes the shared numerical recipe, not one arm. It preserves
within-track fairness and provides the relevant public explanation for the
discarded FP16 campaign. Mixed-precision outputs remain diagnostics and cannot
enter result tables.

## D8: H100 is the canonical core hardware

All 32 core cells restart from scratch on H100 GPUs under one code revision and
one strict-FP32 environment. V100 runs continue only as engineering diagnostics.
The paper cannot combine V100 and H100 core cells, even when an experiment name
matches.

## D9: Corrected training targets

The label adapter treats HIGH/MEDIUM vessel rows as positives and LOW-confidence
rows as ignores. HIGH/MEDIUM non-vessel rows cannot become positive centers. A
source audit identified the earlier caller-side conversion error; the scorer
itself required no change.

## D10: Checkpoint-bound operating points

Each reportable development score binds a checkpoint hash, threshold, and
metric record. A threshold selected for `last.ckpt` cannot score `best.ckpt`.
Held-out evaluation remains blocked until an immutable core cohort and its
operating points exist.

## D11: Deadline and complete paper profiles

The result generator accepts only finite values from verified H100 `DONE`
records. A deadline report may compare a label fraction only after all eight
arms at that fraction finish. Other cells appear as status entries without
numeric comparisons. The complete profile requires all 32 core cells.

The class paper reports corrected development-selection metrics if the sealed
50-scene evaluation has not run by the deadline. It must label that limitation
and cannot imply final held-out performance.

## D12: Public submission structure

The class report uses one-inch margins. Its main body, from Introduction through
Conclusion, may occupy no more than five pages. References and appendices follow
outside that limit. A longer AIPR manuscript remains a supplemental work in
progress and uses the same machine-readable H100 snapshot.

The public Git tree retains frozen split and statistics metadata. The Canvas
archive excludes the entire `data/` directory and all imagery, labels, weights,
checkpoints, run trees, environments, credentials, and private infrastructure
paths. The MIT grant applies only to original project code and documentation;
third-party data, checkpoints, software, and publication assets retain their
upstream terms.
