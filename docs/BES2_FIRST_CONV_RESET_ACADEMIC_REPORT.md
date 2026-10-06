# Diagnosing Negative Transfer from BigEarthNet-S2 in xView3

## A controlled first-convolution weight-reset study

**Report status:** DEV-only technical report  
**Diagnostic source revision:** `4b14689dae104697d0bc408139429ad372cbb0de`  
**Frozen campaign comparator revision:** `1a82d508fbeb9fdf6868a9637611e9018952fb43`  
**Result package identity:** `dd1bb97ac86b088c11e61c6f6c08fd2843eab6eebad48ae30cfbbb8b7af9b87d`  
**Evaluation scope:** TRAIN plus fixed DEV8; no TEST or final-evaluation data accessed  
**Run seed:** 0  
**Label fractions:** 10% (`f10`) and 50% (`f50`)

---

## Abstract

This study investigated why an optical remote-sensing ConvNeXt-V2-Base
initialization pretrained on BigEarthNet-S2 underperformed random
initialization when transferred to Sentinel-1 synthetic-aperture-radar (SAR)
vessel detection in xView3. The leading hypothesis concerned the input
interface rather than the post-stem representation. The source checkpoint has
ten Sentinel-2 spectral input channels, whereas the target detector consumes
three SAR-derived channels, `[VH, VV, VH−VV]`. The production loader converted
the source kernel by retaining its first three optical-band slices and scaling
them by `10/3`. We tested this mechanism through a tightly controlled
intervention: retain every transferred BigEarthNet-S2 backbone tensor except
`stem.0.weight`, for which the seeded three-channel target model's native
random initialization was preserved. The convolution bias, stem
normalization, post-stem backbone, detector head, optimization, data order,
and evaluation procedure were unchanged.

Across complete 50-epoch paired runs, the reset improved best fixed-DEV8 F1
from 0.7992 to 0.8371 at `f10` and from 0.8408 to 0.8883 at `f50`. The gains
were primarily recall-driven. Mechanistic audits found that the production
conversion increased the initial first-convolution weight norm by 5.85× and
the empirical post-convolution activation variance by approximately 233×
relative to the reset. Layer-wise learning-rate decay assigned the stem an
early learning-rate scale of only 0.005688, and the production kernel remained
close to its initialization at the selected checkpoint. These results support
the predeclared classification that the current stem conversion explains most
of the observed deficit. The inference is deliberately narrow: it concerns
this checkpoint, adapter rule, seed, optimization recipe, and DEV split. It
does not imply that optical remote-sensing pretraining is intrinsically
harmful. On the contrary, after removing the incompatible kernel transfer, the
retained optical representation exceeded the frozen random-initialization
baseline at both tested label fractions.

## 1. Research question

The experiment asks:

> Is the apparent negative transfer from the BigEarthNet-S2 initialization
> caused primarily by the repository's ten-to-three-channel conversion of the
> first convolution, or by the transferred post-stem optical representation
> and its interaction with target-task optimization?

This distinction matters scientifically. A result in which the complete
optical checkpoint trails random initialization can arise from at least two
different mechanisms:

1. **Representation-level negative transfer.** The source task or optical
   representation is poorly aligned with SAR vessel detection, even after a
   valid target input interface is established.
2. **Interface-induced negative transfer.** A non-native source-to-target
   channel conversion distorts the signal before it reaches an otherwise
   useful representation.

The first-convolution reset was designed to separate these explanations with
the smallest possible intervention. It is not a new pretraining method and it
does not alter the architecture. It replaces exactly one incompatible tensor
with the target model's seeded initialization.

## 2. Source and target mismatch

### 2.1 Source checkpoint

The source is the public
[BigEarthNet v2.0 Sentinel-2 ConvNeXt-V2-Base checkpoint](https://huggingface.co/BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0).
Its configuration declares ten input channels. The channel order is:

`B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12`.

These are multispectral optical bands. The checkpoint was trained for
BigEarthNet land-cover classification, not SAR point-object detection. The
underlying reBEN release describes a large paired Sentinel-1/Sentinel-2
archive and publishes modality-specific pretrained models [1].

### 2.2 Target input

The xView3 detector consumes the fixed three-channel representation

$$
x = [x_{VH},\;x_{VV},\;x_{VH-VV}].
$$

The third channel is deterministically derived from the first two. Before
normalization, the audit confirmed the exact relation

$$
x_{VH-VV} = x_{VH} - x_{VV}.
$$

After the study's fixed per-channel normalization and centering, the sampled
data satisfied

$$
\tilde{x}_3 \approx 0.70056\tilde{x}_1 - 0.71360\tilde{x}_2,
$$

with residual RMSE `6.50e-7` and covariance rank two. Thus, the third channel
adds no independent linear degree of freedom at the first convolution. This
does not make the representation invalid—the same input is held fixed for all
arms—but it makes the first-layer parameterization especially important.

### 2.3 Production ten-to-three conversion

The production helper in
[`src/models/init_loaders.py`](../src/models/init_loaders.py) implements a
repeat-and-rescale rule. Let the source kernel be

$$
W \in \mathbb{R}^{128\times10\times4\times4}.
$$

For a three-channel target, the implemented rule reduces to

$$
\tilde{W}_{:,j,:,:} = \frac{10}{3}W_{:,j,:,:},\qquad j\in\{0,1,2\}.
$$

Consequently, only the kernels associated with optical bands B02, B03, and
B04 are retained; the other seven source-band kernels are discarded. The
retained kernels are then applied, respectively, to `VH`, `VV`, and `VH−VV`,
despite there being no physical band correspondence between those optical and
SAR channels.

Because the target input has rank two, the converted convolution is
equivalent, before its bias, to

$$
y = \frac{10}{3}\left[(W_0+W_2)*x_{VH} +
                       (W_1-W_2)*x_{VV}\right],
$$

where `*` denotes convolution. The conversion therefore does more than select
three source channels: it creates two effective SAR kernels through sums and
differences of arbitrary optical-band filters and magnifies them by `10/3`.

This operation is not a built-in ten-to-three adaptation formula in `timm`.
The upstream `adapt_input_conv` implementation supports its documented
three-to-`N` transformations and raises `NotImplementedError` for unsupported
source-channel layouts [4]. The upstream builder's fallback on that exception
is to omit the incompatible input-convolution weight and leave the target
layer randomly initialized [3]. The diagnostic reset implements that fallback
policy explicitly for this custom checkpoint-loading path.

## 3. Intervention and causal contrast

### 3.1 Variants

Two variants were initialized from the same seed and trained under the same
recipe:

- **`current_replay`** used the unchanged production `bigearthnet_s2` loader,
  including the ten-to-three conversion above.
- **`first_conv_reset`** constructed the seeded three-channel target model and
  loaded every mapped BigEarthNet-S2 backbone tensor except
  `stem.0.weight`. The convolution bias, stem normalization parameters, and
  all post-stem tensors were loaded from the checkpoint. The omitted weight
  remained byte-identical to the fresh seed-0 target model.

The term *first-convolution reset* is therefore shorthand for a reset of the
**kernel weight only**. It is not a reset of the full stem module.

### 3.2 Isolation guarantees

The initialization audit established all of the following before training:

| Property | Verified value |
|---|---:|
| Mapped backbone tensors | 378 |
| Source tensors intentionally dropped | 2 classifier-head tensors |
| Additional target tensors omitted in reset | exactly `stem.0.weight` |
| Parameters in reset tensor | 6,144 |
| ConvNeXt backbone parameters | 87,692,800 |
| Fraction of backbone reset | 0.007006% |
| Fraction of backbone retained | 99.992994% |
| Detector-head parameters | 6,296,065 |
| Detector-head hash equality | exact |
| Backbone parameter parity | exact |
| Feature/adapter geometry | unchanged |

The current converted-kernel SHA-256 was
`09bcbd55c21145a4c10460d7c0a7f8609ebf0099c50ca5e28186107c61e31469`;
the seed-0 reset-kernel SHA-256 was
`d3ab6509237ab6324010cce191d2aa3213541c3585cb8dd4ecd6dd10473a2487`.
The detector-head state had the same SHA-256 in both variants
(`b8d9e0b02c807eb9b5a2d8892cfa98324a97e5f82cd035c6c3fd48fa641a39b5`).
The implementation restored the random-number generator state after
reconstructing the seeded target model, preventing the diagnostic operation
itself from perturbing later initialization or data streams.

This is a strong local intervention: at training step zero, only one parameter
tensor differs between variants. It does not by itself support a population
causal claim because only one seed and one held-out development split were
used.

### 3.3 Estimands

For fraction `f`, the primary paired contrast was

$$
\Delta_f = F1_{reset,f} - F1_{current,f}.
$$

The predeclared recovery statistic was

$$
R_f = \frac{F1_{reset,f}-F1_{current,f}}
            {F1_{random,f}-F1_{current,f}}.
$$

The decision rule classified `R_f ≥ 0.50` as evidence that the stem conversion
explained most of the deficit. Because this ratio is unstable when its
denominator is near zero, the direct paired difference `Δ_f` remains the more
interpretable quantity at `f10`.

## 4. Experimental design

### 4.1 Training protocol

Each variant was run from scratch at `f10` and `f50`, producing four complete
runs. All runs used:

- seed 0;
- the fixed TRAIN scene selection for the corresponding label fraction;
- the same fixed eight-scene DEV set;
- a maximum and schedule horizon of 50 epochs;
- micro-batch 16, gradient accumulation 1, and effective batch 16;
- layer-wise learning-rate decay of 0.65;
- one H100 GPU per run, one process per GPU, and no DDP;
- PyTorch Lightning precision `32-true`;
- IEEE FP32 CUDA matmul and cuDNN convolution, with TF32 disabled;
- the same detector head, losses, augmentations, optimizer, scheduler,
  datamodule, data order, centralized ground-truth converter, and frozen
  scorer.

All four runs reached epoch 50 and remained finite. The complete paired
allocation consumed 131.923 GPU-hours; total BES2 diagnostic consumption,
including prior probes and audits, was 154.436 GPU-hours, below the approved
190-GPU-hour ceiling.

### 4.2 Evaluation and reproducibility

The reported operating point is the best checkpoint-bound DEV F1, including
its exact confidence threshold, precision, recall, counts, epoch, and
checkpoint hash. TEST imagery, TEST labels, verified-final imagery, and
verified-final labels were neither staged nor read.

Each selected checkpoint was evaluated repeatedly. At `f50`, both variants
were bitwise identical at the raw-candidate level across repeats. At `f10`,
the reset was bitwise identical; the current replay differed at one raw score
position by `0.00048828125`. That perturbation changed neither the selected
predictions nor the operating point, per-scene metrics, or aggregate metrics.
The evidence is therefore **outcome-stable**, although not universally
bitwise deterministic.

The test gate at source revision `4b14689...` passed all 543 tests. The result
package was independently verified against its manifest and content hashes.

## 5. Performance results

### 5.1 Primary paired result

| Fraction | Variant | Best epoch | F1 | Precision | Recall | TP | FP | FN | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `f10` | `current_replay` | 29 | 0.799197 | 0.830898 | 0.769826 | 398 | 81 | 119 | 0.749023 |
| `f10` | `first_conv_reset` | 49 | **0.837117** | **0.854839** | **0.820116** | **424** | **72** | **93** | 0.825195 |
| `f50` | `current_replay` | 49 | 0.840841 | 0.871369 | 0.812379 | 420 | **62** | 97 | 0.873047 |
| `f50` | `first_conv_reset` | 29 | **0.888252** | **0.877358** | **0.899420** | **465** | 65 | **52** | 0.884766 |

The paired effects were:

| Fraction | ΔF1 | ΔPrecision | ΔRecall | ΔTP | ΔFP | ΔFN |
|---|---:|---:|---:|---:|---:|---:|
| `f10` | **+0.037921** | +0.023941 | +0.050290 | +26 | −9 | −26 |
| `f50` | **+0.047411** | +0.005989 | +0.087041 | +45 | +3 | −45 |

The improvement was therefore not produced merely by moving along a
precision-recall tradeoff. At `f10`, both precision and recall improved. At
`f50`, precision was nearly unchanged while recall increased by 8.70 points,
with 45 additional true positives and 45 fewer false negatives at the
checkpoint-bound operating point.

### 5.2 Context from frozen campaign comparators

The following table places the paired diagnostic runs beside immutable
single-seed DEV results from the original H100 cohort. Random and S1 were not
rerun concurrently and are contextual comparators rather than members of the
paired intervention.

| Fraction | Paired current S2 | Paired S2 + reset | Frozen S2 | Frozen random | Frozen BigEarthNet-S1 |
|---|---:|---:|---:|---:|---:|
| `f10` | 0.799197 | **0.837117** | 0.799197 | 0.799205 | 0.876209 |
| `f50` | 0.840841 | **0.888252** | 0.858576 | 0.866405 | 0.899807 |

At `f10`, the frozen S2 and random results were effectively tied—their
difference was only `7.98e-6`—so the formal recovery ratio was numerically
ill-conditioned (`4749.45`). The scientifically useful quantity is the direct
paired reset gain of 0.0379 F1. At `f50`, the recovery denominator was 0.02556
and the recovery ratio was 1.855: the reset recovered more than the complete
current-versus-random deficit and exceeded the frozen random result by 0.02185
F1.

The `f10` current replay exactly reproduced the frozen S2 value. The `f50`
current replay was 0.01773 below the frozen S2 value, within the predeclared
absolute replay tolerance of 0.02. This near-boundary discrepancy warrants
caution but did not make the result indeterminate under the approved rule.

The reset remained below the frozen S1 result by 0.03909 at `f10` and 0.01156
at `f50`. This is not a clean modality control: the S1 arm uses the same shared
channel-conversion helper for a two-to-three-channel mapping, and that mapping
was intentionally outside this diagnostic. It remains a separate unresolved
compliance issue.

### 5.3 Scene sensitivity

The reset improved per-scene F1 on six of eight DEV scenes at `f10` and seven
of eight at `f50`. Leave-one-DEV-scene-out recomputation gave:

| Fraction | Variant | Retained-seven F1 range |
|---|---|---:|
| `f10` | `current_replay` | 0.783314–0.826636 |
| `f10` | `first_conv_reset` | 0.821549–0.854482 |
| `f50` | `current_replay` | 0.823261–0.875576 |
| `f50` | `first_conv_reset` | 0.876623–0.908681 |

These are deterministic sensitivity ranges, not confidence intervals. They
show that the aggregate effect was not attributable to one favorable DEV
scene, but they provide no estimate of split or population uncertainty.

## 6. Mechanistic evidence

### 6.1 Kernel scale and effective input operator

The reset changed only 6,144 weights, yet it substantially changed the signal
presented to the retained network:

| Initial quantity | Current conversion | First-conv reset | Current/reset ratio |
|---|---:|---:|---:|
| First-convolution weight norm | 9.1598 | 1.5650 | 5.85× |
| Covariance-predicted output variance | 0.399590 | 0.012198 | 32.76× |
| Raw-dB covariance-predicted variance | 17.3658 | 0.515207 | 33.70× |
| Empirical post-convolution variance | 2.71956 | 0.011683 | 232.78× |
| Empirical post-stem variance | 0.955035 | 0.578305 | 1.65× |
| Stage-1 output variance | 6.57534 | 1.59106 | 4.13× |

The covariance calculation and activation-hook calculation summarize
different objects—the former is a centered channel projection, whereas the
latter includes the learned spatial filters, spatial correlations, boundary
effects, and bias—so their ratios are not expected to agree numerically. They
nevertheless agree in direction and show that the production conversion
delivered a much larger early signal.

The empirical participation-ratio effective rank immediately after the first
convolution was 1.461 for the current conversion and 1.716 for the reset. No
nonfinite activation was observed. The issue was therefore not an overt
numerical explosion; it was a large, finite distributional and geometric shift
at the source-target interface.

### 6.2 Propagation through the backbone

Normalization attenuated but did not erase the difference. At initialization,
the post-stem and stage-0 variances remained higher for the current conversion,
and the gap expanded again at stage 1. Substantial differences persisted into
later stages:

| Hook | Current variance | Reset variance |
|---|---:|---:|
| Pre-stem input | 0.606747 | 0.606747 |
| Post first convolution | 2.719555 | 0.011683 |
| Post stem | 0.955035 | 0.578305 |
| Stage 0 | 1.06450 | 0.615234 |
| Stage 1 | 6.57534 | 1.59106 |
| Stage 2 | 56.8807 | 41.1099 |
| Stage 3 | 32.5681 | 28.8195 |

The identical pre-stem row is an internal negative control: the variants saw
the same normalized samples. Divergence begins exactly at the intervened
layer.

### 6.3 Optimization under layer-wise learning-rate decay

The first convolution received the smallest layer-wise learning-rate scale,
`0.005688009`, corresponding to an early learning rate of approximately
`1.14e-7`. Its gradients were large relative to its parameter norm, especially
for the reset, but the actual parameter update was strongly attenuated.

Initialization-to-selected-checkpoint drift was:

| Fraction | Variant | Initial norm | Update norm | Relative displacement | Cosine similarity |
|---|---|---:|---:|---:|---:|
| `f10` | current | 9.1598 | 0.06313 | 0.689% | 0.999976 |
| `f10` | reset | 1.5650 | 0.06508 | 4.158% | 0.999135 |
| `f50` | current | 9.1598 | 0.18170 | 1.984% | 0.999803 |
| `f50` | reset | 1.5650 | 0.12126 | 7.748% | 0.996994 |

The production kernel therefore remained very close in direction to its
converted optical initialization. Its post-convolution variance at the best
checkpoint remained approximately 233× the reset at `f10` and 260× at `f50`.
The optimizer did not simply relearn a target-appropriate stem during the
50-epoch schedule.

This observation identifies an interaction between initialization and
optimization. Layer-wise decay is shared and fair across arms, but it assumes
that early pretrained layers are worth preserving. When the first layer is an
unsupported cross-modal conversion, that assumption can lock in the very
tensor most in need of adaptation.

### 6.4 Causal mechanism consistent with the evidence

The evidence supports the following bounded causal account:

> Ten-channel optical kernel → retain B02/B03/B04 and multiply by `10/3` →
> semantically arbitrary, oversized effective SAR kernels → persistent early
> activation shift → minimal correction under strong layer-wise decay → lower
> DEV recall and F1.

This chain is supported by intervention, initialization hashes, input
covariance, activation statistics, and parameter drift. The experiment cannot
assign a unique fraction of the effect to semantic mismatch versus scale,
because both are changed together when the incompatible tensor is omitted.

## 7. Interpretation

### 7.1 What the result supports

Under the predeclared decision rule, both label fractions were classified as
`stem-conversion-explains-most-deficit`; the aggregate result was concordant
and determinate. The strongest evidence is the paired improvement after a
single-tensor intervention, supported by the localization of activation
divergence to the first convolution and by the weak subsequent movement of the
production kernel.

The appropriate interpretation is **implementation-induced negative
transfer**. A source checkpoint with a useful post-stem representation was
coupled to the target task through a non-native, poorly scaled optical-to-SAR
input operator. Once the input weight was reset, retaining more than 99.99% of
the pretrained backbone by parameter count, the model exceeded frozen random
initialization at both tested fractions.

### 7.2 What the result does not support

The experiment does not establish that:

- optical remote-sensing pretraining is generally inferior for SAR;
- every BigEarthNet-S2 or ConvNeXt checkpoint has the same behavior;
- resetting the first convolution is optimal among all possible adapters;
- the observed DEV gains will transfer unchanged to TEST or final evaluation;
- the effect is statistically significant across seeds or data splits; or
- BigEarthNet-S1 is a valid positive control for the channel-conversion rule.

No hand-designed band permutation, averaging, PCA projection, or learned
adapter was tested. Those alternatives would answer different questions and
would introduce method choices not authorized by the study design.

## 8. Threats to validity

1. **Single seed.** All runs used seed 0. The design identifies a paired effect
   for that seed but cannot estimate initialization or training variance.
2. **Small development split.** DEV contains eight scenes. Leave-one-scene-out
   analysis tests influence, not sampling uncertainty.
3. **Development-set reuse.** DEV is used for checkpoint and operating-point
   selection as well as the diagnostic comparison. The result is suitable for
   root-cause selection, not an unbiased final performance estimate.
4. **Two label fractions.** Only `f10` and `f50` were rerun to conserve compute.
   The mechanism was not directly re-estimated at `f25` or `f100`.
5. **Near-boundary replay at `f50`.** The current replay was within, but close
   to, the predeclared 0.02 reproducibility limit.
6. **Ill-conditioned `f10` recovery ratio.** Random and current S2 were almost
   identical at `f10`; the direct paired effect is more reliable than the
   normalized recovery statistic.
7. **Weight-only intervention.** The convolution bias and normalization were
   retained. The result isolates the kernel tensor rather than testing a fully
   reinitialized stem.
8. **Joint scale-and-semantics intervention.** Resetting the source kernel
   simultaneously removes its scaling and its optical spectral semantics.
   Their separate contributions are not identified.
9. **Checkpoint and recipe specificity.** Conclusions are conditional on the
   named checkpoint, fixed `[VH,VV,VH−VV]` representation, ConvNeXt-V2-Base,
   detector, layer decay, and training horizon.
10. **No held-out final evidence.** TEST and final data were deliberately not
    accessed. No final generalization claim is made.

## 9. Decision and study consequence

The diagnostic result met the predeclared eligibility condition for using the
documented unsupported-input-convolution fallback: preserve the seeded target
kernel while transferring every compatible source tensor. The owner has
subsequently approved this first-convolution fallback as the Arm-6 replacement
initialization.

For the controlled study, the replacement should be represented as a distinct,
fully provenance-bound initialization. It must not overwrite the historical
Arm-6 result or be spliced into the existing cohort. Because initialization is
the only allowed within-track variable, approval of the replacement implies a
fresh 32-cell cohort in a new namespace, with all eight arms and four label
fractions rerun under one sealed H100 environment and the unchanged shared
recipe. The original cohort, including its hashes and evaluation record,
remains immutable.

Recommended reporting language is:

> The original BigEarthNet-S2 arm used a generalized ten-to-three-channel
> kernel conversion that induced negative transfer. A controlled ablation
> showed that retaining the target model's seeded first-convolution weight
> while transferring all compatible BigEarthNet-S2 backbone tensors improved
> fixed-DEV8 F1 at both tested label fractions. We therefore treated the
> original conversion as an interface failure, not as evidence against the
> transferred post-stem optical representation.

## 10. Reproducibility record

The narrow verified result package contains the audit, four run metrics,
execution receipt, causal decision, report, and necessary logs. The principal
identities are:

| Artifact | SHA-256 / identity |
|---|---|
| Diagnostic Git revision | `4b14689dae104697d0bc408139429ad372cbb0de` |
| Source checkpoint | `b09d0e41cc683878243a9128a6f4724d6a71d562318beeae716f0dce9cbbf454` |
| Result identity | `dd1bb97ac86b088c11e61c6f6c08fd2843eab6eebad48ae30cfbbb8b7af9b87d` |
| Package manifest | `eeadf738badf31b6305bc5a78580928f0501c94694ac4acd28ab373bbb50ce5f` |
| Package `READY.json` | `890672c08c966d5e1e06600d0cb2675430fdd1d6c883feaff951d856a53448e8` |
| Package `SHA256SUMS` | `020fe3d0523df236fba50200735d355e44e18d8328d329acfa6d54e64590fbfb` |

Canonical package-relative evidence includes:

- `evidence/control/BES2_ROOT_CAUSE.json`
- `evidence/control/BES2_ROOT_CAUSE.md`
- `evidence/control/BES2_AUDIT.json`
- `evidence/control/BES2_EXECUTION_COMPLETE.json`
- `evidence/diagnostic_runs/f10/current_replay/full/diagnostic_metrics.json`
- `evidence/diagnostic_runs/f10/first_conv_reset/full/diagnostic_metrics.json`
- `evidence/diagnostic_runs/f50/current_replay/full/diagnostic_metrics.json`
- `evidence/diagnostic_runs/f50/first_conv_reset/full/diagnostic_metrics.json`

## References

1. Clasen, K. N., Hackel, L., Burgert, T., Sumbul, G., Demir, B., and
   Markl, V. *reBEN: Refined BigEarthNet Dataset for Remote Sensing Image
   Analysis*. IGARSS, 2025; arXiv:2407.03653.
   <https://arxiv.org/abs/2407.03653>
2. Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., and Xie, S.
   *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*.
   CVPR, 2023.
   <https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html>
3. `timm` model builder, unsupported input-convolution fallback.
   <https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_builder.py>
4. `timm` `adapt_input_conv` implementation.
   <https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/_manipulate.py>
5. BIFOLD, *BigEarthNet v2.0 ConvNeXt-V2-Base Sentinel-2 model card*.
   <https://huggingface.co/BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0>
6. BIFOLD, *BigEarthNet v2.0 model collection and Sentinel-2 channel order*.
   <https://huggingface.co/BIFOLD-BigEarthNetv2-0>
7. BIFOLD, *ConvNeXt-V2-Base S2 configuration (`channels: 10`)*.
   <https://huggingface.co/BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0/commit/70ebf16500ab1467e34713ce4a2a6d1267567c9b>

