# Judy BigEarthNet-S2 diagnostic runbook

This runbook executes only the owner-approved TRAIN+fixed-DEV8 root-cause
diagnostic. It does not authorize a replacement, a new core campaign, TEST
access, or verified-final access. It never acquires or controls a V100 lease.

## Preconditions

- The reviewed sprint-10a-bes2-root-cause commit is clean and contains the
  approved dev ancestor.
- Canonical Judy H100_READY.json exists, is mode 0444, and remains unmodified.
- No final_eval.lock, final data-view/consumption/completion receipt, or
  per-cell final_verified_metrics.json exists under the original Judy runs
  root.
- The accepted sealed venv, base/runtime packages, base-Python closure, and
  TRAIN/DEV staging inputs remain at their verified persistent paths.
- Use a fresh Box folder for each amendment and result direction. Folder IDs,
  JWT paths, tokens, and URLs are runtime inputs only and must never be written
  into this repository, site snapshots, logs, or package controls.

The submit script repeats the final-consumption, source, package, hash, budget,
path-isolation, and stage-order gates before it creates any diagnostic state.

## Build and send the code-only amendment

Run from the reviewed source checkout with the transfer environment and the
fresh amendment Box folder selected only in the process environment.

```bash
python -B -m scripts.handoff build-bes2-amendment \
  --repo /absolute/reviewed/sprint10a \
  --h100-ready-json /absolute/verified/H100_READY.json \
  --weights-root /absolute/data/weights \
  --output-dir /absolute/new/amendment-output
```

Record the printed package ID and READY, manifest, SHA256SUMS, source-commit,
and bundle hashes outside the repository. Verify locally, then upload to the
fresh empty folder; READY is published last.

```bash
python -B -m scripts.handoff verify-bes2-amendment \
  --package-root /absolute/amendment-package \
  --expected-h100-ready-sha256 "$H100_READY_SHA256"
```

```bash
python -B -m scripts.handoff upload-bes2-amendment \
  --repo /absolute/reviewed/sprint10a \
  --package-root /absolute/amendment-package \
  --expected-h100-ready-sha256 "$H100_READY_SHA256" \
  --receipt /absolute/outside-repo/bes2-amendment-upload.json
```

Generate the standalone hash-pinned puller outside all repository worktrees.

```bash
python -B -m scripts.handoff build-bes2-bootstrap \
  --repo /absolute/reviewed/sprint10a \
  --package-root /absolute/amendment-package \
  --expected-h100-ready-sha256 "$H100_READY_SHA256" \
  --output /absolute/outside-repo/pull-bes2.sh
```

## Pull and bind on Judy

On a Judy transfer/login host, set `TRANSFER_PYTHON`, `XVIEW3_TARGET_ROOT`,
`BOX_JWT_CONFIG`, and `BOX_FOLDER_ID` only in the invoking process environment,
without logging their values, then run the generated bootstrap. The target root
must already be an empty, canonical, non-symlink directory. The bootstrap
verifies the exact remote tree, downloads through partial files, reconstructs
the Git bundle, and publishes a clean content-addressed checkout.

```bash
/absolute/pull-bes2.sh
```

Preserve its printed package_root, bundle, checkout, commit, and hashes. Create
an untracked mode-0600 BES2 site file outside the checkout, or at the ignored
slurm/h100/bes2-site.env path. Reuse the accepted H100 base/runtime/venv fields
from the canonical Judy site contract and add exactly these diagnostic values:

    BES2_PACKAGE_ROOT=/bootstrap/package_root
    BES2_BUNDLE=/bootstrap/bundle
    BES2_DIAGNOSTIC_ROOT=/new/persistent/diagnostic-namespace
    BES2_ORIGINAL_RUNS_ROOT=/canonical/original/h100-runs
    BES2_JOB_LOG_DIR=/new/persistent/diagnostic-job-logs
    BES2_FORECAST_GPU_HOURS=<fresh-f10-f50-forecast-at-or-below-125>

BES2_ORIGINAL_RUNS_ROOT must equal the existing H100_RUNS_ROOT. The diagnostic
root, job-log root, scratch root, original runs root, package, bootstrap
checkout, base/runtime packages, wheelhouse, and venv must satisfy the
submitter's pairwise isolation checks. Do not create a Judy path that pretends
to be a V100 filesystem. Recompute the forecast from accepted TRAIN/DEV timing
evidence for the f50 five-epoch probe pair plus the full f10 and f50 pairs; do
not retain the superseded f100 estimate. A value above 125 GPU-hours is a STOP
for owner approval.

## Execute the Judy gates and pair

Submit each stage only after the previous job has completed and its immutable
marker has been reviewed. The audit requests one H100; probe and full request
two. Slurm supplies the GPUs—do not use gpu get.

```bash
BES2_SITE_ENV=/absolute/bes2-site.env \
  ./slurm/h100/submit_bes2_diagnostic.sh audit
```

Audit must produce .control/BES2_DIAGNOSTIC_READY.json after:

- the complete sealed-venv test suite;
- exact package/source/frozen-file checks;
- one-H100 strict IEEE FP32 parent/child probes;
- value-sensitive BigEarthNet-S2 load and exact paired initialization audit;
- a numerical-only snapshot of the immutable H100 BigEarthNet-S2 and
  CNN-random f10/f50 best-DEV results (four final_metrics.json files only;
  no checkpoint, cohort, TEST, or verified-final reads);
- deterministic input covariance, exact two-channel stem projection,
  activation, and sample-manifest evidence;
- finite batch-16 forward/backward probes for both variants; and
- the finite forecast at or below 125 GPU-hours.

A failed audit is a STOP. Do not relax a guard or continue to training.

```bash
BES2_SITE_ENV=/absolute/bes2-site.env \
  ./slurm/h100/submit_bes2_diagnostic.sh probe
```

Both fresh five-epoch runs use f50, seed 0, and the original 50-epoch schedule
horizon. BES2_PROBES_COMPLETE.json must bind both finite f50 diagnostic metrics
before the full stage is submitted.

```bash
BES2_SITE_ENV=/absolute/bes2-site.env \
  ./slurm/h100/submit_bes2_diagnostic.sh full
```

The full allocation first starts the fresh f10 pair and then the fresh f50
pair. Within each fraction, current replay and first-convolution reset run
concurrently on two H100s with one process per GPU and no DDP. Every run
records observational gradient and actual-update norms at optimizer steps
1, 10, 100, and 500 without changing training. A Slurm USR1 signal reaches
the active controller, which waits for every live worker to checkpoint before
the outer batch calls the real scontrol requeue. Allocation scratch is
reconstructed after requeue; persistent checkpoints and controller state
remain under BES2_DIAGNOSTIC_ROOT.

Successful full completion requires all four full metrics and produces
BES2_EXECUTION_COMPLETE.json, BES2_ROOT_CAUSE.json, and
BES2_ROOT_CAUSE.md. The f10 and f50 decisions are kept separate and are never
pooled. Checkpoints stay in the persistent Judy namespace.

## Build and return narrow evidence

Build in the sealed Judy venv into a new output parent outside the diagnostic
root. This command reopens and hashes the retained best checkpoints to validate
the checkpoint-bound operating points, but it never copies checkpoint bytes.

```bash
/absolute/sealed-venv/bin/python -B -m scripts.handoff \
  build-bes2-results \
  --repo /bootstrap/checkout \
  --diagnostic-root /persistent/diagnostic-namespace \
  --original-runs-root /canonical/original/h100-runs \
  --job-log-dir /persistent/diagnostic-job-logs \
  --output-dir /new/bes2-result-output \
  --max-part-bytes "$BOX_MAXIMUM_FILE_BYTES"
```

The package contains only validated JSON, the Markdown report, controller/test
logs, strict-runtime JSON, and Slurm logs. It rejects checkpoints, core/TEST/
final/cohort artifacts, secrets, Box settings, endpoints, and extra files.
Verify it locally, then upload it to a separate fresh Box folder.

```bash
python -B -m scripts.handoff verify-bes2-results \
  --package-root /absolute/bes2-result-package \
  --expected-source-git-sha "$SPRINT10A_GIT_SHA"
```

```bash
python -B -m scripts.handoff upload-bes2-results \
  --repo /bootstrap/checkout \
  --package-root /absolute/bes2-result-package \
  --expected-source-git-sha "$SPRINT10A_GIT_SHA" \
  --receipt /absolute/outside-repo/bes2-result-upload.json
```

On the receiving side, use the printed package controls and the verified
downloader; a manual directory copy is not transfer proof.

```bash
python -B -m scripts.handoff download-bes2-results \
  --repo /absolute/reviewed/sprint10a \
  --package-root /absolute/absent/result-destination \
  --expected-source-git-sha "$SPRINT10A_GIT_SHA" \
  --expected-ready-sha256 "$READY_SHA256" \
  --expected-manifest-sha256 "$MANIFEST_SHA256" \
  --expected-sha256sums-sha256 "$SHA256SUMS_SHA256" \
  --expected-package-id "$PACKAGE_ID"
```

## Decision and mandatory stop

The summarize command applies the frozen replay tolerance and recovery
thresholds independently at f10 and f50. It records precision/recall, exact
threshold curves, frozen-scorer localization-distance distributions, score
distributions, trajectories, activations, early gradient/update traces,
initialization-to-best drift, and leave-one-DEV-scene-out sensitivity without
significance or seed-variance claims. A first-convolution-reset replacement is
conditionally eligible only if both fraction decisions are determinate,
stem-dominant, and concordant; disagreement is reported as a blocked
fraction-dependent mixed mechanism.

No replacement code or campaign may start from this runbook. A stem-dominant
result still requires an explicit owner amendment before first-convolution
reset can become a core initialization. Mixed or reset-insufficient results
require a licensed downloaded native-three-channel optical-RS
ConvNeXt-V2-Base checkpoint. If no eligible checkpoint is approved,
replacement is blocked and the original bounded final authorization remains
the only permissible fallback. CNN-random replay and shared layer-decay
ablation are possible evidence-triggered follow-ups only; neither is
authorized by this runbook.
