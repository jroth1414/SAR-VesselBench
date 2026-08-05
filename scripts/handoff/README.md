# Sprint 7f Judy evaluation-contract handoff

The Judy handoff has two immutable layers:

1. the already uploaded and remotely verified Sprint 7d base payload, which
   contains the data, six core checkpoint directories, offline cu126
   wheelhouse, and historical source bundle; and
2. a separate, small, content-addressed schema-2 Sprint 7f runtime amendment
   containing the current `sprint-7f-eval-contract` Git bundle, a deterministic
   source-audited 13,911-row TRAIN+fixed-DEV8 label CSV, and `manifest.json` /
   `SHA256SUMS` / `READY.json` controls. Its identity binds the accepted
   Sprint-7e native-venv ancestor. It contains no TEST/eval-final row, raster,
   chip, weight, wheelhouse, run, venv, or credential.

Judy executes only a native, final-path Python virtual environment. There is no
container build, container launch, or venv activation in the Judy path.
The Apptainer definition retained in the Sprint 7d package is historical,
non-executed provenance; do not remove it from, rebuild, or otherwise mutate
that verified package. The login-built native venv is diagnostic evidence
only. After at least two eligible DGX nodes produce the same full base-runtime
fingerprint, build one fresh sealed venv inside a CPU-only task on an eligible
DGX node at a new persistent path; Sprint 7f does not reuse the login receipt.
Judy and the live V100 host have completely separate filesystems. The Judy
runtime requires `H100_V100_CONTROL_PLANE=box-transfer-v1`; do not create a
dummy Judy path for the V100 runs tree. Smoke and H100 acceptance use only
Judy-local immutable inputs. Corrected R2/R3 evidence, `CUTOVER_READY.json`,
and the human-authored `V100_DIAGNOSTIC_ISOLATION.json` cross Box later as
small dynamic control transfers and never enter either forward package. The
live V100 campaign continues untouched as a non-reportable diagnostic.

## Immutable Sprint 7d base identity

These controls are fixed and must be checked out of band on every receiving
host:

```text
package_id:       xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a
source_git_sha:   2726199efcebbebc89156e708b89df2a3415468a
READY.json:       b0d6ee18f9ddbd0d604cbea06610dcdbae6a9eb6d1f5ff3ea3431bd9e2d55f81
manifest.json:    fccb0b505c89836a148afec709bb799f7af4908d955ea1b142e153154d830896
SHA256SUMS:       21c83b2e3b1b9d67bf00b8abca3ce267a5efd9362c1206b8d29ab21ca3e2d396
repository bundle:
                  df21d64cd7ba2d884fcb4a454c46106b39a6c604459861681e2823ac460f2f4e
environment lock: 7c651f762b84801fcdf50a48accca05911b2b7f6bc1c536cb1acce0d7fa22154
remote inventory: 201 files, 294,278,292,176 bytes
```

The production runtime-amendment builder and verifier require that exact base
package and run its complete Sprint 7d verifier. A look-alike package, changed
control file, changed source bundle, or changed environment lock fails closed.

## Transfer environment and Box isolation

Use a transfer environment and runtime file outside the repository. They are
not the training venv or Slurm `site.env`:

```bash
python3 -m venv /outside/repo/xview3-box-transfer
/outside/repo/xview3-box-transfer/bin/pip install -r requirements-transfer.txt
chmod 0600 /outside/repo/box-jwt.json
chmod 0600 /outside/repo/xview3-box-transfer.env
source /outside/repo/xview3-box-transfer.env
```

`requirements-transfer.txt` pins the compatible `boxsdk[jwt]` v4 line and the
other transfer-only dependencies. The external runtime file supplies
`BOX_JWT_CONFIG` and the `BOX_FOLDER_ID` for the one dedicated folder used by
the current transfer. Keep both files outside the repository at mode `0600`;
their paths, contents, and folder ID must never be logged, archived, embedded
in a package/bootstrap, or copied into Slurm `site.env`.

## Dynamic cross-site control evidence

V100 continues while Judy is qualified and after H100 launch, so dynamic
evidence cannot be part of the immutable Sprint 7f runtime package. The
`build-control`, `verify-control`, `upload-control`, and
`download-control` commands implement closed, content-addressed packages:

1. the immutable `REFERENCE_CAMPAIGN.json` plus corrected R2/R3
   `final_metrics.json` and `runtime_provenance.json` travel V100→Judy as kind
   `references`;
2. `CUTOVER_READY.json` travels Judy→V100 as kind `cutover-ready`; and
3. `V100_DIAGNOSTIC_ISOLATION.json` travels V100→Judy as kind
   `diagnostic-isolation`.

Every kind has an exact JSON allowlist and direction, manifest,
`SHA256SUMS`, and READY-last publication. Each uses its own initially empty
Box child folder. Downloads use `.partial` files and atomic rename; upload
receipts and package manifests never record the Box folder ID. Campaign uses
only verified canonical Judy copies and never mounts a V100/reference path.
The isolation attestation proves disjoint namespaces and forbids V100 marker
suppression/resume; it does not stop or archive V100.

The CLI enforces the following closed payloads and binding keys. Build every
package from a clean `sprint-7f-eval-contract` checkout, use output/receipt
paths outside all worktrees, and record the printed package ID, producer Git
SHA, identity SHA-256, manifest SHA-256, READY SHA-256, and SHA256SUMS SHA-256
through a trusted out-of-band channel.

Build and verify the corrected R2/R3 package on the V100 side:

The legacy `scripts/export_results.py` path is intentionally not part of this
flow. It never reads `runs/<exp_id>` or exports an experiment directory.
Corrected references live under the explicit external reference-campaign root;
the exact `build-control --kind references` mapping below, followed by
`verify-control`, is the sole supported reference export path.

```bash
python -m scripts.handoff build-control \
  --repo "$PWD" \
  --kind references \
  --source references/REFERENCE_CAMPAIGN.json=/absolute/path/to/REFERENCE_CAMPAIGN.json \
  --source references/locateanything-zs/final_metrics.json=/absolute/path/to/locateanything-zs/final_metrics.json \
  --source references/locateanything-zs/runtime_provenance.json=/absolute/path/to/locateanything-zs/runtime_provenance.json \
  --source references/yolo26-f100/final_metrics.json=/absolute/path/to/yolo26-f100/final_metrics.json \
  --source references/yolo26-f100/runtime_provenance.json=/absolute/path/to/yolo26-f100/runtime_provenance.json \
  --binding evaluation_contract=vessel-hm-positive-low-ignore-v2 \
  --binding reference_git_sha="$REFERENCE_GIT_SHA" \
  --binding reference_campaign_id="$REFERENCE_CAMPAIGN_ID" \
  --binding v100_core_git_sha="$V100_CORE_GIT_SHA" \
  --binding v100_core_campaign_id="$V100_CORE_CAMPAIGN_ID" \
  --output-dir /outside/repo/control-out

python -m scripts.handoff verify-control \
  --package-root "$REFERENCES_PACKAGE_ROOT" \
  --expected-kind references \
  --expected-binding evaluation_contract=vessel-hm-positive-low-ignore-v2 \
  --expected-binding reference_git_sha="$REFERENCE_GIT_SHA" \
  --expected-binding reference_campaign_id="$REFERENCE_CAMPAIGN_ID" \
  --expected-binding v100_core_git_sha="$V100_CORE_GIT_SHA" \
  --expected-binding v100_core_campaign_id="$V100_CORE_CAMPAIGN_ID"
```

Source the external transfer environment with `BOX_FOLDER_ID` set to a new,
initially empty references-package folder only for the upload, then run:

```bash
python -m scripts.handoff upload-control \
  --repo "$PWD" \
  --package-root "$REFERENCES_PACKAGE_ROOT" \
  --expected-kind references \
  --expected-binding evaluation_contract=vessel-hm-positive-low-ignore-v2 \
  --expected-binding reference_git_sha="$REFERENCE_GIT_SHA" \
  --expected-binding reference_campaign_id="$REFERENCE_CAMPAIGN_ID" \
  --expected-binding v100_core_git_sha="$V100_CORE_GIT_SHA" \
  --expected-binding v100_core_campaign_id="$V100_CORE_CAMPAIGN_ID" \
  --receipt /outside/repo/receipts/references.upload.json
```

On Judy, select that same folder through the external transfer environment
and download against the out-of-band controls. `package-root` must not exist:

```bash
python -m scripts.handoff download-control \
  --repo "$PWD" \
  --package-root /outside/repo/xview3-control-references-IDENTITY64HEX \
  --expected-kind references \
  --expected-binding evaluation_contract=vessel-hm-positive-low-ignore-v2 \
  --expected-binding reference_git_sha="$REFERENCE_GIT_SHA" \
  --expected-binding reference_campaign_id="$REFERENCE_CAMPAIGN_ID" \
  --expected-binding v100_core_git_sha="$V100_CORE_GIT_SHA" \
  --expected-binding v100_core_campaign_id="$V100_CORE_CAMPAIGN_ID" \
  --expected-ready-sha256 "$REFERENCES_READY_SHA256" \
  --expected-manifest-sha256 "$REFERENCES_MANIFEST_SHA256" \
  --expected-sha256sums-sha256 "$REFERENCES_SHA256SUMS_SHA256" \
  --expected-package-id "$REFERENCES_PACKAGE_ID"
```

After Judy acceptance and `slurm/h100/submit.sh cutover-check`, build the
Judy-to-V100 marker package from the exact canonical marker:

```bash
python -m scripts.handoff build-control \
  --repo "$PWD" \
  --kind cutover-ready \
  --source CUTOVER_READY.json=/persistent/h100-runs/.h100/CUTOVER_READY.json \
  --binding h100_campaign_id="$H100_CAMPAIGN_ID" \
  --binding h100_git_sha="$H100_GIT_SHA" \
  --binding h100_ready_sha256="$H100_READY_SHA256" \
  --binding references_manifest_sha256="$REFERENCES_MANIFEST_SHA256" \
  --binding references_package_id="$REFERENCES_PACKAGE_ID" \
  --output-dir /outside/repo/control-out
```

Verify and upload it with `--expected-kind cutover-ready` and those same five
repeated `--expected-binding` values. On the V100 side, use
`download-control` with the exact package ID plus READY, manifest, and
SHA256SUMS hashes, `--expected-kind cutover-ready`, and the same bindings.
The operator verifies `CUTOVER_READY.json`, then authors
`V100_DIAGNOSTIC_ISOLATION.json` against
`slurm/h100/V100_DIAGNOSTIC_ISOLATION.schema.json`. Its V100 execution status
must exactly match the cutover forecast: use
`continues-running-non-reportable-diagnostic` while remaining time is positive,
or `complete-non-reportable-diagnostic` only when remaining time is exactly
zero. The namespaces must be disjoint, all H100 suppression/resume/mixing
fields must be false, and any later V100 stop/archive remains optional rather
than an H100 prerequisite.

Build the return package on the V100 side from that exact attestation:

```bash
python -m scripts.handoff build-control \
  --repo "$PWD" \
  --kind diagnostic-isolation \
  --source V100_DIAGNOSTIC_ISOLATION.json=/absolute/path/to/V100_DIAGNOSTIC_ISOLATION.json \
  --binding cutover_ready_sha256="$CUTOVER_READY_SHA256" \
  --binding h100_campaign_id="$H100_CAMPAIGN_ID" \
  --binding v100_core_campaign_id="$V100_CORE_CAMPAIGN_ID" \
  --binding v100_core_git_sha="$V100_CORE_GIT_SHA" \
  --output-dir /outside/repo/control-out
```

Verify and upload it with `--expected-kind diagnostic-isolation` and those
same four repeated bindings. On Judy, use `download-control` with the exact
package ID plus READY, manifest, and SHA256SUMS hashes,
`--expected-kind diagnostic-isolation`, and the same bindings. The
`producer_git_sha` printed for both `references` and `diagnostic-isolation`
packages must be the H100 Sprint 7f source SHA. `reference_git_sha` and
`reference_campaign_id` identify the corrected R2/R3 execution;
`v100_core_git_sha` and `v100_core_campaign_id` separately identify the live
non-reportable V100 core diagnostic. Record the
downloaded package root and all six printed identity/hash values in the
corresponding `H100_*_PACKAGE_*` fields of the untracked Judy `site.env`.

The Sprint 7f amendment must use its own newly created, initially empty Box
folder. Do not point `BOX_FOLDER_ID` at the Sprint 7d base folder, an active
Sprint 7d download, or a future result-package folder. The Box preflight
performs a disposable write/update/rename/delete probe and discovers the
current maximum file size. Physical files above 50,000,000 bytes use resumable
chunked upload.

## Build and upload the Sprint 7f amendment

Build only from a clean, committed `sprint-7f-eval-contract` checkout. The output
directory must be outside all repository worktrees:

```bash
python -m scripts.handoff preflight --repo "$PWD" \
  --minimum-free-bytes 0
python -m scripts.handoff build-runtime \
  --repo "$PWD" \
  --base-package-root /path/to/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --output-dir /outside/repo/handoff
```

The resulting directory is named:

```text
xview3-h100-runtime-<SPRINT7F_FULL_GIT_SHA>-<RUNTIME_IDENTITY_SHA256>
```

It contains exactly two logical artifacts: the current runtime Git bundle and
the deterministic source-audited 13,911-row TRAIN+fixed-DEV8 label CSV. The
only other files are `manifest.json`, `SHA256SUMS`, and the `READY.json`
written last; no TEST/eval-final row or imagery is present. Its runtime
identity binds the exact Sprint 7d controls, required Sprint-7e ancestor,
Sprint 7f commit and bundle, unchanged environment lock, corrected
evaluation/cohort contract, and native strict-FP32 recipe. Record the printed
package ID and three control hashes through a trusted out-of-band channel,
then verify and upload:

```bash
python -m scripts.handoff verify-runtime \
  --base-package-root /path/to/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --package-root /outside/repo/handoff/xview3-h100-runtime-FULL40CHARSHA-IDENTITY64HEX
python -m scripts.handoff upload-runtime \
  --repo "$PWD" \
  --base-package-root /path/to/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --package-root /outside/repo/handoff/xview3-h100-runtime-FULL40CHARSHA-IDENTITY64HEX \
  --receipt /outside/repo/receipts/xview3-h100-runtime-FULL40CHARSHA-IDENTITY64HEX.upload.json
```

`READY.json` is uploaded last. An ambiguous READY upload or failed final remote
verification removes the remote marker and returns nonzero. Upload comparison
uses Box SHA-1 plus size, not timestamps. The atomic receipt records package
and control identities without recording the Box folder ID.

After source-side verification, generate the mandatory standalone Judy puller.
This is the only approved Judy runtime download/clone path. The output must be
outside the repository and must not already exist:

```bash
python -m scripts.handoff build-runtime-bootstrap \
  --repo "$PWD" \
  --base-package-root /path/to/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --runtime-package-root /outside/repo/handoff/xview3-h100-runtime-FULL40CHARSHA-IDENTITY64HEX \
  --output /outside/repo/pull_runtime_amendment.sh
```

The generated mode-`0700` script embeds only the verified file inventory and
content hashes. On Judy it requires `TRANSFER_PYTHON`, `XVIEW3_TARGET_ROOT`,
`BOX_JWT_CONFIG`, and `BOX_FOLDER_ID` at runtime; the latter two come from the
external transfer environment. It refuses to overwrite an existing package,
bundle, or checkout, downloads through `.partial` files, verifies every byte,
and atomically publishes the exact clean branch and commit.

The general `download-runtime`, `verify-runtime`, and `extract-runtime`
commands deliberately invoke the full immutable Sprint 7d verifier. They are
source/setup-only tools, not a Judy runtime-acquisition path, and are forbidden
inside H100 smoke, acceptance, campaign, or any other pre-cohort allocation:
that full verifier opens and hashes TEST raster and combined-label archives.
Only the generated standalone bootstrap may download and clone Sprint 7f on
Judy.

## Bootstrap Sprint 7f on Judy

The immutable Sprint 7d base must already have been transferred, fully
verified, and extracted during source/setup staging before any H100 job is
submitted. That one-time setup is outside the H100 allocation path. No
full-base transfer or verification command may run inside a pre-cohort
allocation.

Use the generated hash-pinned standalone script to acquire Sprint 7f on Judy.
Supply its trusted SHA-256 and runtime settings out of band:

```bash
source /outside/repo/sprint7f-runtime-transfer.env
export TRANSFER_PYTHON=/path/to/xview3-box-transfer/bin/python
export XVIEW3_TARGET_ROOT=/persistent/path/xview3-handoff
export BOX_JWT_CONFIG=/outside/repo/xview3-jwt.json
export BOX_FOLDER_ID=${BOX_FOLDER_ID:?provided by the external environment}
# Set this only if Judy's shared transfer Python needs its libpython directory.
export H100_BASE_PYTHON_LIB_DIR=/path/to/python/3.11.13/lib
sha256sum -- /outside/repo/pull_runtime_amendment.sh
# Compare that digest with the trusted build-runtime-bootstrap result.
chmod 700 -- /outside/repo/pull_runtime_amendment.sh
/outside/repo/pull_runtime_amendment.sh
```

The bootstrap requires an exact dedicated Box tree, checks Box SHA-1 and size,
downloads every package file through `.partial`, verifies its pinned SHA-256,
and atomically publishes the schema-2 package, reconstructed Git bundle, and
clean SHA-addressed checkout. Point `H100_RUNTIME_PACKAGE_ROOT` at the printed
package path and `H100_RUNTIME_BUNDLE` at the printed bundle path. The checkout
is an operator inspection/bootstrap copy; each Slurm allocation clones the
verified bundle independently, and phase staging reconstructs the audited
TRAIN+fixed-DEV8 CSV only for the training view.

## Build the final-path native H100 venv

Provide Judy's exact Python 3.11.13 executable and canonical libpython
directory. First require the complete base-runtime fingerprint to agree on at
least two eligible DGX compute nodes and preserve each node/digest record.
Then, from a CPU-only task on an eligible DGX node, remeasure that node and
require the accepted digest immediately before building one fresh venv at a
new permanent path. Never use a login-built receipt for H100 qualification.
The loader path is explicit because the shared interpreter cannot start
without it and Slurm uses `--export=NONE`. The build is offline and consumes
the verified Sprint 7d wheelhouse:

```bash
export H100_BASE_PYTHON_LIB_DIR=/cm/shared/mitre-apps/python/3.11.13/build/lib
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
cd /persistent/path/xview3-handoff/bootstrap-SPRINT7F_FULL_GIT_SHA
/path/to/bootstrap-python -B -m scripts.h100.build_venv build \
  --repo "$PWD" \
  --wheelhouse /scratch/xview3-base-extracted/environment/wheelhouse \
  --base-extraction-receipt /scratch/xview3-base-extracted/HANDOFF_EXTRACTED.json \
  --expected-base-payload-package-id xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --expected-base-payload-manifest-sha256 fccb0b505c89836a148afec709bb799f7af4908d955ea1b142e153154d830896 \
  --base-python /cm/shared/mitre-apps/python/3.11.13/build/bin/python3.11 \
  --output /persistent/venvs/xview3-h100-fp32
```

The builder uses `venv --copies`, `pip --no-index`, and the exact
`locks/env-v100node.txt` contract (`torch==2.11.0+cu126`). It checks the
normalized freeze and `pip check`, proves the wheelhouse tree is the one named
by the verified Sprint-7d `HANDOFF_EXTRACTED.json`, removes bytecode, seals the
venv read-only, and writes these read-only sidecars:

```text
/persistent/venvs/xview3-h100-fp32.sha256
/persistent/venvs/xview3-h100-fp32.build.json
```

The venv is deliberately non-relocatable. Before every Slurm allocation,
verify its complete tree, receipt, final path, environment lock, installed
freeze, exact wheelhouse/extraction receipt, and complete base-Python runtime.
The runtime digest covers the resolved executable, libpython, stdlib/platstdlib
(excluding only caches and base site/dist-packages), and deterministically
probed mapped system libraries:

```bash
export H100_BASE_PYTHON_LIB_DIR=/cm/shared/mitre-apps/python/3.11.13/build/lib
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
cd /persistent/path/xview3-handoff/bootstrap-SPRINT7F_FULL_GIT_SHA
/path/to/bootstrap-python -B -m scripts.h100.build_venv verify \
  --repo "$PWD" \
  --venv-root /persistent/venvs/xview3-h100-fp32 \
  --base-python /cm/shared/mitre-apps/python/3.11.13/build/bin/python3.11 \
  --wheelhouse /scratch/xview3-base-extracted/environment/wheelhouse \
  --base-extraction-receipt /scratch/xview3-base-extracted/HANDOFF_EXTRACTED.json \
  --expected-venv-sha256 VENV_TREE_64HEX \
  --expected-receipt-sha256 VENV_BUILD_JSON_64HEX \
  --expected-base-python-sha256 BASE_PYTHON_EXECUTABLE_64HEX \
  --expected-base-python-runtime-sha256 BASE_PYTHON_RUNTIME_64HEX \
  --expected-wheelhouse-sha256 WHEELHOUSE_TREE_64HEX \
  --expected-base-extraction-receipt-sha256 EXTRACTION_RECEIPT_64HEX \
  --expected-base-payload-package-id xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --expected-base-payload-manifest-sha256 fccb0b505c89836a148afec709bb799f7af4908d955ea1b142e153154d830896
```

Use only the module-form builder/verifier above. Judy endpoint controls killed
the literal direct `scripts/h100/build_venv.py` entry, while the module form
completed and correctly enforced the full base-runtime receipt. Preserve any
login-built venv as diagnostic evidence, and do not normalize a node-runtime
difference.

Jobs invoke `/persistent/venvs/xview3-h100-fp32/bin/python` directly under a
clean environment. They never activate the venv and never execute a container.
`NVIDIA_TF32_OVERRIDE=0` is set before CUDA initialization; Lightning remains
`32-true`, micro-batch/effective-batch 16, one process per GPU, and no DDP.

## Target test split and acceptance

The complete test collection is partitioned between two exact environments:

- The host transfer Python runs
  `tests/test_h100_handoff.py tests/test_experiment_manifest.py`, which require
  host `git`/`zstd`, and writes a source-receipt-bound host-test receipt.
- The sealed venv runs the entire remaining pytest collection with those two
  files ignored.

The underlying commands are:

```bash
/path/to/transfer-python -m pytest -q \
  tests/test_h100_handoff.py tests/test_experiment_manifest.py
/persistent/venvs/xview3-h100-fp32/bin/python -m pytest -q \
  --ignore=tests/test_h100_handoff.py \
  --ignore=tests/test_experiment_manifest.py
```

The Slurm pack records the first command through
`scripts.h100.host_test_gate`; do not substitute an unbound manual pass for its
receipt. `slurm/h100/submit.sh acceptance` combines the non-overlapping
receipts in `runs/.h100/PYTEST_ACCEPTANCE.json`; aggregate coverage must equal
the entire repository test suite. It also requires eight H100s at compute
capability 9.0,
strict IEEE FP32 backend assertions in child processes, all six value-sensitive
checkpoint loads, finite ViT/CNN train and full-scene inference probes, the
deterministic evaluation-GT count receipt, and the 200-step throughput gate.
V100 remains running unchanged throughout. H100 launch additionally requires
Slurm signal/requeue/resume smoke, corrected R2/R3 packages, CUTOVER_READY, and
the external diagnostic-isolation package.

See `slurm/h100/README.md` and the ignored `slurm/h100/site.env` interface for
the full target commands and receipt fields.

## Reverse result package

After all 32 H100 cells have valid completion markers, retain the sealed venv
build receipt as:

```text
runs/.h100/venv_build.json
```

Use another dedicated, initially empty Box folder. First source its external
Box settings and run preflight with the transfer venv; record the positive
`maximum_file_bytes` value from the JSON output as `H100_MAX_PART_BYTES`:

```bash
source /outside/repo/xview3-results-transfer.env
"$TRANSFER_PYTHON" -B -m scripts.handoff preflight --repo "$PWD"
export H100_MAX_PART_BYTES=REPLACE_WITH_PRINTED_MAXIMUM_FILE_BYTES
```

Then unset the Box settings and build with the sealed H100 venv. Construction
does not authenticate to Box and does not require `boxsdk`; the output argument
is a parent, and the builder creates the exact content-addressed child name only
after all archive/member hashes are known:

```bash
unset BOX_JWT_CONFIG BOX_FOLDER_ID
"$H100_VENV_ROOT/bin/python" -I -B -m scripts.handoff build-results \
  --repo "$PWD" \
  --runs-root /persistent/h100-runs \
  --campaign-manifest /persistent/h100-runs/.h100/campaign_manifest.json \
  --output-dir /outside/repo/result-packages \
  --max-part-bytes "$H100_MAX_PART_BYTES"
```

Record the printed package root, package ID, result-identity SHA-256, READY
SHA-256, manifest SHA-256, and SHA256SUMS SHA-256 out of band. Re-source the
transfer settings and upload that exact generated directory:

```bash
export RESULT_PACKAGE_ROOT=REPLACE_WITH_PRINTED_PACKAGE_ROOT
export RESULT_PACKAGE_ID=REPLACE_WITH_PRINTED_PACKAGE_ID
export RESULT_IDENTITY_SHA256=REPLACE_WITH_PRINTED_RESULT_IDENTITY_SHA256
export RESULT_READY_SHA256=REPLACE_WITH_PRINTED_READY_SHA256
export RESULT_MANIFEST_SHA256=REPLACE_WITH_PRINTED_MANIFEST_SHA256
export RESULT_SHA256SUMS_SHA256=REPLACE_WITH_PRINTED_SHA256SUMS_SHA256
source /outside/repo/xview3-results-transfer.env
"$TRANSFER_PYTHON" -B -m scripts.handoff upload \
  --repo "$PWD" \
  --package-root "$RESULT_PACKAGE_ROOT" \
  --receipt /outside/repo/receipts/results.upload.json
```

The reverse package includes the immutable 32-cell `TRAINING_COHORT.json`,
schema-2 training markers, separate immutable `test_metrics.json` artifacts,
configs, logs, best/last checkpoints, campaign/control provenance,
strict-FP32 hardware evidence, `venv_build.json`, and the dual base/runtime
transfer identities. Standalone R2/R3 files are excluded, but the exact
accepted R2/R3 metrics and provenance embedded in the validated
`CUTOVER_READY.json` remain part of campaign provenance.

On the receiving side, use the transfer venv and a destination that does not
exist. Supply every out-of-band identity; download uses `.partial` files and
atomic rename, then perform an explicit verification and extraction:

```bash
source /outside/repo/xview3-results-transfer.env
"$TRANSFER_PYTHON" -B -m scripts.handoff download \
  --repo "$PWD" \
  --package-root "/outside/repo/$RESULT_PACKAGE_ID" \
  --expected-ready-sha256 "$RESULT_READY_SHA256" \
  --expected-manifest-sha256 "$RESULT_MANIFEST_SHA256" \
  --expected-sha256sums-sha256 "$RESULT_SHA256SUMS_SHA256" \
  --expected-package-id "$RESULT_PACKAGE_ID"
"$TRANSFER_PYTHON" -B -m scripts.handoff verify \
  --package-root "/outside/repo/$RESULT_PACKAGE_ID"
"$TRANSFER_PYTHON" -B -m scripts.handoff extract \
  --package-root "/outside/repo/$RESULT_PACKAGE_ID" \
  --destination /outside/repo/xview3-h100-results-extracted
```

Status: the immutable Sprint 7d base upload and its remote verification are
complete. Earlier Sprint 7e runtime amendments remain immutable provenance.
Only a new content-addressed Sprint 7f package may execute the corrected
campaign, and it must go to a separate initially empty Box folder. The
login-built Judy venv is diagnostic only; cross-DGX agreement and a fresh
compute-built receipt, Sprint 7f transfer, H100 acceptance, corrected
reference/control receipts, Slurm smoke, and training remain pending.
