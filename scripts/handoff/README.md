# Sprint 7e Judy native-venv handoff

The Judy handoff has two immutable layers:

1. the already uploaded and remotely verified Sprint 7d base payload, which
   contains the data, six core checkpoint directories, offline cu126
   wheelhouse, and historical source bundle; and
2. a separate, small, content-addressed Sprint 7e runtime amendment containing
   the current `sprint-7e-judy-venv` Git bundle.

Judy executes only a native, final-path Python virtual environment. There is no
container build, container launch, or venv activation in the Sprint 7e path.
The Apptainer definition retained in the Sprint 7d package is historical,
non-executed provenance; do not remove it from, rebuild, or otherwise mutate
that verified package. This native venv applies to the H100 lane only; it
does not change the live V100 environment, references, or fallback campaign.

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

Use a transfer environment outside the repository. It is not the training
venv:

```bash
python3 -m venv /outside/repo/xview3-box-transfer
/outside/repo/xview3-box-transfer/bin/pip install -r requirements-transfer.txt
chmod 0600 /outside/repo/box-jwt.json
export BOX_JWT_CONFIG=/outside/repo/box-jwt.json
export BOX_FOLDER_ID=REPLACE_WITH_NEW_EMPTY_RUNTIME_AMENDMENT_FOLDER_ID
```

`requirements-transfer.txt` pins the compatible `boxsdk[jwt]` v4 line and the
other transfer-only dependencies. Keep the JWT outside the repository at mode
`0600`; its path and contents must never be logged or archived.

The Sprint 7e amendment must use its own newly created, initially empty Box
folder. Do not point `BOX_FOLDER_ID` at the Sprint 7d base folder, an active
Sprint 7d download, or a future result-package folder. The Box preflight
performs a disposable write/update/rename/delete probe and discovers the
current maximum file size. Physical files above 50,000,000 bytes use resumable
chunked upload.

## Build and upload the Sprint 7e amendment

Build only from a clean, committed `sprint-7e-judy-venv` checkout. The output
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
xview3-h100-runtime-<SPRINT7E_FULL_GIT_SHA>-<RUNTIME_IDENTITY_SHA256>
```

It contains only the current runtime Git bundle plus `manifest.json`,
`SHA256SUMS`, and the `READY.json` written last. Its runtime identity binds the
exact Sprint 7d controls, Sprint 7e commit and bundle, unchanged environment
lock, and native strict-FP32 contract. Record the printed package ID and three
control hashes through a trusted out-of-band channel, then verify and upload:

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

## Download and extract on Judy

Keep a verified local copy of the Sprint 7d base available to validate the
amendment. If it is not already local, let any existing transfer finish or
download it once from its dedicated base-payload Box folder into a destination
that does not yet exist:

```bash
export BOX_JWT_CONFIG=/outside/repo/box-jwt.json
export BOX_FOLDER_ID=SPRINT7D_BASE_FOLDER_ID
python -m scripts.handoff download \
  --repo /path/to/bootstrap/repo \
  --package-root /scratch/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --expected-ready-sha256 b0d6ee18f9ddbd0d604cbea06610dcdbae6a9eb6d1f5ff3ea3431bd9e2d55f81 \
  --expected-manifest-sha256 fccb0b505c89836a148afec709bb799f7af4908d955ea1b142e153154d830896 \
  --expected-sha256sums-sha256 21c83b2e3b1b9d67bf00b8abca3ce267a5efd9362c1206b8d29ab21ca3e2d396 \
  --expected-package-id xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a
python -m scripts.handoff verify \
  --package-root /scratch/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a
```

Use the distinct, empty Sprint 7e Box folder for the amendment download:

```bash
export BOX_FOLDER_ID=RUNTIME_AMENDMENT_FOLDER_ID
python -m scripts.handoff download-runtime \
  --repo /path/to/bootstrap/repo \
  --base-package-root /scratch/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --package-root /scratch/xview3-h100-runtime-FULL40CHARSHA-IDENTITY64HEX \
  --expected-ready-sha256 RUNTIME_READY_64HEX \
  --expected-manifest-sha256 RUNTIME_MANIFEST_64HEX \
  --expected-sha256sums-sha256 RUNTIME_SHA256SUMS_64HEX \
  --expected-package-id xview3-h100-runtime-FULL40CHARSHA-IDENTITY64HEX
python -m scripts.handoff verify-runtime \
  --base-package-root /scratch/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --package-root /scratch/xview3-h100-runtime-FULL40CHARSHA-IDENTITY64HEX
python -m scripts.handoff extract-runtime \
  --base-package-root /scratch/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --package-root /scratch/xview3-h100-runtime-FULL40CHARSHA-IDENTITY64HEX \
  --destination /scratch/xview3-runtime-amendment
```

Downloads use private sibling staging and `.partial` files, validate Box SHA-1
and size, verify package SHA-256 values and both package identities, then
atomically rename the complete tree. Extraction atomically produces
`code/xview3-runtime.bundle` and `RUNTIME_AMENDMENT_EXTRACTED.json`. Clone only
that Sprint 7e bundle for H100 execution:

```bash
git clone --branch sprint-7e-judy-venv --single-branch \
  /scratch/xview3-runtime-amendment/code/xview3-runtime.bundle \
  /scratch/xview3-sprint7e
```

Verify and extract the Sprint 7d base separately for its data, checkpoints,
and wheelhouse, with at least 500 GB (500,000,000,000 bytes) free before
extraction:

```bash
python -m scripts.handoff extract \
  --package-root /scratch/xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --destination /scratch/xview3-base-extracted
```

Its historical Git bundle and Apptainer definition are rehashed as evidence
but are never the Judy execution checkout or runtime.

## Build the final-path native H100 venv

Provide Judy's exact Python 3.11.13 executable and canonical libpython
directory, then build directly at the permanent path. The loader path is
explicit because the shared interpreter cannot start without it and Slurm uses
`--export=NONE`. The build is offline and consumes the verified Sprint 7d
wheelhouse:

```bash
export H100_BASE_PYTHON_LIB_DIR=/cm/shared/mitre-apps/python/3.11.13/build/lib
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
cd /scratch/xview3-sprint7e
/path/to/bootstrap-python -m scripts.h100.build_venv build \
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
cd /scratch/xview3-sprint7e
/path/to/bootstrap-python -m scripts.h100.build_venv verify \
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
checkpoint loads, finite ViT/CNN train and full-scene inference probes, and the
200-step throughput gate. V100 remains running until this acceptance, Slurm
signal/requeue/resume smoke, current R2/R3 markers, and operator cutover gates
all pass.

See `slurm/h100/README.md` and the ignored `slurm/h100/site.env` interface for
the full target commands and receipt fields.

## Reverse result package

After all 32 H100 cells have valid completion markers, retain the sealed venv
build receipt as:

```text
runs/.h100/venv_build.json
```

Build and upload the reverse package to another dedicated, initially empty Box
folder:

```bash
python -m scripts.handoff build-results \
  --repo "$PWD" \
  --runs-root /persistent/h100-runs \
  --campaign-manifest /persistent/h100-runs/.h100/campaign_manifest.json \
  --output /outside/repo/xview3-h100-results-FULL40CHARSHA-IDENTITY64HEX \
  --max-part-bytes BYTES_FROM_BOX_PREFLIGHT
python -m scripts.handoff upload \
  --repo "$PWD" \
  --package-root /outside/repo/xview3-h100-results-FULL40CHARSHA-IDENTITY64HEX \
  --receipt /outside/repo/receipts/xview3-h100-results-FULL40CHARSHA-IDENTITY64HEX.upload.json
```

The reverse package includes metrics, configs, logs, best/last checkpoints,
campaign and cutover provenance, strict-FP32 hardware evidence,
`venv_build.json`, and the dual base/runtime transfer identities.

Status: the immutable Sprint 7d base upload and its remote verification are
complete. The first uploaded Sprint 7e runtime amendment binds Python 3.11.15
and is retained as superseded provenance. Only a new content-addressed package
from the owner-approved 3.11.13 branch tip may be used, and it must go to a
separate initially empty Box folder. Judy venv build, H100 acceptance,
throughput decision, cutover, and training remain pending.
