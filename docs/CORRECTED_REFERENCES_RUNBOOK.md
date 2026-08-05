# Corrected V100 references runbook

This is the only production launch path for the Sprint 7f R2/R3 correction.
It is independent of `runs/launch/run_fresh34_fp32.py` and never starts,
signals, pauses, unlocks, or writes into the live diagnostic core campaign.
R2 and R3 run serially on one separately leased V100.

The commands below are intentionally explicit. Do not shorten them by using
the old controller, an old reference result directory, or a hand-written
campaign manifest.

## 1. Preconditions and immutable identities

Use a fresh checkout of the final, committed `sprint-7f-eval-contract` SHA.
The checkout must remain clean for manifest creation and for the full duration
of each reference run. Ignored data/checkpoint staging is allowed only in this
fresh checkout; never add it to the live checkout.

The diagnostic core identity is already frozen by its read-only campaign
manifest:

```bash
export REFERENCE_REPO=/absolute/path/to/fresh-sprint7f-checkout
export REFERENCE_PYTHON=/nfs/WRIVA/jroth/xView3/.venv-v100/bin/python
export REFERENCE_CAMPAIGN_ID=sprint7f-v100-references-20260804
export V100_CORE_CAMPAIGN_ID=fresh34-v100-fp32-20260726
export V100_CORE_GIT_SHA=48e10534a8c7baf0662acd548f52928da69f23c8
cd "$REFERENCE_REPO"
export REFERENCE_GIT_SHA="$(git -c "safe.directory=$REFERENCE_REPO" -C "$REFERENCE_REPO" rev-parse HEAD)"
export REFERENCE_ENV_LOCK="$REFERENCE_REPO/locks/env-v100node.txt"
export REFERENCE_ENV_SHA="$($REFERENCE_PYTHON -c 'from pathlib import Path; from src.references.runtime_provenance import normalized_environment_lock_sha256; import sys; print(normalized_environment_lock_sha256(Path(sys.argv[1])))' "$REFERENCE_ENV_LOCK")"

test "$(git -c "safe.directory=$REFERENCE_REPO" -C "$REFERENCE_REPO" branch --show-current)" = sprint-7f-eval-contract
test -z "$(git -c "safe.directory=$REFERENCE_REPO" -C "$REFERENCE_REPO" status --porcelain --untracked-files=all)"
test "${#REFERENCE_GIT_SHA}" -eq 40
```

The Python command calculates the normalized package-set digest encoded by the
lock. Manifest creation independently compares that value with the packages
installed in `REFERENCE_PYTHON`; a merely well-formed digest cannot pass.

Do not use the diagnostic campaign's former reference JSON files. They bind
the old source and evaluation contract and are not reportable.

## 2. Create a fresh external campaign root

The campaign manifest and all results must be outside every Git checkout and
outside the live `runs/` tree. The launcher requires the exact layout below.

```bash
export REFERENCE_CAMPAIGN_ROOT="/nfs/WRIVA/jroth/reference-campaigns/$REFERENCE_CAMPAIGN_ID"
export REFERENCE_MANIFEST="$REFERENCE_CAMPAIGN_ROOT/REFERENCE_CAMPAIGN.json"
export REFERENCE_RESULTS_ROOT="$REFERENCE_CAMPAIGN_ROOT/results"
export REFERENCE_DATA_CONFIG="$REFERENCE_CAMPAIGN_ROOT/data-v100.yaml"

test ! -e "$REFERENCE_CAMPAIGN_ROOT"
install -d -m 0700 "$REFERENCE_CAMPAIGN_ROOT" "$REFERENCE_RESULTS_ROOT"
```

`REFERENCE_CAMPAIGN.json` is schema 1 and has exactly these fields:

- `schema`: `1`
- `campaign_role`: `corrected-v100-references`
- `campaign_id`: the new corrected-reference campaign ID
- `core_campaign_id`: the live diagnostic core campaign ID
- `core_git_sha`: the live diagnostic core Git SHA
- `git_sha`: the clean Sprint 7f reference Git SHA
- `environment_sha256`: normalized exact installed/locked package digest
- `environment_lock_sha256`: byte SHA-256 of `locks/env-v100node.txt`
- `runtime_launcher_sha256`: byte SHA-256 of
  `scripts/run_corrected_references.py`

Create it once; the launcher writes it atomically, marks it read-only, and
refuses replacement:

```bash
cd "$REFERENCE_REPO"
"$REFERENCE_PYTHON" -m scripts.run_corrected_references manifest \
  --expected-git-sha "$REFERENCE_GIT_SHA" \
  --reference-campaign-id "$REFERENCE_CAMPAIGN_ID" \
  --core-campaign-id "$V100_CORE_CAMPAIGN_ID" \
  --core-git-sha "$V100_CORE_GIT_SHA" \
  --environment-sha256 "$REFERENCE_ENV_SHA" \
  --environment-lock "$REFERENCE_ENV_LOCK" \
  --campaign-manifest "$REFERENCE_MANIFEST"
```

Record the printed manifest SHA-256 out of band.

## 3. Stage read-only inputs without mutating the live campaign

Create one external data config. Its four paths must be absolute. `splits` and
`stats` must resolve to the frozen files in the clean reference checkout;
`raw_xview3` and `chips` may point to the existing physical data because both
reference scorers open them read-only.

```bash
export LIVE_DATA_ROOT=/nfs/WRIVA/jroth/xView3/data
"$REFERENCE_PYTHON" - "$REFERENCE_DATA_CONFIG" "$REFERENCE_REPO" "$LIVE_DATA_ROOT" <<'PY'
from pathlib import Path
import sys
import yaml

destination, repo, live = map(Path, sys.argv[1:])
payload = yaml.safe_load((repo / "configs/data.yaml").read_text())
payload["paths"].update(
    raw_xview3=str((live / "raw/xview3").resolve()),
    chips=str((live / "chips").resolve()),
    splits=str((repo / "data/splits.json").resolve()),
    stats=str((repo / "data/stats.json").resolve()),
)
destination.write_text(yaml.safe_dump(payload, sort_keys=False))
PY
chmod 0400 "$REFERENCE_DATA_CONFIG"
```

If a storage mount requires repository-shaped paths, symlink only read-only
raw/chip directories inside the fresh reference checkout. Never create or
change a link in `/nfs/WRIVA/jroth/xView3`, and never symlink a results path.

R2 must use a physical copy of the preserved checkpoint at its canonical path
inside the fresh checkout. A symlink is rejected:

```bash
export R2_SOURCE=/nfs/WRIVA/jroth/xView3/runs/yolo26-f100/weights/best.pt
export R2_WEIGHTS="$REFERENCE_REPO/runs/yolo26-f100/weights/best.pt"
test ! -e "$R2_WEIGHTS"
install -D -m 0444 "$R2_SOURCE" "$R2_WEIGHTS"
test "$(sha256sum "$R2_WEIGHTS" | awk '{print $1}')" = 15520cb6cff9d4b01ed5c4a7e039fab763e8e5b0ca5b8e6bffd591ef0d7b8064
```

Stage a physical R3 payload outside the live tree. The R3 entrypoint verifies
all 39 non-cache files, total bytes, full payload-manifest hash, source note,
license, and pinned model revision before inference:

```bash
export R3_SOURCE=/nfs/WRIVA/jroth/xView3/data/weights/locateanything
export R3_WEIGHTS="$REFERENCE_CAMPAIGN_ROOT/inputs/locateanything"
test ! -e "$R3_WEIGHTS"
install -d -m 0700 "$REFERENCE_CAMPAIGN_ROOT/inputs"
cp -a --reflink=auto "$R3_SOURCE" "$R3_WEIGHTS"
chmod -R a-w "$R3_WEIGHTS"
```

Do not symlink the R2 checkpoint, R3 payload root, campaign manifest, or result
directories. The control-package builder later requires physical result JSONs.

## 4. Obtain one separate V100 lease

Run `gpu info`, then obtain exactly one independently available card with the
site's normal lease command:

```bash
gpu get --lock 1
```

Continue inside the returned leased environment. Use only the local index
shown there (normally `0` for a one-card lease), never a physical host ID:

```bash
nvidia-smi -L
export REFERENCE_GPU=0
```

If no card is independently leasable, stop and wait. Do not unlock, signal, or
borrow a card from `fresh34-v100-fp32-20260726`. The launcher masks its child
to the one explicit index, and the reference runtime additionally requires
Torch to see exactly one V100 with compute capability 7.0.

## 5. Run R2 and R3 serially

R2 is a corrected rescore of the preserved `best.pt`; this command cannot
train or export YOLO data:

```bash
cd "$REFERENCE_REPO"
"$REFERENCE_PYTHON" -m scripts.run_corrected_references r2 \
  --expected-git-sha "$REFERENCE_GIT_SHA" \
  --reference-campaign-id "$REFERENCE_CAMPAIGN_ID" \
  --core-campaign-id "$V100_CORE_CAMPAIGN_ID" \
  --core-git-sha "$V100_CORE_GIT_SHA" \
  --environment-sha256 "$REFERENCE_ENV_SHA" \
  --environment-lock "$REFERENCE_ENV_LOCK" \
  --campaign-manifest "$REFERENCE_MANIFEST" \
  --data-config "$REFERENCE_DATA_CONFIG" \
  --results-root "$REFERENCE_RESULTS_ROOT" \
  --r2-weights "$R2_WEIGHTS" \
  --gpu "$REFERENCE_GPU"
```

Wait for R2 to finish and publish both files before starting R3. The campaign
lock rejects overlapping invocations:

```bash
"$REFERENCE_PYTHON" -m scripts.run_corrected_references r3 \
  --expected-git-sha "$REFERENCE_GIT_SHA" \
  --reference-campaign-id "$REFERENCE_CAMPAIGN_ID" \
  --core-campaign-id "$V100_CORE_CAMPAIGN_ID" \
  --core-git-sha "$V100_CORE_GIT_SHA" \
  --environment-sha256 "$REFERENCE_ENV_SHA" \
  --environment-lock "$REFERENCE_ENV_LOCK" \
  --campaign-manifest "$REFERENCE_MANIFEST" \
  --data-config "$REFERENCE_DATA_CONFIG" \
  --results-root "$REFERENCE_RESULTS_ROOT" \
  --r3-weights "$R3_WEIGHTS" \
  --gpu "$REFERENCE_GPU"
```

`make references` is a thin wrapper around the same launcher and intentionally
requires `REFERENCE_ACTION=manifest|r2|r3` plus the full `REFERENCE_ARGS`.

Each result is committed by `final_metrics.json`, written only after its
read-only `runtime_provenance.json` sidecar:

```text
<campaign-root>/results/yolo26-f100/final_metrics.json
<campaign-root>/results/yolo26-f100/runtime_provenance.json
<campaign-root>/results/locateanything-zs/final_metrics.json
<campaign-root>/results/locateanything-zs/runtime_provenance.json
```

If a result directory exists, the launcher refuses to reuse or overwrite it.
Preserve partial evidence and start a new campaign ID/root unless the directory
was never created. Never repair a marker by hand.

## 6. CPU validation and handoff

The launcher and schema tests do not initialize CUDA:

```bash
cd "$REFERENCE_REPO"
"$REFERENCE_PYTHON" -m pytest -q \
  tests/test_corrected_reference_launcher.py \
  tests/test_eval_contract_diagnostic.py \
  tests/test_h100_runtime_amendment.py
```

After both result pairs exist, follow `scripts/handoff/README.md` to build the
narrow references control package from five physical JSON files: the immutable
`REFERENCE_CAMPAIGN.json` plus the four R2/R3 metrics/provenance files. Bind the
package to `REFERENCE_CAMPAIGN_ID` (not the old core campaign ID), the
Sprint 7f Git SHA, and the evaluation-contract version. Upload it only to a
new, initially empty Box child folder. The live V100 core campaign continues
as a diagnostic throughout this workflow.
