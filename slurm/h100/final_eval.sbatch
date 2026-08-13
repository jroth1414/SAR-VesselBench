#!/usr/bin/env bash
#SBATCH --job-name=xview3-h100-final-eval
#SBATCH --account=geofam
#SBATCH --partition=minor-use-case
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
# The 50-scene pass is projected near 3.1 h from TEST32 telemetry. Reserve a
# conservative non-requeueable window so scheduler timeout cannot consume the
# once-only lock; this does not change the scientific computation.
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail

# This must precede every Python process, nvidia-smi invocation, or CUDA import.
export NVIDIA_TF32_OVERRIDE=0
# Every Git operation in this job is read-only. The staged checkout is sealed
# 0555/0444, so forbid optional index refreshes and their index.lock side effect.
export GIT_OPTIONAL_LOCKS=0

compute_site="${1:?submit_final_eval.sh must pass the immutable compute-site snapshot}"
compute_site_sha256="${2:?submit_final_eval.sh must pass the compute-site SHA-256}"
if [[ ! "$compute_site_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "compute-site SHA-256 must be lowercase 64-hex" >&2
  exit 2
fi
if [[ -L "$compute_site" || ! -f "$compute_site" ||
      "$(stat -c '%a' "$compute_site")" != "444" ||
      "$(sha256sum "$compute_site" | awk '{print $1}')" != "$compute_site_sha256" ]]; then
  echo "final compute-site snapshot is absent, writable, linked, or hash-mismatched" >&2
  exit 2
fi
readonly compute_site compute_site_sha256
# shellcheck disable=SC1090
source "$compute_site"
unset BOX_JWT_CONFIG BOX_FOLDER_ID

if [[ "${SLURM_RESTART_COUNT:-0}" != "0" ]]; then
  echo "verified-final evaluation is non-requeueable; restart count must be zero" >&2
  exit 2
fi
if [[ ! "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "verified-final evaluation requires a numeric Slurm job ID" >&2
  exit 2
fi
if [[ "${SLURM_NTASKS:-1}" != "1" || "${SLURM_GPUS_ON_NODE:-8}" != "8" ]]; then
  echo "verified-final allocation must expose one task and exactly eight GPUs" >&2
  exit 2
fi
if [[ "$H100_BASE_PYTHON_LIB_DIR" != /* || -L "$H100_BASE_PYTHON_LIB_DIR" ||
      ! -d "$H100_BASE_PYTHON_LIB_DIR" ]]; then
  echo "snapshotted H100_BASE_PYTHON_LIB_DIR is invalid" >&2
  exit 2
fi
H100_BASE_PYTHON_LIB_DIR="$(realpath -e -- "$H100_BASE_PYTHON_LIB_DIR")"
if [[ ! -r "$H100_BASE_PYTHON_LIB_DIR/libpython3.11.so.1.0" ]]; then
  echo "snapshotted Judy libpython is absent" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
readonly H100_BASE_PYTHON_LIB_DIR LD_LIBRARY_PATH

canonical_existing_dir() {
  local name="$1"
  local raw="${!name}"
  local resolved
  if [[ "$raw" != /* || -L "$raw" || ! -d "$raw" ]]; then
    echo "$name must be an existing absolute non-symlink directory" >&2
    return 2
  fi
  resolved="$(realpath -e -- "$raw")"
  if [[ "$resolved" == "/" ]]; then
    echo "$name must not resolve to /" >&2
    return 2
  fi
  printf '%s\n' "$resolved"
}

H100_RUNS_ROOT="$(canonical_existing_dir H100_RUNS_ROOT)"
H100_SCRATCH_ROOT="$(canonical_existing_dir H100_SCRATCH_ROOT)"
H100_JOB_LOG_DIR="$(canonical_existing_dir H100_JOB_LOG_DIR)"
H100_FINAL_PACKAGE_ROOT="$(canonical_existing_dir H100_FINAL_PACKAGE_ROOT)"
H100_VENV_ROOT="$(canonical_existing_dir H100_VENV_ROOT)"
H100_WHEELHOUSE="$(canonical_existing_dir H100_WHEELHOUSE)"
H100_BASE_PACKAGE_ROOT="$(canonical_existing_dir H100_BASE_PACKAGE_ROOT)"
H100_FINAL_PROJECT_ROOT="$(canonical_existing_dir H100_FINAL_PROJECT_ROOT)"
readonly H100_RUNS_ROOT H100_SCRATCH_ROOT H100_JOB_LOG_DIR
readonly H100_FINAL_PACKAGE_ROOT H100_VENV_ROOT H100_WHEELHOUSE
readonly H100_BASE_PACKAGE_ROOT
readonly H100_FINAL_PROJECT_ROOT
if [[ "$(realpath -e -- "$H100_TRANSFER_PYTHON")" != \
      "$(realpath -e -- "$H100_VENV_ROOT/bin/python")" ]]; then
  echo "snapshotted transfer Python differs from the accepted sealed venv" >&2
  exit 2
fi

assert_disjoint() {
  local left_name="$1" left="$2" right_name="$3" right="$4"
  if [[ "$left" == "$right" || "$left" == "$right/"* ||
        "$right" == "$left/"* ]]; then
    echo "$left_name overlaps $right_name: $left and $right" >&2
    return 2
  fi
}
assert_disjoint H100_RUNS_ROOT "$H100_RUNS_ROOT" H100_SCRATCH_ROOT "$H100_SCRATCH_ROOT"
assert_disjoint H100_RUNS_ROOT "$H100_RUNS_ROOT" H100_JOB_LOG_DIR "$H100_JOB_LOG_DIR"
assert_disjoint H100_SCRATCH_ROOT "$H100_SCRATCH_ROOT" H100_JOB_LOG_DIR "$H100_JOB_LOG_DIR"
assert_disjoint H100_FINAL_PROJECT_ROOT "$H100_FINAL_PROJECT_ROOT" \
  H100_FINAL_PACKAGE_ROOT "$H100_FINAL_PACKAGE_ROOT"
assert_disjoint H100_FINAL_PROJECT_ROOT "$H100_FINAL_PROJECT_ROOT" \
  H100_VENV_ROOT "$H100_VENV_ROOT"
assert_disjoint H100_FINAL_PROJECT_ROOT "$H100_FINAL_PROJECT_ROOT" \
  H100_WHEELHOUSE "$H100_WHEELHOUSE"
assert_disjoint H100_FINAL_PROJECT_ROOT "$H100_FINAL_PROJECT_ROOT" \
  H100_BASE_PACKAGE_ROOT "$H100_BASE_PACKAGE_ROOT"
for protected_name in final-project final-package venv wheelhouse base-package; do
  case "$protected_name" in
    final-project) protected="$H100_FINAL_PROJECT_ROOT" ;;
    final-package) protected="$H100_FINAL_PACKAGE_ROOT" ;;
    venv) protected="$H100_VENV_ROOT" ;;
    wheelhouse) protected="$H100_WHEELHOUSE" ;;
    base-package) protected="$H100_BASE_PACKAGE_ROOT" ;;
  esac
  assert_disjoint H100_RUNS_ROOT "$H100_RUNS_ROOT" "$protected_name" "$protected"
  assert_disjoint H100_SCRATCH_ROOT "$H100_SCRATCH_ROOT" "$protected_name" "$protected"
  assert_disjoint H100_JOB_LOG_DIR "$H100_JOB_LOG_DIR" "$protected_name" "$protected"
done

allocation_scratch=""
final_data_view=""
staged_data_view_sha256=""

archive_failed_prelock_data_view() {
  local expected_receipt="$H100_RUNS_ROOT/.h100/FINAL_DATA_VIEW.json"
  local lockfile="$H100_RUNS_ROOT/final_eval.lock"
  local attempts_root="$H100_RUNS_ROOT/.h100/final-data-view-attempts"
  local destination="$attempts_root/FINAL_DATA_VIEW-${SLURM_JOB_ID}-prelock-failed.json"
  local actual_sha256=""
  local receipt_resolved=""
  local receipt_mode=""
  local attempts_resolved=""
  local attempts_mode=""
  local attempts_owner=""
  local destination_mode=""
  local destination_sha256=""

  # Nothing was published by this allocation, or semantic access has already
  # crossed the permanent lock boundary. In either case, never move anything.
  if [[ -z "$final_data_view" ]]; then
    return 0
  fi
  if [[ -e "$lockfile" || -L "$lockfile" ]]; then
    return 0
  fi
  # A signal can arrive after the package has fully validated and atomically
  # published its canonical receipt but before the parent shell assigns the
  # digest below. Recover only a receipt bound to this allocation-private view
  # and both exact source identities; never infer ownership from its filename.
  if [[ -z "$staged_data_view_sha256" ]]; then
    if [[ ! -e "$final_data_view" && ! -L "$final_data_view" ]]; then
      return 0
    fi
    staged_data_view_sha256="$(
      PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
        "$H100_TRANSFER_PYTHON" -B - \
          "$final_data_view" "$final_view" "$H100_FINAL_PACKAGE_ID" \
          "$H100_FINAL_EXPECTED_GIT_SHA" "$H100_EXPECTED_GIT_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

receipt_path, view_root = map(Path, sys.argv[1:3])
package_id, evaluator_sha, campaign_sha = sys.argv[3:]
raw = receipt_path.read_bytes()
payload = json.loads(raw)
assert payload.get("schema") == 1
assert payload.get("status") == "final-eval-data-view-staged"
assert payload.get("package", {}).get("package_id") == package_id
assert payload.get("source", {}).get("git_commit") == evaluator_sha
assert payload.get("source", {}).get("required_campaign_commit") == campaign_sha
assert payload.get("view", {}).get("root") == str(view_root)
assert payload.get("view", {}).get("repo") == str(view_root / "repo")
print(hashlib.sha256(raw).hexdigest())
PY
    )" || {
      staged_data_view_sha256=""
      echo "refusing unbound pre-lock FINAL_DATA_VIEW" >&2
      return 1
    }
  fi
  receipt_resolved="$(
    realpath -e -- "$final_data_view" 2>/dev/null || true
  )"
  receipt_mode="$(
    stat -c '%a' "$final_data_view" 2>/dev/null || true
  )"
  if [[ "$final_data_view" != "$expected_receipt" ||
        -L "$final_data_view" || ! -f "$final_data_view" ||
        "$receipt_resolved" != "$expected_receipt" ||
        "$receipt_mode" != "444" ]]; then
    echo "refusing to archive unsafe pre-lock FINAL_DATA_VIEW: $final_data_view" >&2
    return 1
  fi
  actual_sha256="$(sha256sum "$final_data_view" | awk '{print $1}')"
  if [[ "$actual_sha256" != "$staged_data_view_sha256" ]]; then
    echo "refusing to archive drifted pre-lock FINAL_DATA_VIEW" >&2
    return 1
  fi

  if [[ -e "$attempts_root" || -L "$attempts_root" ]]; then
    attempts_resolved="$(
      realpath -e -- "$attempts_root" 2>/dev/null || true
    )"
    attempts_mode="$(
      stat -c '%a' "$attempts_root" 2>/dev/null || true
    )"
    attempts_owner="$(
      stat -c '%u' "$attempts_root" 2>/dev/null || true
    )"
    if [[ -L "$attempts_root" || ! -d "$attempts_root" ||
          "$attempts_resolved" != "$attempts_root" ||
          "$attempts_mode" != "700" ||
          "$attempts_owner" != "$(id -u)" ]]; then
      echo "refusing unsafe final-data-view attempt archive: $attempts_root" >&2
      return 1
    fi
  else
    (umask 077; mkdir -m 700 -- "$attempts_root") || return 1
    chmod 0700 -- "$attempts_root" || return 1
  fi
  if [[ -e "$destination" || -L "$destination" ]]; then
    echo "pre-lock final-data-view archive destination exists: $destination" >&2
    return 1
  fi

  # GNU mv --no-clobber gives an atomic same-filesystem rename in the normal
  # case. The explicit postconditions detect a destination race without ever
  # deleting or overwriting either receipt.
  mv -T --no-clobber -- "$final_data_view" "$destination" || return 1
  destination_mode="$(
    stat -c '%a' "$destination" 2>/dev/null || true
  )"
  destination_sha256="$(
    sha256sum "$destination" 2>/dev/null | awk '{print $1}'
  )"
  if [[ -e "$final_data_view" || -L "$final_data_view" ||
        -L "$destination" || ! -f "$destination" ||
        "$destination_mode" != "444" ||
        "$destination_sha256" != "$staged_data_view_sha256" ]]; then
    echo "pre-lock final-data-view archive postcondition failed" >&2
    return 1
  fi

  # Persist the rename and the archive-directory creation. Failure is still
  # fail-closed: no job is resubmitted, and whichever receipt path exists is
  # retained for human review.
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$H100_TRANSFER_PYTHON" -B - "$destination" "$attempts_root" \
      "$H100_RUNS_ROOT/.h100" <<'PY'
import os
import sys

for index, raw in enumerate(sys.argv[1:]):
    flags = os.O_RDONLY
    if index:
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(raw, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
  echo "archived failed pre-lock final-data receipt: $destination" >&2
}

remove_allocation_scratch() {
  local actual="" expected=""
  if [[ -z "$allocation_scratch" ]]; then
    return 0
  fi
  expected="$H100_SCRATCH_ROOT/xview3-${SLURM_JOB_ID:-invalid}-r${SLURM_RESTART_COUNT:-0}"
  if [[ ! -e "$allocation_scratch" && ! -L "$allocation_scratch" ]]; then
    allocation_scratch=""
    return 0
  fi
  actual="$(realpath -e -- "$allocation_scratch" 2>/dev/null || true)"
  if [[ "$allocation_scratch" != "$expected" || "$actual" != "$allocation_scratch" ||
        -L "$allocation_scratch" || ! -d "$allocation_scratch" ||
        "$(stat -c '%u' "$allocation_scratch" 2>/dev/null || true)" != "$(id -u)" ]]; then
    echo "refusing unsafe allocation-scratch cleanup: $allocation_scratch" >&2
    return 1
  fi
  find "$allocation_scratch" -xdev -type d -exec chmod u+rwx -- {} + || return 1
  find "$allocation_scratch" -xdev -depth -mindepth 1 -delete || return 1
  rmdir -- "$allocation_scratch" || return 1
  allocation_scratch=""
}

cleanup_allocation_scratch() {
  local original_rc="$?"
  trap - EXIT
  if (( original_rc != 0 )); then
    if ! archive_failed_prelock_data_view; then
      echo "pre-lock FINAL_DATA_VIEW archival failed closed" >&2
    fi
  fi
  if ! remove_allocation_scratch && (( original_rc == 0 )); then
    original_rc=1
  fi
  exit "$original_rc"
}

expected_scratch="$H100_SCRATCH_ROOT/xview3-${SLURM_JOB_ID:?SLURM_JOB_ID is required}-r0"
if [[ -e "$expected_scratch" || -L "$expected_scratch" ]]; then
  echo "allocation scratch must start absent: $expected_scratch" >&2
  exit 1
fi
allocation_scratch="$expected_scratch"
trap cleanup_allocation_scratch EXIT
mkdir -m 700 -- "$allocation_scratch"
chmod 0700 -- "$allocation_scratch"
if [[ -L "$allocation_scratch" || ! -d "$allocation_scratch" ||
      "$(realpath -e -- "$allocation_scratch")" != "$allocation_scratch" ||
      "$(stat -c '%a' "$allocation_scratch")" != "700" ||
      "$(stat -c '%u' "$allocation_scratch")" != "$(id -u)" ]]; then
  echo "allocation scratch failed canonical ownership/mode validation" >&2
  exit 1
fi
scratch="$allocation_scratch"
readonly scratch
export TMPDIR="$scratch"
readonly TMPDIR
scratch_free="$(df -PB1 "$scratch" | awk 'NR==2 {print $4}')"
if (( scratch_free < 500000000000 )); then
  echo "allocation scratch has less than 500 GB free" >&2
  exit 1
fi

check_sha() {
  local expected="$1" path="$2" actual
  if [[ -L "$path" || ! -f "$path" ]]; then
    echo "required regular non-symlink file is absent: $path" >&2
    exit 1
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 1
  fi
}

meta_root="$H100_RUNS_ROOT/.h100"
if [[ -L "$meta_root" || ! -d "$meta_root" ||
      "$(realpath -e -- "$meta_root")" != "$meta_root" ||
      "$(stat -c '%u' "$meta_root")" != "$(id -u)" ]]; then
  echo "persistent H100 metadata root is unsafe: $meta_root" >&2
  exit 2
fi
h100_ready="$meta_root/H100_READY.json"
owner_amendment="$meta_root/FINAL_EVAL_OWNER_AMENDMENT.json"
final_data_view="$meta_root/FINAL_DATA_VIEW.json"
for absent in "$H100_RUNS_ROOT/final_eval.lock" "$final_data_view"; do
  if [[ -e "$absent" || -L "$absent" ]]; then
    echo "once-only final state appeared after submission: $absent" >&2
    exit 2
  fi
done
if [[ "$(realpath -m -- "$H100_FINAL_OWNER_AMENDMENT")" != "$owner_amendment" ]]; then
  echo "snapshotted owner-amendment path is not canonical" >&2
  exit 2
fi
for immutable in "$h100_ready" "$owner_amendment"; do
  if [[ -L "$immutable" || ! -f "$immutable" ||
        "$(stat -c '%a' "$immutable")" != "444" ]]; then
    echo "required immutable receipt is unsafe: $immutable" >&2
    exit 2
  fi
done
check_sha "$H100_READY_SHA256" "$h100_ready"
check_sha "$H100_FINAL_OWNER_AMENDMENT_SHA256" "$owner_amendment"
check_sha "$H100_FINAL_MANIFEST_SHA256" "$H100_FINAL_PACKAGE_ROOT/manifest.json"
check_sha "$H100_FINAL_READY_SHA256" "$H100_FINAL_PACKAGE_ROOT/READY.json"
check_sha "$H100_FINAL_SHA256SUMS_SHA256" "$H100_FINAL_PACKAGE_ROOT/SHA256SUMS"
check_sha "$H100_VENV_BUILD_SHA256" "$H100_VENV_BUILD_JSON"
check_sha "$H100_BASE_PYTHON_SHA256" "$H100_BASE_PYTHON"
(cd "$H100_FINAL_PACKAGE_ROOT" && sha256sum --check SHA256SUMS)
if [[ "$(realpath -m -- "$H100_VENV_BUILD_JSON")" != \
      "$(realpath -m -- "${H100_VENV_ROOT}.build.json")" ]]; then
  echo "snapshotted venv build receipt is not canonical" >&2
  exit 2
fi

bootstrap_repo="$scratch/bootstrap-repo"
final_view="$scratch/final-view"
[[ ! -e "$bootstrap_repo" && ! -e "$final_view" ]] || {
  echo "scratch bootstrap/final-view destinations must start absent" >&2
  exit 1
}
if [[ "$(git -C "$H100_FINAL_PROJECT_ROOT" rev-parse HEAD)" != \
      "$H100_FINAL_EXPECTED_GIT_SHA" ]]; then
  echo "snapshotted final project checkout changed after submission" >&2
  exit 2
fi
git -C "$H100_FINAL_PROJECT_ROOT" diff --quiet
git -C "$H100_FINAL_PROJECT_ROOT" diff --cached --quiet
mapfile -t final_project_status < <(
  git -C "$H100_FINAL_PROJECT_ROOT" status \
    --porcelain=v1 --untracked-files=all
)
for line in "${final_project_status[@]}"; do
  if [[ "$line" != "?? slurm/h100/site.env" ]]; then
    echo "snapshotted final project checkout is not clean: $line" >&2
    exit 2
  fi
done
git clone --no-local "$H100_FINAL_PROJECT_ROOT" "$bootstrap_repo"
git -C "$bootstrap_repo" checkout --detach "$H100_FINAL_EXPECTED_GIT_SHA"
if [[ "$(git -C "$bootstrap_repo" rev-parse HEAD)" != "$H100_FINAL_EXPECTED_GIT_SHA" ]]; then
  echo "staged final evaluator SHA mismatch" >&2
  exit 1
fi
git -C "$bootstrap_repo" merge-base --is-ancestor \
  "$H100_EXPECTED_GIT_SHA" "$H100_FINAL_EXPECTED_GIT_SHA" || {
    echo "staged final evaluator is not a campaign descendant" >&2
    exit 1
  }
git -C "$bootstrap_repo" diff --quiet
git -C "$bootstrap_repo" diff --cached --quiet

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$bootstrap_repo" \
  "$H100_TRANSFER_PYTHON" -B -m scripts.handoff.final_eval_package verify \
  --package-root "$H100_FINAL_PACKAGE_ROOT"
repo="$bootstrap_repo"
check_sha "$H100_ENV_LOCK_SHA256" "$repo/locks/env-v100node.txt"
check_sha "$H100_DETECTOR_SHA256" "$repo/configs/detector.yaml"
check_sha "$H100_SCORER_SHA256" "$repo/src/eval/scorer.py"
check_sha "$H100_SPLITS_SHA256" "$repo/data/splits.json"
check_sha "$H100_STATS_SHA256" "$repo/data/stats.json"
check_sha "$H100_LSSSDD_SHA256" "$repo/data/lsssdd_split.json"

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$repo" \
  "$H100_TRANSFER_PYTHON" -B -m scripts.h100.build_venv verify \
  --repo "$repo" \
  --venv-root "$H100_VENV_ROOT" \
  --base-python "$H100_BASE_PYTHON" \
  --wheelhouse "$H100_WHEELHOUSE" \
  --base-extraction-receipt "$H100_BASE_EXTRACTION_RECEIPT" \
  --expected-venv-sha256 "$H100_VENV_SHA256" \
  --expected-receipt-sha256 "$H100_VENV_BUILD_SHA256" \
  --expected-base-python-sha256 "$H100_BASE_PYTHON_SHA256" \
  --expected-base-python-runtime-sha256 "$H100_BASE_PYTHON_RUNTIME_SHA256" \
  --expected-wheelhouse-sha256 "$H100_WHEELHOUSE_SHA256" \
  --expected-base-extraction-receipt-sha256 "$H100_BASE_EXTRACTION_RECEIPT_SHA256" \
  --expected-base-payload-package-id "$H100_BASE_PACKAGE_ID" \
  --expected-base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256"

runtime_home="$scratch/runtime-home"
mkdir -m 700 "$runtime_home"
mkdir -m 700 "$runtime_home/cache" "$runtime_home/torch" "$runtime_home/triton"
native_env=(
  /usr/bin/env -i
  "HOME=$runtime_home"
  "XDG_CACHE_HOME=$runtime_home/cache"
  "TORCH_HOME=$runtime_home/torch"
  "TRITON_CACHE_DIR=$runtime_home/triton"
  "TMPDIR=$scratch"
  "PATH=$H100_VENV_ROOT/bin:/usr/local/bin:/usr/bin:/bin"
  "PYTHONPATH=$final_view/repo/scripts/h100:$final_view/repo:$bootstrap_repo/scripts/h100:$bootstrap_repo"
  "PYTHONNOUSERSITE=1"
  "PYTHONDONTWRITEBYTECODE=1"
  "PIP_NO_INDEX=1"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "NVIDIA_TF32_OVERRIDE=0"
  "GIT_OPTIONAL_LOCKS=0"
  "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
  "LD_LIBRARY_PATH=$H100_BASE_PYTHON_LIB_DIR"
  "OMP_NUM_THREADS=6"
  "SLURM_JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
  "SLURM_RESTART_COUNT=0"
  "SLURM_CLUSTER_NAME=${SLURM_CLUSTER_NAME:-unknown}"
  "SLURM_NTASKS=1"
  "SLURM_PROCID=${SLURM_PROCID:-0}"
  "SLURM_LOCALID=${SLURM_LOCALID:-0}"
  "SLURM_NODEID=${SLURM_NODEID:-0}"
  "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-8}"
  "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-48}"
  "SLURMD_NODENAME=${SLURMD_NODENAME:-unknown}"
  "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unknown}"
)
run_venv() { "${native_env[@]}" "$H100_VENV_ROOT/bin/python" "$@"; }

hardware="$meta_root/FINAL_H100_RUNTIME-${SLURM_JOB_ID}.json"
if [[ -e "$hardware" || -L "$hardware" ]]; then
  echo "final allocation hardware receipt already exists: $hardware" >&2
  exit 2
fi
hardware_payload="$(
  cd "$repo"
  run_venv -B -m scripts.h100.strict_fp32_probe --expected-gpus 8
)"
printf '%s\n' "$hardware_payload" | \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$H100_TRANSFER_PYTHON" -B -c '
import os, sys
from pathlib import Path

path = Path(sys.argv[1])
payload = sys.stdin.buffer.read()
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
flags |= getattr(os, "O_NOFOLLOW", 0)
descriptor = os.open(path, flags, 0o444)
try:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short final hardware receipt write")
        view = view[written:]
    os.fchmod(descriptor, 0o444)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
' "$hardware"

# Bind the new allocation to the exact accepted H100 hardware class, sealed
# venv, campaign source, and strict-IEEE backend before publishing any final
# data-view receipt or invoking the once-only evaluator.
(
  cd "$repo"
  run_venv -B -c '
import json, sys
from pathlib import Path
from scripts.h100.campaign import hardware_class

ready_path, hardware_path = map(Path, sys.argv[1:3])
campaign_sha, detector_sha, venv_root, venv_sha, build_sha = sys.argv[3:]
ready = json.loads(ready_path.read_text(encoding="utf-8"))
hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
if ready.get("schema") != 2 or ready.get("status") != "ready":
    raise RuntimeError("H100_READY is not schema-2 ready")
if ready.get("source", {}).get("git_sha") != campaign_sha:
    raise RuntimeError("H100_READY campaign SHA mismatch")
if ready.get("source", {}).get("frozen_sha256", {}).get("configs/detector.yaml") != detector_sha:
    raise RuntimeError("H100_READY detector hash mismatch")
accepted_venv = ready.get("venv", {})
if accepted_venv.get("path") != venv_root:
    raise RuntimeError("H100_READY sealed-venv path mismatch")
if accepted_venv.get("sha256") != venv_sha:
    raise RuntimeError("H100_READY sealed-venv tree mismatch")
if accepted_venv.get("venv_build_sha256") != build_sha:
    raise RuntimeError("H100_READY sealed-venv build receipt mismatch")
if set(ready.get("strict_fp32", {}).values()) != {"ieee"}:
    raise RuntimeError("H100_READY is not strict IEEE FP32")
if hardware.get("backend") != ready.get("strict_fp32"):
    raise RuntimeError("final allocation strict backend differs from H100_READY")
if hardware_class(hardware) != hardware_class(ready["hardware"]):
    raise RuntimeError("final allocation hardware class differs from H100_READY")
' "$h100_ready" "$hardware" "$H100_EXPECTED_GIT_SHA" \
    "$H100_DETECTOR_SHA256" "$H100_VENV_ROOT" "$H100_VENV_SHA256" \
    "$H100_VENV_BUILD_SHA256"
)

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$bootstrap_repo" \
  "$H100_TRANSFER_PYTHON" -B -m scripts.handoff.final_eval_package stage \
  --package-root "$H100_FINAL_PACKAGE_ROOT" \
  --destination "$final_view" \
  --receipt "$final_data_view" \
  --expected-package-id "$H100_FINAL_PACKAGE_ID" \
  --expected-ready-sha256 "$H100_FINAL_READY_SHA256" \
  --expected-manifest-sha256 "$H100_FINAL_MANIFEST_SHA256" \
  --expected-sha256sums-sha256 "$H100_FINAL_SHA256SUMS_SHA256"
if [[ -L "$final_data_view" || ! -f "$final_data_view" ||
      "$(stat -c '%a' "$final_data_view")" != "444" ]]; then
  echo "final package did not publish canonical immutable FINAL_DATA_VIEW" >&2
  exit 1
fi
staged_data_view_sha256="$(sha256sum "$final_data_view" | awk '{print $1}')"
if [[ ! "$staged_data_view_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "staged FINAL_DATA_VIEW SHA-256 is malformed" >&2
  exit 1
fi

repo="$final_view/repo"
if [[ -L "$repo" || ! -d "$repo" ||
      "$(realpath -e -- "$repo")" != "$repo" ||
      "$(git -C "$repo" rev-parse HEAD)" != "$H100_FINAL_EXPECTED_GIT_SHA" ]]; then
  echo "final package staged an invalid evaluator checkout" >&2
  exit 1
fi
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
check_sha "$H100_ENV_LOCK_SHA256" "$repo/locks/env-v100node.txt"
check_sha "$H100_DETECTOR_SHA256" "$repo/configs/detector.yaml"
check_sha "$H100_SCORER_SHA256" "$repo/src/eval/scorer.py"
check_sha "$H100_SPLITS_SHA256" "$repo/data/splits.json"
check_sha "$H100_STATS_SHA256" "$repo/data/stats.json"
check_sha "$H100_LSSSDD_SHA256" "$repo/data/lsssdd_split.json"

cd "$repo"
"${native_env[@]}" "$H100_VENV_ROOT/bin/python" -B -m src.eval.final_eval \
  --repo "$repo" \
  --runs-root "$H100_RUNS_ROOT" \
  --device cuda \
  --workers 8 \
  --hardware-json "$hardware" \
  --data-view-json "$final_data_view" \
  --data-package-id "$H100_FINAL_PACKAGE_ID" \
  --owner-amendment "$owner_amendment" \
  --i-am-sure
