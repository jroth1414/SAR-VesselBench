#!/usr/bin/env bash
set -euo pipefail

# Submit the once-only, owner-amended 32-cell verified-final evaluation.
# This is intentionally separate from submit.sh/campaign.sbatch: the completed
# training campaign remains immutable, and this job is never requeued.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/../.." && pwd)"
site_env="$(realpath -m "${H100_SITE_ENV:-$script_dir/site.env}")"
if [[ -L "$site_env" || ! -f "$site_env" ]]; then
  echo "missing regular untracked final-eval site configuration: $site_env" >&2
  exit 2
fi
site_rel="$(realpath --relative-to="$repo" "$site_env")"
if [[ "$site_rel" != ../* ]] && \
   git -C "$repo" ls-files --error-unmatch -- "$site_rel" >/dev/null 2>&1; then
  echo "refusing tracked site.env: $site_env" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$site_env"

required=(
  H100_ACCOUNT H100_PARTITION H100_PROJECT H100_JOB_LOG_DIR
  H100_TRANSFER_PYTHON H100_SCRATCH_ROOT H100_RUNS_ROOT H100_CAMPAIGN_ID
  H100_EXPECTED_GIT_SHA H100_BASE_PACKAGE_ROOT H100_BASE_PACKAGE_ID
  H100_BASE_MANIFEST_SHA256 H100_WHEELHOUSE H100_WHEELHOUSE_SHA256
  H100_BASE_EXTRACTION_RECEIPT H100_BASE_EXTRACTION_RECEIPT_SHA256
  H100_BASE_PYTHON H100_BASE_PYTHON_LIB_DIR H100_BASE_PYTHON_SHA256
  H100_BASE_PYTHON_RUNTIME_SHA256 H100_VENV_ROOT H100_VENV_SHA256
  H100_VENV_BUILD_JSON H100_VENV_BUILD_SHA256 H100_ENV_LOCK_SHA256
  H100_DETECTOR_SHA256 H100_SCORER_SHA256 H100_SPLITS_SHA256
  H100_STATS_SHA256 H100_LSSSDD_SHA256
  H100_FINAL_PROJECT_ROOT H100_FINAL_EXPECTED_GIT_SHA
  H100_FINAL_PACKAGE_ROOT H100_FINAL_PACKAGE_ID
  H100_FINAL_MANIFEST_SHA256 H100_FINAL_READY_SHA256
  H100_FINAL_SHA256SUMS_SHA256 H100_FINAL_OWNER_AMENDMENT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "final-eval site.env is missing $name" >&2
    exit 2
  fi
done

git_names=(H100_EXPECTED_GIT_SHA H100_FINAL_EXPECTED_GIT_SHA)
for name in "${git_names[@]}"; do
  if [[ ! "${!name}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "$name must be a lowercase full 40-character Git SHA" >&2
    exit 2
  fi
done
hash_names=(
  H100_BASE_MANIFEST_SHA256 H100_WHEELHOUSE_SHA256
  H100_BASE_EXTRACTION_RECEIPT_SHA256 H100_BASE_PYTHON_SHA256
  H100_BASE_PYTHON_RUNTIME_SHA256 H100_VENV_SHA256
  H100_VENV_BUILD_SHA256 H100_ENV_LOCK_SHA256 H100_DETECTOR_SHA256
  H100_SCORER_SHA256 H100_SPLITS_SHA256 H100_STATS_SHA256
  H100_LSSSDD_SHA256
  H100_FINAL_MANIFEST_SHA256 H100_FINAL_READY_SHA256
  H100_FINAL_SHA256SUMS_SHA256
)
for name in "${hash_names[@]}"; do
  if [[ ! "${!name}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be a lowercase 64-character SHA-256" >&2
    exit 2
  fi
done
if [[ "$H100_FINAL_EXPECTED_GIT_SHA" == "$H100_EXPECTED_GIT_SHA" ]]; then
  echo "final evaluator SHA must be a distinct descendant of the campaign SHA" >&2
  exit 2
fi

canonical_dir() {
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

assert_disjoint() {
  local left_name="$1" left="$2" right_name="$3" right="$4"
  if [[ "$left" == "$right" || "$left" == "$right/"* ||
        "$right" == "$left/"* ]]; then
    echo "$left_name overlaps $right_name: $left and $right" >&2
    return 2
  fi
}

H100_FINAL_PROJECT_ROOT="$(canonical_dir H100_FINAL_PROJECT_ROOT)"
H100_FINAL_PACKAGE_ROOT="$(canonical_dir H100_FINAL_PACKAGE_ROOT)"
H100_RUNS_ROOT="$(canonical_dir H100_RUNS_ROOT)"
H100_SCRATCH_ROOT="$(canonical_dir H100_SCRATCH_ROOT)"
H100_JOB_LOG_DIR="$(canonical_dir H100_JOB_LOG_DIR)"
H100_VENV_ROOT="$(canonical_dir H100_VENV_ROOT)"
H100_WHEELHOUSE="$(canonical_dir H100_WHEELHOUSE)"
H100_BASE_PACKAGE_ROOT="$(canonical_dir H100_BASE_PACKAGE_ROOT)"
readonly H100_FINAL_PROJECT_ROOT H100_FINAL_PACKAGE_ROOT H100_RUNS_ROOT
readonly H100_SCRATCH_ROOT H100_JOB_LOG_DIR H100_VENV_ROOT
readonly H100_WHEELHOUSE H100_BASE_PACKAGE_ROOT

if [[ "$H100_FINAL_PROJECT_ROOT" != "$repo" ]]; then
  echo "H100_FINAL_PROJECT_ROOT does not match this checkout: $repo" >&2
  exit 2
fi
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

if [[ "$H100_BASE_PYTHON_LIB_DIR" != /* || -L "$H100_BASE_PYTHON_LIB_DIR" ||
      ! -d "$H100_BASE_PYTHON_LIB_DIR" ]]; then
  echo "H100_BASE_PYTHON_LIB_DIR must be an existing absolute non-symlink directory" >&2
  exit 2
fi
H100_BASE_PYTHON_LIB_DIR="$(realpath -e -- "$H100_BASE_PYTHON_LIB_DIR")"
if [[ ! -r "$H100_BASE_PYTHON_LIB_DIR/libpython3.11.so.1.0" ]]; then
  echo "H100_BASE_PYTHON_LIB_DIR lacks libpython3.11.so.1.0" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
readonly H100_BASE_PYTHON_LIB_DIR LD_LIBRARY_PATH
if [[ ! -x "$H100_TRANSFER_PYTHON" || ! -x "$H100_BASE_PYTHON" ||
      ! -x "$H100_VENV_ROOT/bin/python" ]]; then
  echo "transfer, base, and sealed-venv Python interpreters must be executable" >&2
  exit 2
fi
if [[ "$(realpath -e -- "$H100_TRANSFER_PYTHON")" != \
      "$(realpath -e -- "$H100_VENV_ROOT/bin/python")" ]]; then
  echo "final package control and scoring must use the accepted sealed-venv Python" >&2
  exit 2
fi

if [[ "$(git -C "$repo" rev-parse HEAD)" != "$H100_FINAL_EXPECTED_GIT_SHA" ]]; then
  echo "checkout HEAD differs from H100_FINAL_EXPECTED_GIT_SHA" >&2
  exit 2
fi
git -C "$repo" merge-base --is-ancestor \
  "$H100_EXPECTED_GIT_SHA" "$H100_FINAL_EXPECTED_GIT_SHA" || {
    echo "final evaluator is not a descendant of the completed campaign" >&2
    exit 2
  }
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
mapfile -t worktree_status < <(
  git -C "$repo" status --porcelain=v1 --untracked-files=all
)
for line in "${worktree_status[@]}"; do
  if [[ "$line" != "?? slurm/h100/site.env" ]]; then
    echo "final evaluator checkout is not clean: $line" >&2
    exit 2
  fi
done

check_sha() {
  local expected="$1" path="$2" actual
  if [[ -L "$path" || ! -f "$path" ]]; then
    echo "required regular non-symlink file is absent: $path" >&2
    exit 2
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 2
  fi
}

check_sha "$H100_ENV_LOCK_SHA256" "$repo/locks/env-v100node.txt"
check_sha "$H100_DETECTOR_SHA256" "$repo/configs/detector.yaml"
check_sha "$H100_SCORER_SHA256" "$repo/src/eval/scorer.py"
check_sha "$H100_SPLITS_SHA256" "$repo/data/splits.json"
check_sha "$H100_STATS_SHA256" "$repo/data/stats.json"
check_sha "$H100_LSSSDD_SHA256" "$repo/data/lsssdd_split.json"
check_sha "$H100_FINAL_MANIFEST_SHA256" "$H100_FINAL_PACKAGE_ROOT/manifest.json"
check_sha "$H100_FINAL_READY_SHA256" "$H100_FINAL_PACKAGE_ROOT/READY.json"
check_sha "$H100_FINAL_SHA256SUMS_SHA256" "$H100_FINAL_PACKAGE_ROOT/SHA256SUMS"
check_sha "$H100_VENV_BUILD_SHA256" "$H100_VENV_BUILD_JSON"
check_sha "$H100_BASE_PYTHON_SHA256" "$H100_BASE_PYTHON"

(cd "$H100_FINAL_PACKAGE_ROOT" && sha256sum --check SHA256SUMS)
# Do not reconstruct 50 large scene archives in the login node's implicit
# /tmp.  Physical bytes are checked above against the pinned SHA256SUMS; this
# lightweight control check binds them to the exact package/source identity.
# The full semantic package verifier runs inside the compute allocation with
# TMPDIR forced to the contracted allocation-private scratch tree.
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$H100_TRANSFER_PYTHON" -B -c '
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
expected_id, evaluator_sha, campaign_sha = sys.argv[2:]
manifest = json.loads((root / "manifest.json").read_text())
ready = json.loads((root / "READY.json").read_text())
assert manifest["format_version"] == ready["format_version"] == 1
assert manifest["package_type"] == ready["package_type"] == "h100-final-eval-inputs"
assert manifest["package_id"] == ready["package_id"] == expected_id
assert manifest["source"]["git_commit"] == ready["git_commit"] == evaluator_sha
assert manifest["source"]["required_campaign_commit"] == campaign_sha
assert ready["status"] == "READY"
' "$H100_FINAL_PACKAGE_ROOT" "$H100_FINAL_PACKAGE_ID" \
    "$H100_FINAL_EXPECTED_GIT_SHA" "$H100_EXPECTED_GIT_SHA"

h100_ready="$H100_RUNS_ROOT/.h100/H100_READY.json"
owner_amendment="$H100_RUNS_ROOT/.h100/FINAL_EVAL_OWNER_AMENDMENT.json"
final_data_view="$H100_RUNS_ROOT/.h100/FINAL_DATA_VIEW.json"
if [[ "$(realpath -m -- "$H100_FINAL_OWNER_AMENDMENT")" != "$owner_amendment" ]]; then
  echo "H100_FINAL_OWNER_AMENDMENT must be $owner_amendment" >&2
  exit 2
fi
for immutable in "$h100_ready" "$owner_amendment"; do
  if [[ -L "$immutable" || ! -f "$immutable" ||
        "$(stat -c '%a' "$immutable")" != "444" ]]; then
    echo "required receipt must be an immutable regular file: $immutable" >&2
    exit 2
  fi
done
H100_FINAL_OWNER_AMENDMENT_SHA256="$(
  sha256sum "$owner_amendment" | awk '{print $1}'
)"
export H100_FINAL_OWNER_AMENDMENT_SHA256
for absent in "$H100_RUNS_ROOT/final_eval.lock" "$final_data_view"; do
  if [[ -e "$absent" || -L "$absent" ]]; then
    echo "refusing final-eval submission because once-only state exists: $absent" >&2
    exit 2
  fi
done
H100_READY_SHA256="$(sha256sum "$h100_ready" | awk '{print $1}')"
export H100_READY_SHA256

active="$(
  squeue -h -u "$USER" -o '%A|%T|%j|%R' 2>/dev/null |
    grep -E 'xview3.*h100' || true
)"
if [[ -n "$active" ]]; then
  echo "refusing final evaluation while an xView3 H100 job is active:" >&2
  printf '%s\n' "$active" >&2
  exit 2
fi

mkdir -p "$H100_RUNS_ROOT/.h100/slurm"
snapshot_names=(
  "${required[@]}"
  H100_READY_SHA256 H100_FINAL_OWNER_AMENDMENT_SHA256
)
snapshot_tmp="$(mktemp "$H100_RUNS_ROOT/.h100/slurm/.final-compute-site.XXXXXX")"
snapshot_cleanup() { rm -f -- "$snapshot_tmp"; }
trap snapshot_cleanup EXIT
chmod 0600 "$snapshot_tmp"
for name in "${snapshot_names[@]}"; do
  printf '%s=%q\n' "$name" "${!name}" >> "$snapshot_tmp"
done
compute_site_sha256="$(sha256sum "$snapshot_tmp" | awk '{print $1}')"
compute_site="$H100_RUNS_ROOT/.h100/slurm/final-compute-site-${compute_site_sha256}.env"
if [[ -e "$compute_site" || -L "$compute_site" ]]; then
  if [[ -L "$compute_site" || ! -f "$compute_site" ||
        "$(sha256sum "$compute_site" | awk '{print $1}')" != "$compute_site_sha256" ]]; then
    echo "existing final compute-site snapshot is unsafe: $compute_site" >&2
    exit 2
  fi
else
  chmod 0444 "$snapshot_tmp"
  if ! ln -- "$snapshot_tmp" "$compute_site"; then
    if [[ -L "$compute_site" || ! -f "$compute_site" ||
          "$(sha256sum "$compute_site" | awk '{print $1}')" != "$compute_site_sha256" ]]; then
      echo "final compute-site snapshot installation raced" >&2
      exit 2
    fi
  fi
fi
chmod 0444 "$compute_site"
snapshot_cleanup
trap - EXIT

sbatch_args=(
  --account="$H100_ACCOUNT"
  --partition="$H100_PARTITION"
  --job-name="${H100_PROJECT}-h100-final-eval"
  --output="$H100_JOB_LOG_DIR/%x-%j.out"
  --export=NONE
  --no-requeue
)
if [[ -n "${H100_RESERVATION:-}" ]]; then
  sbatch_args+=(--reservation="$H100_RESERVATION")
fi
if [[ -n "${H100_MAIL_USER:-}" ]]; then
  sbatch_args+=(--mail-user="$H100_MAIL_USER")
fi
if [[ -n "${H100_MAIL_TYPE:-}" ]]; then
  sbatch_args+=(--mail-type="$H100_MAIL_TYPE")
fi

env -u BOX_JWT_CONFIG -u BOX_FOLDER_ID sbatch \
  "${sbatch_args[@]}" \
  "$script_dir/final_eval.sbatch" "$compute_site" "$compute_site_sha256"
