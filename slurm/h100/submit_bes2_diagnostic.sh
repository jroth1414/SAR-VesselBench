#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "audit" && "$mode" != "probe" && "$mode" != "full" ]]; then
  echo "usage: $0 audit|probe|full" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/../.." && pwd)"
site_env="$(realpath -m "${BES2_SITE_ENV:-$script_dir/bes2-site.env}")"
if [[ ! -f "$site_env" || -L "$site_env" ]]; then
  echo "missing regular untracked BES2 site configuration: $site_env" >&2
  exit 2
fi
site_rel="$(realpath --relative-to="$repo" "$site_env")"
if [[ "$site_rel" != ../* ]] &&
   git -C "$repo" ls-files --error-unmatch -- "$site_rel" >/dev/null 2>&1; then
  echo "refusing tracked BES2 site configuration: $site_env" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$site_env"

required=(
  H100_PROJECT_ROOT H100_SCRATCH_ROOT H100_RUNS_ROOT H100_TRANSFER_PYTHON
  H100_BASE_PACKAGE_ROOT H100_BASE_PACKAGE_ID H100_BASE_MANIFEST_SHA256
  H100_RUNTIME_PACKAGE_ROOT H100_RUNTIME_PACKAGE_ID H100_RUNTIME_GIT_SHA
  H100_RUNTIME_MANIFEST_SHA256 H100_RUNTIME_READY_SHA256
  H100_RUNTIME_SHA256SUMS_SHA256 H100_EXPECTED_GIT_SHA
  H100_WHEELHOUSE H100_WHEELHOUSE_SHA256 H100_BASE_EXTRACTION_RECEIPT
  H100_BASE_EXTRACTION_RECEIPT_SHA256 H100_BASE_PYTHON
  H100_BASE_PYTHON_LIB_DIR H100_BASE_PYTHON_SHA256
  H100_BASE_PYTHON_RUNTIME_SHA256 H100_VENV_ROOT H100_VENV_SHA256
  H100_VENV_BUILD_JSON H100_VENV_BUILD_SHA256 H100_ENV_LOCK_SHA256
  H100_DETECTOR_SHA256 H100_SCORER_SHA256 H100_SPLITS_SHA256
  H100_STATS_SHA256 H100_LSSSDD_SHA256
  BES2_PACKAGE_ROOT BES2_BUNDLE BES2_DIAGNOSTIC_ROOT
  BES2_ORIGINAL_RUNS_ROOT BES2_JOB_LOG_DIR BES2_FORECAST_GPU_HOURS
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "BES2 site configuration is missing $name" >&2
    exit 2
  fi
done

if [[ "$H100_EXPECTED_GIT_SHA" != "1a82d508fbeb9fdf6868a9637611e9018952fb43" ||
      "$H100_RUNTIME_GIT_SHA" != "$H100_EXPECTED_GIT_SHA" ]]; then
  echo "BES2 staging must bind the accepted original H100 campaign source" >&2
  exit 2
fi
if [[ "$(realpath -m -- "$H100_RUNS_ROOT")" != \
      "$(realpath -m -- "$BES2_ORIGINAL_RUNS_ROOT")" ]]; then
  echo "BES2_ORIGINAL_RUNS_ROOT must be the canonical original H100 runs root" >&2
  exit 2
fi

git_sha="$(git -C "$repo" rev-parse HEAD)"
branch="$(git -C "$repo" branch --show-current)"
dirty="$(git -C "$repo" status --porcelain=v1 --untracked-files=all)"
if [[ "$branch" != "sprint-10b-bes2-budget-amendment" ||
      ! "$git_sha" =~ ^[0-9a-f]{40}$ || -n "$dirty" ]]; then
  echo "submit from the clean sprint-10b-bes2-budget-amendment bootstrap checkout" >&2
  exit 2
fi
if ! git -C "$repo" merge-base --is-ancestor \
  61bc8391a3baa19b053ffa85e201caf36e0f53a4 "$git_sha"
then
  echo "BES2 source lacks the reviewed Sprint 10a ancestor" >&2
  exit 2
fi
BES2_EXPECTED_GIT_SHA="$git_sha"
BES2_PROJECT_ROOT="$repo"
export BES2_EXPECTED_GIT_SHA BES2_PROJECT_ROOT

hash_names=(
  H100_BASE_MANIFEST_SHA256 H100_RUNTIME_MANIFEST_SHA256
  H100_RUNTIME_READY_SHA256 H100_RUNTIME_SHA256SUMS_SHA256
  H100_WHEELHOUSE_SHA256 H100_BASE_EXTRACTION_RECEIPT_SHA256
  H100_BASE_PYTHON_SHA256 H100_BASE_PYTHON_RUNTIME_SHA256
  H100_VENV_SHA256 H100_VENV_BUILD_SHA256 H100_ENV_LOCK_SHA256
  H100_DETECTOR_SHA256 H100_SCORER_SHA256 H100_SPLITS_SHA256
  H100_STATS_SHA256 H100_LSSSDD_SHA256
)
for name in "${hash_names[@]}"; do
  if [[ ! "${!name}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be lowercase SHA-256" >&2
    exit 2
  fi
done
if [[ ! "$BES2_FORECAST_GPU_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v value="$BES2_FORECAST_GPU_HOURS" \
     'BEGIN { exit !(value > 0 && value <= 190) }'
then
  echo "BES2 forecast must be positive and no more than approved 190 GPU-hours" >&2
  exit 2
fi

canonical_root() {
  local name="$1"
  local raw="${!name}"
  local canonical
  if [[ "$raw" != /* ]]; then
    echo "$name must be absolute" >&2
    return 2
  fi
  canonical="$(realpath -m -- "$raw")"
  if [[ "$canonical" == "/" ]]; then
    echo "$name must not resolve to /" >&2
    return 2
  fi
  printf '%s\n' "$canonical"
}

assert_disjoint() {
  local left_name="$1"
  local left="$2"
  local right_name="$3"
  local right="$4"
  if [[ "$left" == "$right" || "$left" == "$right/"* || "$right" == "$left/"* ]]; then
    echo "$left_name overlaps $right_name: $left / $right" >&2
    return 2
  fi
}

BES2_DIAGNOSTIC_ROOT="$(canonical_root BES2_DIAGNOSTIC_ROOT)"
BES2_ORIGINAL_RUNS_ROOT="$(canonical_root BES2_ORIGINAL_RUNS_ROOT)"
BES2_JOB_LOG_DIR="$(canonical_root BES2_JOB_LOG_DIR)"
H100_SCRATCH_ROOT="$(canonical_root H100_SCRATCH_ROOT)"
BES2_PACKAGE_ROOT="$(canonical_root BES2_PACKAGE_ROOT)"
BES2_PROJECT_ROOT="$(canonical_root BES2_PROJECT_ROOT)"
export BES2_DIAGNOSTIC_ROOT BES2_ORIGINAL_RUNS_ROOT BES2_JOB_LOG_DIR
export H100_SCRATCH_ROOT BES2_PACKAGE_ROOT BES2_PROJECT_ROOT
for root_name in BES2_DIAGNOSTIC_ROOT BES2_JOB_LOG_DIR; do
  root_value="${!root_name}"
  if [[ -L "$root_value" || ! -d "$root_value" ||
        "$(realpath -e -- "$root_value")" != "$root_value" ]]; then
    echo "$root_name must be an existing canonical non-symlink directory" >&2
    exit 2
  fi
done
if [[ "$mode" == "audit" ]]; then
  if [[ -n "$(find "$BES2_DIAGNOSTIC_ROOT" -mindepth 1 -print -quit)" ]]; then
    echo "initial BES2 diagnostic namespace must start empty" >&2
    exit 2
  fi
  if [[ -n "$(find "$BES2_JOB_LOG_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "initial BES2 job-log namespace must start empty" >&2
    exit 2
  fi
fi
if [[ "$BES2_PROJECT_ROOT" != "$repo" || -L "$BES2_PROJECT_ROOT" ||
      ! -d "$BES2_PROJECT_ROOT" ||
      "$(realpath -e -- "$BES2_PROJECT_ROOT")" != "$BES2_PROJECT_ROOT" ]]; then
  echo "BES2 project root must be the canonical bootstrap checkout" >&2
  exit 2
fi
if [[ -L "$BES2_PACKAGE_ROOT" || ! -d "$BES2_PACKAGE_ROOT" ||
      "$(realpath -e -- "$BES2_PACKAGE_ROOT")" != "$BES2_PACKAGE_ROOT" ]]; then
  echo "BES2 package root must be canonical and non-symlink" >&2
  exit 2
fi
if [[ "$BES2_BUNDLE" != /* || -L "$BES2_BUNDLE" || ! -f "$BES2_BUNDLE" ||
      "$(realpath -e -- "$BES2_BUNDLE")" != "$BES2_BUNDLE" ]]; then
  echo "BES2 bundle must be the canonical bootstrap bundle" >&2
  exit 2
fi
assert_disjoint BES2_DIAGNOSTIC_ROOT "$BES2_DIAGNOSTIC_ROOT" \
  BES2_ORIGINAL_RUNS_ROOT "$BES2_ORIGINAL_RUNS_ROOT"
assert_disjoint BES2_DIAGNOSTIC_ROOT "$BES2_DIAGNOSTIC_ROOT" \
  BES2_PACKAGE_ROOT "$BES2_PACKAGE_ROOT"
assert_disjoint BES2_DIAGNOSTIC_ROOT "$BES2_DIAGNOSTIC_ROOT" \
  BES2_PROJECT_ROOT "$BES2_PROJECT_ROOT"
assert_disjoint BES2_DIAGNOSTIC_ROOT "$BES2_DIAGNOSTIC_ROOT" \
  H100_SCRATCH_ROOT "$H100_SCRATCH_ROOT"
assert_disjoint BES2_DIAGNOSTIC_ROOT "$BES2_DIAGNOSTIC_ROOT" \
  BES2_JOB_LOG_DIR "$BES2_JOB_LOG_DIR"
assert_disjoint BES2_ORIGINAL_RUNS_ROOT "$BES2_ORIGINAL_RUNS_ROOT" \
  H100_SCRATCH_ROOT "$H100_SCRATCH_ROOT"
assert_disjoint BES2_ORIGINAL_RUNS_ROOT "$BES2_ORIGINAL_RUNS_ROOT" \
  BES2_JOB_LOG_DIR "$BES2_JOB_LOG_DIR"

if [[ ! -x "$H100_TRANSFER_PYTHON" || ! -x "$H100_BASE_PYTHON" ||
      ! -x "$H100_VENV_ROOT/bin/python" ||
      ! -f "$H100_BASE_EXTRACTION_RECEIPT" ||
      ! -f "$H100_VENV_BUILD_JSON" || ! -d "$H100_WHEELHOUSE" ]]; then
  echo "BES2 submit requires the accepted sealed Judy runtime inputs" >&2
  exit 2
fi
if [[ "$H100_BASE_PYTHON_LIB_DIR" != /* ||
      -L "$H100_BASE_PYTHON_LIB_DIR" ||
      ! -d "$H100_BASE_PYTHON_LIB_DIR" ||
      "$(realpath -e -- "$H100_BASE_PYTHON_LIB_DIR")" != "$H100_BASE_PYTHON_LIB_DIR" ||
      ! -r "$H100_BASE_PYTHON_LIB_DIR/libpython3.11.so.1.0" ]]; then
  echo "invalid Judy libpython snapshot" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"

h100_ready="$BES2_ORIGINAL_RUNS_ROOT/.h100/H100_READY.json"
if [[ -L "$h100_ready" || ! -f "$h100_ready" ||
      "$(stat -c '%a' "$h100_ready")" != "444" ]]; then
  echo "missing immutable original H100_READY" >&2
  exit 2
fi
BES2_H100_READY_SHA256="$(sha256sum "$h100_ready" | awk '{print $1}')"
export BES2_H100_READY_SHA256

assert_final_unconsumed() {
  local relative
  for relative in \
    final_eval.lock \
    .h100/FINAL_DATA_VIEW.json \
    .h100/FINAL_GROUND_TRUTH_CONSUMED.json \
    .h100/FINAL_NORMALIZED_GROUND_TRUTH.json \
    .h100/FINAL_EVAL_COMPLETE.json
  do
    if [[ -e "$BES2_ORIGINAL_RUNS_ROOT/$relative" ||
          -L "$BES2_ORIGINAL_RUNS_ROOT/$relative" ]]; then
      echo "joint final access is no longer legal: $relative exists" >&2
      return 1
    fi
  done
  if compgen -G "$BES2_ORIGINAL_RUNS_ROOT/*/final_verified_metrics.json" >/dev/null; then
    echo "joint final access is no longer legal: final cell output exists" >&2
    return 1
  fi
}
assert_final_unconsumed

PYTHONNOUSERSITE=1 PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" \
  -B -m scripts.handoff verify-bes2-amendment \
  --package-root "$BES2_PACKAGE_ROOT" \
  --expected-h100-ready-sha256 "$BES2_H100_READY_SHA256" >/dev/null

manifest_query='import json,sys;'
manifest_query+='p=json.load(open(sys.argv[1]));'
manifest_query+='print(p["package_id"]);'
manifest_query+='print(p["source"]["git_commit"]);'
manifest_query+='print(p["source"]["git_bundle_sha256"])'
mapfile -t amendment_values < <(
  "$H100_TRANSFER_PYTHON" -c "$manifest_query" \
    "$BES2_PACKAGE_ROOT/manifest.json"
)
if [[ "${#amendment_values[@]}" -ne 3 ||
      "${amendment_values[1]}" != "$BES2_EXPECTED_GIT_SHA" ]]; then
  echo "BES2 amendment manifest differs from the submitting source" >&2
  exit 2
fi
BES2_PACKAGE_ID="${amendment_values[0]}"
BES2_BUNDLE_SHA256="${amendment_values[2]}"
BES2_MANIFEST_SHA256="$(sha256sum "$BES2_PACKAGE_ROOT/manifest.json" | awk '{print $1}')"
BES2_READY_SHA256="$(sha256sum "$BES2_PACKAGE_ROOT/READY.json" | awk '{print $1}')"
BES2_SHA256SUMS_SHA256="$(sha256sum "$BES2_PACKAGE_ROOT/SHA256SUMS" | awk '{print $1}')"
if [[ "$(sha256sum "$BES2_BUNDLE" | awk '{print $1}')" != "$BES2_BUNDLE_SHA256" ]]; then
  echo "bootstrap BES2 bundle differs from its amendment manifest" >&2
  exit 2
fi
export BES2_PACKAGE_ID BES2_BUNDLE_SHA256 BES2_MANIFEST_SHA256
export BES2_READY_SHA256 BES2_SHA256SUMS_SHA256

if [[ "$mode" != "audit" &&
      ( ! -f "$BES2_DIAGNOSTIC_ROOT/.control/BES2_DIAGNOSTIC_READY.json" ||
        -L "$BES2_DIAGNOSTIC_ROOT/.control/BES2_DIAGNOSTIC_READY.json" ) ]]; then
  echo "probe/full submission requires completed diagnostic audit" >&2
  exit 2
fi
if [[ "$mode" == "full" &&
      ( ! -f "$BES2_DIAGNOSTIC_ROOT/.control/BES2_PROBES_COMPLETE.json" ||
        -L "$BES2_DIAGNOSTIC_ROOT/.control/BES2_PROBES_COMPLETE.json" ) ]]; then
  echo "full submission requires completed paired probes" >&2
  exit 2
fi

mkdir -m 700 -p "$BES2_DIAGNOSTIC_ROOT/.control/slurm"
mkdir -m 700 -p "$BES2_JOB_LOG_DIR"
snapshot_names=(
  H100_PROJECT_ROOT H100_SCRATCH_ROOT H100_TRANSFER_PYTHON
  H100_BASE_PACKAGE_ROOT H100_BASE_PACKAGE_ID H100_BASE_MANIFEST_SHA256
  H100_RUNTIME_PACKAGE_ROOT H100_RUNTIME_PACKAGE_ID H100_RUNTIME_GIT_SHA
  H100_RUNTIME_MANIFEST_SHA256 H100_RUNTIME_READY_SHA256
  H100_RUNTIME_SHA256SUMS_SHA256 H100_EXPECTED_GIT_SHA
  H100_WHEELHOUSE H100_WHEELHOUSE_SHA256 H100_BASE_EXTRACTION_RECEIPT
  H100_BASE_EXTRACTION_RECEIPT_SHA256 H100_BASE_PYTHON
  H100_BASE_PYTHON_LIB_DIR H100_BASE_PYTHON_SHA256
  H100_BASE_PYTHON_RUNTIME_SHA256 H100_VENV_ROOT H100_VENV_SHA256
  H100_VENV_BUILD_JSON H100_VENV_BUILD_SHA256 H100_ENV_LOCK_SHA256
  H100_DETECTOR_SHA256 H100_SCORER_SHA256 H100_SPLITS_SHA256
  H100_STATS_SHA256 H100_LSSSDD_SHA256
  BES2_PACKAGE_ROOT BES2_PACKAGE_ID BES2_BUNDLE BES2_BUNDLE_SHA256
  BES2_MANIFEST_SHA256 BES2_READY_SHA256 BES2_SHA256SUMS_SHA256
  BES2_EXPECTED_GIT_SHA BES2_PROJECT_ROOT BES2_DIAGNOSTIC_ROOT
  BES2_ORIGINAL_RUNS_ROOT BES2_JOB_LOG_DIR BES2_FORECAST_GPU_HOURS
  BES2_H100_READY_SHA256
)
if [[ -n "${H100_REAL_SCONTROL:-}" ]]; then
  snapshot_names+=(H100_REAL_SCONTROL)
fi
snapshot_tmp="$(mktemp "$BES2_DIAGNOSTIC_ROOT/.control/slurm/.compute-site.XXXXXX")"
snapshot_cleanup() {
  rm -f -- "$snapshot_tmp"
}
trap snapshot_cleanup EXIT
chmod 0600 "$snapshot_tmp"
for name in "${snapshot_names[@]}"; do
  printf '%s=%q\n' "$name" "${!name}" >> "$snapshot_tmp"
done
snapshot_sha256="$(sha256sum "$snapshot_tmp" | awk '{print $1}')"
snapshot="$BES2_DIAGNOSTIC_ROOT/.control/slurm/compute-site-${snapshot_sha256}.env"
if [[ -e "$snapshot" || -L "$snapshot" ]]; then
  if [[ -L "$snapshot" || ! -f "$snapshot" ||
        "$(sha256sum "$snapshot" | awk '{print $1}')" != "$snapshot_sha256" ]]; then
    echo "existing BES2 compute-site snapshot drifted" >&2
    exit 2
  fi
else
  chmod 0444 "$snapshot_tmp"
  if ! ln -- "$snapshot_tmp" "$snapshot"; then
    if [[ -L "$snapshot" || ! -f "$snapshot" ||
          "$(sha256sum "$snapshot" | awk '{print $1}')" != "$snapshot_sha256" ]]; then
      echo "BES2 compute-site snapshot publication raced" >&2
      exit 2
    fi
  fi
fi
chmod 0444 "$snapshot"
snapshot_cleanup
trap - EXIT

gpus=2
cpus=32
walltime="2-18:00:00"
if [[ "$mode" == "audit" ]]; then
  gpus=1
  cpus=16
  walltime="08:00:00"
elif [[ "$mode" == "probe" ]]; then
  walltime="12:00:00"
fi
sbatch_args=(
  --account="${H100_ACCOUNT:-geofam}"
  --partition="${H100_PARTITION:-minor-use-case}"
  --job-name="${H100_PROJECT:-xview3}-bes2-$mode"
  --output="$BES2_JOB_LOG_DIR/%x-%j.out"
  --gpus-per-node="$gpus"
  --cpus-per-task="$cpus"
  --time="$walltime"
  --export=NONE
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
  "$script_dir/bes2_diagnostic.sbatch" \
  "$mode" "$snapshot" "$snapshot_sha256"
