#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "acceptance" && \
      "$mode" != "cutover-check" && "$mode" != "campaign" ]]; then
  echo "usage: $0 smoke|acceptance|cutover-check|campaign" >&2
  exit 2
fi
if [[ "$mode" == "cutover-check" || "$mode" == "campaign" ]]; then
  echo "$mode is not authorized until the content-addressed dynamic Box control-transfer protocol is implemented and reviewed" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/../.." && pwd)"
site_env="$(realpath -m "${H100_SITE_ENV:-$script_dir/site.env}")"
if [[ ! -f "$site_env" ]]; then
  echo "missing untracked site configuration: $site_env" >&2
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
  H100_BASE_PACKAGE_ROOT H100_BASE_PACKAGE_ID H100_BASE_GIT_SHA
  H100_BASE_MANIFEST_SHA256 H100_BASE_READY_SHA256
  H100_BASE_SHA256SUMS_SHA256 H100_BASE_REPO_BUNDLE_SHA256
  H100_WHEELHOUSE H100_WHEELHOUSE_SHA256 H100_BASE_EXTRACTION_RECEIPT
  H100_BASE_EXTRACTION_RECEIPT_SHA256
  H100_RUNTIME_PACKAGE_ROOT H100_RUNTIME_PACKAGE_ID H100_RUNTIME_GIT_SHA
  H100_RUNTIME_BUNDLE H100_RUNTIME_BUNDLE_SHA256
  H100_RUNTIME_MANIFEST_SHA256 H100_RUNTIME_READY_SHA256
  H100_RUNTIME_SHA256SUMS_SHA256 H100_VENV_ROOT H100_VENV_SHA256
  H100_VENV_BUILD_JSON H100_VENV_BUILD_SHA256 H100_RUNS_ROOT
  H100_EXPECTED_GIT_SHA H100_JOB_LOG_DIR H100_V100_CONTROL_PLANE
  H100_PROJECT H100_PROJECT_ROOT H100_TRANSFER_PYTHON H100_ENV_LOCK_SHA256
  H100_BASE_PYTHON H100_BASE_PYTHON_LIB_DIR H100_BASE_PYTHON_SHA256
  H100_BASE_PYTHON_RUNTIME_SHA256
  H100_DETECTOR_SHA256 H100_SCORER_SHA256
  H100_SPLITS_SHA256 H100_STATS_SHA256 H100_LSSSDD_SHA256
)
if [[ "$mode" == "acceptance" || "$mode" == "cutover-check" ]]; then
  required+=(H100_REMAINING_V100_WALL_HOURS)
fi
if [[ "$mode" == "cutover-check" || "$mode" == "campaign" ]]; then
  required+=(
    H100_EXPECTED_REFERENCE_GIT_SHA H100_REFERENCE_CAMPAIGN_ID H100_CUTOVER_READY
  )
fi
if [[ "$mode" == "cutover-check" ]]; then
  required+=(H100_REFERENCES_ROOT)
fi
if [[ "$mode" == "campaign" ]]; then
  required+=(
    H100_CAMPAIGN_ID H100_CUTOVER_READY_SHA256
    H100_V100_CORE_ARCHIVED H100_V100_CORE_ARCHIVED_SHA256
    H100_V100_ARCHIVE_MANIFEST H100_V100_ARCHIVE_MANIFEST_SHA256
  )
fi
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "site.env is missing $name" >&2
    exit 2
  fi
done
if [[ "$H100_V100_CONTROL_PLANE" != "box-transfer-v1" ]]; then
  echo "H100_V100_CONTROL_PLANE must be box-transfer-v1" >&2
  exit 2
fi
if [[ "$H100_RUNTIME_GIT_SHA" != "$H100_EXPECTED_GIT_SHA" ]]; then
  echo "H100_RUNTIME_GIT_SHA must equal H100_EXPECTED_GIT_SHA" >&2
  exit 2
fi
for name in H100_BASE_GIT_SHA H100_RUNTIME_GIT_SHA H100_EXPECTED_GIT_SHA; do
  if [[ ! "${!name}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "$name must be a lowercase full 40-character Git SHA" >&2
    exit 2
  fi
done
hash_names=(
  H100_BASE_MANIFEST_SHA256 H100_BASE_READY_SHA256
  H100_BASE_SHA256SUMS_SHA256 H100_BASE_REPO_BUNDLE_SHA256
  H100_WHEELHOUSE_SHA256 H100_BASE_EXTRACTION_RECEIPT_SHA256
  H100_RUNTIME_BUNDLE_SHA256 H100_RUNTIME_MANIFEST_SHA256
  H100_RUNTIME_READY_SHA256 H100_RUNTIME_SHA256SUMS_SHA256
  H100_VENV_SHA256 H100_VENV_BUILD_SHA256 H100_BASE_PYTHON_SHA256
  H100_BASE_PYTHON_RUNTIME_SHA256
  H100_ENV_LOCK_SHA256 H100_DETECTOR_SHA256 H100_SCORER_SHA256
  H100_SPLITS_SHA256 H100_STATS_SHA256 H100_LSSSDD_SHA256
)
for name in "${hash_names[@]}"; do
  if [[ ! "${!name}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be a lowercase 64-character SHA-256" >&2
    exit 2
  fi
done

# Judy and the live V100 host have separate filesystems. Canonicalize the two
# Judy-owned persistent write roots and reject overlap in either direction
# before the first mkdir, receipt write, or Slurm submission. V100 references
# enter Judy only as transferred, provenance-bound cutover evidence.
canonical_write_root() {
  local name="$1"
  local raw="${!name}"
  local canonical
  if [[ "$raw" != /* ]]; then
    echo "$name must be an absolute path" >&2
    return 2
  fi
  canonical="$(realpath -m -- "$raw")"
  if [[ "$canonical" == "/" ]]; then
    echo "$name must not resolve to /" >&2
    return 2
  fi
  printf '%s\n' "$canonical"
}

assert_disjoint_from_protected_root() {
  local name="$1"
  local candidate="$2"
  local protected_name="$3"
  local protected_root="$4"
  if [[ "$candidate" == "$protected_root" ||
        "$candidate" == "$protected_root/"* ||
        "$protected_root" == "$candidate/"* ]]; then
    echo "$name overlaps $protected_name $protected_root: $candidate" >&2
    return 2
  fi
}

h100_runs_root="$(canonical_write_root H100_RUNS_ROOT)"
h100_job_log_root="$(canonical_write_root H100_JOB_LOG_DIR)"
assert_disjoint_from_protected_root \
  H100_JOB_LOG_DIR "$h100_job_log_root" "H100 runs root" "$h100_runs_root"
for protected_name in \
  H100_PROJECT_ROOT H100_BASE_PACKAGE_ROOT H100_RUNTIME_PACKAGE_ROOT \
  H100_WHEELHOUSE H100_VENV_ROOT
do
  protected_value="${!protected_name}"
  if [[ "$protected_value" != /* ]]; then
    echo "$protected_name must be an absolute protected path" >&2
    exit 2
  fi
  protected_root="$(realpath -m -- "$protected_value")"
  assert_disjoint_from_protected_root \
    H100_RUNS_ROOT "$h100_runs_root" "$protected_name" "$protected_root"
  assert_disjoint_from_protected_root \
    H100_JOB_LOG_DIR "$h100_job_log_root" "$protected_name" "$protected_root"
done

reference_runs_root=""
if [[ "$mode" == "cutover-check" ]]; then
  reference_runs_root="$(canonical_write_root H100_REFERENCES_ROOT)"
  if [[ ! -d "$reference_runs_root" ]]; then
    echo "H100_REFERENCES_ROOT must resolve to verified transferred V100 reference evidence: $reference_runs_root" >&2
    exit 2
  fi
  assert_disjoint_from_protected_root \
    H100_RUNS_ROOT "$h100_runs_root" "V100 reference runs root" "$reference_runs_root"
  assert_disjoint_from_protected_root \
    H100_JOB_LOG_DIR "$h100_job_log_root" "V100 reference runs root" "$reference_runs_root"
  H100_REFERENCES_ROOT="$reference_runs_root"
else
  unset H100_REFERENCES_ROOT
fi
H100_RUNS_ROOT="$h100_runs_root"
H100_JOB_LOG_DIR="$h100_job_log_root"
readonly H100_RUNS_ROOT H100_JOB_LOG_DIR H100_REFERENCES_ROOT

if [[ "$(realpath -m "$H100_PROJECT_ROOT")" != "$repo" ]]; then
  echo "H100_PROJECT_ROOT does not match this checkout: $repo" >&2
  exit 2
fi
if [[ "$H100_BASE_PYTHON_LIB_DIR" != /* || ! -d "$H100_BASE_PYTHON_LIB_DIR" ]]; then
  echo "H100_BASE_PYTHON_LIB_DIR must be an absolute existing directory" >&2
  exit 2
fi
H100_BASE_PYTHON_LIB_DIR="$(realpath -e -- "$H100_BASE_PYTHON_LIB_DIR")"
if [[ ! -r "$H100_BASE_PYTHON_LIB_DIR/libpython3.11.so.1.0" ]]; then
  echo "H100_BASE_PYTHON_LIB_DIR must contain readable libpython3.11.so.1.0" >&2
  exit 2
fi
# Judy's shared CPython build does not carry an executable-relative libpython
# search path. Never inherit a mutable submit-host loader path: use this one
# canonical, snapshot-bound directory for transfer, base, and venv Python.
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
readonly H100_BASE_PYTHON_LIB_DIR LD_LIBRARY_PATH
if [[ ! -x "$H100_BASE_PYTHON" || ! -x "$H100_VENV_ROOT/bin/python" ]]; then
  echo "base Python and exact native venv Python must both be executable" >&2
  exit 2
fi
if [[ ! -d "$H100_WHEELHOUSE" || ! -f "$H100_BASE_EXTRACTION_RECEIPT" ]]; then
  echo "verified persistent wheelhouse/extraction receipt are required" >&2
  exit 2
fi
canonical_venv_receipt="${H100_VENV_ROOT}.build.json"
if [[ "$(realpath -m "$H100_VENV_BUILD_JSON")" != "$(realpath -m "$canonical_venv_receipt")" ]]; then
  echo "H100_VENV_BUILD_JSON must be the builder's canonical $canonical_venv_receipt" >&2
  exit 2
fi
if [[ "$(realpath -m "$H100_BASE_PACKAGE_ROOT")" == "$(realpath -m "$H100_RUNTIME_PACKAGE_ROOT")" ]]; then
  echo "base payload and runtime amendment roots must be distinct" >&2
  exit 2
fi
if [[ "$mode" == "cutover-check" || "$mode" == "campaign" ]]; then
  expected_cutover="$H100_RUNS_ROOT/.h100/CUTOVER_READY.json"
  if [[ "$(realpath -m "$H100_CUTOVER_READY")" != "$(realpath -m "$expected_cutover")" ]]; then
    echo "H100_CUTOVER_READY must be $expected_cutover" >&2
    exit 2
  fi
fi

mkdir -p "$H100_RUNS_ROOT/.h100/slurm"
mkdir -p "$H100_JOB_LOG_DIR"
if [[ "$mode" == "cutover-check" ]]; then
  PYTHONNOUSERSITE=1 PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" -m scripts.h100.cutover \
    --h100-ready "$H100_RUNS_ROOT/.h100/H100_READY.json" \
    --r2-run-dir "$H100_REFERENCES_ROOT/yolo26-f100" \
    --r3-run-dir "$H100_REFERENCES_ROOT/locateanything-zs" \
    --expected-h100-git-sha "$H100_EXPECTED_GIT_SHA" \
    --expected-reference-git-sha "$H100_EXPECTED_REFERENCE_GIT_SHA" \
    --expected-venv-sha256 "$H100_VENV_SHA256" \
    --expected-venv-build-sha256 "$H100_VENV_BUILD_SHA256" \
    --expected-base-python-sha256 "$H100_BASE_PYTHON_SHA256" \
    --expected-base-python-runtime-sha256 "$H100_BASE_PYTHON_RUNTIME_SHA256" \
    --expected-wheelhouse-sha256 "$H100_WHEELHOUSE_SHA256" \
    --expected-base-extraction-receipt-sha256 "$H100_BASE_EXTRACTION_RECEIPT_SHA256" \
    --expected-base-payload-package-id "$H100_BASE_PACKAGE_ID" \
    --expected-base-payload-git-sha "$H100_BASE_GIT_SHA" \
    --expected-base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256" \
    --expected-base-payload-ready-sha256 "$H100_BASE_READY_SHA256" \
    --expected-base-payload-sha256sums-sha256 "$H100_BASE_SHA256SUMS_SHA256" \
    --expected-base-payload-repo-bundle-sha256 "$H100_BASE_REPO_BUNDLE_SHA256" \
    --expected-runtime-amendment-package-id "$H100_RUNTIME_PACKAGE_ID" \
    --expected-runtime-amendment-git-sha "$H100_RUNTIME_GIT_SHA" \
    --expected-runtime-amendment-manifest-sha256 "$H100_RUNTIME_MANIFEST_SHA256" \
    --expected-runtime-amendment-ready-sha256 "$H100_RUNTIME_READY_SHA256" \
    --expected-runtime-amendment-sha256sums-sha256 "$H100_RUNTIME_SHA256SUMS_SHA256" \
    --expected-runtime-amendment-bundle-sha256 "$H100_RUNTIME_BUNDLE_SHA256" \
    --expected-frozen-sha256 "$H100_DETECTOR_SHA256" \
    --expected-frozen-sha256 "$H100_SCORER_SHA256" \
    --expected-frozen-sha256 "$H100_SPLITS_SHA256" \
    --expected-frozen-sha256 "$H100_STATS_SHA256" \
    --expected-frozen-sha256 "$H100_LSSSDD_SHA256" \
    --smoke-ready "$H100_RUNS_ROOT/.h100/slurm-smoke/SLURM_SMOKE_READY.json" \
    --expected-reference-campaign-id "$H100_REFERENCE_CAMPAIGN_ID" \
    --current-remaining-v100-wall-hours "$H100_REMAINING_V100_WALL_HOURS" \
    --output "$H100_CUTOVER_READY"
  echo "CUTOVER_READY written; no Slurm job was submitted." >&2
  echo "Gracefully stop/archive V100 core diagnostics, then create the external operator receipt." >&2
  exit 0
fi

if [[ "$mode" == "campaign" ]]; then
  canonical_archive_manifest="$H100_RUNS_ROOT/.h100/V100_CORE_ARCHIVE_MANIFEST.json"
  PYTHONNOUSERSITE=1 PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" -m scripts.h100.operator_cutover \
    --cutover-ready "$H100_CUTOVER_READY" \
    --cutover-ready-sha256 "$H100_CUTOVER_READY_SHA256" \
    --receipt "$H100_V100_CORE_ARCHIVED" \
    --receipt-sha256 "$H100_V100_CORE_ARCHIVED_SHA256" \
    --archive-manifest "$H100_V100_ARCHIVE_MANIFEST" \
    --archive-manifest-sha256 "$H100_V100_ARCHIVE_MANIFEST_SHA256" \
    --bound-archive-manifest "$canonical_archive_manifest" \
    --persist-meta-root "$H100_RUNS_ROOT/.h100" \
    --expected-h100-git-sha "$H100_EXPECTED_GIT_SHA" \
    --expected-venv-sha256 "$H100_VENV_SHA256" \
    --expected-base-payload-package-id "$H100_BASE_PACKAGE_ID" \
    --expected-base-payload-git-sha "$H100_BASE_GIT_SHA" \
    --expected-base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256" \
    --expected-base-payload-ready-sha256 "$H100_BASE_READY_SHA256" \
    --expected-base-payload-sha256sums-sha256 "$H100_BASE_SHA256SUMS_SHA256" \
    --expected-base-payload-repo-bundle-sha256 "$H100_BASE_REPO_BUNDLE_SHA256" \
    --expected-runtime-amendment-package-id "$H100_RUNTIME_PACKAGE_ID" \
    --expected-runtime-amendment-git-sha "$H100_RUNTIME_GIT_SHA" \
    --expected-runtime-amendment-manifest-sha256 "$H100_RUNTIME_MANIFEST_SHA256" \
    --expected-runtime-amendment-ready-sha256 "$H100_RUNTIME_READY_SHA256" \
    --expected-runtime-amendment-sha256sums-sha256 "$H100_RUNTIME_SHA256SUMS_SHA256" \
    --expected-runtime-amendment-bundle-sha256 "$H100_RUNTIME_BUNDLE_SHA256" \
    --expected-reference-git-sha "$H100_EXPECTED_REFERENCE_GIT_SHA" \
    --expected-reference-campaign-id "$H100_REFERENCE_CAMPAIGN_ID"
  H100_V100_CORE_ARCHIVED="$H100_RUNS_ROOT/.h100/V100_CORE_ARCHIVED.json"
  H100_V100_ARCHIVE_MANIFEST="$canonical_archive_manifest"
  export H100_V100_CORE_ARCHIVED H100_V100_ARCHIVE_MANIFEST
  echo "Canonical operator evidence persisted under $H100_RUNS_ROOT/.h100" >&2
fi

batch_script="$script_dir/campaign.sbatch"
job_suffix="fp32"
if [[ "$mode" == "smoke" ]]; then
  batch_script="$script_dir/smoke.sbatch"
  job_suffix="slurm-smoke"
fi

# Persist a content-addressed, read-only allowlist of compute inputs. The
# original untracked site.env can be edited after submission without changing
# queued or requeued allocations, and transfer credentials never enter this
# snapshot. Campaign mode runs this after canonical operator evidence paths
# have replaced their mutable source paths above.
snapshot_names=("${required[@]}")
for optional_name in H100_REAL_SCONTROL; do
  if [[ -n "${!optional_name:-}" ]]; then
    snapshot_names+=("$optional_name")
  fi
done
snapshot_tmp="$(mktemp "$H100_RUNS_ROOT/.h100/slurm/.compute-site.XXXXXX")"
snapshot_cleanup() {
  rm -f -- "$snapshot_tmp"
}
trap snapshot_cleanup EXIT
chmod 0600 "$snapshot_tmp"
for name in "${snapshot_names[@]}"; do
  printf '%s=%q\n' "$name" "${!name}" >> "$snapshot_tmp"
done
compute_site_sha256="$(sha256sum "$snapshot_tmp" | awk '{print $1}')"
compute_site="$H100_RUNS_ROOT/.h100/slurm/compute-site-${compute_site_sha256}.env"
if [[ -e "$compute_site" || -L "$compute_site" ]]; then
  if [[ -L "$compute_site" || ! -f "$compute_site" ||
        "$(sha256sum "$compute_site" | awk '{print $1}')" != "$compute_site_sha256" ]]; then
    echo "existing compute-site snapshot is not the expected regular content-addressed file: $compute_site" >&2
    exit 2
  fi
else
  chmod 0444 "$snapshot_tmp"
  if ! ln -- "$snapshot_tmp" "$compute_site"; then
    if [[ -L "$compute_site" || ! -f "$compute_site" ||
          "$(sha256sum "$compute_site" | awk '{print $1}')" != "$compute_site_sha256" ]]; then
      echo "compute-site snapshot installation raced with different content" >&2
      exit 2
    fi
  fi
fi
chmod 0444 "$compute_site"
snapshot_cleanup
trap - EXIT

# Box credentials are transfer-host inputs and must not enter Slurm's exported
# environment or a native training process. Mode, sanitized snapshot, and its
# digest are positional batch arguments; no user environment is exported.
env -u BOX_JWT_CONFIG -u BOX_FOLDER_ID sbatch \
  --account="${H100_ACCOUNT:-geofam}" \
  --partition="${H100_PARTITION:-minor-use-case}" \
  --reservation="${H100_RESERVATION:-geofam}" \
  --job-name="${H100_PROJECT}-h100-${job_suffix}" \
  --output="$H100_JOB_LOG_DIR/%x-%j.out" \
  ${H100_MAIL_USER:+--mail-user="$H100_MAIL_USER"} \
  ${H100_MAIL_TYPE:+--mail-type="$H100_MAIL_TYPE"} \
  --export=NONE \
  "$batch_script" "$mode" "$compute_site" "$compute_site_sha256"
