#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "smoke" && "$mode" != "acceptance" && \
      "$mode" != "cutover-check" && "$mode" != "campaign" ]]; then
  echo "usage: $0 smoke|acceptance|cutover-check|campaign" >&2
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
  H100_RUNTIME_PACKAGE_ROOT H100_RUNTIME_PACKAGE_ID H100_RUNTIME_GIT_SHA
  H100_RUNTIME_BUNDLE H100_RUNTIME_BUNDLE_SHA256
  H100_RUNTIME_MANIFEST_SHA256 H100_RUNTIME_READY_SHA256
  H100_RUNTIME_SHA256SUMS_SHA256 H100_VENV_ROOT H100_VENV_SHA256
  H100_VENV_BUILD_JSON H100_VENV_BUILD_SHA256 H100_RUNS_ROOT
  H100_EXPECTED_GIT_SHA H100_JOB_LOG_DIR
  H100_PROJECT H100_PROJECT_ROOT H100_TRANSFER_PYTHON H100_ENV_LOCK_SHA256
  H100_BASE_PYTHON H100_BASE_PYTHON_SHA256 H100_DETECTOR_SHA256 H100_SCORER_SHA256
  H100_SPLITS_SHA256 H100_STATS_SHA256 H100_LSSSDD_SHA256
)
if [[ "$mode" == "acceptance" || "$mode" == "cutover-check" ]]; then
  required+=(H100_REMAINING_V100_WALL_HOURS)
fi
if [[ "$mode" == "cutover-check" || "$mode" == "campaign" ]]; then
  required+=(
    H100_REFERENCES_ROOT H100_EXPECTED_REFERENCE_GIT_SHA
    H100_REFERENCE_CAMPAIGN_ID H100_CUTOVER_READY
  )
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
  H100_RUNTIME_BUNDLE_SHA256 H100_RUNTIME_MANIFEST_SHA256
  H100_RUNTIME_READY_SHA256 H100_RUNTIME_SHA256SUMS_SHA256
  H100_VENV_SHA256 H100_VENV_BUILD_SHA256 H100_BASE_PYTHON_SHA256
  H100_ENV_LOCK_SHA256 H100_DETECTOR_SHA256 H100_SCORER_SHA256
  H100_SPLITS_SHA256 H100_STATS_SHA256 H100_LSSSDD_SHA256
)
for name in "${hash_names[@]}"; do
  if [[ ! "${!name}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be a lowercase 64-character SHA-256" >&2
    exit 2
  fi
done
if [[ "$(realpath -m "$H100_PROJECT_ROOT")" != "$repo" ]]; then
  echo "H100_PROJECT_ROOT does not match this checkout: $repo" >&2
  exit 2
fi
if [[ ! -x "$H100_BASE_PYTHON" || ! -x "$H100_VENV_ROOT/bin/python" ]]; then
  echo "base Python and exact native venv Python must both be executable" >&2
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

# Box credentials are transfer-host inputs and must not enter Slurm's exported
# environment or a native training process. Mode and site path are positional
# batch arguments; no user environment is exported into the allocation.
env -u BOX_JWT_CONFIG -u BOX_FOLDER_ID sbatch \
  --account="${H100_ACCOUNT:-geofam}" \
  --partition="${H100_PARTITION:-minor-use-case}" \
  --reservation="${H100_RESERVATION:-geofam}" \
  --job-name="${H100_PROJECT}-h100-${job_suffix}" \
  --output="$H100_JOB_LOG_DIR/%x-%j.out" \
  ${H100_MAIL_USER:+--mail-user="$H100_MAIL_USER"} \
  ${H100_MAIL_TYPE:+--mail-type="$H100_MAIL_TYPE"} \
  --export=NONE \
  "$batch_script" "$mode" "$site_env"
