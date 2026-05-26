#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.yml"
ENV_NAME="${ASSET_QUALITY_ENV_NAME:-asset-quality-scorer}"
RECREATE="${ASSET_QUALITY_RECREATE:-0}"

if ! command -v conda >/dev/null 2>&1; then
  for candidate in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/root/miniconda3/etc/profile.d/conda.sh" \
    "/root/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck disable=SC1090
      source "${candidate}"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda was not found in PATH or common install locations." >&2
  echo "        Install Miniconda/Anaconda first, then rerun:" >&2
  echo "        bash ${ROOT_DIR}/scripts/create_env.sh" >&2
  exit 127
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${RECREATE}" == "1" ]]; then
    echo "[info] removing existing conda env: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
    echo "[info] creating conda env: ${ENV_NAME}"
    conda env create -f "${ENV_FILE}"
  else
    echo "[info] updating existing conda env: ${ENV_NAME}"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
  fi
else
  echo "[info] creating conda env: ${ENV_NAME}"
  conda env create -f "${ENV_FILE}"
fi

echo
echo "[done] activate with:"
echo "       conda activate ${ENV_NAME}"
