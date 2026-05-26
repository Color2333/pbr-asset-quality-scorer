#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "/storage/home/haojiang/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "/storage/home/haojiang/miniconda3/etc/profile.d/conda.sh"
  conda activate asset-quality-scorer
fi

CHANNEL="${1:-roughness}"
shift || true

if [[ "${CHANNEL}" == "all" ]]; then
  python -u asset_quality_scorer/scripts/train_phase1_ordinal.py --all "$@"
else
  python -u asset_quality_scorer/scripts/train_phase1_ordinal.py --channel "${CHANNEL}" "$@"
fi
