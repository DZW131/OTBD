#!/usr/bin/env bash
set -euo pipefail

MODALITY="${1:-}"
FOLD="${2:-}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"

if [[ "${MODALITY}" != "ct" && "${MODALITY}" != "mr" ]]; then
  echo "Usage: $0 {ct|mr} {0|1|2|3|4|all}" >&2
  exit 1
fi

if [[ -z "${FOLD}" ]]; then
  echo "Usage: $0 {ct|mr} {0|1|2|3|4|all}" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ "${MODALITY}" == "ct" ]]; then
  DATASET_ID=401
else
  DATASET_ID=402
fi

echo "Training ${MODALITY^^} dataset ${DATASET_ID}, configuration=${CONFIGURATION}, fold=${FOLD}"
nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" --npz
