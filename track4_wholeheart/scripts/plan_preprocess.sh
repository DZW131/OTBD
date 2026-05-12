#!/usr/bin/env bash
set -euo pipefail

MODALITY="${1:-}"
if [[ "${MODALITY}" != "ct" && "${MODALITY}" != "mr" ]]; then
  echo "Usage: $0 {ct|mr}" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ "${MODALITY}" == "ct" ]]; then
  DATASET_ID=401
else
  DATASET_ID=402
fi

echo "Planning/preprocessing ${MODALITY^^} dataset ${DATASET_ID}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"

nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --verify_dataset_integrity
