#!/usr/bin/env bash
set -euo pipefail

TRACK4_ROOT="${TRACK4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRACK4_DATASET_ROOT="${TRACK4_DATASET_ROOT:-${TRACK4_ROOT}/DATASET}"

export TRACK4_ROOT
export TRACK4_DATASET_ROOT
export nnUNet_raw="${nnUNet_raw:-${TRACK4_DATASET_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${TRACK4_DATASET_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${TRACK4_DATASET_ROOT}/nnUNet_result}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"
