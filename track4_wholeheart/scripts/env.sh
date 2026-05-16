#!/usr/bin/env bash

# Keep this file safe to source from interactive shells. The training and
# preprocessing entrypoints set their own strict shell options.
if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _TRACK4_ENV_FILE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  eval '_TRACK4_ENV_FILE="${(%):-%x}"'
else
  _TRACK4_ENV_FILE="$0"
fi

TRACK4_ROOT="${TRACK4_ROOT:-$(cd "$(dirname "${_TRACK4_ENV_FILE}")/.." && pwd)}"
TRACK4_DATASET_ROOT="${TRACK4_DATASET_ROOT:-${TRACK4_ROOT}/DATASET}"

export TRACK4_ROOT
export TRACK4_DATASET_ROOT
export nnUNet_raw="${nnUNet_raw:-${TRACK4_DATASET_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${TRACK4_DATASET_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${TRACK4_DATASET_ROOT}/nnUNet_result}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

unset _TRACK4_ENV_FILE
