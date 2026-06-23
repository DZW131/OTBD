#!/usr/bin/env bash

# Environment wrapper for the TaWald/nnU-Net nnSSL fine-tuning branch.
# Source this before MAE preprocessing or pretrained training commands.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${TRACK4_ROOT}/.." && pwd)}"
MAE_NNUNET_ROOT="${MAE_NNUNET_ROOT:-${PROJECT_ROOT}/third_party/TaWald_nnUNet_nnssl}"
MAE_CONDA_PREFIX="${MAE_CONDA_PREFIX:-/home/jingkun/miniconda3/envs/track4_wholeheart_mae}"
MAE_BIN_DIR="${MAE_BIN_DIR:-${MAE_CONDA_PREFIX}/bin}"
MAE_PRETRAINED_ROOT="${MAE_PRETRAINED_ROOT:-${PROJECT_ROOT}/pretrained/MAE}"
MAE_CHECKPOINT="${MAE_CHECKPOINT:-${MAE_PRETRAINED_ROOT}/ResEncL-OpenMind-MAE/checkpoint_final.pth}"
MAE_PRETRAINING_NAME="${MAE_PRETRAINING_NAME:-ResEncL_OpenMind_MAE}"

export PROJECT_ROOT
export MAE_NNUNET_ROOT
export MAE_CONDA_PREFIX
export MAE_BIN_DIR
export MAE_PRETRAINED_ROOT
export MAE_CHECKPOINT
export MAE_PRETRAINING_NAME
export nnssl_pretrained_models="${nnssl_pretrained_models:-${MAE_PRETRAINED_ROOT}}"
export nnUNet_results="${MAE_NNUNET_RESULTS:-${TRACK4_DATASET_ROOT}/nnUNet_result_mae}"
export PYTHONPATH="${MAE_NNUNET_ROOT}:${PYTHONPATH:-}"
export PATH="${MAE_BIN_DIR}:${PATH}"

mkdir -p "${nnUNet_results}" "${nnssl_pretrained_models}"
