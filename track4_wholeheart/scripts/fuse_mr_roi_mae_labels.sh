#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  BASE_ROOT=/path/to/tuned_roi_preds \
  DONOR_ROOT=/path/to/mae_preds \
  OUTPUT_ROOT=/path/to/output \
  bash track4_wholeheart/scripts/fuse_mr_roi_mae_labels.sh

Fuse MR tuned-ROI predictions with MAE predictions in train-label space.

Default strategy, validated on five-fold OOF:
  base: tuned AO+Myo ROI prediction
  donor: MAE prediction
  selected labels from donor: LV,RV,Myo,PA = 1,2,5,7
  protected base labels: LA,RA,AO = 3,4,6
  mode: add-only
  optional postprocess: class-aware-hd

Supported layouts:
  OOF:  BASE_ROOT/fold_0/validation/*.nii.gz ... fold_4/validation/*.nii.gz
  flat: BASE_ROOT/*.nii.gz and DONOR_ROOT/*.nii.gz

Environment overrides:
  SELECTED_LABELS="1,2,5,7"
  PROTECT_LABELS="3,4,6"
  FUSION_MODE=add-only
  FOLDS="0 1 2 3 4"
  POSTPROCESS=1
  POSTPROCESS_PRESET=class-aware-hd
  OVERWRITE=1
  PYTHON=python
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

BASE_ROOT="${BASE_ROOT:?Set BASE_ROOT to tuned ROI prediction root or flat prediction dir.}"
DONOR_ROOT="${DONOR_ROOT:?Set DONOR_ROOT to MAE prediction root or flat prediction dir.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRACK4_ROOT}/outputs/mr_roi_mae_fusion}"
SELECTED_LABELS="${SELECTED_LABELS:-1,2,5,7}"
PROTECT_LABELS="${PROTECT_LABELS:-3,4,6}"
FUSION_MODE="${FUSION_MODE:-add-only}"
FOLDS="${FOLDS:-0 1 2 3 4}"
POSTPROCESS="${POSTPROCESS:-1}"
POSTPROCESS_PRESET="${POSTPROCESS_PRESET:-class-aware-hd}"
OVERWRITE="${OVERWRITE:-1}"
PYTHON="${PYTHON:-python}"

if [[ "${FUSION_MODE}" != "add-only" && "${FUSION_MODE}" != "replace-protect" ]]; then
  echo "ERROR: FUSION_MODE must be add-only or replace-protect, got ${FUSION_MODE}" >&2
  exit 1
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}"

cat <<EOF
MR tuned ROI + MAE label fusion
base_root=${BASE_ROOT}
donor_root=${DONOR_ROOT}
output_root=${OUTPUT_ROOT}
selected_labels=${SELECTED_LABELS}
protect_labels=${PROTECT_LABELS}
fusion_mode=${FUSION_MODE}
postprocess=${POSTPROCESS}
postprocess_preset=${POSTPROCESS_PRESET}
EOF

run_fusion_pair() {
  local base_dir="$1"
  local donor_dir="$2"
  local fused_dir="$3"
  local final_dir="$4"

  if [[ ! -d "${base_dir}" ]]; then
    echo "ERROR: missing base dir: ${base_dir}" >&2
    exit 1
  fi
  if [[ ! -d "${donor_dir}" ]]; then
    echo "ERROR: missing donor dir: ${donor_dir}" >&2
    exit 1
  fi

  "${PYTHON}" "${TRACK4_ROOT}/scripts/fuse_selected_labels.py" \
    --base-dir "${base_dir}" \
    --donor-dir "${donor_dir}" \
    --output-dir "${fused_dir}" \
    --selected-labels "${SELECTED_LABELS}" \
    --protect-labels "${PROTECT_LABELS}" \
    --mode "${FUSION_MODE}" \
    --overwrite

  if [[ "${POSTPROCESS}" == "1" ]]; then
    "${PYTHON}" "${TRACK4_ROOT}/scripts/postprocess_predictions.py" \
      --input-dir "${fused_dir}" \
      --output-dir "${final_dir}" \
      --label-space train \
      --preset "${POSTPROCESS_PRESET}"
  else
    mkdir -p "$(dirname "${final_dir}")"
    if [[ "${fused_dir}" != "${final_dir}" ]]; then
      cp -a "${fused_dir}" "${final_dir}"
    fi
  fi
}

if [[ -d "${BASE_ROOT}/fold_0/validation" ]]; then
  for fold in ${FOLDS}; do
    echo "=== fold ${fold} ==="
    run_fusion_pair \
      "${BASE_ROOT}/fold_${fold}/validation" \
      "${DONOR_ROOT}/fold_${fold}/validation" \
      "${OUTPUT_ROOT}/fused_raw/fold_${fold}/validation" \
      "${OUTPUT_ROOT}/fused_classaware/fold_${fold}/validation"
  done
  echo "Done."
  echo "Raw fused predictions: ${OUTPUT_ROOT}/fused_raw"
  echo "Final train-label predictions: ${OUTPUT_ROOT}/fused_classaware"
else
  run_fusion_pair \
    "${BASE_ROOT}" \
    "${DONOR_ROOT}" \
    "${OUTPUT_ROOT}/fused_raw" \
    "${OUTPUT_ROOT}/fused_classaware"
  echo "Done."
  echo "Raw fused predictions: ${OUTPUT_ROOT}/fused_raw"
  echo "Final train-label predictions: ${OUTPUT_ROOT}/fused_classaware"
fi
