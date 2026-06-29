#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash track4_wholeheart/scripts/run_mr_grata_lite_oof.sh

Runs a conservative MR OOF GraTa-lite experiment on the baseline nnU-Net model:
  same validation cases with steps=0 baseline
  same validation cases with GraTa-lite adaptation
  class-aware-hd postprocess for both
  class-wise DSC/HD/ASSD comparison and center summary

Common environment overrides:
  FOLDS="0"                     validation folds to run
  GPU=2                         visible GPU id for this experiment
  MAX_CASES_PER_CENTER=1        smoke-test one C/D and one E case per fold
  MAX_CASES=                    optional total case cap per fold
  CHECKPOINT=checkpoint_final.pth
  STEPS=1
  LR=1e-5
  PARAM_SCOPE=norm-affine       norm-affine|decoder-last|norm-and-decoder-last
  ALIGNMENT_THRESHOLD=0.0
  RUN_NAME=mr_grata_lite_baseline_smoke
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DATASET_ID="${DATASET_ID:-402}"
DATASET_NAME="${DATASET_NAME:-Dataset402_CARE2026_WholeHeart_MR}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
FOLDS="${FOLDS:-0}"
GPU="${GPU:-2}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"
RUN_NAME="${RUN_NAME:-mr_grata_lite_baseline_smoke}"
OUT_ROOT="${OUT_ROOT:-${TRACK4_ROOT}/outputs/internal_val_${RUN_NAME}}"
OVERWRITE="${OVERWRITE:-1}"
POSTPROCESS_PRESET="${POSTPROCESS_PRESET:-class-aware-hd}"

STEPS="${STEPS:-1}"
LR="${LR:-1e-5}"
PARAM_SCOPE="${PARAM_SCOPE:-norm-affine}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-1.0}"
CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-1.0}"
ALIGNMENT_THRESHOLD="${ALIGNMENT_THRESHOLD:-0.0}"
NOISE_STD="${NOISE_STD:-0.01}"
BRIGHTNESS_STD="${BRIGHTNESS_STD:-0.02}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
TILE_STEP_SIZE="${TILE_STEP_SIZE:-0.5}"
MAX_CASES="${MAX_CASES:-}"
MAX_CASES_PER_CENTER="${MAX_CASES_PER_CENTER:-}"
SAVE_PROBABILITIES="${SAVE_PROBABILITIES:-0}"
DISABLE_MIRRORING="${DISABLE_MIRRORING:-0}"

MODEL_DIR="${MODEL_DIR:-${nnUNet_results}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIGURATION}}"
INPUT_DIR="${INPUT_DIR:-${nnUNet_raw}/${DATASET_NAME}/imagesTr}"
SPLITS_JSON="${SPLITS_JSON:-${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json}"
GT_DIR="${GT_DIR:-${nnUNet_preprocessed}/${DATASET_NAME}/gt_segmentations}"
METRICS_DIR="${METRICS_DIR:-${TRACK4_ROOT}/outputs/metrics}"
LOG_DIR="${LOG_DIR:-${TRACK4_ROOT}/outputs/logs}"

mkdir -p "${OUT_ROOT}" "${METRICS_DIR}" "${LOG_DIR}"

safe_clean_dir() {
  local target="$1"
  if [[ "${OVERWRITE}" != "1" ]]; then
    return 0
  fi
  case "${target}" in
    "${TRACK4_ROOT}/outputs/"*)
      rm -rf "${target}"
      ;;
    *)
      echo "ERROR: refusing to remove non-output directory: ${target}" >&2
      exit 1
      ;;
  esac
}

require_path() {
  local kind="$1"
  local path="$2"
  if [[ "${kind}" == "dir" && ! -d "${path}" ]]; then
    echo "ERROR: missing directory: ${path}" >&2
    exit 1
  fi
  if [[ "${kind}" == "file" && ! -f "${path}" ]]; then
    echo "ERROR: missing file: ${path}" >&2
    exit 1
  fi
}

read -r -a FOLD_ARRAY <<< "${FOLDS}"
EXTRA_CASE_ARGS=()
if [[ -n "${MAX_CASES}" ]]; then
  EXTRA_CASE_ARGS+=(--max-cases "${MAX_CASES}")
fi
if [[ -n "${MAX_CASES_PER_CENTER}" ]]; then
  EXTRA_CASE_ARGS+=(--max-cases-per-center "${MAX_CASES_PER_CENTER}")
fi
if [[ "${SAVE_PROBABILITIES}" == "1" ]]; then
  EXTRA_CASE_ARGS+=(--save-probabilities)
fi
if [[ "${DISABLE_MIRRORING}" == "1" ]]; then
  EXTRA_CASE_ARGS+=(--disable-mirroring)
fi

require_path dir "${MODEL_DIR}"
require_path dir "${INPUT_DIR}"
require_path file "${SPLITS_JSON}"
require_path dir "${GT_DIR}"

cat <<EOF
MR GraTa-lite OOF experiment
track4_root=${TRACK4_ROOT}
dataset=${DATASET_ID} ${DATASET_NAME}
model_dir=${MODEL_DIR}
input_dir=${INPUT_DIR}
splits_json=${SPLITS_JSON}
gt_dir=${GT_DIR}
folds=${FOLDS}
gpu=${GPU}
checkpoint=${CHECKPOINT}
out_root=${OUT_ROOT}
postprocess=${POSTPROCESS_PRESET}
tta=$([[ "${DISABLE_MIRRORING}" == "1" ]] && echo disabled || echo enabled)
baseline_steps=0
grata_steps=${STEPS}
grata_lr=${LR}
grata_param_scope=${PARAM_SCOPE}
grata_alignment_threshold=${ALIGNMENT_THRESHOLD}
max_cases=${MAX_CASES:-<none>}
max_cases_per_center=${MAX_CASES_PER_CENTER:-<none>}
EOF

run_predict() {
  local fold="$1"
  local output_dir="$2"
  local steps="$3"
  local label="$4"
  local log_file="${LOG_DIR}/${RUN_NAME}_${label}_fold${fold}.log"

  safe_clean_dir "${output_dir}"
  mkdir -p "${output_dir}"
  {
    echo "[start] $(date -Iseconds)"
    echo "label=${label}"
    echo "fold=${fold}"
    echo "steps=${steps}"
    CUDA_VISIBLE_DEVICES="${GPU}" python "${TRACK4_ROOT}/scripts/predict_grata_lite.py" \
      --model-dir "${MODEL_DIR}" \
      --input-dir "${INPUT_DIR}" \
      --output-dir "${output_dir}" \
      --folds "${fold}" \
      --checkpoint "${CHECKPOINT}" \
      --splits-json "${SPLITS_JSON}" \
      --use-split-val \
      --device cuda \
      --tile-step-size "${TILE_STEP_SIZE}" \
      --overwrite \
      --steps "${steps}" \
      --lr "${LR}" \
      --param-scope "${PARAM_SCOPE}" \
      --entropy-weight "${ENTROPY_WEIGHT}" \
      --consistency-weight "${CONSISTENCY_WEIGHT}" \
      --alignment-threshold "${ALIGNMENT_THRESHOLD}" \
      --noise-std "${NOISE_STD}" \
      --brightness-std "${BRIGHTNESS_STD}" \
      --max-grad-norm "${MAX_GRAD_NORM}" \
      "${EXTRA_CASE_ARGS[@]}"
    echo "[done] $(date -Iseconds)"
  } 2>&1 | tee "${log_file}"
}

postprocess_dir() {
  local input_dir="$1"
  local output_dir="$2"

  safe_clean_dir "${output_dir}"
  mkdir -p "${output_dir}"
  python "${TRACK4_ROOT}/scripts/postprocess_predictions.py" \
    --input-dir "${input_dir}" \
    --output-dir "${output_dir}" \
    --label-space train \
    --preset "${POSTPROCESS_PRESET}"
}

BASE_ROOT="${OUT_ROOT}/baseline_postprocessed"
GRATA_ROOT="${OUT_ROOT}/grata_postprocessed"

for fold in "${FOLD_ARRAY[@]}"; do
  BASE_RAW_DIR="${OUT_ROOT}/baseline_raw/fold_${fold}/validation"
  GRATA_RAW_DIR="${OUT_ROOT}/grata_raw/fold_${fold}/validation"
  BASE_POST_DIR="${BASE_ROOT}/fold_${fold}/validation"
  GRATA_POST_DIR="${GRATA_ROOT}/fold_${fold}/validation"

  echo "=== fold ${fold}: no-adapt baseline ==="
  run_predict "${fold}" "${BASE_RAW_DIR}" 0 "baseline"
  echo "=== fold ${fold}: GraTa-lite ==="
  run_predict "${fold}" "${GRATA_RAW_DIR}" "${STEPS}" "grata"

  echo "=== fold ${fold}: MR class-aware-hd postprocess ==="
  postprocess_dir "${BASE_RAW_DIR}" "${BASE_POST_DIR}"
  postprocess_dir "${GRATA_RAW_DIR}" "${GRATA_POST_DIR}"
done

CASE_CSV="${METRICS_DIR}/${RUN_NAME}_case_dsc_hd_assd.csv"
SUMMARY_CSV="${METRICS_DIR}/${RUN_NAME}_summary.csv"
SUMMARY_MD="${METRICS_DIR}/${RUN_NAME}_summary.md"
COMPARE_CSV="${METRICS_DIR}/${RUN_NAME}_vs_baseline.csv"
COMPARE_MD="${METRICS_DIR}/${RUN_NAME}_vs_baseline.md"
CENTER_CSV="${METRICS_DIR}/${RUN_NAME}_center_summary.csv"
CENTER_MD="${METRICS_DIR}/${RUN_NAME}_center_summary.md"

python "${TRACK4_ROOT}/scripts/evaluate_segmentation_metrics.py" \
  --run "baseline=${BASE_ROOT}" \
  --run "grata_lite=${GRATA_ROOT}" \
  --folds "${FOLD_ARRAY[@]}" \
  --gt-dir "${GT_DIR}" \
  --label-space train \
  --case-csv "${CASE_CSV}" \
  --summary-csv "${SUMMARY_CSV}" \
  --summary-md "${SUMMARY_MD}"

python "${TRACK4_ROOT}/scripts/compare_metric_summaries.py" \
  --summary-csv "${SUMMARY_CSV}" \
  --baseline-run baseline \
  --output-csv "${COMPARE_CSV}" \
  --output-md "${COMPARE_MD}" \
  --include-folds

python "${TRACK4_ROOT}/scripts/summarize_metrics_by_center.py" \
  --case-csv "${CASE_CSV}" \
  --output-csv "${CENTER_CSV}" \
  --output-md "${CENTER_MD}"

echo "Done."
echo "Case metrics: ${CASE_CSV}"
echo "Summary: ${SUMMARY_MD}"
echo "Baseline comparison: ${COMPARE_MD}"
echo "Center summary: ${CENTER_MD}"
