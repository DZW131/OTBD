# Track4 Whole Heart Runbook

This folder adapts the previous nnU-Net competition workflow to MICCAI CARE 2026
Track4 / CARE-Whole Heart.

The baseline trains CT and MR as two separate nnU-Net v2 datasets:

```text
CT -> Dataset401_CARE2026_WholeHeart_CT
MR -> Dataset402_CARE2026_WholeHeart_MR
```

The old liver checkpoints and plans are not reused.

## Folder Contents

```text
track4_wholeheart/
  configs/wholeheart_labels.json
  scripts/common.py
  scripts/convert_to_nnunet.py
  scripts/audit_dataset.py
  scripts/restore_label_values.py
  scripts/postprocess_predictions.py
  scripts/summarize_fold_class_metrics.py
  scripts/install_wholeheart_trainer.py
  scripts/env.sh
  scripts/plan_preprocess.sh
  scripts/train_nnunet.sh
  scripts/predict_val.sh
  scripts/refine_mr_with_medsam2.py
  nnunet_extensions/nnUNetTrainerWholeHeartAug.py
  setup_conda_env.sh
  DATASET/                  # generated, ignored by git
  outputs/                  # generated predictions, ignored by git
```

## Server Setup

Clone the branch:

```bash
git clone -b track4_wholeheart https://github.com/DZW131/OTBD.git
cd OTBD
```

Create or activate the conda environment:

```bash
bash track4_wholeheart/setup_conda_env.sh
conda activate track4_wholeheart
```

If the environment already exists:

```bash
conda activate track4_wholeheart
```

Check PyTorch, CUDA, and nnU-Net:

```bash
python - <<'PY'
import torch, nnunetv2
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("nnunetv2:", nnunetv2.__file__)
PY
```

Known good server check:

```text
torch: 2.5.1+cu121
cuda available: True
cuda version: 12.1
gpu count: 1
gpu: NVIDIA GeForce RTX 4090
```

## Data Layout

Recommended layout on the server:

```text
/root/data/
  Wholeheart_Train_Dataset/
    A ct_train/
      Case1001_image.nii.gz
      Case1001_label.nii.gz
    B ct_train/
    G ct_train/
    C and D mr_train/
    E mr_train/
  Wholeheart_Val_Dataset/
    ct_val/
      CaseXXXX_image.nii.gz
    mr_val/
      CaseYYYY_image.nii.gz
```

Expected local/released training split:

| Split | Images | Labels |
| --- | ---: | ---: |
| A ct_train | 20 | 20 |
| B ct_train | 20 | 20 |
| G ct_train | 20 | 20 |
| C and D mr_train | 20 | 20 |
| E mr_train | 26 | 26 |
| CT train total | 60 | 60 |
| MR train total | 46 | 46 |
| all train total | 106 | 106 |

Expected validation split:

| Split | Images |
| --- | ---: |
| ct_val | 30 |
| mr_val | 20 |
| all val total | 50 |

Count files after upload:

```bash
find /root/data/Wholeheart_Train_Dataset -name "*_image.nii.gz" | wc -l
find /root/data/Wholeheart_Train_Dataset -name "*_label.nii.gz" | wc -l
find /root/data/Wholeheart_Val_Dataset -name "*_image.nii.gz" | wc -l
```

Expected:

```text
106
106
50
```

If training data is still uploading, only check validation:

```bash
find /root/data/Wholeheart_Val_Dataset -name "*_image.nii.gz" | wc -l
find /root/data/Wholeheart_Val_Dataset/ct_val -name "*_image.nii.gz" | wc -l
find /root/data/Wholeheart_Val_Dataset/mr_val -name "*_image.nii.gz" | wc -l
```

Expected:

```text
50
30
20
```

Do not run the real conversion until both train and val counts are complete.

## What Dry-Run Does

Dry-run is a safe preflight check. It:

- discovers CT/MR train and validation folders
- counts train and validation images
- checks whether each training image has a matching label
- avoids copying files
- avoids writing the final nnU-Net dataset
- does not require SimpleITK

Run:

```bash
python track4_wholeheart/scripts/convert_to_nnunet.py \
  --train-root /root/data/Wholeheart_Train_Dataset \
  --val-root /root/data/Wholeheart_Val_Dataset \
  --task both \
  --dry-run
```

Expected:

```text
[CT] dry run
[CT] training images: 60
[CT] validation images: 30
[CT] missing labels: 0
[MR] dry run
[MR] training images: 46
[MR] validation images: 20
[MR] missing labels: 0
```

## Convert Data

Run the real conversion after dry-run is correct:

```bash
python track4_wholeheart/scripts/convert_to_nnunet.py \
  --train-root /root/data/Wholeheart_Train_Dataset \
  --val-root /root/data/Wholeheart_Val_Dataset \
  --task both
```

The converter:

- discovers folders such as `A ct_train`, `B ct_train`, `G ct_train`,
  `C and D mr_train`, `E mr_train`, `ct_val`, and `mr_val`
- renames images from `Case1001_image.nii.gz` to `Case1001_0000.nii.gz`
- maps official training label values to contiguous nnU-Net labels
- writes `dataset.json`
- writes `conversion_mapping.json` for restoring validation output names

Generated layout:

```text
track4_wholeheart/DATASET/
  nnUNet_raw/
    Dataset401_CARE2026_WholeHeart_CT/
      imagesTr/
      labelsTr/
      imagesTs/
      dataset.json
      conversion_mapping.json
    Dataset402_CARE2026_WholeHeart_MR/
      imagesTr/
      labelsTr/
      imagesTs/
      dataset.json
      conversion_mapping.json
  nnUNet_preprocessed/
  nnUNet_result/
```

If official data ever uses separate image and label directories, pass explicit
directories, for example:

```bash
python track4_wholeheart/scripts/convert_to_nnunet.py \
  --ct-train-images /path/to/ct/images \
  --ct-train-labels /path/to/ct/labels \
  --ct-val-images /path/to/ct/val \
  --task ct
```

## Plan and Preprocess

```bash
bash track4_wholeheart/scripts/plan_preprocess.sh ct
bash track4_wholeheart/scripts/plan_preprocess.sh mr
```

These scripts set:

```text
nnUNet_raw=track4_wholeheart/DATASET/nnUNet_raw
nnUNet_preprocessed=track4_wholeheart/DATASET/nnUNet_preprocessed
nnUNet_results=track4_wholeheart/DATASET/nnUNet_result
```

Then they run:

```bash
nnUNetv2_plan_and_preprocess -d 401 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 402 --verify_dataset_integrity
```

## Train

Train one fold first:

```bash
bash track4_wholeheart/scripts/train_nnunet.sh ct 0
bash track4_wholeheart/scripts/train_nnunet.sh mr 0
```

Full five-fold cross-validation trains five independent models per modality:

```text
fold 0: 80% train, 20% internal validation
fold 1: 80% train, 20% internal validation
fold 2: 80% train, 20% internal validation
fold 3: 80% train, 20% internal validation
fold 4: 80% train, 20% internal validation
```

Run CT five-fold training:

```bash
mkdir -p track4_wholeheart/outputs/logs
set -o pipefail

for f in 0 1 2 3 4; do
  bash track4_wholeheart/scripts/train_nnunet.sh ct "$f" 2>&1 | tee -a "track4_wholeheart/outputs/logs/ct_fold${f}.log"
done
```

Run MR five-fold training:

```bash
mkdir -p track4_wholeheart/outputs/logs
set -o pipefail

for f in 0 1 2 3 4; do
  bash track4_wholeheart/scripts/train_nnunet.sh mr "$f" 2>&1 | tee -a "track4_wholeheart/outputs/logs/mr_fold${f}.log"
done
```

Default configuration:

```text
CONFIGURATION=3d_fullres
```

Override if needed:

```bash
CONFIGURATION=2d bash track4_wholeheart/scripts/train_nnunet.sh ct 0
```

Use the modality-aware intensity augmentation trainer for the first innovation
run:

```bash
TRAINER=nnUNetTrainerWholeHeartAug bash track4_wholeheart/scripts/train_nnunet.sh ct 0
TRAINER=nnUNetTrainerWholeHeartAug bash track4_wholeheart/scripts/train_nnunet.sh mr 0
```

This copies only `nnUNetTrainerWholeHeartAug.py` into the active nnU-Net
environment and launches `nnUNetv2_train` with `-tr nnUNetTrainerWholeHeartAug`.
The default command above still uses the original
`nnUNetTrainer` baseline.

Random histogram matching and mean-teacher variants are exposed as separate
trainer names so ablations can be run from the command line:

```bash
# Strong CT/MR augmentation + random histogram matching
TRAINER=nnUNetTrainerWholeHeartRHM bash track4_wholeheart/scripts/train_nnunet.sh ct 0
TRAINER=nnUNetTrainerWholeHeartRHM bash track4_wholeheart/scripts/train_nnunet.sh mr 0

# Random histogram matching + mean-teacher consistency
TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher bash track4_wholeheart/scripts/train_nnunet.sh ct 0
TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher bash track4_wholeheart/scripts/train_nnunet.sh mr 0
```

The mean-teacher trainer uses labeled `imagesTr/labelsTr` for supervised
Dice/CE loss, and by default samples unlabeled patches from the converted
official validation images in `imagesTs`. Those validation images have no
labels and are used only for consistency loss. To avoid repeatedly reading
raw `.nii.gz` volumes after mean teacher starts, it caches sampled unlabeled
patches on CPU; the default is `WHOLEHEART_MT_PATCH_CACHE=128`. Use the
`imagesTs` path only if the challenge rules allow unlabeled validation images
during training; otherwise set `WHOLEHEART_MT_UNLABELED_MODE=labeled_batch`.
The original official files are copied there by `convert_to_nnunet.py` from
paths such as:

```text
/home/data/jingkun/duyanhong/dataspace/track4_wholeheart/Wholeheart_Val_Dataset/ct_val/CaseCTVal001_image.nii.gz
/home/data/jingkun/duyanhong/dataspace/track4_wholeheart/Wholeheart_Val_Dataset/mr_val/CaseMRVal001_image.nii.gz
```

Useful ablation switches:

```bash
# Shorten a pilot run to 600 epochs instead of the nnU-Net default
WHOLEHEART_NUM_EPOCHS=600 TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher \
  bash track4_wholeheart/scripts/train_nnunet.sh mr 0

# Disable random histogram matching even when using the RHM trainer
WHOLEHEART_RHM_PROB=0 TRAINER=nnUNetTrainerWholeHeartRHM \
  bash track4_wholeheart/scripts/train_nnunet.sh mr 0

# Keep mean teacher but use only labeled training batches as unlabeled views
WHOLEHEART_MT_UNLABELED_MODE=labeled_batch \
  TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher \
  bash track4_wholeheart/scripts/train_nnunet.sh mr 0

# Keep imagesTs unlabeled data but reduce CPU patch cache memory
WHOLEHEART_MT_PATCH_CACHE=64 \
  TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher \
  bash track4_wholeheart/scripts/train_nnunet.sh ct 0

# Disable mean teacher inside the combined trainer
WHOLEHEART_MT=0 TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher \
  bash track4_wholeheart/scripts/train_nnunet.sh mr 0

# Tune random histogram matching probability
WHOLEHEART_RHM_PROB=0.3 TRAINER=nnUNetTrainerWholeHeartRHM \
  bash track4_wholeheart/scripts/train_nnunet.sh ct 0

# Enable CT-aware random window/level augmentation.
# HU windows are mapped through nnU-Net CT foreground mean/std before applying
# clip-rescale in normalized training space.
WHOLEHEART_CT_WINDOW_AUG=1 \
WHOLEHEART_CT_WINDOW_PROB=0.25 \
WHOLEHEART_CT_WINDOW_LOWER_RANGE=-50,100 \
WHOLEHEART_CT_WINDOW_UPPER_RANGE=600,1200 \
WHOLEHEART_CT_WINDOW_BLEND=0.7 \
  TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher \
  bash track4_wholeheart/scripts/train_nnunet.sh ct 0

# Tune mean-teacher schedule
WHOLEHEART_MT_START_EPOCH=40 WHOLEHEART_MT_RAMPUP_EPOCHS=80 WHOLEHEART_MT_MAX_WEIGHT=1.0 \
  TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher \
  bash track4_wholeheart/scripts/train_nnunet.sh mr 0
```

Notes:

- On a single RTX 4090, run one training job at a time.
- `WHOLEHEART_NUM_EPOCHS` changes the trainer's total epoch count and therefore
  the nnU-Net learning-rate schedule. Use a separate `nnUNet_results` directory
  for shortened runs so 600-epoch experiments do not resume from or overwrite
  older 1000-epoch runs.
- CT window augmentation is disabled by default and only runs when
  `WHOLEHEART_CT_WINDOW_AUG=1` and the current dataset modality is CT. It is
  intended for CT domain generalization experiments, not for MR.
- If training is interrupted, rerun the same command; nnU-Net resumes from
  `checkpoint_latest.pth` when available.
- Fold `all` is not the same as five-fold cross-validation. Use folds `0 1 2 3 4`
  for cross-validation and ensemble prediction.

Check completed CT fold checkpoints:

```bash
find track4_wholeheart/DATASET/nnUNet_result \
  -path "*Dataset401_CARE2026_WholeHeart_CT*" \
  -path "*fold_*" \
  -name "checkpoint_final.pth" | sort
```

Check completed MR fold checkpoints:

```bash
find track4_wholeheart/DATASET/nnUNet_result \
  -path "*Dataset402_CARE2026_WholeHeart_MR*" \
  -path "*fold_*" \
  -name "checkpoint_final.pth" | sort
```

## Predict Validation Set

After training:

```bash
FOLDS="0" bash track4_wholeheart/scripts/predict_val.sh ct
FOLDS="0" bash track4_wholeheart/scripts/predict_val.sh mr
```

For whole-heart trainer variants, pass the same trainer name used during
training:

```bash
TRAINER=nnUNetTrainerWholeHeartRHM FOLDS="0" bash track4_wholeheart/scripts/predict_val.sh mr
TRAINER=nnUNetTrainerWholeHeartRHMMeanTeacher FOLDS="0" bash track4_wholeheart/scripts/predict_val.sh mr
```

Use all five folds for ensemble prediction after full cross-validation:

```bash
FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct
FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh mr
```

Enable anatomical connected-component post-processing before restoring official
label values. The default `legacy` preset keeps the historical rule that each
class keeps only its largest connected component:

```bash
POSTPROCESS=1 FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct
POSTPROCESS=1 FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh mr
```

For submission-oriented HD reduction, use the class-aware preset. It keeps one
main component for LV/RV/LA/RA/Myo, fills small chamber holes without overwriting
other labels, and preserves multiple near-heart AO/PA vessel components while
removing remote islands:

```bash
POSTPROCESS=1 POSTPROCESS_PRESET=class-aware-hd \
  FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct

POSTPROCESS=1 POSTPROCESS_PRESET=class-aware-hd \
  FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh mr
```

`predict_val.sh` prints the active modality, trainer, folds, TTA status,
probability saving, post-processing preset, expected preset, and sanity-check
setting before inference. The current submission-oriented defaults are:

```text
CT: POSTPROCESS=1 POSTPROCESS_PRESET=legacy
MR: POSTPROCESS=1 POSTPROCESS_PRESET=class-aware-hd
```

The script warns, without aborting, when inference is not using `FOLDS="0 1 2 3 4"`,
when post-processing is disabled, or when a modality uses a preset different
from the current best-known CT/MR setting. These warnings are intended to catch
submission mix-ups while still allowing raw ablations.

CT AO/PA softmax patch experiments can be enabled after the CT legacy
post-processing step. This is intentionally CT-only and the script aborts if
`CT_AOPA_SOFT=1` is used without `POSTPROCESS=1 POSTPROCESS_PRESET=legacy`.
The raw nnU-Net `.npz` probabilities are read from the prediction directory;
their spatial shape must match the legacy prediction shape exactly.

```bash
# CT-0: current rollback baseline
SANITY_CHECK=1 POSTPROCESS=1 POSTPROCESS_PRESET=legacy \
  FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct

# CT-1: fixed-threshold AO/PA hysteresis, no closing
SANITY_CHECK=1 POSTPROCESS=1 POSTPROCESS_PRESET=legacy \
  CT_AOPA_SOFT=1 CT_AOPA_THRESHOLD_MODE=fixed CT_AOPA_P_CLASS_THRESHOLD=0.25 \
  CT_AOPA_COMPONENT_SCORE=0 CT_AOPA_ENABLE_CLOSING=0 \
  FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct

# CT-2: seed-adaptive AO/PA hysteresis, no closing
SANITY_CHECK=1 POSTPROCESS=1 POSTPROCESS_PRESET=legacy \
  CT_AOPA_SOFT=1 CT_AOPA_THRESHOLD_MODE=adaptive \
  CT_AOPA_COMPONENT_SCORE=0 CT_AOPA_ENABLE_CLOSING=0 \
  FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct

# CT-3: CT-2 plus component-score cleanup
SANITY_CHECK=1 POSTPROCESS=1 POSTPROCESS_PRESET=legacy \
  CT_AOPA_SOFT=1 CT_AOPA_THRESHOLD_MODE=adaptive CT_AOPA_COMPONENT_SCORE=1 \
  CT_AOPA_MIN_MEAN_PROB=0.35 CT_AOPA_MIN_P10_PROB=0.15 \
  CT_AOPA_ENABLE_CLOSING=0 \
  FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct

# CT-4: CT-3 plus light AO/PA radius=1 closing
SANITY_CHECK=1 POSTPROCESS=1 POSTPROCESS_PRESET=legacy \
  CT_AOPA_SOFT=1 CT_AOPA_THRESHOLD_MODE=adaptive CT_AOPA_COMPONENT_SCORE=1 \
  CT_AOPA_ENABLE_CLOSING=1 CT_AOPA_CLOSING_RADIUS_VOXEL=1 \
  FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct
```

`CT_AOPA_THRESHOLD_MODE=adaptive` computes each case/class threshold from the
legacy seed probability as `max(0.18, min(0.30, 0.45 * seed_median_prob))`.
The patch only writes AO/PA into background or same-class voxels by default,
blocks high-confidence competing classes, and restricts growth to the seed
bbox/distance neighborhood. The refined train-label output is written to
`track4_wholeheart/outputs/ct_val_nnunet_postprocessed_aopa_soft/`, then restored
to `track4_wholeheart/outputs/ct_val_official_labels_postprocessed_aopa_soft/`.

Raw nnU-Net predictions:

```text
track4_wholeheart/outputs/ct_val_nnunet/
track4_wholeheart/outputs/mr_val_nnunet/
```

Restored official-label predictions:

```text
track4_wholeheart/outputs/ct_val_official_labels/
track4_wholeheart/outputs/mr_val_official_labels/
```

## Anatomical Post-Processing

Apply connected-component cleanup manually before restoring official labels.
Use `--preset legacy` for the original largest-component cleanup, or
`--preset class-aware-hd` for the HD-oriented class-aware cleanup:

```bash
python track4_wholeheart/scripts/postprocess_predictions.py \
  --input-dir track4_wholeheart/outputs/ct_val_nnunet \
  --output-dir track4_wholeheart/outputs/ct_val_nnunet_postprocessed \
  --label-space train \
  --preset class-aware-hd \
  --min-component-size 0

python track4_wholeheart/scripts/postprocess_predictions.py \
  --input-dir track4_wholeheart/outputs/mr_val_nnunet \
  --output-dir track4_wholeheart/outputs/mr_val_nnunet_postprocessed \
  --label-space train \
  --preset class-aware-hd \
  --min-component-size 0
```

The class-aware preset can be tuned without code changes:

```bash
python track4_wholeheart/scripts/postprocess_predictions.py \
  --input-dir track4_wholeheart/outputs/mr_val_nnunet \
  --output-dir track4_wholeheart/outputs/mr_val_nnunet_postprocessed \
  --label-space train \
  --preset class-aware-hd \
  --wholeheart-distance-mm 25 \
  --vessel-min-component-size 20
```

To run the CT AO/PA soft patch manually after CT legacy post-processing:

```bash
python track4_wholeheart/scripts/postprocess_ct_aopa_soft.py \
  --input-dir track4_wholeheart/outputs/ct_val_nnunet_postprocessed \
  --probability-dir track4_wholeheart/outputs/ct_val_nnunet \
  --output-dir track4_wholeheart/outputs/ct_val_nnunet_postprocessed_aopa_soft \
  --threshold-mode adaptive \
  --component-score
```

Then restore official label values from the postprocessed directory:

```bash
python track4_wholeheart/scripts/restore_label_values.py \
  --pred-dir track4_wholeheart/outputs/ct_val_nnunet_postprocessed \
  --mapping-json track4_wholeheart/DATASET/nnUNet_raw/Dataset401_CARE2026_WholeHeart_CT/conversion_mapping.json \
  --output-dir track4_wholeheart/outputs/ct_val_official_labels_postprocessed

python track4_wholeheart/scripts/restore_label_values.py \
  --pred-dir track4_wholeheart/outputs/mr_val_nnunet_postprocessed \
  --mapping-json track4_wholeheart/DATASET/nnUNet_raw/Dataset402_CARE2026_WholeHeart_MR/conversion_mapping.json \
  --output-dir track4_wholeheart/outputs/mr_val_official_labels_postprocessed
```

If predictions have already been restored to official labels, run
`postprocess_predictions.py` with `--label-space official`.

## Prediction Sanity Check

Enable the submission sanity check during validation prediction by setting
`SANITY_CHECK=1`. This runs after official label restoration and checks image
geometry, allowed official label values, empty predictions, missing classes,
unexpected files, and missing outputs:

```bash
SANITY_CHECK=1 POSTPROCESS=1 POSTPROCESS_PRESET=class-aware-hd \
  FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh mr
```

To check an existing restored official-label output directory manually:

```bash
python track4_wholeheart/scripts/check_prediction_sanity.py \
  --input-dir track4_wholeheart/DATASET/nnUNet_raw/Dataset402_CARE2026_WholeHeart_MR/imagesTs \
  --pred-dir track4_wholeheart/outputs/mr_val_official_labels_postprocessed \
  --mapping-json track4_wholeheart/DATASET/nnUNet_raw/Dataset402_CARE2026_WholeHeart_MR/conversion_mapping.json \
  --label-space official
```

## MR AO/Myo ROI Refinement Pipeline

The current MR ROI candidate is an automatic inference-time patch on top of the
stable MR baseline:

```text
MR baseline 5-fold nnU-Net
  -> class-aware-hd postprocess
  -> AO ROI add-only probability paste
  -> Myo ROI add-only probability paste
  -> official label restoration
  -> optional sanity check
```

It deliberately does not modify PA. Internal OOF validation showed PA ROI paste
was negative on fold 0, while AO and Myo had stable positive deltas.

Default one-click submission-style inference:

```bash
cd /home/data/jingkun/duyanhong/workspace/OTBD

MR_ROI_PRESET=score SANITY_CHECK=1 \
  bash track4_wholeheart/scripts/predict_mr_aomyo_roi.sh
```

The two built-in presets are:

| Preset | AO gate | Myo gate | OOF mean DSC delta vs MR baseline | Note |
| --- | ---: | ---: | ---: | --- |
| `score` | `p>=0.90`, `d<=3mm` | `p>=0.90`, `d<=2mm` | `+0.000681` | best DSC in 5-fold OOF |
| `safe` | `p>=0.90`, `d<=2mm` | `p>=0.90`, `d<=2mm` | `+0.000637` | slightly more conservative AO expansion |

Outputs are written under `track4_wholeheart/outputs/${RUN_NAME}`:

```text
base_raw/                 # raw MR baseline probabilities and labels
base_class_aware_hd/      # first-stage train-label masks used for ROI crops
ao_roi/                   # AO crops, metadata, and ROI probabilities
ao_pasted_train_labels/   # AO-refined train-label masks
myo_roi/                  # Myo crops, metadata, and ROI probabilities
train_labels/             # final AO+Myo refined train-label masks
official_labels/          # restored official challenge labels
```

Useful overrides:

```bash
# More conservative hidden-test option
MR_ROI_PRESET=safe SANITY_CHECK=1 \
  bash track4_wholeheart/scripts/predict_mr_aomyo_roi.sh

# Reuse an existing class-aware MR baseline instead of rerunning nnU-Net
SKIP_BASE_PREDICT=1 \
BASE_POSTPROCESSED_DIR=track4_wholeheart/outputs/mr_val_nnunet_postprocessed \
RUN_NAME=mr_aomyo_roi_reuse_baseline \
SANITY_CHECK=1 \
  bash track4_wholeheart/scripts/predict_mr_aomyo_roi.sh

# Override individual gates if a new validation sweep supports it
AO_DISTANCE_MM=2 MYO_DISTANCE_MM=2 AO_PROB_THRESHOLD=0.90 MYO_PROB_THRESHOLD=0.90 \
  bash track4_wholeheart/scripts/predict_mr_aomyo_roi.sh
```

The script uses training labels internally (`0..7`) and restores official label
values at the end. It keeps nnU-Net mirror TTA enabled by default and uses full
`FOLDS="0 1 2 3 4"` unless explicitly overridden.

## MR MedSAM2 Refinement

Use this only as a development-time refinement between raw MR nnU-Net prediction
and anatomical post-processing. The intended first comparison is therefore:

```text
raw nnU-Net validation predictions
raw nnU-Net validation predictions refined by MedSAM2 for Myo/AO/PA only
```

This keeps the known post-processing gain out of the ablation. The refinement
script prompts MedSAM2 from the existing nnU-Net masks for labels `5,6,7`
(`Myo`, `AO`, `PA`), then only accepts candidates that stay near the original
mask and pass conservative volume and IoU checks.

The default thresholds are intentionally conservative: target candidates must
keep volume within `0.90..1.10` of the original mask and have IoU at least
`0.85` with the original mask. Treat this as an ablation gate; only carry the
refined predictions into later post-processing if the raw validation metrics
improve.

Server layout used for the separate MedSAM2 environment:

```text
/home/data/jingkun/duyanhong/workspace/OTBD/
  third_party/MedSAM2/
  pretrained/MedSAM2/MedSAM2_latest.pt
  .conda_envs/medsam2refine/
```

Run a one-case smoke test first:

```bash
cd /home/data/jingkun/duyanhong/workspace/OTBD

DATASET_ROOT="$PWD/track4_wholeheart/DATASET"
BASE_ROOT="$DATASET_ROOT/nnUNet_result/Dataset402_CARE2026_WholeHeart_MR/nnUNetTrainer__nnUNetPlans__3d_fullres"
IMG_DIR="$DATASET_ROOT/nnUNet_raw/Dataset402_CARE2026_WholeHeart_MR/imagesTr"
SAM2_ROOT="$PWD/track4_wholeheart/outputs/internal_val/mr_nnunet_medsam2_raw"

CUDA_VISIBLE_DEVICES=0 .conda_envs/medsam2refine/bin/python \
  track4_wholeheart/scripts/refine_mr_with_medsam2.py \
  --pred-root "$BASE_ROOT" \
  --output-root "$SAM2_ROOT" \
  --folds 0 \
  --image-dir "$IMG_DIR" \
  --checkpoint pretrained/MedSAM2/MedSAM2_latest.pt \
  --medsam2-root third_party/MedSAM2 \
  --max-cases 1 \
  --decisions-csv track4_wholeheart/outputs/metrics/mr_medsam2_refine_decisions_smoke.csv
```

If the smoke test writes a NIfTI file and decisions CSV, run the full target
fold comparison without post-processing:

```bash
CUDA_VISIBLE_DEVICES=0 .conda_envs/medsam2refine/bin/python \
  track4_wholeheart/scripts/refine_mr_with_medsam2.py \
  --pred-root "$BASE_ROOT" \
  --output-root "$SAM2_ROOT" \
  --folds 0 \
  --image-dir "$IMG_DIR" \
  --checkpoint pretrained/MedSAM2/MedSAM2_latest.pt \
  --medsam2-root third_party/MedSAM2 \
  --decisions-csv track4_wholeheart/outputs/metrics/mr_medsam2_refine_decisions_fold0.csv
```

Then evaluate raw nnU-Net against raw nnU-Net + MedSAM2:

```bash
GT_DIR="$DATASET_ROOT/nnUNet_preprocessed/Dataset402_CARE2026_WholeHeart_MR/gt_segmentations"

/home/jingkun/miniconda3/envs/track4_wholeheart/bin/python \
  track4_wholeheart/scripts/evaluate_segmentation_metrics.py \
  --run nnunet="$BASE_ROOT" \
  --run nnunet_medsam2="$SAM2_ROOT" \
  --folds 0 \
  --gt-dir "$GT_DIR" \
  --case-csv track4_wholeheart/outputs/metrics/mr_nnunet_vs_medsam2_fold0_case.csv \
  --summary-csv track4_wholeheart/outputs/metrics/mr_nnunet_vs_medsam2_fold0_summary.csv \
  --summary-md track4_wholeheart/outputs/metrics/mr_nnunet_vs_medsam2_fold0_summary.md
```

Print only the three target classes:

```bash
/home/jingkun/miniconda3/envs/track4_wholeheart/bin/python - <<'PY'
import pandas as pd

summary = pd.read_csv("track4_wholeheart/outputs/metrics/mr_nnunet_vs_medsam2_fold0_summary.csv")
print(summary[summary["class"].isin(["Myo", "AO", "PA"])].to_string(index=False))
PY
```

## Per-Class Fold Metrics

Summarize the seven whole-heart classes per fold from nnU-Net validation
summaries, training logs, or prediction/reference folders:

```bash
python track4_wholeheart/scripts/summarize_fold_class_metrics.py \
  --modality ct \
  --results-root track4_wholeheart/DATASET/nnUNet_result/Dataset401_CARE2026_WholeHeart_CT/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --training-log-dir track4_wholeheart/outputs/logs \
  --output-csv track4_wholeheart/outputs/metrics/ct_fold_class_metrics.csv \
  --output-md track4_wholeheart/outputs/metrics/ct_fold_class_metrics.md

python track4_wholeheart/scripts/summarize_fold_class_metrics.py \
  --modality mr \
  --results-root track4_wholeheart/DATASET/nnUNet_result/Dataset402_CARE2026_WholeHeart_MR/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --training-log-dir track4_wholeheart/outputs/logs \
  --output-csv track4_wholeheart/outputs/metrics/mr_fold_class_metrics.csv \
  --output-md track4_wholeheart/outputs/metrics/mr_fold_class_metrics.md
```

For the augmentation trainer, replace `nnUNetTrainer__...` with
`nnUNetTrainerWholeHeartAug__...`. When validation references are available,
you can also pass `--pred-dir` and `--ref-dir` to compute Dice directly from
NIfTI files.

## DSC / HD / ASSD Fold Metrics

Compare competition-style segmentation metrics across runs on the same internal
validation folds. This computes per-case and per-class DSC, symmetric
Hausdorff distance, and ASSD from NIfTI predictions and labels:

```bash
python track4_wholeheart/scripts/evaluate_segmentation_metrics.py \
  --run baseline="$DATASET_ROOT/nnUNet_result/Dataset401_CARE2026_WholeHeart_CT/nnUNetTrainer__nnUNetPlans__3d_fullres" \
  --run mt="$DATASET_ROOT/nnUNet_result_ct_mt200_w02_ep600/Dataset401_CARE2026_WholeHeart_CT/nnUNetTrainerWholeHeartRHMMeanTeacher__nnUNetPlans__3d_fullres" \
  --run mt_ctwindow="$DATASET_ROOT/nnUNet_result_ct_mt200_w02_ep600_ctwindow/Dataset401_CARE2026_WholeHeart_CT/nnUNetTrainerWholeHeartRHMMeanTeacher__nnUNetPlans__3d_fullres" \
  --folds 0 1 \
  --gt-dir "$DATASET_ROOT/nnUNet_preprocessed/Dataset401_CARE2026_WholeHeart_CT/gt_segmentations" \
  --case-csv track4_wholeheart/outputs/metrics/ct_fold01_case_dsc_hd_assd.csv \
  --summary-csv track4_wholeheart/outputs/metrics/ct_fold01_summary_dsc_hd_assd.csv \
  --summary-md track4_wholeheart/outputs/metrics/ct_fold01_summary_dsc_hd_assd.md
```

The script expects training-label predictions (`0..7`) by default. Use
`--label-space official` only when predictions and references have already been
restored to official label values.

Always compare new CT/MR experiments against the current submission baseline
before deciding whether they are useful:

```text
CT baseline run name: ct-baseline-legacy
MR baseline run name: mr-baseline-class-aware-hd
```

After `evaluate_segmentation_metrics.py` writes a summary CSV containing the
baseline and candidate runs, generate a class-wise delta table:

```bash
python track4_wholeheart/scripts/compare_metric_summaries.py \
  --summary-csv track4_wholeheart/outputs/metrics/ct_experiment_summary_dsc_hd_assd.csv \
  --baseline-run ct-baseline-legacy \
  --output-csv track4_wholeheart/outputs/metrics/ct_experiment_vs_baseline.csv \
  --output-md track4_wholeheart/outputs/metrics/ct_experiment_vs_baseline.md \
  --include-folds

python track4_wholeheart/scripts/compare_metric_summaries.py \
  --summary-csv track4_wholeheart/outputs/metrics/mr_experiment_summary_dsc_hd_assd.csv \
  --baseline-run mr-baseline-class-aware-hd \
  --output-csv track4_wholeheart/outputs/metrics/mr_experiment_vs_baseline.csv \
  --output-md track4_wholeheart/outputs/metrics/mr_experiment_vs_baseline.md \
  --include-folds
```

The comparison output keeps one row per class (`LV/RV/LA/RA/Myo/AO/PA`) and
reports the experiment metric, baseline metric, and delta for DSC, HD, and ASSD.

## Label Mapping

Training uses contiguous nnU-Net labels:

| Structure | nnU-Net label | Official output value |
| --- | ---: | ---: |
| background | 0 | 0 |
| LV | 1 | 500 |
| RV | 2 | 600 |
| LA | 3 | 420 |
| RA | 4 | 550 |
| Myo | 5 | 205 |
| AO | 6 | 820 |
| PA | 7 | 850 |

Observed note:

```text
Case3010_label.nii.gz contains value 421.
421 is mapped to nnU-Net label 3, the same class as LA value 420.
```

The mapping lives in:

```text
track4_wholeheart/configs/wholeheart_labels.json
```

After prediction, `restore_label_values.py` converts labels `0..7` back to the
official values expected by the challenge.

## Quick Command Block

Once data upload is complete, this is the usual baseline sequence:

```bash
cd ~/OTBD
conda activate track4_wholeheart

find /root/data/Wholeheart_Train_Dataset -name "*_image.nii.gz" | wc -l
find /root/data/Wholeheart_Train_Dataset -name "*_label.nii.gz" | wc -l
find /root/data/Wholeheart_Val_Dataset -name "*_image.nii.gz" | wc -l

python track4_wholeheart/scripts/convert_to_nnunet.py \
  --train-root /root/data/Wholeheart_Train_Dataset \
  --val-root /root/data/Wholeheart_Val_Dataset \
  --task both \
  --dry-run

python track4_wholeheart/scripts/convert_to_nnunet.py \
  --train-root /root/data/Wholeheart_Train_Dataset \
  --val-root /root/data/Wholeheart_Val_Dataset \
  --task both

bash track4_wholeheart/scripts/plan_preprocess.sh ct
bash track4_wholeheart/scripts/plan_preprocess.sh mr

bash track4_wholeheart/scripts/train_nnunet.sh ct 0
bash track4_wholeheart/scripts/train_nnunet.sh mr 0

TRAINER=nnUNetTrainerWholeHeartAug bash track4_wholeheart/scripts/train_nnunet.sh ct 0
TRAINER=nnUNetTrainerWholeHeartAug bash track4_wholeheart/scripts/train_nnunet.sh mr 0
```

## Troubleshooting Notes

- If dry-run reports `0` train images, check whether the data is still uploading
  or whether the folder names contain `ct`/`mr` and `train`.
- If dry-run reports missing labels, check whether each `*_image.nii.gz` has a
  matching `*_label.nii.gz`.
- If preprocessing cannot find a dataset, rerun conversion and check
  `track4_wholeheart/DATASET/nnUNet_raw/`.
- If CUDA is not available, verify the active conda environment and PyTorch CUDA
  wheel before starting training.
