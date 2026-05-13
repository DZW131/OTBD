# OTBD - CARE 2026 Track4 Whole Heart Baseline

This branch contains the MICCAI CARE 2026 Track4 / CARE-Whole Heart baseline under
`track4_wholeheart/`.

The current goal is to run a clean nnU-Net v2 baseline for CT and MR whole-heart
segmentation. Extra modules such as pseudo-label filtering, TTA drift gates, and
SAM/MedSAM refinement are intentionally left for later.

For the detailed runbook, see [`track4_wholeheart/README.md`](track4_wholeheart/README.md).

## 1. Clone on the Server

```bash
git clone -b track4_wholeheart https://github.com/DZW131/OTBD.git
cd OTBD
```

Expected server project path:

```text
~/OTBD
```

## 2. Activate Environment

If the environment already exists:

```bash
conda activate track4_wholeheart
```

If it still needs to be created:

```bash
bash track4_wholeheart/setup_conda_env.sh
conda activate track4_wholeheart
```

Quick GPU check:

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

## 3. Put Data in Place

Recommended server layout:

```text
/root/data/
  Wholeheart_Train_Dataset/
    A ct_train/
    B ct_train/
    G ct_train/
    C and D mr_train/
    E mr_train/
  Wholeheart_Val_Dataset/
    ct_val/
    mr_val/
```

Expected counts after upload:

```text
train images: 106
train labels: 106
val images:    50
```

Check them with:

```bash
find /root/data/Wholeheart_Train_Dataset -name "*_image.nii.gz" | wc -l
find /root/data/Wholeheart_Train_Dataset -name "*_label.nii.gz" | wc -l
find /root/data/Wholeheart_Val_Dataset -name "*_image.nii.gz" | wc -l
```

If only validation data has finished uploading, check it first and wait before
running conversion:

```bash
find /root/data/Wholeheart_Val_Dataset/ct_val -name "*_image.nii.gz" | wc -l
find /root/data/Wholeheart_Val_Dataset/mr_val -name "*_image.nii.gz" | wc -l
```

Expected:

```text
30
20
```

## 4. Dry-Run Conversion

Dry-run checks discovery, counts, and train image-label pairing. It does not copy
files or create the final nnU-Net dataset.

```bash
python track4_wholeheart/scripts/convert_to_nnunet.py \
  --train-root /root/data/Wholeheart_Train_Dataset \
  --val-root /root/data/Wholeheart_Val_Dataset \
  --task both \
  --dry-run
```

Expected:

```text
[CT] training images: 60
[CT] validation images: 30
[CT] missing labels: 0
[MR] training images: 46
[MR] validation images: 20
[MR] missing labels: 0
```

## 5. Convert to nnU-Net Format

Run this only after the dry-run counts are correct:

```bash
python track4_wholeheart/scripts/convert_to_nnunet.py \
  --train-root /root/data/Wholeheart_Train_Dataset \
  --val-root /root/data/Wholeheart_Val_Dataset \
  --task both
```

Generated datasets:

```text
track4_wholeheart/DATASET/nnUNet_raw/
  Dataset401_CARE2026_WholeHeart_CT/
  Dataset402_CARE2026_WholeHeart_MR/
```

## 6. Plan and Preprocess

```bash
bash track4_wholeheart/scripts/plan_preprocess.sh ct
bash track4_wholeheart/scripts/plan_preprocess.sh mr
```

## 7. Train

Single fold first:

```bash
bash track4_wholeheart/scripts/train_nnunet.sh ct 0
bash track4_wholeheart/scripts/train_nnunet.sh mr 0
```

Full five-fold cross-validation:

```bash
mkdir -p track4_wholeheart/outputs/logs
set -o pipefail

for f in 0 1 2 3 4; do
  bash track4_wholeheart/scripts/train_nnunet.sh ct "$f" 2>&1 | tee -a "track4_wholeheart/outputs/logs/ct_fold${f}.log"
done

for f in 0 1 2 3 4; do
  bash track4_wholeheart/scripts/train_nnunet.sh mr "$f" 2>&1 | tee -a "track4_wholeheart/outputs/logs/mr_fold${f}.log"
done
```

On a single RTX 4090, run folds sequentially instead of launching CT and MR
training at the same time. If training is interrupted, rerun the same command;
nnU-Net resumes from `checkpoint_latest.pth` when available.

Check completed fold checkpoints:

```bash
find track4_wholeheart/DATASET/nnUNet_result \
  -path "*Dataset401_CARE2026_WholeHeart_CT*" \
  -path "*fold_*" \
  -name "checkpoint_final.pth" | sort

find track4_wholeheart/DATASET/nnUNet_result \
  -path "*Dataset402_CARE2026_WholeHeart_MR*" \
  -path "*fold_*" \
  -name "checkpoint_final.pth" | sort
```

## 8. Predict Validation Set

Single-fold prediction:

```bash
FOLDS="0" bash track4_wholeheart/scripts/predict_val.sh ct
FOLDS="0" bash track4_wholeheart/scripts/predict_val.sh mr
```

Five-fold ensemble prediction after all folds are trained:

```bash
FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh ct
FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh mr
```

Official-label outputs are written to:

```text
track4_wholeheart/outputs/ct_val_official_labels/
track4_wholeheart/outputs/mr_val_official_labels/
```

## Label Mapping

Training uses contiguous nnU-Net labels and prediction output is restored to the
official values:

| Structure | nnU-Net train label | Official value |
| --- | ---: | ---: |
| background | 0 | 0 |
| LV | 1 | 500 |
| RV | 2 | 600 |
| LA | 3 | 420 |
| RA | 4 | 550 |
| Myo | 5 | 205 |
| AO | 6 | 820 |
| PA | 7 | 850 |

Observed note: `Case3010_label.nii.gz` contains value `421`; this branch maps
`421` to LA, the same training class as official LA value `420`.

## Old 2025 Liver Package

The old liver inference package remains in the repository for reference, but it
is not used by the Track4 baseline.
