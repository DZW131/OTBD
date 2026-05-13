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
  scripts/env.sh
  scripts/plan_preprocess.sh
  scripts/train_nnunet.sh
  scripts/predict_val.sh
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

Train all five folds:

```bash
for f in 0 1 2 3 4; do bash track4_wholeheart/scripts/train_nnunet.sh ct "$f"; done
for f in 0 1 2 3 4; do bash track4_wholeheart/scripts/train_nnunet.sh mr "$f"; done
```

Default configuration:

```text
CONFIGURATION=3d_fullres
```

Override if needed:

```bash
CONFIGURATION=2d bash track4_wholeheart/scripts/train_nnunet.sh ct 0
```

## Predict Validation Set

After training:

```bash
bash track4_wholeheart/scripts/predict_val.sh ct
bash track4_wholeheart/scripts/predict_val.sh mr
```

Use specific folds if needed:

```bash
FOLDS="0" bash track4_wholeheart/scripts/predict_val.sh ct
FOLDS="0 1 2 3 4" bash track4_wholeheart/scripts/predict_val.sh mr
```

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
