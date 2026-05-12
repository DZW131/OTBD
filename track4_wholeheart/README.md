# Track4 Whole Heart Baseline

This branch adapts the 2025 nnU-Net competition framework to MICCAI CARE 2026 Track4 / CARE-Whole Heart.

The first goal is a clean baseline that can train as soon as the official training set is available. This branch intentionally does not yet include probe filtering, SAM/MedSAM refinement, or other experimental modules.

## Task

CARE-Whole Heart is a CT/MR whole-heart segmentation task with seven foreground structures.

Training uses contiguous nnU-Net labels:

| Class | nnU-Net label | Official output value |
| --- | ---: | ---: |
| background | 0 | 0 |
| LV | 1 | 500 |
| RV | 2 | 600 |
| LA | 3 | 420 |
| RA | 4 | 550 |
| Myo | 5 | 205 |
| AO | 6 | 820 |
| PA | 7 | 850 |

Predictions are restored back to the official label values after inference.

## Layout

```text
track4_wholeheart/
  configs/wholeheart_labels.json
  scripts/audit_dataset.py
  scripts/convert_to_nnunet.py
  scripts/plan_preprocess.sh
  scripts/train_nnunet.sh
  scripts/predict_val.sh
  scripts/restore_label_values.py
  setup_conda_env.sh
  DATASET/                  # generated, ignored by git
```

## Conda Setup

```bash
cd /path/to/OTBD
bash track4_wholeheart/setup_conda_env.sh
conda activate track4_wholeheart
```

If the server driver does not support the default CUDA wheel index, set `TORCH_INDEX_URL`, for example:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash track4_wholeheart/setup_conda_env.sh
```

## Convert Data

When the official training set is available, place or point it like this if possible:

```text
Wholeheart_Train_Dataset/
  ct_train/
    CaseCT001_image.nii.gz
    CaseCT001_label.nii.gz
  mr_train/
    CaseMR001_image.nii.gz
    CaseMR001_label.nii.gz

Wholeheart_Val_Dataset/
  ct_val/
    CaseCTVal001_image.nii.gz
  mr_val/
    CaseMRVal001_image.nii.gz
```

Then run:

```bash
python track4_wholeheart/scripts/convert_to_nnunet.py \
  --train-root /path/to/Wholeheart_Train_Dataset \
  --val-root /path/to/Wholeheart_Val_Dataset \
  --task both
```

If the official training labels are in separate folders, use explicit arguments such as `--ct-train-images`, `--ct-train-labels`, `--mr-train-images`, and `--mr-train-labels`.

## Plan and Preprocess

```bash
bash track4_wholeheart/scripts/plan_preprocess.sh ct
bash track4_wholeheart/scripts/plan_preprocess.sh mr
```

## Train

Train one fold:

```bash
bash track4_wholeheart/scripts/train_nnunet.sh ct 0
bash track4_wholeheart/scripts/train_nnunet.sh mr 0
```

Train all five folds:

```bash
for f in 0 1 2 3 4; do bash track4_wholeheart/scripts/train_nnunet.sh ct "$f"; done
for f in 0 1 2 3 4; do bash track4_wholeheart/scripts/train_nnunet.sh mr "$f"; done
```

## Predict Validation Set

After training:

```bash
bash track4_wholeheart/scripts/predict_val.sh ct
bash track4_wholeheart/scripts/predict_val.sh mr
```

Restored official-label predictions are written to:

```text
track4_wholeheart/outputs/ct_val_official_labels/
track4_wholeheart/outputs/mr_val_official_labels/
```

## Notes

- CT and MR are separated into two nnU-Net datasets first. This keeps the baseline simple and avoids mixing very different intensity distributions before we have training data.
- The old liver checkpoints and plans are not reused for Track4.
- The old Docker-only path assumptions are not used here. All paths are rooted under `track4_wholeheart/` unless overridden by environment variables.
