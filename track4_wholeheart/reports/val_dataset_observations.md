# Validation Dataset Observations

Local validation path inspected: `D:\work\Wholeheart_Val_Dataset`.

## CT Validation

- Cases: 30
- Naming: `CaseCTVal###_image.nii.gz`
- Median shape: approximately `[512, 512, 225]`
- Median spacing: approximately `[0.488, 0.488, 0.625]`
- Sampled intensity median: approximately `-593.5`

## MR Validation

- Cases: 20
- Naming: `CaseMRVal###_image.nii.gz`
- Median shape: approximately `[272, 272, 140]`
- Median spacing: approximately `[1.204, 1.204, 1.370]`
- Sampled intensity median: approximately `72.5`

## Engineering Implication

CT and MR should start as separate nnU-Net datasets. The spacing and intensity distributions differ enough that a shared first baseline would add avoidable uncertainty.

## Training Dataset Notes

Local training path inspected: `D:\work\Wholeheart_Train_Dataset`.

- CT folders: `A ct_train`, `B ct_train`, `G ct_train`
- MR folders: `C and D mr_train`, `E mr_train`
- CT training cases discovered: 60
- MR training cases discovered: 46
- Missing labels in dry run: 0
- Label values observed across training set: `0, 205, 420, 421, 500, 550, 600, 820, 850`
- The only non-standard value found was `421` in `C and D mr_train\Case3010_label.nii.gz`; it is mapped to LA together with official value `420`.
