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
