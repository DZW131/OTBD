from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from common import (
    case_id_from_image,
    dataset_name,
    default_dataset_root,
    discover_split_dirs,
    find_images,
    find_label_for_image,
    load_label_config,
    make_dataset_json,
    mapping_arrays,
    nnunet_raw_root,
    output_label_name_from_image,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CARE Whole Heart data to nnU-Net format.")
    parser.add_argument("--train-root", type=Path, default=None,
                        help="Root containing ct_train and/or mr_train.")
    parser.add_argument("--val-root", type=Path, default=None,
                        help="Root containing ct_val and/or mr_val.")
    parser.add_argument("--task", choices=["ct", "mr", "both"], default="both")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                        help="Output root that will contain nnUNet_raw, nnUNet_preprocessed and nnUNet_result.")
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report discovered cases and missing labels. Does not require SimpleITK.")

    parser.add_argument("--ct-train-images", type=Path, nargs="*", default=None)
    parser.add_argument("--ct-train-labels", type=Path, nargs="*", default=None)
    parser.add_argument("--ct-val-images", type=Path, nargs="*", default=None)
    parser.add_argument("--mr-train-images", type=Path, nargs="*", default=None)
    parser.add_argument("--mr-train-labels", type=Path, nargs="*", default=None)
    parser.add_argument("--mr-val-images", type=Path, nargs="*", default=None)
    return parser.parse_args()


def remap_label_to_train_values(label_path: Path, output_path: Path, official_to_train: dict[int, int]) -> None:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(image)
    unique = set(int(v) for v in np.unique(arr))

    contiguous_values = set(range(8))
    official_values = set(official_to_train.keys())

    if unique.issubset(contiguous_values):
        out = arr.astype(np.uint8, copy=False)
    elif unique.issubset(official_values):
        out = np.zeros_like(arr, dtype=np.uint8)
        for source_value, train_value in official_to_train.items():
            out[arr == source_value] = train_value
    else:
        raise ValueError(
            f"Unexpected label values in {label_path}: {sorted(unique)}. "
            f"Expected contiguous 0-7 or official values {sorted(official_values)}."
        )

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(output_path))


def copy_training_cases(
    modality: str,
    train_images_dir: list[Path],
    train_labels_dir: list[Path] | None,
    dataset_dir: Path,
    official_to_train: dict[int, int],
) -> list[dict]:
    images = find_images(train_images_dir)
    records = []
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        label_path = find_label_for_image(image_path, train_labels_dir)
        if label_path is None:
            raise FileNotFoundError(f"No label found for {image_path}")

        case_id = case_id_from_image(image_path)
        target_image = images_tr / f"{case_id}_0000.nii.gz"
        target_label = labels_tr / f"{case_id}.nii.gz"

        shutil.copy2(image_path, target_image)
        remap_label_to_train_values(label_path, target_label, official_to_train)

        records.append({
            "case_id": case_id,
            "modality": modality,
            "source_image": str(image_path),
            "source_label": str(label_path),
            "nnunet_image": str(target_image),
            "nnunet_label": str(target_label)
        })
    return records


def copy_validation_cases(modality: str, val_images_dir: list[Path], dataset_dir: Path) -> list[dict]:
    images = find_images(val_images_dir)
    records = []
    images_ts = dataset_dir / "imagesTs"
    images_ts.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        case_id = case_id_from_image(image_path)
        target_image = images_ts / f"{case_id}_0000.nii.gz"
        shutil.copy2(image_path, target_image)
        records.append({
            "case_id": case_id,
            "modality": modality,
            "source_image": str(image_path),
            "nnunet_image": str(target_image),
            "output_label_name": output_label_name_from_image(image_path)
        })
    return records


def dry_run_modality(modality: str, train_images_dir: list[Path], train_labels_dir: list[Path] | None,
                     val_images_dir: list[Path]) -> None:
    train_images = find_images(train_images_dir)
    val_images = find_images(val_images_dir)
    missing_labels = [
        image_path for image_path in train_images
        if find_label_for_image(image_path, train_labels_dir) is None
    ]
    print(f"[{modality.upper()}] dry run")
    print(f"[{modality.upper()}] training images: {len(train_images)}")
    print(f"[{modality.upper()}] validation images: {len(val_images)}")
    print(f"[{modality.upper()}] missing labels: {len(missing_labels)}")
    for path in missing_labels[:10]:
        print(f"  missing label for {path}")


def convert_modality(modality: str, args: argparse.Namespace, cfg: dict) -> None:
    raw_root = nnunet_raw_root(args.dataset_root)
    dataset_dir = raw_root / dataset_name(modality, cfg)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if modality == "ct":
        train_images = args.ct_train_images or discover_split_dirs(args.train_root, "ct", "train")
        train_labels = args.ct_train_labels
        val_images = args.ct_val_images or discover_split_dirs(args.val_root, "ct", "val")
    else:
        train_images = args.mr_train_images or discover_split_dirs(args.train_root, "mr", "train")
        train_labels = args.mr_train_labels
        val_images = args.mr_val_images or discover_split_dirs(args.val_root, "mr", "val")

    print(f"[{modality.upper()}] train image dirs: {[str(p) for p in train_images]}")
    print(f"[{modality.upper()}] val image dirs: {[str(p) for p in val_images]}")

    if args.dry_run:
        dry_run_modality(modality, train_images, train_labels, val_images)
        return

    official_to_train, _ = mapping_arrays(cfg)

    train_records = copy_training_cases(modality, train_images, train_labels, dataset_dir, official_to_train)
    val_records = copy_validation_cases(modality, val_images, dataset_dir)

    write_json(dataset_dir / "dataset.json", make_dataset_json(modality, cfg, len(train_records)))
    write_json(dataset_dir / "conversion_mapping.json", {
        "modality": modality,
        "dataset_name": dataset_name(modality, cfg),
        "train": train_records,
        "validation": val_records
    })

    print(f"[{modality.upper()}] wrote {len(train_records)} training and {len(val_records)} validation cases to {dataset_dir}")


def main() -> None:
    args = parse_args()
    cfg = load_label_config(args.label_config) if args.label_config else load_label_config()

    tasks = ["ct", "mr"] if args.task == "both" else [args.task]
    for modality in tasks:
        convert_modality(modality, args, cfg)

    for folder in ["nnUNet_preprocessed", "nnUNet_result"]:
        (args.dataset_root / folder).mkdir(parents=True, exist_ok=True)

    print("Done.")
    print(f"Set nnUNet_raw={args.dataset_root / 'nnUNet_raw'}")
    print(f"Set nnUNet_preprocessed={args.dataset_root / 'nnUNet_preprocessed'}")
    print(f"Set nnUNet_results={args.dataset_root / 'nnUNet_result'}")


if __name__ == "__main__":
    main()
