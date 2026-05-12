from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from common import load_label_config, mapping_arrays, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore nnU-Net 0-7 labels to official CARE Whole Heart values.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, default=None)
    return parser.parse_args()


def restore_one(pred_path: Path, output_path: Path, train_to_official: dict[int, int]) -> None:
    image = sitk.ReadImage(str(pred_path))
    arr = sitk.GetArrayFromImage(image)
    unique = set(int(v) for v in np.unique(arr))
    if not unique.issubset(set(train_to_official.keys())):
        raise ValueError(f"Unexpected prediction labels in {pred_path}: {sorted(unique)}")

    out = np.zeros_like(arr, dtype=np.uint16)
    for train_value, official_value in train_to_official.items():
        out[arr == train_value] = official_value

    out_img = sitk.GetImageFromArray(out)
    out_img.CopyInformation(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(output_path))


def main() -> None:
    args = parse_args()
    cfg = load_label_config(args.label_config) if args.label_config else load_label_config()
    _, train_to_official = mapping_arrays(cfg)

    mapping = read_json(args.mapping_json)
    validation_records = mapping.get("validation", [])
    if not validation_records:
        raise ValueError(f"No validation records found in {args.mapping_json}")

    restored = 0
    for record in validation_records:
        case_id = record["case_id"]
        pred_path = args.pred_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            print(f"Skipping missing prediction: {pred_path}")
            continue
        output_path = args.output_dir / record["output_label_name"]
        restore_one(pred_path, output_path, train_to_official)
        restored += 1
        print(f"Restored {pred_path.name} -> {output_path}")

    print(f"Restored {restored} predictions to official labels.")


if __name__ == "__main__":
    main()
