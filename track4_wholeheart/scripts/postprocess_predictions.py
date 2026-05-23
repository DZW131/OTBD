from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

try:
    from .common import load_label_config, mapping_arrays
except ImportError:
    from common import load_label_config, mapping_arrays


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def largest_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    cc, num = ndimage.label(mask, structure=structure)
    if num <= 1:
        return mask

    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    return cc == int(sizes.argmax())


def remove_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    if min_component_size <= 0 or not np.any(mask):
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    cc, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return mask

    sizes = np.bincount(cc.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[sizes >= min_component_size] = True
    keep[0] = False
    return keep[cc]


def postprocess_label_array(
    seg: np.ndarray,
    labels: Iterable[int],
    keep_largest: bool = True,
    min_component_size: int = 0,
) -> np.ndarray:
    cleaned = seg.copy()
    for label in labels:
        mask = seg == int(label)
        if keep_largest:
            mask = largest_component(mask)
        mask = remove_small_components(mask, min_component_size)
        cleaned[seg == int(label)] = 0
        cleaned[mask] = int(label)
    return cleaned


def load_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required for NIfTI postprocessing. Install it in the nnU-Net environment.") from exc
    return sitk


def label_values_for_space(label_space: str, cfg: dict, labels: list[int]) -> list[int]:
    if label_space == "train":
        return labels
    _, train_to_official = mapping_arrays(cfg)
    return [train_to_official[label] for label in labels]


def postprocess_file(input_path: Path, output_path: Path, labels: list[int], keep_largest: bool,
                     min_component_size: int) -> None:
    sitk = load_sitk()
    image = sitk.ReadImage(str(input_path))
    arr = sitk.GetArrayFromImage(image)
    cleaned = postprocess_label_array(arr, labels, keep_largest=keep_largest, min_component_size=min_component_size)

    out_img = sitk.GetImageFromArray(cleaned.astype(arr.dtype, copy=False))
    out_img.CopyInformation(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply anatomical connected-component postprocessing to predictions.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing .nii.gz predictions.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for postprocessed predictions.")
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--label-space", choices=["train", "official"], default="train",
                        help="Use train labels 1..7 before restore, or official challenge values after restore.")
    parser.add_argument("--labels", default="1,2,3,4,5,6,7",
                        help="Comma-separated train labels to postprocess. Official values are inferred if needed.")
    parser.add_argument("--min-component-size", type=int, default=0,
                        help="Remove non-largest components below this voxel count. 0 disables size filtering.")
    parser.add_argument("--no-keep-largest", action="store_true",
                        help="Keep all components except those removed by --min-component-size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_label_config(args.label_config) if args.label_config else load_label_config()
    train_labels = parse_int_list(args.labels)
    labels = label_values_for_space(args.label_space, cfg, train_labels)

    files = sorted(args.input_dir.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No .nii.gz predictions found in {args.input_dir}")

    for input_path in files:
        output_path = args.output_dir / input_path.name
        postprocess_file(
            input_path,
            output_path,
            labels=labels,
            keep_largest=not args.no_keep_largest,
            min_component_size=args.min_component_size,
        )
        print(f"Postprocessed {input_path.name} -> {output_path}")

    print(f"Postprocessed {len(files)} files.")


if __name__ == "__main__":
    main()
