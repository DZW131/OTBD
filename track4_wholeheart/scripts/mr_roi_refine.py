from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .common import strip_nii_gz_name
    from .postprocess_predictions import build_class_aware_hd_rules, load_sitk, postprocess_label_array
except ImportError:
    from common import strip_nii_gz_name
    from postprocess_predictions import build_class_aware_hd_rules, load_sitk, postprocess_label_array


DEFAULT_LABELS = [1, 2, 3, 4, 5, 6, 7]


@dataclass(frozen=True)
class CropRecord:
    case_id: str
    image_path: str
    prediction_path: str
    output_image: str
    bbox_zyx: list[list[int]]
    original_shape_zyx: list[int]
    target_label: int
    roi_label: int
    reason: str


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def case_id_from_prediction(path: Path) -> str:
    return strip_nii_gz_name(path)


def case_id_from_image(path: Path) -> str:
    name = strip_nii_gz_name(path)
    return name[:-5] if name.endswith("_0000") else name


def image_path_for_case(images_dir: Path, case_id: str) -> Path:
    candidates = [
        images_dir / f"{case_id}_0000.nii.gz",
        images_dir / f"{case_id}.nii.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for {case_id} in {images_dir}")


def label_path_for_case(labels_dir: Path, case_id: str) -> Path:
    path = labels_dir / f"{case_id}.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Could not find label for {case_id}: {path}")
    return path


def prediction_path_for_case(
    case_id: str,
    pred_dir: Path | None = None,
    pred_root: Path | None = None,
    case_to_fold: dict[str, int] | None = None,
) -> Path:
    if pred_dir is not None:
        path = pred_dir / f"{case_id}.nii.gz"
        if path.exists():
            return path
        raise FileNotFoundError(f"Missing prediction for {case_id}: {path}")

    if pred_root is None:
        raise ValueError("Either pred_dir or pred_root is required")

    if case_to_fold is not None and case_id in case_to_fold:
        path = pred_root / f"fold_{case_to_fold[case_id]}" / "validation" / f"{case_id}.nii.gz"
        if path.exists():
            return path
        raise FileNotFoundError(f"Missing fold prediction for {case_id}: {path}")

    matches = sorted(pred_root.glob(f"fold_*/validation/{case_id}.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"Could not locate OOF prediction for {case_id} under {pred_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple OOF predictions for {case_id}: {matches}")
    return matches[0]


def load_case_to_fold(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    splits = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, int] = {}
    for fold, split in enumerate(splits):
        for case_id in split.get("val", []):
            mapping[str(case_id)] = int(fold)
    return mapping


def margin_voxels(spacing_zyx: Iterable[float], margin_mm: float, min_margin_voxels: int) -> np.ndarray:
    spacing = np.asarray(tuple(float(v) for v in spacing_zyx), dtype=np.float64)
    spacing = np.maximum(spacing, 1e-6)
    return np.maximum(np.ceil(float(margin_mm) / spacing).astype(int), int(min_margin_voxels))


def bbox_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if not np.any(mask):
        return None
    coords = np.argwhere(mask)
    lower = coords.min(axis=0)
    upper = coords.max(axis=0) + 1
    return lower.astype(int), upper.astype(int)


def padded_bbox(
    mask: np.ndarray,
    spacing_zyx: Iterable[float],
    margin_mm: float,
    min_margin_voxels: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    raw = bbox_from_mask(mask)
    if raw is None:
        return None
    lower, upper = raw
    margin = margin_voxels(spacing_zyx, margin_mm, min_margin_voxels)
    lower = np.maximum(lower - margin, 0)
    upper = np.minimum(upper + margin, np.asarray(mask.shape, dtype=int))
    return lower, upper


def select_roi_bbox(
    pred: np.ndarray,
    label: np.ndarray | None,
    target_label: int,
    spacing_zyx: Iterable[float],
    margin_mm: float,
    min_margin_voxels: int,
    include_gt_guard: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    pred_mask = pred == int(target_label)
    reason = "prediction_target"
    crop_mask = pred_mask

    if include_gt_guard and label is not None:
        gt_mask = label == int(target_label)
        candidate = pred_mask | gt_mask
        if np.any(candidate):
            crop_mask = candidate
            reason = "prediction_target_plus_gt_guard" if np.any(pred_mask) else "gt_target_fallback"

    if not np.any(crop_mask):
        foreground = pred > 0
        if np.any(foreground):
            crop_mask = foreground
            reason = "prediction_foreground_fallback"
        elif label is not None and np.any(label == int(target_label)):
            crop_mask = label == int(target_label)
            reason = "gt_target_fallback"
        else:
            crop_mask = np.ones_like(pred, dtype=bool)
            reason = "full_image_fallback"

    bbox = padded_bbox(crop_mask, spacing_zyx, margin_mm, min_margin_voxels)
    if bbox is None:
        raise RuntimeError("ROI bbox selection failed unexpectedly")
    return bbox[0], bbox[1], reason


def crop_sitk_image(image, lower_zyx: np.ndarray, upper_zyx: np.ndarray):
    sitk = load_sitk()
    size_xyz = [
        int(upper_zyx[2] - lower_zyx[2]),
        int(upper_zyx[1] - lower_zyx[1]),
        int(upper_zyx[0] - lower_zyx[0]),
    ]
    index_xyz = [int(lower_zyx[2]), int(lower_zyx[1]), int(lower_zyx[0])]
    return sitk.RegionOfInterest(image, size=size_xyz, index=index_xyz)


def save_array_like(array: np.ndarray, reference_image, output_path: Path) -> None:
    sitk = load_sitk()
    output = sitk.GetImageFromArray(array)
    output.CopyInformation(reference_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(output_path))


def write_dataset_json(path: Path, dataset_name: str, target_name: str, num_training: int) -> None:
    payload = {
        "channel_names": {"0": "MR"},
        "labels": {"background": 0, target_name: 1},
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
        "name": dataset_name,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_dataset(args: argparse.Namespace) -> None:
    sitk = load_sitk()
    prepare_output_dir(args.output_dataset_dir, args.overwrite)
    images_out = args.output_dataset_dir / "imagesTr"
    labels_out = args.output_dataset_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    case_to_fold = load_case_to_fold(args.splits_json)
    records: list[CropRecord] = []
    source_images = sorted(args.source_images_dir.glob("*.nii.gz"))
    if not source_images:
        raise SystemExit(f"No source images found in {args.source_images_dir}")

    for image_path in source_images:
        case_id = case_id_from_image(image_path)
        label_path = label_path_for_case(args.source_labels_dir, case_id)
        pred_path = prediction_path_for_case(case_id, pred_root=args.roi_pred_root, case_to_fold=case_to_fold)

        image = sitk.ReadImage(str(image_path))
        label_img = sitk.ReadImage(str(label_path))
        pred_img = sitk.ReadImage(str(pred_path))
        image_arr = sitk.GetArrayFromImage(image)
        label_arr = sitk.GetArrayFromImage(label_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)
        if image_arr.shape != label_arr.shape or image_arr.shape != pred_arr.shape:
            raise ValueError(f"Shape mismatch for {case_id}: image={image_arr.shape}, label={label_arr.shape}, pred={pred_arr.shape}")

        lower, upper, reason = select_roi_bbox(
            pred=pred_arr,
            label=label_arr,
            target_label=args.target_label,
            spacing_zyx=tuple(reversed(image.GetSpacing())),
            margin_mm=args.margin_mm,
            min_margin_voxels=args.min_margin_voxels,
            include_gt_guard=not args.no_gt_guard,
        )
        cropped_image = crop_sitk_image(image, lower, upper)
        cropped_label_img = crop_sitk_image(label_img, lower, upper)
        cropped_label = (sitk.GetArrayFromImage(cropped_label_img) == int(args.target_label)).astype(np.uint8)

        out_image = images_out / f"{case_id}_0000.nii.gz"
        out_label = labels_out / f"{case_id}.nii.gz"
        sitk.WriteImage(cropped_image, str(out_image))
        save_array_like(cropped_label, cropped_label_img, out_label)
        records.append(
            CropRecord(
                case_id=case_id,
                image_path=str(image_path),
                prediction_path=str(pred_path),
                output_image=str(out_image),
                bbox_zyx=[[int(v) for v in lower], [int(v) for v in upper]],
                original_shape_zyx=[int(v) for v in image_arr.shape],
                target_label=int(args.target_label),
                roi_label=1,
                reason=reason,
            )
        )
        print(f"{case_id}: {reason} bbox={records[-1].bbox_zyx} -> {out_image}")

    write_dataset_json(args.output_dataset_dir / "dataset.json", args.dataset_name, args.target_name, len(records))
    if args.splits_json is not None:
        shutil.copy2(args.splits_json, args.output_dataset_dir / "splits_final.json")
    (args.output_dataset_dir / "roi_metadata.json").write_text(
        json.dumps({"records": [asdict(record) for record in records]}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} ROI training cases to {args.output_dataset_dir}")


def crop_inference(args: argparse.Namespace) -> None:
    sitk = load_sitk()
    prepare_output_dir(args.output_images_ts_dir, args.overwrite)
    records: list[CropRecord] = []
    pred_files = sorted(args.pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise SystemExit(f"No predictions found in {args.pred_dir}")

    for pred_path in pred_files:
        case_id = case_id_from_prediction(pred_path)
        image_path = image_path_for_case(args.images_dir, case_id)
        image = sitk.ReadImage(str(image_path))
        pred_img = sitk.ReadImage(str(pred_path))
        image_arr = sitk.GetArrayFromImage(image)
        pred_arr = sitk.GetArrayFromImage(pred_img)
        if image_arr.shape != pred_arr.shape:
            raise ValueError(f"Shape mismatch for {case_id}: image={image_arr.shape}, pred={pred_arr.shape}")

        lower, upper, reason = select_roi_bbox(
            pred=pred_arr,
            label=None,
            target_label=args.target_label,
            spacing_zyx=tuple(reversed(image.GetSpacing())),
            margin_mm=args.margin_mm,
            min_margin_voxels=args.min_margin_voxels,
            include_gt_guard=False,
        )
        cropped_image = crop_sitk_image(image, lower, upper)
        out_image = args.output_images_ts_dir / f"{case_id}_0000.nii.gz"
        sitk.WriteImage(cropped_image, str(out_image))
        records.append(
            CropRecord(
                case_id=case_id,
                image_path=str(image_path),
                prediction_path=str(pred_path),
                output_image=str(out_image),
                bbox_zyx=[[int(v) for v in lower], [int(v) for v in upper]],
                original_shape_zyx=[int(v) for v in image_arr.shape],
                target_label=int(args.target_label),
                roi_label=1,
                reason=reason,
            )
        )
        print(f"{case_id}: {reason} bbox={records[-1].bbox_zyx} -> {out_image}")

    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps({"records": [asdict(record) for record in records]}, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} ROI inference crops to {args.output_images_ts_dir}")


def load_metadata(path: Path) -> dict[str, CropRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for item in data.get("records", []):
        record = CropRecord(**item)
        records[record.case_id] = record
    return records


def paste_refinement(args: argparse.Namespace) -> None:
    sitk = load_sitk()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata_json)
    protect_labels = set(parse_int_list(args.protect_labels))
    class_rules = build_class_aware_hd_rules(DEFAULT_LABELS) if args.postprocess == "class-aware-hd" else None

    for base_path in sorted(args.base_pred_dir.glob("*.nii.gz")):
        case_id = case_id_from_prediction(base_path)
        if case_id not in metadata:
            raise KeyError(f"No crop metadata for {case_id}")
        record = metadata[case_id]
        roi_path = args.roi_pred_dir / f"{case_id}.nii.gz"
        if not roi_path.exists():
            raise FileNotFoundError(f"Missing ROI prediction for {case_id}: {roi_path}")

        base_img = sitk.ReadImage(str(base_path))
        roi_img = sitk.ReadImage(str(roi_path))
        base = sitk.GetArrayFromImage(base_img)
        roi = sitk.GetArrayFromImage(roi_img)
        lower = np.asarray(record.bbox_zyx[0], dtype=int)
        upper = np.asarray(record.bbox_zyx[1], dtype=int)
        expected_shape = tuple(int(v) for v in (upper - lower))
        if roi.shape != expected_shape:
            raise ValueError(f"ROI shape mismatch for {case_id}: roi={roi.shape}, expected={expected_shape}")

        refined = base.copy()
        slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
        patch = refined[slices].copy()
        original_patch = patch.copy()
        if args.merge_mode == "replace":
            patch[patch == int(args.target_label)] = 0
        roi_mask = roi == int(args.roi_label)
        if protect_labels:
            protected = np.isin(original_patch, list(protect_labels))
            roi_mask = roi_mask & ~protected
        patch[roi_mask] = int(args.target_label)
        refined[slices] = patch

        if class_rules is not None:
            refined = postprocess_label_array(
                refined,
                DEFAULT_LABELS,
                class_rules=class_rules,
                spacing=tuple(reversed(base_img.GetSpacing())),
            )

        out_path = args.output_dir / base_path.name
        save_array_like(refined.astype(base.dtype, copy=False), base_img, out_path)
        print(f"Pasted {roi_path.name} into {base_path.name} -> {out_path}")

    print(f"Wrote pasted refinements to {args.output_dir}")


def add_common_roi_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-label", type=int, default=6, help="Whole-heart train-space class label to refine.")
    parser.add_argument("--target-name", default="AO", help="Target class name used in ROI dataset.json.")
    parser.add_argument("--margin-mm", type=float, default=20.0)
    parser.add_argument("--min-margin-voxels", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and apply MR ROI refinement datasets for small whole-heart classes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="Create an nnU-Net ROI refinement training dataset.")
    build.add_argument("--source-images-dir", type=Path, required=True)
    build.add_argument("--source-labels-dir", type=Path, required=True)
    build.add_argument("--roi-pred-root", type=Path, required=True, help="OOF prediction root with fold_*/validation.")
    build.add_argument("--splits-json", type=Path, required=True)
    build.add_argument("--output-dataset-dir", type=Path, required=True)
    build.add_argument("--dataset-name", default="Dataset452_CARE2026_MR_AO_ROIRefine")
    build.add_argument("--no-gt-guard", action="store_true", help="Use prediction bbox only, even if it clips the GT target.")
    add_common_roi_args(build)
    build.set_defaults(func=build_dataset)

    crop = subparsers.add_parser("crop-inference", help="Create ROI imagesTs crops from a first-stage prediction directory.")
    crop.add_argument("--images-dir", type=Path, required=True)
    crop.add_argument("--pred-dir", type=Path, required=True)
    crop.add_argument("--output-images-ts-dir", type=Path, required=True)
    crop.add_argument("--metadata-json", type=Path, required=True)
    add_common_roi_args(crop)
    crop.set_defaults(func=crop_inference)

    paste = subparsers.add_parser("paste", help="Paste ROI binary predictions back into whole-heart train-label masks.")
    paste.add_argument("--base-pred-dir", type=Path, required=True)
    paste.add_argument("--roi-pred-dir", type=Path, required=True)
    paste.add_argument("--metadata-json", type=Path, required=True)
    paste.add_argument("--output-dir", type=Path, required=True)
    paste.add_argument("--target-label", type=int, default=6)
    paste.add_argument("--roi-label", type=int, default=1)
    paste.add_argument("--protect-labels", default="1,2,3,4,5,7")
    paste.add_argument(
        "--merge-mode",
        choices=["replace", "add-only"],
        default="replace",
        help="replace clears the target class inside the crop before pasting; add-only preserves baseline target voxels and only adds ROI positives.",
    )
    paste.add_argument("--postprocess", choices=["none", "class-aware-hd"], default="class-aware-hd")
    paste.set_defaults(func=paste_refinement)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
