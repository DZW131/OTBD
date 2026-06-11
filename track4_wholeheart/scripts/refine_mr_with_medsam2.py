from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from scipy import ndimage


TARGET_LABELS = [5, 6, 7]
LABEL_NAMES = {5: "Myo", 6: "AO", 7: "PA"}


@dataclass(frozen=True)
class RefinementDecision:
    case: str
    label: int
    class_name: str
    original_voxels: int
    candidate_voxels: int
    refined_voxels: int
    volume_ratio: float
    iou_with_original: float
    accepted: bool
    reason: str
    key_slice: int | None
    bbox: str


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def strip_nii_gz(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def image_path_for_prediction(pred_path: Path, image_dir: Path) -> Path:
    return image_dir / f"{strip_nii_gz(pred_path)}_0000.nii.gz"


def normalize_volume_to_uint8(
    image: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    use_nonzero: bool = True,
) -> np.ndarray:
    values = image[np.isfinite(image)]
    if use_nonzero:
        nonzero = values[values != 0]
        if nonzero.size:
            values = nonzero
    if values.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    lower, upper = np.percentile(values, [lower_percentile, upper_percentile])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(values.min())
        upper = float(values.max())
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.uint8)

    clipped = np.clip(image.astype(np.float32, copy=False), lower, upper)
    scaled = (clipped - lower) / (upper - lower)
    return np.rint(scaled * 255.0).clip(0, 255).astype(np.uint8)


def resize_grayscale_to_rgb_tensor(volume_uint8: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    frames = np.zeros((volume_uint8.shape[0], 3, image_size, image_size), dtype=np.float32)
    for idx, frame in enumerate(volume_uint8):
        resized = Image.fromarray(frame).convert("RGB").resize((image_size, image_size))
        frames[idx] = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0

    tensor = torch.from_numpy(frames).to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=device)[:, None, None]
    return (tensor - mean) / std


def central_bbox_prompt(mask: np.ndarray, padding: int = 5) -> tuple[int | None, np.ndarray | None]:
    if not np.any(mask):
        return None, None

    per_slice_area = mask.reshape(mask.shape[0], -1).sum(axis=1)
    key_slice = int(np.argmax(per_slice_area))
    slice_mask = mask[key_slice]
    if not np.any(slice_mask):
        return None, None

    y_indices, x_indices = np.where(slice_mask)
    height, width = slice_mask.shape
    x_min = max(0, int(x_indices.min()) - padding)
    x_max = min(width - 1, int(x_indices.max()) + padding)
    y_min = max(0, int(y_indices.min()) - padding)
    y_max = min(height - 1, int(y_indices.max()) + padding)
    return key_slice, np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return math.nan
    return float(np.logical_and(a, b).sum() / union)


def constrained_candidate(
    medsam_mask: np.ndarray,
    original_mask: np.ndarray,
    current_seg: np.ndarray,
    label: int,
    dilation_radius: int,
) -> np.ndarray:
    if dilation_radius > 0:
        nearby = ndimage.binary_dilation(original_mask, iterations=int(dilation_radius))
    else:
        nearby = original_mask
    allowed = np.logical_or(current_seg == 0, current_seg == int(label))
    return medsam_mask.astype(bool) & nearby & allowed


def should_accept_candidate(
    original_mask: np.ndarray,
    candidate: np.ndarray,
    min_volume_ratio: float,
    max_volume_ratio: float,
    min_iou: float,
) -> tuple[bool, str, float, float]:
    original_voxels = int(original_mask.sum())
    candidate_voxels = int(candidate.sum())
    if original_voxels == 0:
        return False, "empty_original", math.nan, math.nan
    if candidate_voxels == 0:
        return False, "empty_candidate", 0.0, 0.0

    volume_ratio = float(candidate_voxels / original_voxels)
    if volume_ratio < min_volume_ratio:
        return False, "too_small", volume_ratio, binary_iou(candidate, original_mask)
    if volume_ratio > max_volume_ratio:
        return False, "too_large", volume_ratio, binary_iou(candidate, original_mask)

    iou = binary_iou(candidate, original_mask)
    if iou < min_iou:
        return False, "low_iou", volume_ratio, iou
    return True, "accepted", volume_ratio, iou


def run_medsam2_for_label(
    predictor,
    inference_state: dict,
    key_slice: int,
    bbox: np.ndarray,
) -> np.ndarray:
    num_frames = int(inference_state["num_frames"])
    output = np.zeros(
        (num_frames, int(inference_state["video_height"]), int(inference_state["video_width"])),
        dtype=bool,
    )

    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(next(predictor.parameters()).device).startswith("cuda")
        else nullcontext()
    )

    with torch.inference_mode(), autocast_context:
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(key_slice),
            obj_id=1,
            box=bbox,
        )
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            output[int(out_frame_idx)] |= (out_mask_logits[0] > 0.0).detach().cpu().numpy()[0]

        predictor.reset_state(inference_state)
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=int(key_slice),
            obj_id=1,
            box=bbox,
        )
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            output[int(out_frame_idx)] |= (out_mask_logits[0] > 0.0).detach().cpu().numpy()[0]

        predictor.reset_state(inference_state)
    return output


def radius_for_label(label: int, myo_radius: int, vessel_radius: int, default_radius: int) -> int:
    if int(label) == 5:
        return int(myo_radius)
    if int(label) in {6, 7}:
        return int(vessel_radius)
    return int(default_radius)


def refine_case(
    image_path: Path,
    pred_path: Path,
    output_path: Path,
    predictor,
    args: argparse.Namespace,
) -> list[RefinementDecision]:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path))
    pred_image = sitk.ReadImage(str(pred_path))
    image_arr = sitk.GetArrayFromImage(image)
    seg = sitk.GetArrayFromImage(pred_image)
    current_seg = seg.copy()

    volume_uint8 = normalize_volume_to_uint8(
        image_arr,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        use_nonzero=not args.use_all_intensities,
    )
    frames = resize_grayscale_to_rgb_tensor(volume_uint8, args.image_size, args.device)
    inference_state = predictor.init_state(frames, volume_uint8.shape[1], volume_uint8.shape[2])

    decisions: list[RefinementDecision] = []
    for label in args.labels:
        original_mask = seg == int(label)
        original_voxels = int(original_mask.sum())
        key_slice, bbox = central_bbox_prompt(original_mask, padding=args.box_padding)
        if key_slice is None or bbox is None:
            decisions.append(
                RefinementDecision(
                    case=pred_path.name,
                    label=int(label),
                    class_name=LABEL_NAMES.get(int(label), str(label)),
                    original_voxels=original_voxels,
                    candidate_voxels=0,
                    refined_voxels=original_voxels,
                    volume_ratio=math.nan,
                    iou_with_original=math.nan,
                    accepted=False,
                    reason="empty_original",
                    key_slice=None,
                    bbox="",
                )
            )
            continue

        medsam_mask = run_medsam2_for_label(predictor, inference_state, key_slice, bbox)
        candidate = constrained_candidate(
            medsam_mask,
            original_mask,
            current_seg,
            int(label),
            dilation_radius=radius_for_label(label, args.myo_radius, args.vessel_radius, args.dilation_radius),
        )
        accepted, reason, volume_ratio, iou = should_accept_candidate(
            original_mask,
            candidate,
            min_volume_ratio=args.min_volume_ratio,
            max_volume_ratio=args.max_volume_ratio,
            min_iou=args.min_iou_with_original,
        )
        if accepted:
            current_seg[current_seg == int(label)] = 0
            current_seg[candidate] = int(label)

        decisions.append(
            RefinementDecision(
                case=pred_path.name,
                label=int(label),
                class_name=LABEL_NAMES.get(int(label), str(label)),
                original_voxels=original_voxels,
                candidate_voxels=int(candidate.sum()),
                refined_voxels=int((current_seg == int(label)).sum()),
                volume_ratio=volume_ratio,
                iou_with_original=iou,
                accepted=accepted,
                reason=reason,
                key_slice=int(key_slice),
                bbox=",".join(str(int(round(v))) for v in bbox.tolist()),
            )
        )

    out_image = sitk.GetImageFromArray(current_seg.astype(seg.dtype, copy=False))
    out_image.CopyInformation(pred_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_image, str(output_path))
    del frames, inference_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return decisions


def iter_prediction_pairs(args: argparse.Namespace) -> Iterable[tuple[Path, Path]]:
    if args.pred_dir is not None:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required with --pred-dir.")
        yield Path(args.pred_dir), Path(args.output_dir)
        return

    if args.pred_root is None or args.output_root is None:
        raise SystemExit("Use either --pred-dir/--output-dir or --pred-root/--output-root.")
    for fold in args.folds:
        yield (
            Path(args.pred_root) / f"fold_{int(fold)}" / "validation",
            Path(args.output_root) / f"fold_{int(fold)}" / "validation",
        )


def write_decisions(path: Path, rows: Sequence[RefinementDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RefinementDecision.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine MR nnU-Net predictions with MedSAM2 for Myo/AO/PA before postprocessing."
    )
    parser.add_argument("--pred-root", type=Path, default=None,
                        help="nnU-Net trainer root containing fold_X/validation directories.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--pred-dir", type=Path, default=None,
                        help="Flat prediction directory for challenge validation predictions.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Raw MR imagesTr/imagesTs directory containing CaseXXXX_0000.nii.gz.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("pretrained/MedSAM2/MedSAM2_latest.pt"))
    parser.add_argument("--medsam2-root", type=Path, default=Path("third_party/MedSAM2"))
    parser.add_argument("--cfg", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--labels", type=parse_int_list, default=TARGET_LABELS)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lower-percentile", type=float, default=0.5)
    parser.add_argument("--upper-percentile", type=float, default=99.5)
    parser.add_argument("--use-all-intensities", action="store_true")
    parser.add_argument("--box-padding", type=int, default=5)
    parser.add_argument("--dilation-radius", type=int, default=3)
    parser.add_argument("--myo-radius", type=int, default=2)
    parser.add_argument("--vessel-radius", type=int, default=5)
    parser.add_argument("--min-volume-ratio", type=float, default=0.8)
    parser.add_argument("--max-volume-ratio", type=float, default=1.2)
    parser.add_argument("--min-iou-with-original", type=float, default=0.5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--decisions-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("CUDA is not available. Use --device cpu or run on a GPU node.")

    import sys

    medsam2_root = args.medsam2_root.resolve()
    sys.path.insert(0, str(medsam2_root))
    from sam2.build_sam import build_sam2_video_predictor_npz

    predictor = build_sam2_video_predictor_npz(
        args.cfg,
        str(args.checkpoint),
        device=args.device,
    )

    all_decisions: list[RefinementDecision] = []
    processed = 0
    for pred_dir, out_dir in iter_prediction_pairs(args):
        pred_files = sorted(pred_dir.glob("*.nii.gz"))
        if args.max_cases is not None:
            pred_files = pred_files[: int(args.max_cases)]
        print(f"{pred_dir}: {len(pred_files)} predictions", flush=True)
        for idx, pred_path in enumerate(pred_files, start=1):
            out_path = out_dir / pred_path.name
            if args.skip_existing and out_path.exists():
                print(f"  [{idx}/{len(pred_files)}] skip existing {pred_path.name}", flush=True)
                continue
            image_path = image_path_for_prediction(pred_path, args.image_dir)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {pred_path}: {image_path}")
            print(f"  [{idx}/{len(pred_files)}] refine {pred_path.name}", flush=True)
            all_decisions.extend(refine_case(image_path, pred_path, out_path, predictor, args))
            processed += 1

    if args.decisions_csv:
        write_decisions(args.decisions_csv, all_decisions)
        print(f"wrote decisions: {args.decisions_csv}")
    print(f"processed cases: {processed}")


if __name__ == "__main__":
    main()
