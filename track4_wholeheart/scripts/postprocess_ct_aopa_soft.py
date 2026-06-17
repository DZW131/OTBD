from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

try:
    from .postprocess_predictions import largest_component, load_sitk, normalize_spacing, parse_int_list
except ImportError:
    from postprocess_predictions import largest_component, load_sitk, normalize_spacing, parse_int_list


PREFERRED_PROBABILITY_KEYS = ("probabilities", "softmax", "probability", "arr_0")


@dataclass(frozen=True)
class HysteresisStats:
    class_idx: int
    threshold: float
    seed_voxels: int
    candidate_voxels: int
    grown_voxels: int
    added_voxels: int


@dataclass(frozen=True)
class ComponentCleanupStats:
    class_idx: int
    components_seen: int
    components_kept: int
    components_removed: int


def adaptive_seed_threshold(
    class_prob: np.ndarray,
    seed_mask: np.ndarray,
    min_threshold: float = 0.18,
    max_threshold: float = 0.30,
    median_scale: float = 0.45,
    fallback_threshold: float = 0.25,
) -> float:
    seed_values = np.asarray(class_prob)[np.asarray(seed_mask, dtype=bool)]
    if seed_values.size == 0:
        return round(float(fallback_threshold), 6)
    threshold = float(np.median(seed_values)) * float(median_scale)
    threshold = max(float(min_threshold), min(float(max_threshold), threshold))
    return round(threshold, 6)


def load_case_probability(path: Path) -> np.ndarray:
    with np.load(path) as data:
        key = next((candidate for candidate in PREFERRED_PROBABILITY_KEYS if candidate in data.files), None)
        if key is None:
            if len(data.files) != 1:
                raise ValueError(
                    f"{path} contains probability keys {data.files}; expected one of {PREFERRED_PROBABILITY_KEYS}"
                )
            key = data.files[0]
        prob = np.asarray(data[key], dtype=np.float32)

    if prob.ndim != 4:
        raise ValueError(f"{path} probability array must have shape [C, Z, Y, X], got {prob.shape}")
    return prob


def _expanded_bbox_mask(mask: np.ndarray, spacing: Iterable[float], margin_mm: float) -> np.ndarray:
    bbox = np.zeros_like(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return bbox

    spacing_values = normalize_spacing(spacing, mask.ndim)
    margin_voxels = np.ceil(float(margin_mm) / np.asarray(spacing_values)).astype(int)
    starts = np.maximum(coords.min(axis=0) - margin_voxels, 0)
    stops = np.minimum(coords.max(axis=0) + margin_voxels + 2, np.asarray(mask.shape))
    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
    bbox[slices] = True
    return bbox


def _other_class_max(prob: np.ndarray, class_idx: int) -> np.ndarray:
    class_ids = [idx for idx in range(1, prob.shape[0]) if idx != int(class_idx)]
    if not class_ids:
        return np.zeros(prob.shape[1:], dtype=np.float32)
    return np.max(prob[class_ids], axis=0)


def _seed_touching_candidate(candidate: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(candidate.ndim, candidate.ndim)
    cc, num = ndimage.label(candidate, structure=structure)
    if num == 0:
        return np.zeros_like(candidate, dtype=bool)
    seed_ids = np.unique(cc[seed_mask])
    seed_ids = seed_ids[seed_ids != 0]
    if seed_ids.size == 0:
        return np.zeros_like(candidate, dtype=bool)
    return np.isin(cc, seed_ids)


def vessel_hysteresis_growing(
    pred_legacy: np.ndarray,
    prob: np.ndarray,
    class_idx: int,
    spacing: Iterable[float],
    p_class_threshold: float = 0.25,
    threshold_mode: str = "fixed",
    strong_other_threshold: float = 0.65,
    bbox_margin_mm: float = 15.0,
    max_distance_mm: float = 15.0,
    allow_overwrite_low_conf_other: bool = False,
) -> tuple[np.ndarray, HysteresisStats]:
    class_idx = int(class_idx)
    pred = np.asarray(pred_legacy)
    if prob.shape[1:] != pred.shape:
        raise ValueError(f"Probability spatial shape {prob.shape[1:]} does not match prediction shape {pred.shape}")
    if class_idx >= prob.shape[0]:
        raise ValueError(f"class_idx={class_idx} is outside probability channel count {prob.shape[0]}")

    seed_mask = largest_component(pred == class_idx)
    seed_voxels = int(seed_mask.sum())
    if seed_voxels == 0:
        stats = HysteresisStats(class_idx, float(p_class_threshold), 0, 0, 0, 0)
        return pred.copy(), stats

    if threshold_mode == "adaptive":
        threshold = adaptive_seed_threshold(prob[class_idx], seed_mask, fallback_threshold=p_class_threshold)
    elif threshold_mode == "fixed":
        threshold = round(float(p_class_threshold), 6)
    else:
        raise ValueError(f"Unsupported threshold_mode={threshold_mode!r}; expected fixed or adaptive")

    spacing_values = normalize_spacing(spacing, pred.ndim)
    seed_distance = ndimage.distance_transform_edt(~seed_mask, sampling=spacing_values)
    bbox_mask = _expanded_bbox_mask(seed_mask, spacing_values, bbox_margin_mm)
    other_max = _other_class_max(prob, class_idx)

    candidate = prob[class_idx] >= threshold
    candidate &= other_max <= float(strong_other_threshold)
    candidate &= bbox_mask | (seed_distance <= float(max_distance_mm))
    candidate |= seed_mask

    grown = _seed_touching_candidate(candidate, seed_mask)
    writable = (pred == 0) | (pred == class_idx)
    if allow_overwrite_low_conf_other:
        writable |= other_max <= float(strong_other_threshold)

    refined = pred.copy()
    add_mask = grown & writable & (refined != class_idx)
    refined[add_mask] = class_idx
    stats = HysteresisStats(
        class_idx=class_idx,
        threshold=threshold,
        seed_voxels=seed_voxels,
        candidate_voxels=int(candidate.sum()),
        grown_voxels=int(grown.sum()),
        added_voxels=int(add_mask.sum()),
    )
    return refined, stats


def _probability_entropy(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob.astype(np.float32, copy=False), 1e-8, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=0)


def component_score_cleanup(
    pred: np.ndarray,
    prob: np.ndarray,
    class_idx: int,
    spacing: Iterable[float],
    seed_mask: np.ndarray | None = None,
    min_voxels: int = 30,
    min_mean_prob: float = 0.35,
    min_p10_prob: float = 0.15,
    max_mean_entropy: float | None = None,
    max_distance_mm: float = 15.0,
) -> tuple[np.ndarray, ComponentCleanupStats]:
    class_idx = int(class_idx)
    if prob.shape[1:] != pred.shape:
        raise ValueError(f"Probability spatial shape {prob.shape[1:]} does not match prediction shape {pred.shape}")
    if class_idx >= prob.shape[0]:
        raise ValueError(f"class_idx={class_idx} is outside probability channel count {prob.shape[0]}")

    vessel_mask = pred == class_idx
    if seed_mask is None:
        seed_mask = largest_component(vessel_mask)
    else:
        seed_mask = np.asarray(seed_mask, dtype=bool)

    structure = ndimage.generate_binary_structure(pred.ndim, pred.ndim)
    cc, num = ndimage.label(vessel_mask, structure=structure)
    if num == 0:
        return pred.copy(), ComponentCleanupStats(class_idx, 0, 0, 0)

    reference_mask = (pred != 0) & (pred != class_idx)
    if not np.any(reference_mask):
        reference_mask = seed_mask
    distance = ndimage.distance_transform_edt(~reference_mask, sampling=normalize_spacing(spacing, pred.ndim))
    entropy = _probability_entropy(prob) if max_mean_entropy is not None else None

    keep = np.zeros(num + 1, dtype=bool)
    keep[0] = False
    for component_id in range(1, num + 1):
        component = cc == component_id
        if np.any(component & seed_mask):
            keep[component_id] = True
            continue

        values = prob[class_idx, component]
        mean_prob = float(np.mean(values)) if values.size else 0.0
        p10_prob = float(np.percentile(values, 10)) if values.size else 0.0
        mean_entropy = float(np.mean(entropy[component])) if entropy is not None and values.size else 0.0
        min_distance = float(np.min(distance[component])) if values.size else float("inf")
        keep[component_id] = (
            int(component.sum()) >= int(min_voxels)
            and mean_prob >= float(min_mean_prob)
            and p10_prob >= float(min_p10_prob)
            and min_distance <= float(max_distance_mm)
            and (max_mean_entropy is None or mean_entropy <= float(max_mean_entropy))
        )

    cleaned = pred.copy()
    remove_mask = vessel_mask & ~keep[cc]
    cleaned[remove_mask] = 0
    kept_count = int(keep[1:].sum())
    stats = ComponentCleanupStats(
        class_idx=class_idx,
        components_seen=int(num),
        components_kept=kept_count,
        components_removed=int(num - kept_count),
    )
    return cleaned, stats


def vessel_binary_closing(
    pred: np.ndarray,
    prob: np.ndarray,
    class_idx: int,
    spacing: Iterable[float],
    radius_voxel: int = 1,
    strong_other_threshold: float = 0.65,
    bbox_margin_mm: float = 15.0,
) -> np.ndarray:
    if radius_voxel <= 0:
        return pred.copy()

    vessel_mask = pred == int(class_idx)
    if not np.any(vessel_mask):
        return pred.copy()

    structure = ndimage.iterate_structure(
        ndimage.generate_binary_structure(pred.ndim, 1),
        int(radius_voxel),
    )
    closed = ndimage.binary_closing(vessel_mask, structure=structure)
    bbox_mask = _expanded_bbox_mask(vessel_mask, normalize_spacing(spacing, pred.ndim), bbox_margin_mm)
    other_max = _other_class_max(prob, int(class_idx))
    add_mask = closed & ~vessel_mask & bbox_mask & (other_max <= float(strong_other_threshold)) & (pred == 0)

    refined = pred.copy()
    refined[add_mask] = int(class_idx)
    return refined


def refine_ct_aopa_array(
    pred: np.ndarray,
    prob: np.ndarray,
    spacing: Iterable[float],
    classes: Iterable[int] = (6, 7),
    threshold_mode: str = "adaptive",
    p_class_threshold: float = 0.25,
    strong_other_threshold: float = 0.65,
    bbox_margin_mm: float = 15.0,
    max_distance_mm: float = 15.0,
    enable_component_score: bool = False,
    min_voxels: int = 30,
    min_mean_prob: float = 0.35,
    min_p10_prob: float = 0.15,
    max_mean_entropy: float | None = None,
    enable_closing: bool = False,
    closing_radius_voxel: int = 1,
) -> tuple[np.ndarray, list[HysteresisStats], list[ComponentCleanupStats]]:
    refined = np.asarray(pred).copy()
    grow_stats: list[HysteresisStats] = []
    cleanup_stats: list[ComponentCleanupStats] = []

    for class_idx in [int(label) for label in classes]:
        seed_mask = largest_component(refined == class_idx)
        refined, stats = vessel_hysteresis_growing(
            refined,
            prob,
            class_idx=class_idx,
            spacing=spacing,
            p_class_threshold=p_class_threshold,
            threshold_mode=threshold_mode,
            strong_other_threshold=strong_other_threshold,
            bbox_margin_mm=bbox_margin_mm,
            max_distance_mm=max_distance_mm,
        )
        grow_stats.append(stats)
        if enable_closing:
            refined = vessel_binary_closing(
                refined,
                prob,
                class_idx=class_idx,
                spacing=spacing,
                radius_voxel=closing_radius_voxel,
                strong_other_threshold=strong_other_threshold,
                bbox_margin_mm=bbox_margin_mm,
            )
        if enable_component_score:
            refined, cleanup = component_score_cleanup(
                refined,
                prob,
                class_idx=class_idx,
                spacing=spacing,
                seed_mask=seed_mask,
                min_voxels=min_voxels,
                min_mean_prob=min_mean_prob,
                min_p10_prob=min_p10_prob,
                max_mean_entropy=max_mean_entropy,
                max_distance_mm=max_distance_mm,
            )
            cleanup_stats.append(cleanup)

    return refined, grow_stats, cleanup_stats


def probability_path_for_prediction(probability_dir: Path, pred_path: Path) -> Path:
    name = pred_path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = pred_path.stem
    return probability_dir / f"{name}.npz"


def refine_file(
    input_path: Path,
    probability_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> tuple[list[HysteresisStats], list[ComponentCleanupStats]]:
    sitk = load_sitk()
    image = sitk.ReadImage(str(input_path))
    pred = sitk.GetArrayFromImage(image)
    prob = load_case_probability(probability_path)
    if prob.shape[1:] != pred.shape:
        raise ValueError(
            f"{probability_path} probability shape {prob.shape[1:]} does not match {input_path.name} label shape {pred.shape}; "
            "refusing to transpose silently"
        )
    if prob.shape[0] < max(args.classes) + 1:
        raise ValueError(f"{probability_path} has {prob.shape[0]} channels, but classes={args.classes}")

    spacing_zyx = tuple(reversed(image.GetSpacing()))
    refined, grow_stats, cleanup_stats = refine_ct_aopa_array(
        pred,
        prob,
        spacing=spacing_zyx,
        classes=args.classes,
        threshold_mode=args.threshold_mode,
        p_class_threshold=args.p_class_threshold,
        strong_other_threshold=args.strong_other_threshold,
        bbox_margin_mm=args.bbox_margin_mm,
        max_distance_mm=args.max_distance_mm,
        enable_component_score=args.component_score,
        min_voxels=args.min_voxels,
        min_mean_prob=args.min_mean_prob,
        min_p10_prob=args.min_p10_prob,
        max_mean_entropy=args.max_mean_entropy,
        enable_closing=args.enable_closing,
        closing_radius_voxel=args.closing_radius_voxel,
    )

    out_img = sitk.GetImageFromArray(refined.astype(pred.dtype, copy=False))
    out_img.CopyInformation(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(output_path))
    return grow_stats, cleanup_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CT-only AO/PA softmax hysteresis refinement after legacy postprocessing.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Legacy-postprocessed train-label prediction directory.")
    parser.add_argument("--probability-dir", type=Path, required=True, help="nnU-Net prediction directory containing matching .npz files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for refined train-label predictions.")
    parser.add_argument("--classes", default="6,7", help="Comma-separated internal vessel labels. Defaults to AO=6,PA=7.")
    parser.add_argument("--threshold-mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--p-class-threshold", type=float, default=0.25)
    parser.add_argument("--strong-other-threshold", type=float, default=0.65)
    parser.add_argument("--bbox-margin-mm", type=float, default=15.0)
    parser.add_argument("--max-distance-mm", type=float, default=15.0)
    parser.add_argument("--component-score", action="store_true")
    parser.add_argument("--min-voxels", type=int, default=30)
    parser.add_argument("--min-mean-prob", type=float, default=0.35)
    parser.add_argument("--min-p10-prob", type=float, default=0.15)
    parser.add_argument("--max-mean-entropy", type=float, default=None)
    parser.add_argument("--enable-closing", action="store_true")
    parser.add_argument("--closing-radius-voxel", type=int, default=1)
    args = parser.parse_args()
    args.classes = parse_int_list(args.classes)
    if not args.classes:
        raise SystemExit("--classes must contain at least one label")
    return args


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No .nii.gz predictions found in {args.input_dir}")

    for input_path in files:
        probability_path = probability_path_for_prediction(args.probability_dir, input_path)
        if not probability_path.exists():
            raise SystemExit(f"Missing probability file for {input_path.name}: {probability_path}")
        output_path = args.output_dir / input_path.name
        grow_stats, cleanup_stats = refine_file(input_path, probability_path, output_path, args)
        grow_text = ", ".join(
            f"class={s.class_idx}:thr={s.threshold:.3f}:seed={s.seed_voxels}:add={s.added_voxels}"
            for s in grow_stats
        )
        cleanup_text = ", ".join(
            f"class={s.class_idx}:kept={s.components_kept}:removed={s.components_removed}"
            for s in cleanup_stats
        )
        if cleanup_text:
            cleanup_text = f"; cleanup {cleanup_text}"
        print(f"CT AOPA soft refined {input_path.name} -> {output_path} ({grow_text}{cleanup_text})")

    print(f"CT AOPA soft refined {len(files)} files.")


if __name__ == "__main__":
    main()
