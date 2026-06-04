from __future__ import annotations

import argparse
import json
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


def normalize_spacing(spacing: Iterable[float] | None, ndim: int) -> tuple[float, ...]:
    if spacing is None:
        return tuple(1.0 for _ in range(ndim))
    values = tuple(float(v) for v in spacing)
    if len(values) != ndim:
        raise ValueError(f"Expected {ndim} spacing values, got {values}")
    return values


def build_class_aware_hd_rules(
    labels: list[int],
    max_distance_to_heart_mm: float = 25.0,
    vessel_min_component_size: int = 20,
) -> dict[int, dict[str, object]]:
    """Default HD-oriented rules for LV/RV/LA/RA/Myo/AO/PA in the provided label space."""
    defaults = [
        {"keep_top_k": 1, "fill_holes": True},
        {"keep_top_k": 1, "fill_holes": True},
        {"keep_top_k": 1, "fill_holes": True},
        {"keep_top_k": 1, "fill_holes": True},
        {"keep_top_k": 1},
        {
            "keep_top_k": 2,
            "min_component_size": vessel_min_component_size,
            "max_distance_to_heart_mm": max_distance_to_heart_mm,
        },
        {
            "keep_top_k": 3,
            "min_component_size": vessel_min_component_size,
            "max_distance_to_heart_mm": max_distance_to_heart_mm,
        },
    ]
    return {int(label): dict(rule) for label, rule in zip(labels, defaults)}


def merge_rule_overrides(
    rules: dict[int, dict[str, object]],
    overrides: dict[int, dict[str, object]] | None,
) -> dict[int, dict[str, object]]:
    merged = {int(label): dict(rule) for label, rule in rules.items()}
    if not overrides:
        return merged
    for label, override in overrides.items():
        merged.setdefault(int(label), {}).update(override)
    return merged


def load_rule_overrides(path: Path | None) -> dict[int, dict[str, object]] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(label): dict(rule) for label, rule in data.items()}


def largest_foreground_component(seg: np.ndarray, labels: list[int]) -> np.ndarray:
    foreground = np.isin(seg, labels)
    return largest_component(foreground)


def component_filter(
    mask: np.ndarray,
    rule: dict[str, object],
    heart_distance: np.ndarray | None = None,
) -> np.ndarray:
    if not np.any(mask):
        return mask

    structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    cc, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return mask

    sizes = np.bincount(cc.ravel())
    component_ids = np.arange(1, num + 1)
    min_component_size = int(rule.get("min_component_size", 0) or 0)
    keep_ids = component_ids[sizes[component_ids] >= min_component_size]

    max_distance = rule.get("max_distance_to_heart_mm")
    if max_distance is not None and heart_distance is not None and keep_ids.size:
        distances = np.atleast_1d(ndimage.minimum(heart_distance, labels=cc, index=keep_ids)).astype(float)
        keep_ids = keep_ids[distances <= float(max_distance)]

    if not keep_ids.size:
        return np.zeros_like(mask, dtype=bool)

    keep_top_k = rule.get("keep_top_k")
    keep_largest = bool(rule.get("keep_largest", False))
    if keep_top_k is not None:
        top_k = int(keep_top_k)
    elif keep_largest:
        top_k = 1
    else:
        top_k = 0

    if top_k > 0 and keep_ids.size > top_k:
        order = np.argsort(sizes[keep_ids])[::-1]
        keep_ids = keep_ids[order[:top_k]]

    return np.isin(cc, keep_ids)


def postprocess_label_array(
    seg: np.ndarray,
    labels: Iterable[int],
    keep_largest: bool = True,
    min_component_size: int = 0,
    class_rules: dict[int, dict[str, object]] | None = None,
    spacing: Iterable[float] | None = None,
) -> np.ndarray:
    cleaned = seg.copy()
    label_values = [int(label) for label in labels]
    heart_distance = None
    if class_rules is not None:
        heart_body = largest_foreground_component(seg, label_values)
        if np.any(heart_body):
            heart_distance = ndimage.distance_transform_edt(
                ~heart_body,
                sampling=normalize_spacing(spacing, seg.ndim),
            )

    for label in label_values:
        mask = seg == int(label)
        if class_rules is None:
            if keep_largest:
                mask = largest_component(mask)
            mask = remove_small_components(mask, min_component_size)
        else:
            rule = class_rules.get(int(label), {})
            mask = component_filter(mask, rule, heart_distance=heart_distance)
            if bool(rule.get("fill_holes", False)):
                filled = ndimage.binary_fill_holes(mask)
                mask = mask | (filled & (seg == 0))
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


def postprocess_file(
    input_path: Path,
    output_path: Path,
    labels: list[int],
    keep_largest: bool,
    min_component_size: int,
    class_rules: dict[int, dict[str, object]] | None = None,
) -> None:
    sitk = load_sitk()
    image = sitk.ReadImage(str(input_path))
    arr = sitk.GetArrayFromImage(image)
    spacing_zyx = tuple(reversed(image.GetSpacing()))
    cleaned = postprocess_label_array(
        arr,
        labels,
        keep_largest=keep_largest,
        min_component_size=min_component_size,
        class_rules=class_rules,
        spacing=spacing_zyx,
    )

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
    parser.add_argument("--preset", choices=["legacy", "class-aware-hd"], default="legacy",
                        help="legacy keeps the historical largest-component cleanup; class-aware-hd uses per-class HD rules.")
    parser.add_argument("--rules-json", type=Path, default=None,
                        help="Optional JSON rule overrides keyed by label value in the selected label space.")
    parser.add_argument("--wholeheart-distance-mm", type=float, default=25.0,
                        help="Maximum distance from the whole-heart body for AO/PA components in class-aware-hd mode.")
    parser.add_argument("--vessel-min-component-size", type=int, default=20,
                        help="Minimum AO/PA component size in voxels for class-aware-hd mode.")
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
    class_rules = None
    if args.preset == "class-aware-hd":
        class_rules = build_class_aware_hd_rules(
            labels=labels,
            max_distance_to_heart_mm=args.wholeheart_distance_mm,
            vessel_min_component_size=args.vessel_min_component_size,
        )
        class_rules = merge_rule_overrides(class_rules, load_rule_overrides(args.rules_json))
    elif args.rules_json is not None:
        class_rules = load_rule_overrides(args.rules_json)

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
            class_rules=class_rules,
        )
        print(f"Postprocessed {input_path.name} -> {output_path}")

    print(f"Postprocessed {len(files)} files.")


if __name__ == "__main__":
    main()
