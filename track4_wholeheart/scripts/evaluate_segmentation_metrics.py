from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

try:
    from .common import load_label_config, mapping_arrays
except ImportError:
    from common import load_label_config, mapping_arrays


DEFAULT_LABELS = [1, 2, 3, 4, 5, 6, 7]


@dataclass(frozen=True)
class BinaryMetrics:
    dsc: float
    hd_mm: float
    assd_mm: float


@dataclass(frozen=True)
class RunSpec:
    name: str
    root: Path


def dice_score(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    denom = int(pred.sum() + ref.sum())
    if denom == 0:
        return math.nan
    return float(2 * np.logical_and(pred, ref).sum() / denom)


def surface_distances_mm(pred: np.ndarray, ref: np.ndarray, spacing_xyz: Sequence[float]) -> np.ndarray:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    if pred.sum() == 0 and ref.sum() == 0:
        return np.array([], dtype=np.float64)
    if pred.sum() == 0 or ref.sum() == 0:
        return np.array([math.inf], dtype=np.float64)

    pred_surface = pred ^ binary_erosion(pred)
    ref_surface = ref ^ binary_erosion(ref)
    spacing_zyx = tuple(float(v) for v in spacing_xyz[::-1])

    ref_distance = distance_transform_edt(~ref_surface, sampling=spacing_zyx)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing_zyx)
    return np.concatenate([ref_distance[pred_surface], pred_distance[ref_surface]]).astype(np.float64)


def crop_to_union_foreground(
    pred: np.ndarray,
    ref: np.ndarray,
    padding: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    union = np.logical_or(pred, ref)
    if not union.any():
        return pred, ref

    coords = np.argwhere(union)
    lower = np.maximum(coords.min(axis=0) - int(padding), 0)
    upper = np.minimum(coords.max(axis=0) + int(padding) + 1, pred.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    return pred[slices], ref[slices]


def compute_binary_metrics(pred: np.ndarray, ref: np.ndarray, spacing_xyz: Sequence[float]) -> BinaryMetrics:
    pred, ref = crop_to_union_foreground(pred, ref)
    distances = surface_distances_mm(pred, ref, spacing_xyz)
    if distances.size == 0:
        hd = 0.0
        assd = 0.0
    elif np.isinf(distances).any():
        hd = math.inf
        assd = math.inf
    else:
        hd = float(np.max(distances))
        assd = float(np.mean(distances))
    return BinaryMetrics(dsc=dice_score(pred, ref), hd_mm=hd, assd_mm=assd)


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    return mean(finite) if finite else math.nan


def class_name_map(cfg: dict) -> dict[int, str]:
    return {int(value): name for name, value in cfg["labels"].items() if int(value) != 0}


def parse_run_specs(values: Sequence[str]) -> list[RunSpec]:
    specs = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --run value '{value}'. Use name=/path/to/results_root.")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise SystemExit(f"Invalid --run value '{value}': run name is empty.")
        specs.append(RunSpec(name=name, root=Path(path)))
    return specs


def validation_dir(root: Path, fold: int) -> Path:
    candidates = [
        root / f"fold_{fold}" / "validation",
        root / f"fold_{fold}" / "validation_raw",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: object) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_f):
        return "nan"
    if math.isinf(value_f):
        return "inf"
    return f"{value_f:.4f}"


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    headers = ["run", "fold", "class", "n_cases", "dsc", "hd_mm", "assd_mm"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [format_metric(row.get(header)) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required to read NIfTI files.") from exc
    return sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSC, HD, and ASSD for nnU-Net validation folds.")
    parser.add_argument("--run", action="append", required=True,
                        help="Run spec in the form name=/path/to/nnUNetTrainer__plans__config root. Repeatable.")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Ground-truth directory, usually nnUNet_preprocessed/<Dataset>/gt_segmentations.")
    parser.add_argument("--label-space", choices=["train", "official"], default="train")
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_label_config(args.label_config) if args.label_config else load_label_config()
    label_names = class_name_map(cfg)
    labels = DEFAULT_LABELS
    metric_labels = labels
    if args.label_space == "official":
        _, train_to_official = mapping_arrays(cfg)
        metric_labels = [train_to_official[label] for label in labels]
        label_names = {train_to_official[label]: label_names[label] for label in labels}

    sitk = load_sitk()
    case_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spec in parse_run_specs(args.run):
        for fold in args.folds:
            pred_dir = validation_dir(spec.root, fold)
            pred_files = sorted(pred_dir.glob("*.nii.gz"))
            print(f"{spec.name} fold {fold}: {len(pred_files)} predictions from {pred_dir}", flush=True)
            if not pred_files:
                continue

            fold_rows: list[dict[str, object]] = []
            for case_idx, pred_path in enumerate(pred_files, start=1):
                gt_path = args.gt_dir / pred_path.name
                if not gt_path.exists():
                    print(f"missing gt: {gt_path}")
                    continue
                print(f"  [{case_idx}/{len(pred_files)}] {pred_path.name}", flush=True)
                pred_img = sitk.ReadImage(str(pred_path))
                ref_img = sitk.ReadImage(str(gt_path))
                pred = sitk.GetArrayFromImage(pred_img)
                ref = sitk.GetArrayFromImage(ref_img)
                spacing = pred_img.GetSpacing()

                for label in metric_labels:
                    metrics = compute_binary_metrics(pred == label, ref == label, spacing)
                    row = {
                        "run": spec.name,
                        "fold": fold,
                        "case": pred_path.name,
                        "label": label,
                        "class": label_names.get(label, str(label)),
                        "dsc": metrics.dsc,
                        "hd_mm": metrics.hd_mm,
                        "assd_mm": metrics.assd_mm,
                    }
                    case_rows.append(row)
                    fold_rows.append(row)

            for label in metric_labels:
                rows = [row for row in fold_rows if row["label"] == label]
                if not rows:
                    continue
                summary_rows.append({
                    "run": spec.name,
                    "fold": fold,
                    "label": label,
                    "class": label_names.get(label, str(label)),
                    "n_cases": len(rows),
                    "dsc": finite_mean(row["dsc"] for row in rows),
                    "hd_mm": finite_mean(row["hd_mm"] for row in rows),
                    "assd_mm": finite_mean(row["assd_mm"] for row in rows),
                })

    if not case_rows:
        raise SystemExit("No case metrics were computed. Check --run roots, --folds, and --gt-dir.")

    write_csv(args.case_csv, case_rows, ["run", "fold", "case", "label", "class", "dsc", "hd_mm", "assd_mm"])
    write_csv(args.summary_csv, summary_rows, ["run", "fold", "label", "class", "n_cases", "dsc", "hd_mm", "assd_mm"])
    if args.summary_md:
        write_markdown(args.summary_md, summary_rows)

    print(f"wrote case metrics: {args.case_csv}")
    print(f"wrote summary metrics: {args.summary_csv}")


if __name__ == "__main__":
    main()
