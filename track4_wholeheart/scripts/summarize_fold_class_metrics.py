from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np

try:
    from .common import load_label_config, mapping_arrays
except ImportError:
    from common import load_label_config, mapping_arrays


DEFAULT_LABELS = [1, 2, 3, 4, 5, 6, 7]


@dataclass
class FoldLogMetrics:
    path: Path
    fold: int | None = None
    mean_validation_dice: float | None = None
    best_ema_pseudo_dice: float | None = None
    final_pseudo_dice: dict[int, float] | None = None


def dice_per_label(pred: np.ndarray, ref: np.ndarray, labels: Iterable[int]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for label in labels:
        pred_mask = pred == int(label)
        ref_mask = ref == int(label)
        denom = int(pred_mask.sum() + ref_mask.sum())
        if denom == 0:
            scores[int(label)] = math.nan
        else:
            scores[int(label)] = float(2 * np.logical_and(pred_mask, ref_mask).sum() / denom)
    return scores


def worst_class(scores: dict[int, float], label_names: dict[int, str]) -> tuple[str, float]:
    finite = [(label, value) for label, value in scores.items() if not math.isnan(value)]
    if not finite:
        return ("NA", math.nan)
    label, value = min(finite, key=lambda item: item[1])
    return (label_names.get(label, str(label)), value)


def safe_mean(values: Iterable[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return mean(finite) if finite else math.nan


def metrics_from_nnunet_summary(summary: dict, labels: Iterable[int]) -> dict[int, float]:
    mean_metrics = summary.get("mean", {})
    scores = {}
    for label in labels:
        item = mean_metrics.get(str(label), {})
        value = item.get("Dice")
        scores[int(label)] = float(value) if value is not None else math.nan
    return scores


def parse_training_log_lines(lines: Iterable[str], path: Path) -> FoldLogMetrics:
    parsed = FoldLogMetrics(path=path, final_pseudo_dice={})
    final_values: list[float] = []
    for line in lines:
        fold_match = re.search(r"Desired fold for training:\s*(\d+)", line)
        if fold_match:
            parsed.fold = int(fold_match.group(1))

        mean_match = re.search(r"Mean Validation Dice:\s*([0-9.]+)", line)
        if mean_match:
            parsed.mean_validation_dice = float(mean_match.group(1))

        best_match = re.search(r"Yayy! New best EMA pseudo Dice:\s*([0-9.]+)", line)
        if best_match:
            parsed.best_ema_pseudo_dice = float(best_match.group(1))

        pseudo_match = re.search(r"Pseudo dice\s+\[(.*)\]", line)
        if pseudo_match:
            final_values = [
                float(m.group(1) or m.group(2))
                for m in re.finditer(
                    r"np\.float32\(([-+]?[0-9]*\.?[0-9]+)\)|(?<![A-Za-z0-9_.])([-+]?[0-9]*\.?[0-9]+)",
                    pseudo_match.group(1),
                )
            ]

    parsed.final_pseudo_dice = {idx + 1: value for idx, value in enumerate(final_values)}
    return parsed


def parse_training_log(path: Path) -> FoldLogMetrics:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return parse_training_log_lines(f, path)


def scan_fold_summaries(results_root: Path, labels: Iterable[int]) -> dict[int, dict[int, float]]:
    found: dict[int, dict[int, float]] = {}
    for fold_dir in sorted(results_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        candidates = [
            fold_dir / "validation" / "summary.json",
            fold_dir / "validation_raw" / "summary.json",
            fold_dir / "summary.json",
        ]
        for summary_path in candidates:
            if summary_path.exists():
                with summary_path.open("r", encoding="utf-8") as f:
                    found[fold] = metrics_from_nnunet_summary(json.load(f), labels)
                break
    return found


def compute_folder_metrics(pred_dir: Path, ref_dir: Path, labels: Iterable[int]) -> dict[str, dict[int, float]]:
    sitk = load_sitk()
    rows = {}
    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        ref_path = ref_dir / pred_path.name
        if not ref_path.exists():
            continue
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path)))
        ref = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_path)))
        rows[pred_path.name] = dice_per_label(pred, ref, labels)
    return rows


def load_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required to compute Dice from NIfTI files.") from exc
    return sitk


def class_name_map(cfg: dict) -> dict[int, str]:
    return {int(value): name for name, value in cfg["labels"].items() if int(value) != 0}


def write_csv(path: Path, rows: list[dict[str, object]], label_names: dict[int, str], labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["modality", "fold", "source", "mean_validation_dice", "best_ema_pseudo_dice", "worst_class",
              "worst_dice"] + [label_names[label] for label in labels]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], label_names: dict[int, str], labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["modality", "fold", "source", "mean", "worst"] + [label_names[label] for label in labels]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [
            str(row["modality"]),
            str(row["fold"]),
            str(row["source"]),
            format_optional(row.get("mean_validation_dice")),
            f"{row['worst_class']} {format_optional(row.get('worst_dice'))}",
        ]
        values.extend(format_optional(row.get(label_names[label])) for label in labels)
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: object) -> str:
    if value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_f):
        return "nan"
    return f"{value_f:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fold-level Dice by whole-heart class.")
    parser.add_argument("--modality", choices=["ct", "mr"], required=True)
    parser.add_argument("--results-root", type=Path,
                        help="nnU-Net trainer output root containing fold_0..fold_4.")
    parser.add_argument("--training-log-dir", type=Path,
                        help="Directory containing training_log_*.txt files; searched recursively.")
    parser.add_argument("--pred-dir", type=Path, help="Prediction directory for direct Dice computation.")
    parser.add_argument("--ref-dir", type=Path, help="Reference label directory for direct Dice computation.")
    parser.add_argument("--label-space", choices=["train", "official"], default="train")
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_label_config(args.label_config) if args.label_config else load_label_config()
    labels = DEFAULT_LABELS
    label_names = class_name_map(cfg)
    metric_labels = labels
    if args.label_space == "official":
        _, train_to_official = mapping_arrays(cfg)
        metric_labels = [train_to_official[label] for label in labels]
        label_names = {train_to_official[label]: label_names[label] for label in labels}

    rows: list[dict[str, object]] = []

    summary_scores = scan_fold_summaries(args.results_root, metric_labels) if args.results_root else {}
    logs = []
    if args.training_log_dir:
        logs = [parse_training_log(path) for path in sorted(args.training_log_dir.rglob("training_log_*.txt"))]

    by_fold_log = {log.fold: log for log in logs if log.fold is not None}
    for fold in sorted(set(summary_scores) | set(by_fold_log)):
        scores = summary_scores.get(fold)
        source = "summary"
        if scores is None and fold in by_fold_log:
            scores = by_fold_log[fold].final_pseudo_dice or {}
            source = "training_log_final_pseudo"
        if not scores:
            continue
        worst_name, worst_value = worst_class(scores, label_names)
        row: dict[str, object] = {
            "modality": args.modality,
            "fold": fold,
            "source": source,
            "mean_validation_dice": by_fold_log.get(fold).mean_validation_dice if fold in by_fold_log else "",
            "best_ema_pseudo_dice": by_fold_log.get(fold).best_ema_pseudo_dice if fold in by_fold_log else "",
            "worst_class": worst_name,
            "worst_dice": worst_value,
        }
        for label, value in scores.items():
            row[label_names.get(label, str(label))] = value
        rows.append(row)

    if args.pred_dir and args.ref_dir:
        case_scores = compute_folder_metrics(args.pred_dir, args.ref_dir, metric_labels)
        if case_scores:
            mean_scores = {
                label: safe_mean(scores[label] for scores in case_scores.values())
                for label in metric_labels
            }
            worst_name, worst_value = worst_class(mean_scores, label_names)
            row = {
                "modality": args.modality,
                "fold": "folder",
                "source": "pred_vs_ref",
                "mean_validation_dice": safe_mean(mean_scores.values()),
                "best_ema_pseudo_dice": "",
                "worst_class": worst_name,
                "worst_dice": worst_value,
            }
            for label, value in mean_scores.items():
                row[label_names.get(label, str(label))] = value
            rows.append(row)

    if not rows:
        raise SystemExit("No metrics found. Provide --results-root, --training-log-dir, or --pred-dir/--ref-dir.")

    write_csv(args.output_csv, rows, label_names, metric_labels)
    if args.output_md:
        write_markdown(args.output_md, rows, label_names, metric_labels)
    print(f"Wrote {len(rows)} metric rows to {args.output_csv}")


if __name__ == "__main__":
    main()
