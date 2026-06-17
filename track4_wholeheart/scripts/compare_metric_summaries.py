from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


METRICS = ("dsc", "hd_mm", "assd_mm")
OUTPUT_FIELDS = [
    "run",
    "fold",
    "label",
    "class",
    "n_cases",
    "baseline_run",
    "baseline_n_cases",
    "dsc",
    "baseline_dsc",
    "dsc_delta",
    "hd_mm",
    "baseline_hd_mm",
    "hd_mm_delta",
    "assd_mm",
    "baseline_assd_mm",
    "assd_mm_delta",
]


def parse_float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_int(value: object) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if not math.isnan(float(value)) and not math.isinf(float(value))]
    return mean(finite) if finite else math.nan


def rounded(value: float, digits: int = 6) -> float:
    if math.isnan(value) or math.isinf(value):
        return value
    return round(float(value), digits)


def row_key(row: dict[str, object]) -> tuple[str, str]:
    label = str(row.get("label") or "")
    class_name = str(row.get("class") or label)
    return label, class_name


def summarize_by_run_class(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        label, class_name = row_key(row)
        grouped[(str(row.get("run")), label, class_name)].append(row)

    summarized = []
    for (run, label, class_name), items in sorted(grouped.items()):
        summary: dict[str, object] = {
            "run": run,
            "fold": "mean",
            "label": label,
            "class": class_name,
            "n_cases": sum(parse_int(item.get("n_cases")) for item in items),
        }
        for metric in METRICS:
            summary[metric] = rounded(finite_mean(parse_float(item.get(metric)) for item in items))
        summarized.append(summary)
    return summarized


def build_comparison_row(
    row: dict[str, object],
    baseline_row: dict[str, object],
    baseline_run: str,
) -> dict[str, object]:
    compared = {
        "run": row.get("run", ""),
        "fold": row.get("fold", ""),
        "label": row.get("label", ""),
        "class": row.get("class", ""),
        "n_cases": parse_int(row.get("n_cases")),
        "baseline_run": baseline_run,
        "baseline_n_cases": parse_int(baseline_row.get("n_cases")),
    }
    for metric in METRICS:
        value = rounded(parse_float(row.get(metric)))
        baseline_value = rounded(parse_float(baseline_row.get(metric)))
        compared[metric] = value
        compared[f"baseline_{metric}"] = baseline_value
        compared[f"{metric}_delta"] = rounded(value - baseline_value)
    return compared


def compare_rows_at_same_fold(
    rows: list[dict[str, object]],
    baseline_run: str,
) -> list[dict[str, object]]:
    baseline_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        if str(row.get("run")) != baseline_run:
            continue
        label, class_name = row_key(row)
        baseline_by_key[(str(row.get("fold")), label, class_name)] = row

    compared = []
    for row in rows:
        if str(row.get("run")) == baseline_run:
            continue
        label, class_name = row_key(row)
        baseline_row = baseline_by_key.get((str(row.get("fold")), label, class_name))
        if baseline_row is None:
            continue
        compared.append(build_comparison_row(row, baseline_row, baseline_run))
    return compared


def compare_summary_rows(
    rows: list[dict[str, object]],
    baseline_run: str,
    include_folds: bool = False,
) -> list[dict[str, object]]:
    fold_rows = compare_rows_at_same_fold(rows, baseline_run)
    mean_rows = compare_rows_at_same_fold(summarize_by_run_class(rows), baseline_run)
    return fold_rows + mean_rows if include_folds else mean_rows


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: object) -> str:
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
    headers = [
        "run",
        "fold",
        "class",
        "dsc",
        "baseline_dsc",
        "dsc_delta",
        "hd_mm",
        "baseline_hd_mm",
        "hd_mm_delta",
        "assd_mm",
        "baseline_assd_mm",
        "assd_mm_delta",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare evaluate_segmentation_metrics.py summary CSV rows against a fixed baseline run. "
            "Output remains class-wise so LV/RV/LA/RA/Myo/AO/PA deltas are easy to inspect."
        )
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--include-folds", action="store_true", help="Also include per-fold rows before mean rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compare_summary_rows(
        read_csv(args.summary_csv),
        baseline_run=args.baseline_run,
        include_folds=args.include_folds,
    )
    if not rows:
        raise SystemExit(f"No comparable rows found for baseline run {args.baseline_run!r}.")
    write_csv(args.output_csv, rows)
    if args.output_md:
        write_markdown(args.output_md, rows)
    print(f"wrote baseline comparison: {args.output_csv}")


if __name__ == "__main__":
    main()
