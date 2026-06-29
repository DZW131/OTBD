from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def finite_mean(values) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    return mean(finite) if finite else math.nan


def center_for_case(case_name: str) -> str:
    if case_name.startswith("Case3"):
        return "MR_C/D"
    if case_name.startswith("Case5"):
        return "MR_E"
    if case_name.startswith("CaseCT"):
        return "CT_val"
    if case_name.startswith("CaseMR"):
        return "MR_val"
    return "unknown"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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


def write_markdown(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize case-level metric CSV by inferred CARE center group.")
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_rows = read_rows(args.case_csv)
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        case_name = Path(row["case"]).name
        key = (
            row["run"],
            str(row["fold"]),
            center_for_case(case_name),
            row["class"],
        )
        grouped[key].append(row)

    rows: list[dict[str, object]] = []
    for (run, fold, center, class_name), items in sorted(grouped.items()):
        rows.append(
            {
                "run": run,
                "fold": fold,
                "center": center,
                "class": class_name,
                "n_cases": len({Path(item["case"]).name for item in items}),
                "dsc": finite_mean(float(item["dsc"]) for item in items),
                "hd_mm": finite_mean(float(item["hd_mm"]) for item in items),
                "assd_mm": finite_mean(float(item["assd_mm"]) for item in items),
            }
        )

    fields = ["run", "fold", "center", "class", "n_cases", "dsc", "hd_mm", "assd_mm"]
    write_csv(args.output_csv, rows, fields)
    if args.output_md:
        write_markdown(args.output_md, rows, fields)
    print(f"Wrote center summary: {args.output_csv}")


if __name__ == "__main__":
    main()
