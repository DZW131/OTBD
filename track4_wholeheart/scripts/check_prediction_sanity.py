from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .common import load_label_config, output_label_name_from_image, read_json
except ImportError:
    from common import load_label_config, output_label_name_from_image, read_json


@dataclass(frozen=True)
class GeometrySignature:
    size: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]


@dataclass(frozen=True)
class SanityIssue:
    severity: str
    code: str
    message: str
    path: str = ""


def load_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required to sanity-check NIfTI predictions.") from exc
    return sitk


def geometry_from_image(image) -> GeometrySignature:
    return GeometrySignature(
        size=tuple(int(v) for v in image.GetSize()),
        spacing=tuple(float(v) for v in image.GetSpacing()),
        origin=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )


def _allclose(left: Iterable[float], right: Iterable[float], atol: float = 1e-5) -> bool:
    return np.allclose(tuple(left), tuple(right), rtol=0.0, atol=atol)


def compare_geometry(
    input_geometry: GeometrySignature,
    pred_geometry: GeometrySignature,
    path: str = "",
) -> list[SanityIssue]:
    issues: list[SanityIssue] = []
    if input_geometry.size != pred_geometry.size:
        issues.append(
            SanityIssue(
                "error",
                "shape_mismatch",
                f"prediction size {pred_geometry.size} != input size {input_geometry.size}",
                path,
            )
        )
    if not _allclose(input_geometry.spacing, pred_geometry.spacing):
        issues.append(
            SanityIssue(
                "error",
                "spacing_mismatch",
                f"prediction spacing {pred_geometry.spacing} != input spacing {input_geometry.spacing}",
                path,
            )
        )
    if not _allclose(input_geometry.origin, pred_geometry.origin):
        issues.append(
            SanityIssue(
                "error",
                "origin_mismatch",
                f"prediction origin {pred_geometry.origin} != input origin {input_geometry.origin}",
                path,
            )
        )
    if not _allclose(input_geometry.direction, pred_geometry.direction):
        issues.append(
            SanityIssue(
                "error",
                "direction_mismatch",
                "prediction direction does not match input direction",
                path,
            )
        )
    return issues


def check_array_labels(
    pred: np.ndarray,
    allowed_labels: set[int],
    required_labels: set[int],
    path: str = "",
    missing_class_severity: str = "warning",
) -> list[SanityIssue]:
    issues: list[SanityIssue] = []
    if not np.issubdtype(pred.dtype, np.integer):
        issues.append(
            SanityIssue(
                "error",
                "non_integer_dtype",
                f"prediction dtype {pred.dtype} is not an integer label dtype",
                path,
            )
        )

    unique = {int(v) for v in np.unique(pred)}
    unexpected = sorted(unique - set(allowed_labels))
    if unexpected:
        issues.append(
            SanityIssue(
                "error",
                "unexpected_label",
                f"unexpected label values: {unexpected}; allowed: {sorted(allowed_labels)}",
                path,
            )
        )

    foreground = unique - {0}
    if not foreground:
        issues.append(SanityIssue("error", "empty_prediction", "prediction contains no foreground labels", path))

    missing = sorted(set(required_labels) - unique)
    if missing and missing_class_severity != "ignore":
        issues.append(
            SanityIssue(
                missing_class_severity,
                "missing_class",
                f"required class labels have zero voxels: {missing}",
                path,
            )
        )
    return issues


def labels_for_space(cfg: dict, label_space: str) -> tuple[set[int], set[int]]:
    if label_space == "train":
        allowed = {int(k) for k in cfg["train_to_official"].keys()}
    elif label_space == "official":
        allowed = {int(v) for v in cfg["official_label_values"].values()}
    else:
        raise ValueError(f"Unsupported label space: {label_space}")
    return allowed, allowed - {0}


def input_path_for_record(record: dict, input_dir: Path) -> Path | None:
    for key in ("source_image", "nnunet_image"):
        value = record.get(key)
        if value and Path(value).exists():
            return Path(value)

    case_id = record.get("case_id")
    if case_id:
        candidates = [
            input_dir / f"{case_id}_image.nii.gz",
            input_dir / f"{case_id}_0000.nii.gz",
            input_dir / f"{case_id}.nii.gz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def expected_pairs_from_mapping(input_dir: Path, pred_dir: Path, mapping_json: Path) -> tuple[list[tuple[Path, Path]], list[SanityIssue]]:
    mapping = read_json(mapping_json)
    records = mapping.get("validation", [])
    issues: list[SanityIssue] = []
    pairs: list[tuple[Path, Path]] = []
    if not records:
        issues.append(SanityIssue("error", "mapping_empty", f"no validation records found in {mapping_json}"))
        return pairs, issues

    for record in records:
        input_path = input_path_for_record(record, input_dir)
        output_label_name = record.get("output_label_name")
        if input_path is None:
            issues.append(
                SanityIssue(
                    "error",
                    "missing_input",
                    f"could not resolve input image for mapping record: {record}",
                )
            )
            continue
        if not output_label_name:
            issues.append(
                SanityIssue(
                    "error",
                    "missing_output_name",
                    f"mapping record has no output_label_name: {record}",
                    str(input_path),
                )
            )
            continue
        pairs.append((input_path, pred_dir / output_label_name))
    return pairs, issues


def expected_pairs_from_input_dir(input_dir: Path, pred_dir: Path) -> tuple[list[tuple[Path, Path]], list[SanityIssue]]:
    input_paths = sorted(
        path
        for path in input_dir.glob("*.nii.gz")
        if "_label.nii.gz" not in path.name and "_seg.nii.gz" not in path.name
    )
    if not input_paths:
        return [], [SanityIssue("error", "no_inputs", f"no input .nii.gz files found in {input_dir}")]
    return [(path, pred_dir / output_label_name_from_image(path)) for path in input_paths], []


def check_prediction_file(
    input_path: Path,
    pred_path: Path,
    allowed_labels: set[int],
    required_labels: set[int],
    missing_class_severity: str,
) -> list[SanityIssue]:
    if not pred_path.exists():
        return [SanityIssue("error", "missing_prediction", f"missing prediction for {input_path.name}", str(pred_path))]

    sitk = load_sitk()
    input_image = sitk.ReadImage(str(input_path))
    pred_image = sitk.ReadImage(str(pred_path))
    pred_array = sitk.GetArrayFromImage(pred_image)

    issues = compare_geometry(geometry_from_image(input_image), geometry_from_image(pred_image), str(pred_path))
    issues.extend(
        check_array_labels(
            pred_array,
            allowed_labels=allowed_labels,
            required_labels=required_labels,
            path=str(pred_path),
            missing_class_severity=missing_class_severity,
        )
    )
    return issues


def check_predictions(
    input_dir: Path,
    pred_dir: Path,
    allowed_labels: set[int],
    required_labels: set[int],
    mapping_json: Path | None = None,
    missing_class_severity: str = "warning",
) -> tuple[int, list[SanityIssue]]:
    if mapping_json:
        pairs, issues = expected_pairs_from_mapping(input_dir, pred_dir, mapping_json)
    else:
        pairs, issues = expected_pairs_from_input_dir(input_dir, pred_dir)

    expected_names = {pred_path.name for _, pred_path in pairs}
    extra_predictions = sorted(path for path in pred_dir.glob("*.nii.gz") if path.name not in expected_names)
    for path in extra_predictions:
        issues.append(SanityIssue("warning", "extra_prediction", "prediction file is not expected from inputs", str(path)))

    checked = 0
    for input_path, pred_path in pairs:
        issues.extend(
            check_prediction_file(
                input_path,
                pred_path,
                allowed_labels=allowed_labels,
                required_labels=required_labels,
                missing_class_severity=missing_class_severity,
            )
        )
        checked += 1
    return checked, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check CARE whole-heart prediction NIfTI files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input images.")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing prediction labels.")
    parser.add_argument("--mapping-json", type=Path, default=None, help="Optional conversion_mapping.json.")
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--label-space", choices=["official", "train"], default="official")
    parser.add_argument("--missing-class-severity", choices=["warning", "error", "ignore"], default="warning")
    parser.add_argument("--json-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_label_config(args.label_config) if args.label_config else load_label_config()
    allowed_labels, required_labels = labels_for_space(cfg, args.label_space)
    checked, issues = check_predictions(
        input_dir=args.input_dir,
        pred_dir=args.pred_dir,
        mapping_json=args.mapping_json,
        allowed_labels=allowed_labels,
        required_labels=required_labels,
        missing_class_severity=args.missing_class_severity,
    )

    for issue in issues:
        location = f" {issue.path}" if issue.path else ""
        print(f"{issue.severity.upper()} {issue.code}{location}: {issue.message}")

    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    print(f"Checked {checked} predictions: {errors} errors, {warnings} warnings.")

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(
                {
                    "checked": checked,
                    "errors": errors,
                    "warnings": warnings,
                    "issues": [asdict(issue) for issue in issues],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
