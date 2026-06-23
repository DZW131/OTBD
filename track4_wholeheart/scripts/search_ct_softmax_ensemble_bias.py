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
    from .common import load_label_config
    from .postprocess_ct_aopa_soft import load_case_probability
    from .postprocess_predictions import postprocess_label_array
except ImportError:
    from common import load_label_config
    from postprocess_ct_aopa_soft import load_case_probability
    from postprocess_predictions import postprocess_label_array


LABELS = (1, 2, 3, 4, 5, 6, 7)
CLASS_TO_LABEL = {
    "LV": 1,
    "RV": 2,
    "LA": 3,
    "RA": 4,
    "Myo": 5,
    "AO": 6,
    "PA": 7,
}


@dataclass(frozen=True)
class ProbRun:
    name: str
    root: Path


@dataclass(frozen=True)
class Candidate:
    name: str
    weights: dict[str, float]
    bias: dict[int, float]


@dataclass
class MetricAccumulator:
    dsc: list[float]
    hd95_mm: list[float]
    hd_mm: list[float]
    assd_mm: list[float]

    @classmethod
    def empty(cls) -> "MetricAccumulator":
        return cls(dsc=[], hd95_mm=[], hd_mm=[], assd_mm=[])


def finite_mean(values: Iterable[float]) -> float:
    finite = [float(v) for v in values if not math.isnan(float(v)) and not math.isinf(float(v))]
    return mean(finite) if finite else math.nan


def format_float(value: float) -> str:
    if value == 0:
        return "0"
    text = f"{value:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return text


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_prob_run(value: str) -> ProbRun:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Invalid prob run {value!r}; use name=/path")
    name, root = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Probability run name cannot be empty")
    return ProbRun(name=name, root=Path(root))


def parse_weight_spec(value: str, run_names: set[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight item {item!r}; use run=weight")
        name, weight = item.split("=", 1)
        name = name.strip()
        if name not in run_names:
            raise ValueError(f"Weight references unknown probability run {name!r}; available={sorted(run_names)}")
        weights[name] = float(weight)

    if not weights:
        raise ValueError(f"Empty weight spec {value!r}")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError(f"Weights must be non-negative: {weights}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(f"Weights must sum to > 0: {weights}")
    return {name: weight / total for name, weight in sorted(weights.items())}


def parse_bias_spec(value: str) -> dict[int, float]:
    value = value.strip()
    if not value or value.lower() in {"none", "0"}:
        return {}

    bias: dict[int, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid bias item {item!r}; use class=value")
        class_name, bias_value = item.split("=", 1)
        class_name = class_name.strip()
        if class_name not in CLASS_TO_LABEL:
            raise ValueError(f"Unknown class {class_name!r}; available={sorted(CLASS_TO_LABEL)}")
        bias[CLASS_TO_LABEL[class_name]] = float(bias_value)
    return bias


def weight_name(weights: dict[str, float]) -> str:
    return "_".join(f"{name}{weight:.2f}".replace(".", "p") for name, weight in weights.items())


def bias_name(bias: dict[int, float], label_names: dict[int, str]) -> str:
    if not bias:
        return "bnone"
    return "_".join(f"{label_names[label]}{format_float(value)}" for label, value in sorted(bias.items()))


def build_default_weight_specs(prob_runs: Sequence[ProbRun]) -> list[dict[str, float]]:
    names = [run.name for run in prob_runs]
    if len(names) == 1:
        return [{names[0]: 1.0}]
    if set(names) >= {"final", "best"}:
        return [
            {"final": 1.0},
            {"best": 1.0},
            {"final": 0.5, "best": 0.5},
            {"final": 0.7, "best": 0.3},
            {"final": 0.3, "best": 0.7},
        ]
    return [{name: 1.0 / len(names) for name in names}]


def build_bias_specs(args: argparse.Namespace) -> list[dict[int, float]]:
    specs: list[dict[int, float]] = [{}]
    for value in args.bias_spec or []:
        parsed = parse_bias_spec(value)
        if parsed not in specs:
            specs.append(parsed)

    classes = [CLASS_TO_LABEL[name] for name in parse_class_names(args.bias_classes)]
    values = [value for value in parse_float_list(args.bias_values) if value != 0]
    if args.bias_mode == "none":
        return specs
    if args.bias_mode == "one-at-a-time":
        for label in classes:
            for value in values:
                item = {label: value}
                if item not in specs:
                    specs.append(item)
        return specs
    if args.bias_mode == "grid":
        # Keep the full grid explicit because it can become expensive quickly.
        import itertools

        for combo in itertools.product(values + [0.0], repeat=len(classes)):
            item = {label: value for label, value in zip(classes, combo) if value != 0}
            if item not in specs:
                specs.append(item)
        return specs
    raise ValueError(f"Unsupported bias mode: {args.bias_mode}")


def parse_class_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    for name in names:
        if name not in CLASS_TO_LABEL:
            raise ValueError(f"Unknown class {name!r}; available={sorted(CLASS_TO_LABEL)}")
    return names


def build_candidates(args: argparse.Namespace, prob_runs: Sequence[ProbRun], label_names: dict[int, str]) -> list[Candidate]:
    run_names = {run.name for run in prob_runs}
    if args.weight_spec:
        weight_specs = [parse_weight_spec(value, run_names) for value in args.weight_spec]
    else:
        weight_specs = build_default_weight_specs(prob_runs)

    bias_specs = build_bias_specs(args)
    candidates: list[Candidate] = []
    seen = set()
    for weights in weight_specs:
        for bias in bias_specs:
            name = f"{weight_name(weights)}_{bias_name(bias, label_names)}_legacy"
            key = (tuple(sorted(weights.items())), tuple(sorted(bias.items())))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(Candidate(name=name, weights=weights, bias=bias))
    return candidates


def load_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required for CT softmax search.") from exc
    return sitk


def case_name_from_npz(path: Path) -> str:
    return path.with_suffix(".nii.gz").name


def probability_path(run: ProbRun, fold: int, case_name: str) -> Path:
    stem = case_name[:-7] if case_name.endswith(".nii.gz") else Path(case_name).stem
    return run.root / f"fold_{fold}" / "validation" / f"{stem}.npz"


def reference_prediction_path(run: ProbRun, fold: int, case_name: str) -> Path:
    return run.root / f"fold_{fold}" / "validation" / case_name


def discover_cases(prob_runs: Sequence[ProbRun], folds: Sequence[int], max_cases: int | None = None) -> dict[int, list[str]]:
    reference_run = prob_runs[0]
    by_fold: dict[int, list[str]] = {}
    for fold in folds:
        fold_dir = reference_run.root / f"fold_{fold}" / "validation"
        files = sorted(fold_dir.glob("*.npz"))
        if max_cases is not None:
            files = files[: int(max_cases)]
        if not files:
            raise FileNotFoundError(f"No .npz files found in {fold_dir}")
        by_fold[int(fold)] = [case_name_from_npz(path) for path in files]
    return by_fold


def weighted_probability(probabilities: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    items = [(name, weight) for name, weight in weights.items() if weight > 0]
    if not items:
        raise ValueError(f"Empty positive weights: {weights}")
    first = probabilities[items[0][0]]
    out = np.zeros_like(first, dtype=np.float32)
    for name, weight in items:
        prob = probabilities[name]
        if prob.shape != first.shape:
            raise ValueError(f"Probability shape mismatch for {name}: {prob.shape} != {first.shape}")
        out += prob.astype(np.float32, copy=False) * float(weight)
    denom = out.sum(axis=0, keepdims=True)
    np.divide(out, np.maximum(denom, 1e-8), out=out)
    return out


def argmax_with_logit_bias(probability: np.ndarray, bias: dict[int, float]) -> np.ndarray:
    scores = np.log(np.clip(probability, 1e-8, 1.0))
    for label, value in bias.items():
        if label >= scores.shape[0]:
            raise ValueError(f"Bias label {label} outside probability channel count {scores.shape[0]}")
        scores[int(label)] += float(value)
    return np.argmax(scores, axis=0).astype(np.uint8)


def dice_score(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    denom = int(pred.sum() + ref.sum())
    if denom == 0:
        return math.nan
    return float(2 * np.logical_and(pred, ref).sum() / denom)


def crop_to_union(pred: np.ndarray, ref: np.ndarray, padding: int = 2) -> tuple[np.ndarray, np.ndarray]:
    union = np.logical_or(pred, ref)
    if not union.any():
        return pred, ref
    coords = np.argwhere(union)
    lower = np.maximum(coords.min(axis=0) - int(padding), 0)
    upper = np.minimum(coords.max(axis=0) + int(padding) + 1, pred.shape)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    return pred[slices], ref[slices]


def surface_metrics(pred: np.ndarray, ref: np.ndarray, spacing_xyz: Sequence[float]) -> tuple[float, float, float]:
    pred, ref = crop_to_union(pred.astype(bool), ref.astype(bool))
    if pred.sum() == 0 and ref.sum() == 0:
        return 0.0, 0.0, 0.0
    if pred.sum() == 0 or ref.sum() == 0:
        return math.inf, math.inf, math.inf

    pred_surface = pred ^ binary_erosion(pred)
    ref_surface = ref ^ binary_erosion(ref)
    spacing_zyx = tuple(float(v) for v in spacing_xyz[::-1])
    ref_distance = distance_transform_edt(~ref_surface, sampling=spacing_zyx)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing_zyx)
    distances = np.concatenate([ref_distance[pred_surface], pred_distance[ref_surface]]).astype(np.float64)
    if distances.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.percentile(distances, 95)), float(np.max(distances)), float(np.mean(distances))


def update_metric(
    accumulators: dict[tuple[str, int, str], MetricAccumulator],
    candidate_name: str,
    label: int,
    class_name: str,
    dsc: float,
    hd95_mm: float,
    hd_mm: float,
    assd_mm: float,
) -> None:
    key = (candidate_name, int(label), class_name)
    acc = accumulators.setdefault(key, MetricAccumulator.empty())
    acc.dsc.append(float(dsc))
    acc.hd95_mm.append(float(hd95_mm))
    acc.hd_mm.append(float(hd_mm))
    acc.assd_mm.append(float(assd_mm))


def evaluate_candidates(
    candidates: Sequence[Candidate],
    prob_runs: Sequence[ProbRun],
    folds: Sequence[int],
    cases_by_fold: dict[int, list[str]],
    gt_dir: Path,
    label_names: dict[int, str],
    compute_surface: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sitk = load_sitk()
    prob_run_by_name = {run.name: run for run in prob_runs}
    accumulators: dict[tuple[str, int, str], MetricAccumulator] = {}
    case_rows: list[dict[str, object]] = []

    for fold in folds:
        for case_name in cases_by_fold[int(fold)]:
            print(f"fold={fold} case={case_name}", flush=True)
            gt_path = gt_dir / case_name
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing GT for {case_name}: {gt_path}")
            gt_img = sitk.ReadImage(str(gt_path))
            gt = sitk.GetArrayFromImage(gt_img)
            spacing = gt_img.GetSpacing()

            probabilities = {
                run.name: load_case_probability(probability_path(run, int(fold), case_name))
                for run in prob_runs
            }
            for candidate in candidates:
                prob = weighted_probability(probabilities, candidate.weights)
                seg = argmax_with_logit_bias(prob, candidate.bias)
                seg = postprocess_label_array(seg, labels=LABELS, keep_largest=True, min_component_size=0)
                if seg.shape != gt.shape:
                    raise ValueError(f"{candidate.name} {case_name}: seg shape {seg.shape} != gt shape {gt.shape}")

                for label in LABELS:
                    class_name = label_names[int(label)]
                    pred_mask = seg == int(label)
                    gt_mask = gt == int(label)
                    dsc = dice_score(pred_mask, gt_mask)
                    if compute_surface:
                        hd95, hd, assd = surface_metrics(pred_mask, gt_mask, spacing)
                    else:
                        hd95 = hd = assd = math.nan
                    update_metric(accumulators, candidate.name, label, class_name, dsc, hd95, hd, assd)
                    case_rows.append(
                        {
                            "run": candidate.name,
                            "fold": int(fold),
                            "case": case_name,
                            "label": int(label),
                            "class": class_name,
                            "dsc": dsc,
                            "hd95_mm": hd95,
                            "hd_mm": hd,
                            "assd_mm": assd,
                        }
                    )

    summary_rows: list[dict[str, object]] = []
    for (run, label, class_name), acc in sorted(accumulators.items()):
        summary_rows.append(
            {
                "run": run,
                "fold": "mean",
                "label": label,
                "class": class_name,
                "n_cases": len(acc.dsc),
                "dsc": finite_mean(acc.dsc),
                "hd95_mm": finite_mean(acc.hd95_mm),
                "hd_mm": finite_mean(acc.hd_mm),
                "assd_mm": finite_mean(acc.assd_mm),
            }
        )

    # Add an overall mean row per candidate for easy ranking.
    by_run: dict[str, list[dict[str, object]]] = {}
    for row in summary_rows:
        by_run.setdefault(str(row["run"]), []).append(row)
    for run, rows in sorted(by_run.items()):
        summary_rows.append(
            {
                "run": run,
                "fold": "mean",
                "label": "all",
                "class": "mean",
                "n_cases": sum(int(row["n_cases"]) for row in rows),
                "dsc": finite_mean(float(row["dsc"]) for row in rows),
                "hd95_mm": finite_mean(float(row["hd95_mm"]) for row in rows),
                "hd_mm": finite_mean(float(row["hd_mm"]) for row in rows),
                "assd_mm": finite_mean(float(row["assd_mm"]) for row in rows),
            }
        )

    return case_rows, summary_rows


def compare_to_baseline(summary_rows: list[dict[str, object]], baseline_run: str) -> list[dict[str, object]]:
    baseline = {
        (str(row["label"]), str(row["class"])): row
        for row in summary_rows
        if row["run"] == baseline_run
    }
    compared = []
    for row in summary_rows:
        if row["run"] == baseline_run:
            continue
        key = (str(row["label"]), str(row["class"]))
        base = baseline.get(key)
        if base is None:
            continue
        out = {
            "run": row["run"],
            "baseline_run": baseline_run,
            "label": row["label"],
            "class": row["class"],
            "n_cases": row["n_cases"],
        }
        for metric in ("dsc", "hd95_mm", "hd_mm", "assd_mm"):
            value = float(row[metric])
            base_value = float(base[metric])
            out[metric] = value
            out[f"baseline_{metric}"] = base_value
            out[f"{metric}_delta"] = value - base_value
        compared.append(out)
    return compared


def write_csv(path: Path, rows: list[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_top_markdown(path: Path, summary_rows: list[dict[str, object]], compared_rows: list[dict[str, object]]) -> None:
    overall = [row for row in summary_rows if row["class"] == "mean"]
    overall = sorted(overall, key=lambda row: float(row["dsc"]), reverse=True)
    delta_by_run = {
        str(row["run"]): row for row in compared_rows if row["class"] == "mean"
    }
    lines = [
        "| rank | run | mean DSC | delta vs baseline | HD95 | HD | ASSD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(overall[:30], start=1):
        delta = delta_by_run.get(str(row["run"]), {})
        dsc_delta = delta.get("dsc_delta", 0.0 if not delta else math.nan)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row["run"]),
                    f"{float(row['dsc']):.6f}",
                    f"{float(dsc_delta):+.6f}",
                    format_metric(row.get("hd95_mm")),
                    format_metric(row.get("hd_mm")),
                    format_metric(row.get("assd_mm")),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: object) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value_f):
        return "nan"
    if math.isinf(value_f):
        return "inf"
    return f"{value_f:.4f}"


def label_name_map(cfg: dict) -> dict[int, str]:
    return {int(value): name for name, value in cfg["labels"].items() if int(value) != 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search CT checkpoint softmax ensembles and per-class logit bias on OOF validation. "
            "All candidates are converted with argmax, legacy postprocessing, then evaluated against gt_segmentations."
        )
    )
    parser.add_argument("--prob-run", action="append", type=parse_prob_run, required=True,
                        help="Probability run in name=/path format. Path must contain fold_X/validation/*.npz.")
    parser.add_argument("--weight-spec", action="append",
                        help="Candidate weights, for example final=0.7,best=0.3. Repeatable.")
    parser.add_argument("--bias-mode", choices=["none", "one-at-a-time", "grid"], default="one-at-a-time")
    parser.add_argument("--bias-classes", default="Myo,AO,PA")
    parser.add_argument("--bias-values", default="-0.20,-0.15,-0.10,-0.05,0.05,0.10,0.15,0.20")
    parser.add_argument("--bias-spec", action="append",
                        help="Extra explicit bias spec, for example Myo=0.05,AO=-0.05. Repeatable.")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--label-config", type=Path, default=None)
    parser.add_argument("--baseline-run", default=None,
                        help="Run name for deltas. Defaults to the first generated candidate.")
    parser.add_argument("--compute-surface", action="store_true",
                        help="Also compute HD95, HD, and ASSD. Omit for fast DSC-only sweeps.")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Optional smoke-test limit per fold.")
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--top-md", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_label_config(args.label_config) if args.label_config else load_label_config()
    label_names = label_name_map(cfg)
    prob_runs = args.prob_run
    candidates = build_candidates(args, prob_runs, label_names)
    baseline_run = args.baseline_run or candidates[0].name
    if baseline_run not in {candidate.name for candidate in candidates}:
        raise SystemExit(f"--baseline-run {baseline_run!r} is not among generated candidates.")

    print(f"prob_runs={[run.name for run in prob_runs]}")
    print(f"folds={args.folds}")
    print(f"candidates={len(candidates)}")
    print(f"baseline_run={baseline_run}")
    print(f"compute_surface={args.compute_surface}")

    cases_by_fold = discover_cases(prob_runs, args.folds, max_cases=args.max_cases)
    case_rows, summary_rows = evaluate_candidates(
        candidates=candidates,
        prob_runs=prob_runs,
        folds=args.folds,
        cases_by_fold=cases_by_fold,
        gt_dir=args.gt_dir,
        label_names=label_names,
        compute_surface=args.compute_surface,
    )
    compared_rows = compare_to_baseline(summary_rows, baseline_run=baseline_run)

    case_fields = ["run", "fold", "case", "label", "class", "dsc", "hd95_mm", "hd_mm", "assd_mm"]
    summary_fields = ["run", "fold", "label", "class", "n_cases", "dsc", "hd95_mm", "hd_mm", "assd_mm"]
    comparison_fields = [
        "run", "baseline_run", "label", "class", "n_cases",
        "dsc", "baseline_dsc", "dsc_delta",
        "hd95_mm", "baseline_hd95_mm", "hd95_mm_delta",
        "hd_mm", "baseline_hd_mm", "hd_mm_delta",
        "assd_mm", "baseline_assd_mm", "assd_mm_delta",
    ]
    write_csv(args.case_csv, case_rows, case_fields)
    write_csv(args.summary_csv, summary_rows, summary_fields)
    write_csv(args.comparison_csv, compared_rows, comparison_fields)
    if args.top_md:
        write_top_markdown(args.top_md, summary_rows, compared_rows)

    best = max((row for row in summary_rows if row["class"] == "mean"), key=lambda row: float(row["dsc"]))
    print(f"best_by_mean_dsc={best['run']} dsc={float(best['dsc']):.6f}")
    print(f"wrote {args.case_csv}")
    print(f"wrote {args.summary_csv}")
    print(f"wrote {args.comparison_csv}")


if __name__ == "__main__":
    main()
