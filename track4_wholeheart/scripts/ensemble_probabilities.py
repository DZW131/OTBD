from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .postprocess_ct_aopa_soft import PREFERRED_PROBABILITY_KEYS, load_case_probability
    from .postprocess_predictions import load_sitk
except ImportError:
    from postprocess_ct_aopa_soft import PREFERRED_PROBABILITY_KEYS, load_case_probability
    from postprocess_predictions import load_sitk


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def weighted_average_probabilities(probabilities: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    if not probabilities:
        raise ValueError("At least one probability array is required")
    if len(probabilities) != len(weights):
        raise ValueError(f"Expected {len(probabilities)} weights, got {len(weights)}")

    reference_shape = probabilities[0].shape
    for idx, prob in enumerate(probabilities):
        if prob.shape != reference_shape:
            raise ValueError(f"Probability shape mismatch at index {idx}: {prob.shape} != {reference_shape}")
        if prob.ndim != 4:
            raise ValueError(f"Probability arrays must have shape [C, Z, Y, X], got {prob.shape}")

    weight_array = np.asarray(weights, dtype=np.float32)
    if np.any(weight_array < 0):
        raise ValueError(f"Weights must be non-negative, got {weights}")
    weight_sum = float(weight_array.sum())
    if weight_sum <= 0:
        raise ValueError(f"Weights must sum to a positive value, got {weights}")
    weight_array = weight_array / weight_sum

    ensembled = np.zeros(reference_shape, dtype=np.float32)
    for prob, weight in zip(probabilities, weight_array):
        ensembled += prob.astype(np.float32, copy=False) * float(weight)

    denom = ensembled.sum(axis=0, keepdims=True)
    np.divide(ensembled, np.maximum(denom, 1e-8), out=ensembled)
    return ensembled


def argmax_segmentation(probability: np.ndarray) -> np.ndarray:
    if probability.ndim != 4:
        raise ValueError(f"Expected probability shape [C, Z, Y, X], got {probability.shape}")
    return np.argmax(probability, axis=0).astype(np.uint8)


def probability_path_for_case(prob_dir: Path, case_name: str) -> Path:
    if case_name.endswith(".nii.gz"):
        case_name = case_name[:-7]
    else:
        case_name = Path(case_name).stem
    return prob_dir / f"{case_name}.npz"


def save_segmentation_like(reference_path: Path, segmentation: np.ndarray, output_path: Path) -> None:
    sitk = load_sitk()
    reference = sitk.ReadImage(str(reference_path))
    if segmentation.shape != sitk.GetArrayFromImage(reference).shape:
        raise ValueError(
            f"Segmentation shape {segmentation.shape} does not match reference shape "
            f"{sitk.GetArrayFromImage(reference).shape} for {reference_path}"
        )
    output = sitk.GetImageFromArray(segmentation)
    output.CopyInformation(reference)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output, str(output_path))


def ensemble_case(
    probability_paths: Sequence[Path],
    reference_path: Path,
    output_path: Path,
    weights: Sequence[float],
) -> None:
    probabilities = [load_case_probability(path) for path in probability_paths]
    ensembled = weighted_average_probabilities(probabilities, weights)
    segmentation = argmax_segmentation(ensembled)
    save_segmentation_like(reference_path, segmentation, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weighted-average nnU-Net .npz probabilities and save argmax train-label masks.")
    parser.add_argument(
        "--probability-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing .npz probabilities. Repeat in the same order as --weights.",
    )
    parser.add_argument("--weights", required=True, help="Comma-separated non-negative weights, for example 0.7,0.3.")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory with matching .nii.gz predictions used only for geometry and case list.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = parse_float_list(args.weights)
    if len(weights) != len(args.probability_dir):
        raise SystemExit(f"--weights has {len(weights)} values but {len(args.probability_dir)} probability dirs were provided")

    reference_files = sorted(args.reference_dir.glob("*.nii.gz"))
    if not reference_files:
        raise SystemExit(f"No .nii.gz reference predictions found in {args.reference_dir}")

    for reference_path in reference_files:
        prob_paths = [probability_path_for_case(prob_dir, reference_path.name) for prob_dir in args.probability_dir]
        missing = [path for path in prob_paths if not path.exists()]
        if missing:
            raise SystemExit(f"Missing probability files for {reference_path.name}: {missing}")
        output_path = args.output_dir / reference_path.name
        ensemble_case(prob_paths, reference_path, output_path, weights)
        print(
            f"Ensembled {reference_path.name} from "
            f"{', '.join(str(path.parent) for path in prob_paths)} -> {output_path}",
            flush=True,
        )

    print(f"Ensembled {len(reference_files)} files.")


if __name__ == "__main__":
    main()
