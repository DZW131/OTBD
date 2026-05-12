from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from common import find_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CARE Whole Heart image folders.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sample_stats(path: Path) -> dict:
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    flat = arr[np.isfinite(arr)].ravel()
    if flat.size > 200_000:
        rng = np.random.default_rng(0)
        flat = flat[rng.choice(flat.size, 200_000, replace=False)]
    return {
        "file": str(path),
        "shape_sitk_zyx": list(arr.shape),
        "spacing_xyz": [float(v) for v in image.GetSpacing()],
        "min": float(np.min(flat)),
        "p1": float(np.percentile(flat, 1)),
        "median": float(np.median(flat)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat))
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        return {}
    shapes = np.array([r["shape_sitk_zyx"] for r in records], dtype=float)
    spacing = np.array([r["spacing_xyz"] for r in records], dtype=float)
    return {
        "count": len(records),
        "shape_min": shapes.min(axis=0).astype(int).tolist(),
        "shape_median": np.median(shapes, axis=0).tolist(),
        "shape_max": shapes.max(axis=0).astype(int).tolist(),
        "spacing_min": spacing.min(axis=0).tolist(),
        "spacing_median": np.median(spacing, axis=0).tolist(),
        "spacing_max": spacing.max(axis=0).tolist(),
        "intensity_p1_median": float(np.median([r["p1"] for r in records])),
        "intensity_median_median": float(np.median([r["median"] for r in records])),
        "intensity_p99_median": float(np.median([r["p99"] for r in records]))
    }


def main() -> None:
    args = parse_args()
    folders = {
        "ct_train": args.root / "ct_train",
        "mr_train": args.root / "mr_train",
        "ct_val": args.root / "ct_val",
        "mr_val": args.root / "mr_val",
    }
    payload = {}
    for name, folder in folders.items():
        images = find_images(folder)
        records = [sample_stats(p) for p in images]
        payload[name] = {
            "summary": summarize(records),
            "cases": records
        }

    text = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
