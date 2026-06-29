from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise SystemExit("SimpleITK is required for NIfTI label fusion.") from exc
    return sitk


def fuse_arrays(
    base: np.ndarray,
    donor: np.ndarray,
    selected_labels: list[int],
    protect_labels: list[int],
    mode: str,
) -> np.ndarray:
    if base.shape != donor.shape:
        raise ValueError(f"Shape mismatch: base={base.shape}, donor={donor.shape}")

    selected = np.isin(donor, selected_labels)
    protected = np.isin(base, protect_labels)
    fused = base.copy()

    if mode == "add-only":
        write_mask = selected & ~protected & ((base == 0) | np.isin(base, selected_labels))
        fused[write_mask] = donor[write_mask]
        return fused

    if mode == "replace-protect":
        clear_mask = np.isin(fused, selected_labels) & ~protected
        fused[clear_mask] = 0
        write_mask = selected & ~protected
        fused[write_mask] = donor[write_mask]
        return fused

    raise ValueError(f"Unsupported mode: {mode}")


def fuse_file(
    base_path: Path,
    donor_path: Path,
    output_path: Path,
    selected_labels: list[int],
    protect_labels: list[int],
    mode: str,
) -> None:
    sitk = load_sitk()
    base_img = sitk.ReadImage(str(base_path))
    donor_img = sitk.ReadImage(str(donor_path))
    base_arr = sitk.GetArrayFromImage(base_img)
    donor_arr = sitk.GetArrayFromImage(donor_img)
    fused = fuse_arrays(base_arr, donor_arr, selected_labels, protect_labels, mode)

    out_img = sitk.GetImageFromArray(fused.astype(base_arr.dtype, copy=False))
    out_img.CopyInformation(base_img)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse selected labels from a donor prediction into a base prediction while "
            "protecting specified base labels. Label values are expected in train-label space."
        )
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--donor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-labels", required=True, help="Comma-separated labels to import from donor.")
    parser.add_argument("--protect-labels", default="", help="Comma-separated base labels donor cannot overwrite.")
    parser.add_argument("--mode", choices=["replace-protect", "add-only"], default="replace-protect")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_labels = parse_int_list(args.selected_labels)
    protect_labels = parse_int_list(args.protect_labels)
    if not selected_labels:
        raise SystemExit("--selected-labels must not be empty")

    base_files = sorted(args.base_dir.glob("*.nii.gz"))
    if not base_files:
        raise SystemExit(f"No .nii.gz files found in base dir: {args.base_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for base_path in base_files:
        donor_path = args.donor_dir / base_path.name
        output_path = args.output_dir / base_path.name
        if not donor_path.exists():
            raise FileNotFoundError(f"Missing donor prediction for {base_path.name}: {donor_path}")
        if output_path.exists() and not args.overwrite:
            print(f"Skip existing {output_path}")
            continue
        fuse_file(
            base_path=base_path,
            donor_path=donor_path,
            output_path=output_path,
            selected_labels=selected_labels,
            protect_labels=protect_labels,
            mode=args.mode,
        )
        print(f"Fused {base_path.name} -> {output_path}")

    print(f"Fused {len(base_files)} files.")


if __name__ == "__main__":
    main()
