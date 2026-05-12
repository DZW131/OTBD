from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TRACK4_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRACK4_ROOT.parent
DEFAULT_LABEL_CONFIG = TRACK4_ROOT / "configs" / "wholeheart_labels.json"


def load_label_config(path: Path = DEFAULT_LABEL_CONFIG) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def modality_key(modality: str) -> str:
    value = modality.lower()
    if value not in {"ct", "mr"}:
        raise ValueError(f"Unsupported modality: {modality}. Expected 'ct' or 'mr'.")
    return value


def dataset_name(modality: str, cfg: dict) -> str:
    return cfg["dataset_names"][modality_key(modality)]


def dataset_id(modality: str, cfg: dict) -> int:
    return int(cfg["dataset_ids"][modality_key(modality)])


def nnunet_raw_root(dataset_root: Path) -> Path:
    return dataset_root / "nnUNet_raw"


def default_dataset_root() -> Path:
    return TRACK4_ROOT / "DATASET"


def strip_nii_gz_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def case_id_from_image(path: Path) -> str:
    name = strip_nii_gz_name(path)
    if name.endswith("_image"):
        name = name[:-6]
    return sanitize_case_id(name)


def output_label_name_from_image(path: Path) -> str:
    name = path.name
    if name.endswith("_image.nii.gz"):
        return name.replace("_image.nii.gz", "_label.nii.gz")
    if name.endswith(".nii.gz"):
        return name
    return f"{path.stem}_label.nii.gz"


def sanitize_case_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def normalize_dirs(directories: Path | Sequence[Path] | None) -> List[Path]:
    if directories is None:
        return []
    if isinstance(directories, Path):
        directories = [directories]
    return [Path(d) for d in directories if d is not None and Path(d).exists()]


def discover_split_dirs(root: Path | None, modality: str, split: str) -> List[Path]:
    if root is None or not root.exists():
        return []
    key = modality_key(modality)
    split_key = split.lower()

    exact = root / f"{key}_{split_key}"
    if exact.exists():
        return [exact]

    matches = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name.lower()
        if key in name and split_key in name:
            matches.append(child)
    return matches


def find_images(directories: Path | Sequence[Path] | None) -> List[Path]:
    roots = normalize_dirs(directories)
    if not roots:
        return []

    images = sorted(p for root in roots for p in root.rglob("*_image.nii.gz"))
    if images:
        return images
    return sorted(
        p for root in roots for p in root.rglob("*.nii.gz")
        if "_label.nii.gz" not in p.name and "_seg.nii.gz" not in p.name
    )


def label_candidates(image_path: Path, labels_dir: Path | Sequence[Path] | None = None) -> Iterable[Path]:
    image_name = image_path.name
    if image_name.endswith("_image.nii.gz"):
        label_name = image_name.replace("_image.nii.gz", "_label.nii.gz")
    else:
        label_name = f"{strip_nii_gz_name(image_path)}_label.nii.gz"

    yield image_path.with_name(label_name)

    for root in normalize_dirs(labels_dir):
        yield root / label_name
        yield root / image_path.parent.name / label_name


def find_label_for_image(image_path: Path, labels_dir: Path | Sequence[Path] | None = None) -> Path | None:
    for candidate in label_candidates(image_path, labels_dir):
        if candidate.exists():
            return candidate
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_dataset_json(modality: str, cfg: dict, num_training: int) -> dict:
    labels = cfg["labels"]
    return {
        "channel_names": {"0": cfg["channel_names"][modality_key(modality)]},
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz"
    }


def mapping_arrays(cfg: dict) -> Tuple[Dict[int, int], Dict[int, int]]:
    official_to_train = {int(k): int(v) for k, v in cfg["official_to_train"].items()}
    train_to_official = {int(k): int(v) for k, v in cfg["train_to_official"].items()}
    return official_to_train, train_to_official
