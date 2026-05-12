from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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


def find_images(directory: Path | None) -> List[Path]:
    if directory is None or not directory.exists():
        return []
    images = sorted(directory.rglob("*_image.nii.gz"))
    if images:
        return images
    return sorted(
        p for p in directory.rglob("*.nii.gz")
        if "_label.nii.gz" not in p.name and "_seg.nii.gz" not in p.name
    )


def label_candidates(image_path: Path, labels_dir: Path | None = None) -> Iterable[Path]:
    image_name = image_path.name
    if image_name.endswith("_image.nii.gz"):
        label_name = image_name.replace("_image.nii.gz", "_label.nii.gz")
    else:
        label_name = f"{strip_nii_gz_name(image_path)}_label.nii.gz"

    yield image_path.with_name(label_name)

    if labels_dir is not None:
        try:
            rel = image_path.relative_to(image_path.parents[0])
        except ValueError:
            rel = Path(label_name)
        yield labels_dir / rel.parent / label_name
        yield labels_dir / label_name


def find_label_for_image(image_path: Path, labels_dir: Path | None = None) -> Path | None:
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
