from pathlib import Path
import shutil

import nnunetv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NNUNET_ROOT = Path(nnunetv2.__path__[0])

COPIES = [
    (PROJECT_ROOT / "DATASET" / "mT_Dataset.py", NNUNET_ROOT / "training" / "dataloading" / "mT_Dataset.py"),
    (PROJECT_ROOT / "DATASET" / "cotta_wrapper.py", NNUNET_ROOT / "inference" / "cotta_wrapper.py"),
    (PROJECT_ROOT / "DATASET" / "nnUNetTrainer.py", NNUNET_ROOT / "training" / "nnUNetTrainer" / "nnUNetTrainer.py"),
    (PROJECT_ROOT / "DATASET" / "predict_from_raw_data.py", NNUNET_ROOT / "inference" / "predict_from_raw_data.py"),
]


def main() -> None:
    for src, dst in COPIES:
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    main()
