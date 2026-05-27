from __future__ import annotations

from pathlib import Path
import shutil

import nnunetv2


TRACK4_ROOT = Path(__file__).resolve().parents[1]
NNUNET_ROOT = Path(nnunetv2.__path__[0])

SOURCES = [
    TRACK4_ROOT / "nnunet_extensions" / "nnUNetTrainerWholeHeartAug.py",
    TRACK4_ROOT / "nnunet_extensions" / "unlabeled_pool.py",
]
DESTINATION_DIR = NNUNET_ROOT / "training" / "nnUNetTrainer" / "variants" / "wholeheart"


def main() -> None:
    for source in SOURCES:
        if not source.exists():
            raise FileNotFoundError(source)

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    init_file = DESTINATION_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    for source in SOURCES:
        destination = DESTINATION_DIR / source.name
        shutil.copy2(source, destination)
        print(f"Installed {source.name} -> {destination}")


if __name__ == "__main__":
    main()
