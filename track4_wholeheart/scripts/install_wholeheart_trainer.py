from __future__ import annotations

from pathlib import Path
import shutil

import nnunetv2


TRACK4_ROOT = Path(__file__).resolve().parents[1]
NNUNET_ROOT = Path(nnunetv2.__path__[0])

SOURCE = TRACK4_ROOT / "nnunet_extensions" / "nnUNetTrainerWholeHeartAug.py"
DESTINATION_DIR = NNUNET_ROOT / "training" / "nnUNetTrainer" / "variants" / "wholeheart"
DESTINATION = DESTINATION_DIR / SOURCE.name


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    init_file = DESTINATION_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    shutil.copy2(SOURCE, DESTINATION)
    print(f"Installed {SOURCE.name} -> {DESTINATION}")


if __name__ == "__main__":
    main()
