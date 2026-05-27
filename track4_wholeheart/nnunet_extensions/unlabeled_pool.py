from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class WholeHeartRawUnlabeledPool:
    """Patch sampler for official imagesTs volumes used by mean-teacher training."""

    def __init__(
        self,
        image_dir: Path,
        patch_size: Tuple[int, ...],
        modality: str,
        cache_size: int = 2,
        patch_cache_size: int = 128,
    ):
        self.image_dir = Path(image_dir)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.modality = modality
        self.cache_size = max(1, int(cache_size))
        self.patch_cache_size = max(0, int(patch_cache_size))
        self.image_paths = sorted(self.image_dir.glob("*.nii.gz"))
        self._cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._patch_cache: list[torch.Tensor] = []
        if not self.image_paths:
            raise FileNotFoundError(f"No .nii.gz images found in unlabeled image dir: {self.image_dir}")

    def _read_image(self, path: Path) -> torch.Tensor:
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise RuntimeError("SimpleITK is required for raw imagesTs mean-teacher sampling.") from exc

        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
        finite = np.isfinite(array)
        if not np.any(finite):
            array = np.zeros_like(array, dtype=np.float32)
        else:
            values = array[finite]
            if self.modality == "ct":
                lower, upper = np.percentile(values, [0.5, 99.5])
                array = np.clip(array, lower, upper)
                values = array[finite]
            mean = float(values.mean())
            std = float(values.std())
            array = (array - mean) / max(std, 1e-8)
            array[~finite] = 0
        return torch.from_numpy(array.copy()).float()

    def _get_volume(self, path: Path) -> torch.Tensor:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        volume = self._read_image(path)
        self._cache[path] = volume
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return volume

    def _pad_if_needed(self, volume: torch.Tensor) -> torch.Tensor:
        pad = []
        for size, target in zip(reversed(volume.shape), reversed(self.patch_size)):
            missing = max(0, target - int(size))
            pad.extend([missing // 2, missing - missing // 2])
        if any(pad):
            volume = F.pad(volume, pad, mode="constant", value=0)
        return volume

    def _random_patch(self, volume: torch.Tensor) -> torch.Tensor:
        volume = self._pad_if_needed(volume)
        starts = []
        for size, target in zip(volume.shape, self.patch_size):
            max_start = int(size) - int(target)
            starts.append(0 if max_start <= 0 else int(torch.randint(max_start + 1, (), dtype=torch.int64).item()))
        slices = tuple(slice(start, start + target) for start, target in zip(starts, self.patch_size))
        return volume[slices].unsqueeze(0)

    def _ensure_patch_cache(self) -> None:
        if self.patch_cache_size <= 0 or self._patch_cache:
            return

        base_count = self.patch_cache_size // len(self.image_paths)
        extra_count = self.patch_cache_size % len(self.image_paths)
        for path_idx, path in enumerate(self.image_paths):
            patches_for_volume = base_count + (1 if path_idx < extra_count else 0)
            if patches_for_volume <= 0:
                continue
            volume = self._get_volume(path)
            for _ in range(patches_for_volume):
                self._patch_cache.append(self._random_patch(volume).cpu().contiguous())
                if len(self._patch_cache) >= self.patch_cache_size:
                    return

    def sample_batch(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.patch_cache_size > 0:
            self._ensure_patch_cache()
            patches = [
                self._patch_cache[int(torch.randint(len(self._patch_cache), (), dtype=torch.int64).item())]
                for _ in range(batch_size)
            ]
        else:
            patches = []
            for _ in range(batch_size):
                path = self.image_paths[int(torch.randint(len(self.image_paths), (), dtype=torch.int64).item())]
                patches.append(self._random_patch(self._get_volume(path)))
        return torch.stack(patches, dim=0).to(device=device, dtype=dtype, non_blocking=True)
