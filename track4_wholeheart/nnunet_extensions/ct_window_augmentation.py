from __future__ import annotations

from collections.abc import Mapping

import torch


def ct_hu_window_to_normalized(
    lower_hu: float,
    upper_hu: float,
    intensity_properties: Mapping[str, float] | None,
) -> tuple[float, float] | None:
    if intensity_properties is None:
        return None
    mean = intensity_properties.get("mean")
    std = intensity_properties.get("std")
    if mean is None or std is None or float(std) <= 0:
        return None

    lower = (float(lower_hu) - float(mean)) / float(std)
    upper = (float(upper_hu) - float(mean)) / float(std)
    if lower >= upper:
        return None
    return lower, upper


def apply_ct_window_rescale_torch(
    data: torch.Tensor,
    lower: float,
    upper: float,
    blend: float = 1.0,
    channel: int = 0,
) -> torch.Tensor:
    if data.ndim < 3 or data.shape[1] <= channel:
        return data
    if lower >= upper or blend <= 0:
        return data

    blend = max(0.0, min(float(blend), 1.0))
    output = data.clone()
    selected = data[:, channel]
    flat = selected.flatten(start_dim=1)
    target_min = flat.min(dim=1).values.view(-1, *([1] * (selected.ndim - 1)))
    target_max = flat.max(dim=1).values.view(-1, *([1] * (selected.ndim - 1)))
    target_range = target_max - target_min
    valid = target_range > 1e-6
    if not torch.any(valid):
        return output

    clipped = selected.clamp(min=float(lower), max=float(upper))
    scaled = (clipped - float(lower)) / (float(upper) - float(lower))
    scaled = scaled * target_range + target_min
    mixed = (1.0 - blend) * selected + blend * scaled
    output[:, channel] = torch.where(valid, mixed, selected)
    return output
