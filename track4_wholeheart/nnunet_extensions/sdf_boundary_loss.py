from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from torch import nn
import torch.nn.functional as F


def _first_deep_supervision_item(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def _target_to_labels(target, spatial_shape: tuple[int, ...]) -> torch.Tensor:
    target_tensor = _first_deep_supervision_item(target)
    if target_tensor.ndim == len(spatial_shape) + 2 and target_tensor.shape[1] == 1:
        labels = target_tensor[:, 0]
    elif target_tensor.ndim == len(spatial_shape) + 2 and target_tensor.shape[1] > 1:
        labels = torch.argmax(target_tensor, dim=1)
    elif target_tensor.ndim == len(spatial_shape) + 1:
        labels = target_tensor
    else:
        raise ValueError(
            f"Unsupported target shape {tuple(target_tensor.shape)} for SDF boundary loss "
            f"with spatial shape {spatial_shape}."
        )

    if tuple(labels.shape[1:]) != spatial_shape:
        labels = F.interpolate(
            labels.unsqueeze(1).float(),
            size=spatial_shape,
            mode="nearest",
        )[:, 0]
    return labels.long().clamp_min(0)


def _signed_distance_fields(labels: np.ndarray, num_classes: int, normalize: bool) -> np.ndarray:
    output = np.zeros((labels.shape[0], num_classes - 1, *labels.shape[1:]), dtype=np.float32)
    for batch_idx in range(labels.shape[0]):
        for class_idx in range(1, num_classes):
            mask = labels[batch_idx] == class_idx
            if not np.any(mask) or np.all(mask):
                continue
            outside_distance = ndimage.distance_transform_edt(~mask)
            inside_distance = ndimage.distance_transform_edt(mask)
            sdf = outside_distance - inside_distance
            if normalize:
                max_abs = float(np.max(np.abs(sdf)))
                if max_abs > 0:
                    sdf = sdf / max_abs
            output[batch_idx, class_idx - 1] = sdf.astype(np.float32, copy=False)
    return output


def sdf_boundary_loss_from_logits(output, target, normalize: bool = True) -> torch.Tensor:
    logits = _first_deep_supervision_item(output)
    if logits.shape[1] <= 1:
        return logits.new_tensor(0.0)

    logits_float = logits.float()
    labels = _target_to_labels(target, tuple(int(v) for v in logits_float.shape[2:]))
    sdf_np = _signed_distance_fields(
        labels.detach().cpu().numpy().astype(np.int16, copy=False),
        num_classes=int(logits_float.shape[1]),
        normalize=normalize,
    )
    sdf = torch.from_numpy(sdf_np).to(device=logits_float.device, dtype=logits_float.dtype)
    foreground_probabilities = torch.softmax(logits_float, dim=1)[:, 1:]
    return torch.mean(foreground_probabilities * sdf)


class WholeHeartSDFBoundaryLoss(nn.Module):
    """Add supervised SDF boundary loss to the existing nnU-Net segmentation loss."""

    def __init__(self, base_loss, weight: float, normalize: bool = True):
        super().__init__()
        self.base_loss = base_loss
        self.weight = float(weight)
        self.normalize = bool(normalize)
        self.last_sdf_loss = None

    def forward(self, output, target):
        base_loss = self.base_loss(output, target)
        if self.weight <= 0:
            self.last_sdf_loss = base_loss.detach().new_tensor(0.0)
            return base_loss

        sdf_loss = sdf_boundary_loss_from_logits(output, target, normalize=self.normalize)
        self.last_sdf_loss = sdf_loss.detach()
        return base_loss + self.weight * sdf_loss
