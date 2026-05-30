from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import autocast

try:
    from track4_wholeheart.nnunet_extensions.unlabeled_pool import WholeHeartRawUnlabeledPool
except ModuleNotFoundError:  # pragma: no cover - used after install into nnU-Net variants
    from nnunetv2.training.nnUNetTrainer.variants.wholeheart.unlabeled_pool import WholeHeartRawUnlabeledPool

try:
    from torch._dynamo import OptimizedModule
except Exception:  # pragma: no cover - optional torch internals differ by version
    OptimizedModule = ()  # type: ignore[assignment]


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _deep_supervision_outputs(outputs):
    return outputs if isinstance(outputs, (list, tuple)) else [outputs]


def _foreground_like_intensity_mask(volume: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(volume)
    if not torch.any(finite):
        return torch.ones_like(volume, dtype=torch.bool)
    finite_values = volume[finite]
    lower = torch.quantile(finite_values.float(), 0.01)
    upper = torch.quantile(finite_values.float(), 0.99)
    mask = finite & (volume >= lower) & (volume <= upper)
    return mask if torch.any(mask) else finite


def fast_histogram_match_torch(source: torch.Tensor, reference: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    """Approximate random histogram matching used in the CARE liver code, adapted for tensors."""
    if source.numel() == 0 or reference.numel() == 0:
        return source

    src_mask = _foreground_like_intensity_mask(source)
    ref_mask = _foreground_like_intensity_mask(reference)
    src_values = source[src_mask].float()
    ref_values = reference[ref_mask].float()
    if src_values.numel() < 2 or ref_values.numel() < 2:
        return source

    src_min, src_max = src_values.min(), src_values.max()
    ref_min, ref_max = ref_values.min(), ref_values.max()
    if bool(torch.isclose(src_min, src_max)) or bool(torch.isclose(ref_min, ref_max)):
        return source

    src_min_value = float(src_min.item())
    src_max_value = float(src_max.item())
    ref_min_value = float(ref_min.item())
    ref_max_value = float(ref_max.item())

    src_hist = torch.histc(src_values, bins=num_bins, min=src_min_value, max=src_max_value)
    ref_hist = torch.histc(ref_values, bins=num_bins, min=ref_min_value, max=ref_max_value)
    src_cdf = torch.cumsum(src_hist, dim=0)
    ref_cdf = torch.cumsum(ref_hist, dim=0)
    if src_cdf[-1] <= 0 or ref_cdf[-1] <= 0:
        return source

    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    ref_edges = torch.linspace(ref_min_value, ref_max_value, steps=num_bins, device=source.device)

    lookup = torch.empty_like(src_cdf)
    for idx in range(num_bins):
        lookup[idx] = ref_edges[torch.argmin(torch.abs(ref_cdf - src_cdf[idx]))]

    src_edges = torch.linspace(src_min_value, src_max_value, steps=num_bins, device=source.device)
    bucket = torch.bucketize(source.float(), src_edges, right=True).clamp(0, num_bins - 1)
    matched = lookup[bucket].to(dtype=source.dtype)
    output = source.clone()
    output[src_mask] = matched[src_mask]
    return output


def model_state_dict(model: torch.nn.Module) -> dict:
    module = model.module if hasattr(model, "module") else model
    if OptimizedModule and isinstance(module, OptimizedModule):
        module = module._orig_mod
    return module.state_dict()


def load_model_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    module = model.module if hasattr(model, "module") else model
    if OptimizedModule and isinstance(module, OptimizedModule):
        module = module._orig_mod
    module.load_state_dict(state_dict)


class nnUNetTrainerWholeHeartAug(nnUNetTrainer):
    """nnU-Net trainer with stronger CT/MR intensity augmentation for CARE whole-heart."""

    default_rhm_probability = 0.0
    default_rhm_blend = 1.0

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.num_epochs = _env_int("WHOLEHEART_NUM_EPOCHS", self.num_epochs)
        self.rhm_probability = _env_float("WHOLEHEART_RHM_PROB", self.default_rhm_probability)
        self.rhm_num_bins = _env_int("WHOLEHEART_RHM_BINS", 256)
        self.rhm_blend = _env_float("WHOLEHEART_RHM_BLEND", self.default_rhm_blend)
        self._wholeheart_config_logged = False

    def _modality(self) -> str:
        channel_names = self.dataset_json.get("channel_names", {})
        value = str(channel_names.get("0", "")).lower()
        if "mr" in value:
            return "mr"
        if "ct" in value:
            return "ct"

        dataset_name = str(self.plans_manager.dataset_name).lower()
        if "mr" in dataset_name:
            return "mr"
        return "ct"

    def _log_wholeheart_config(self) -> None:
        if self._wholeheart_config_logged:
            return
        self._wholeheart_config_logged = True
        self.print_to_log_file(
            "WholeHeart trainer config: "
            f"trainer={self.__class__.__name__}, modality={self._modality()}, "
            f"num_epochs={self.num_epochs}, "
            f"rhm_probability={self.rhm_probability}, rhm_bins={self.rhm_num_bins}, "
            f"rhm_blend={self.rhm_blend}",
            also_print_to_console=True,
        )

    def on_train_start(self):
        super().on_train_start()
        self._log_wholeheart_config()

    def _apply_random_histogram_matching(self, data: torch.Tensor, probability: float | None = None) -> torch.Tensor:
        probability = self.rhm_probability if probability is None else probability
        if probability <= 0 or data.shape[0] < 2:
            return data

        with torch.no_grad():
            output = data.clone()
            for batch_idx in range(data.shape[0]):
                if torch.rand((), device=data.device).item() >= probability:
                    continue
                candidates = [idx for idx in range(data.shape[0]) if idx != batch_idx]
                ref_idx = candidates[int(torch.randint(len(candidates), (), device=data.device).item())]
                for channel_idx in range(data.shape[1]):
                    matched = fast_histogram_match_torch(
                        data[batch_idx, channel_idx],
                        data[ref_idx, channel_idx],
                        num_bins=self.rhm_num_bins,
                    )
                    output[batch_idx, channel_idx] = (
                        (1 - self.rhm_blend) * data[batch_idx, channel_idx] + self.rhm_blend * matched
                    )
        return output

    def train_step(self, batch: dict) -> dict:
        if self.rhm_probability > 0:
            batch = dict(batch)
            batch["data"] = self._apply_random_histogram_matching(batch["data"])
        return super().train_step(batch)

    def _domain_intensity_transforms(self) -> list[BasicTransform]:
        modality = self._modality()
        if modality == "mr":
            return [
                RandomTransform(
                    GaussianNoiseTransform(
                        noise_variance=(0, 0.15),
                        p_per_channel=1,
                        synchronize_channels=True,
                    ),
                    apply_probability=0.2,
                ),
                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.4, 1.2),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5,
                        benchmark=True,
                    ),
                    apply_probability=0.15,
                ),
                RandomTransform(
                    MultiplicativeBrightnessTransform(
                        multiplier_range=BGContrast((0.65, 1.35)),
                        synchronize_channels=False,
                        p_per_channel=1,
                    ),
                    apply_probability=0.25,
                ),
                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.6, 1.6)),
                        preserve_range=True,
                        synchronize_channels=False,
                        p_per_channel=1,
                    ),
                    apply_probability=0.3,
                ),
                RandomTransform(
                    GammaTransform(
                        gamma=BGContrast((0.6, 1.8)),
                        p_invert_image=0,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1,
                    ),
                    apply_probability=0.35,
                ),
            ]

        return [
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.12),
                    p_per_channel=1,
                    synchronize_channels=True,
                ),
                apply_probability=0.15,
            ),
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5,
                    benchmark=True,
                ),
                apply_probability=0.15,
            ),
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast((0.7, 1.3)),
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.2,
            ),
            RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast((0.65, 1.45)),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1,
                ),
                apply_probability=0.25,
            ),
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.65, 1.6)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.25,
            ),
        ]

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            ignore_axes = None
            patch_size_spatial = patch_size

        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA,
                p_scaling=0.2,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.extend(self._domain_intensity_transforms())

        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2,
            )
        )
        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.08,
            )
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(
                MaskImageTransform(
                    apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    channel_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        transforms.append(RemoveLabelTansform(-1, 0))

        if is_cascaded:
            assert foreground_labels is not None, "Cascade augmentations require foreground_labels."
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True,
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1,
                    ),
                    apply_probability=0.4,
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1,
                    ),
                    apply_probability=0.2,
                )
            )

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0,
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainerWholeHeartRHM(nnUNetTrainerWholeHeartAug):
    """Whole-heart trainer with random histogram matching enabled for ablation."""

    default_rhm_probability = 0.35

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        if "WHOLEHEART_RHM_PROB" not in os.environ and self._modality() == "mr":
            self.rhm_probability = 0.5


class nnUNetTrainerWholeHeartRHMMeanTeacher(nnUNetTrainerWholeHeartRHM):
    """Random histogram matching plus mean-teacher consistency training."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.mt_enabled = _env_bool("WHOLEHEART_MT", True)
        self.mt_start_epoch = _env_int("WHOLEHEART_MT_START_EPOCH", 40)
        self.mt_rampup_epochs = _env_int("WHOLEHEART_MT_RAMPUP_EPOCHS", 80)
        self.mt_max_weight = _env_float("WHOLEHEART_MT_MAX_WEIGHT", 1.0)
        self.mt_ema_decay = _env_float("WHOLEHEART_MT_EMA_DECAY", 0.95)
        self.mt_student_noise_std = _env_float("WHOLEHEART_MT_STUDENT_NOISE_STD", 0.05)
        self.mt_teacher_noise_std = _env_float("WHOLEHEART_MT_TEACHER_NOISE_STD", 0.02)
        self.mt_student_contrast = _env_float("WHOLEHEART_MT_STUDENT_CONTRAST", 0.25)
        self.mt_teacher_contrast = _env_float("WHOLEHEART_MT_TEACHER_CONTRAST", 0.10)
        self.mt_student_rhm_probability = _env_float("WHOLEHEART_MT_STUDENT_RHM_PROB", self.rhm_probability)
        self.mt_unlabeled_mode = os.environ.get("WHOLEHEART_MT_UNLABELED_MODE", "imagesTs").strip().lower()
        self.mt_unlabeled_image_dir = os.environ.get("WHOLEHEART_MT_UNLABELED_IMAGE_DIR", "")
        self.mt_unlabeled_cache_size = _env_int("WHOLEHEART_MT_UNLABELED_CACHE", 2)
        self.mt_unlabeled_patch_cache_size = _env_int("WHOLEHEART_MT_PATCH_CACHE", 128)
        self.mt_unlabeled_pool = None
        self.teacher_network = None

    def initialize(self):
        super().initialize()
        if self.mt_enabled:
            self.teacher_network = deepcopy(self.network)
            self._freeze_teacher()
            self._setup_unlabeled_pool()

    def _log_wholeheart_config(self) -> None:
        if self._wholeheart_config_logged:
            return
        super()._log_wholeheart_config()
        if self.mt_enabled:
            self.print_to_log_file(
                "WholeHeart mean teacher config: "
                f"start_epoch={self.mt_start_epoch}, rampup_epochs={self.mt_rampup_epochs}, "
                f"max_weight={self.mt_max_weight}, ema_decay={self.mt_ema_decay}, "
                f"student_noise_std={self.mt_student_noise_std}, teacher_noise_std={self.mt_teacher_noise_std}, "
                f"student_contrast={self.mt_student_contrast}, teacher_contrast={self.mt_teacher_contrast}, "
                f"student_rhm_probability={self.mt_student_rhm_probability}, "
                f"unlabeled_mode={self.mt_unlabeled_mode}, "
                f"unlabeled_pool_size={len(self.mt_unlabeled_pool.image_paths) if self.mt_unlabeled_pool else 0}, "
                f"unlabeled_patch_cache_size={self.mt_unlabeled_patch_cache_size if self.mt_unlabeled_pool else 0}",
                also_print_to_console=True,
            )

    def _setup_unlabeled_pool(self) -> None:
        if self.mt_unlabeled_mode in {"", "none", "off", "labeled", "labeled_batch"}:
            self.mt_unlabeled_pool = None
            return
        if self.mt_unlabeled_mode not in {"imagests", "images_ts", "official_val", "raw_val"}:
            raise ValueError(
                "Unsupported WHOLEHEART_MT_UNLABELED_MODE="
                f"{self.mt_unlabeled_mode!r}. Use imagesTs or labeled_batch."
            )

        if self.mt_unlabeled_image_dir:
            image_dir = Path(self.mt_unlabeled_image_dir)
        else:
            if nnUNet_raw is None:
                raise RuntimeError("nnUNet_raw is not set; cannot locate imagesTs for unlabeled mean-teacher data.")
            image_dir = Path(nnUNet_raw) / str(self.plans_manager.dataset_name) / "imagesTs"
        try:
            self.mt_unlabeled_pool = WholeHeartRawUnlabeledPool(
                image_dir=image_dir,
                patch_size=tuple(int(v) for v in self.configuration_manager.patch_size),
                modality=self._modality(),
                cache_size=self.mt_unlabeled_cache_size,
                patch_cache_size=self.mt_unlabeled_patch_cache_size,
            )
        except FileNotFoundError:
            self.print_to_log_file(
                f"WholeHeart mean teacher: no imagesTs unlabeled pool found at {image_dir}; "
                "falling back to labeled_batch consistency.",
                also_print_to_console=True,
            )
            self.mt_unlabeled_pool = None

    def _freeze_teacher(self) -> None:
        if self.teacher_network is None:
            return
        self.teacher_network.eval()
        for param in self.teacher_network.parameters():
            param.requires_grad_(False)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.teacher_network is not None:
            self.teacher_network.eval()

    def _consistency_weight(self) -> float:
        if not self.mt_enabled or self.current_epoch < self.mt_start_epoch:
            return 0.0
        if self.mt_rampup_epochs <= 0:
            return self.mt_max_weight
        progress = min(1.0, max(0.0, (self.current_epoch - self.mt_start_epoch) / self.mt_rampup_epochs))
        return float(self.mt_max_weight * np.exp(-8.0 * (1.0 - progress) ** 2))

    def _perturb_for_teacher_training(self, data: torch.Tensor, *, strong: bool) -> torch.Tensor:
        output = data.clone()
        if strong and self.mt_student_rhm_probability > 0:
            output = self._apply_random_histogram_matching(output, probability=self.mt_student_rhm_probability)

        noise_std = self.mt_student_noise_std if strong else self.mt_teacher_noise_std
        contrast = self.mt_student_contrast if strong else self.mt_teacher_contrast
        if noise_std > 0:
            output = output + torch.randn_like(output) * noise_std
        if contrast > 0:
            dims = tuple(range(2, output.ndim))
            mean = output.mean(dim=dims, keepdim=True)
            factors = 1 + (torch.rand((output.shape[0], output.shape[1], *([1] * len(dims))), device=output.device) * 2 - 1) * contrast
            output = (output - mean) * factors + mean
        return output

    def _probabilities_for_consistency(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)

    def _consistency_loss(self, student_outputs, teacher_outputs) -> torch.Tensor:
        student_outputs = _deep_supervision_outputs(student_outputs)
        teacher_outputs = _deep_supervision_outputs(teacher_outputs)
        weights = np.array([1 / (2 ** idx) for idx in range(len(student_outputs))], dtype=np.float32)
        if len(weights) > 1:
            weights[-1] = 0
        weights = weights / weights.sum()

        loss = student_outputs[0].new_tensor(0.0)
        for weight, student, teacher in zip(weights, student_outputs, teacher_outputs):
            if weight <= 0:
                continue
            loss = loss + float(weight) * F.mse_loss(
                self._probabilities_for_consistency(student),
                self._probabilities_for_consistency(teacher.detach()),
            )
        return loss

    @torch.no_grad()
    def _update_teacher_ema(self) -> None:
        if self.teacher_network is None:
            return
        student_state = model_state_dict(self.network)
        teacher_state = model_state_dict(self.teacher_network)
        for key, teacher_value in teacher_state.items():
            student_value = student_state[key].detach()
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(self.mt_ema_decay).add_(student_value, alpha=1 - self.mt_ema_decay)
            else:
                teacher_value.copy_(student_value)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        data = self._apply_random_histogram_matching(data)

        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        lambda_cons = self._consistency_weight()
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            supervised_loss = self.loss(output, target)
            if lambda_cons > 0 and self.teacher_network is not None:
                consistency_data = (
                    self.mt_unlabeled_pool.sample_batch(data.shape[0], self.device, data.dtype)
                    if self.mt_unlabeled_pool is not None
                    else data
                )
                student_input = self._perturb_for_teacher_training(consistency_data, strong=True)
                teacher_input = self._perturb_for_teacher_training(consistency_data, strong=False)
                student_outputs = self.network(student_input)
                with torch.no_grad():
                    teacher_outputs = self.teacher_network(teacher_input)
                consistency_loss = self._consistency_loss(student_outputs, teacher_outputs)
                loss = supervised_loss + lambda_cons * consistency_loss
            else:
                consistency_loss = supervised_loss.new_tensor(0.0)
                loss = supervised_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        if lambda_cons > 0 and self.teacher_network is not None:
            self._update_teacher_ema()

        return {
            "loss": loss.detach().cpu().numpy(),
            "supervised_loss": supervised_loss.detach().cpu().numpy(),
            "consistency_loss": consistency_loss.detach().cpu().numpy(),
        }

    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)
        if self.local_rank == 0 and self.mt_enabled and self.teacher_network is not None and os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
            checkpoint["teacher_network_weights"] = {
                key: value.detach().cpu()
                for key, value in model_state_dict(self.teacher_network).items()
            }
            checkpoint["wholeheart_mean_teacher"] = {
                "mt_start_epoch": self.mt_start_epoch,
                "mt_rampup_epochs": self.mt_rampup_epochs,
                "mt_max_weight": self.mt_max_weight,
                "mt_ema_decay": self.mt_ema_decay,
            }
            torch.save(checkpoint, filename)

    def load_checkpoint(self, filename_or_checkpoint):
        checkpoint = (
            torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
            if isinstance(filename_or_checkpoint, str)
            else filename_or_checkpoint
        )
        super().load_checkpoint(checkpoint)
        if self.mt_enabled and self.teacher_network is not None:
            teacher_weights = checkpoint.get("teacher_network_weights")
            if teacher_weights is not None:
                load_model_state_dict(self.teacher_network, teacher_weights)
            else:
                load_model_state_dict(self.teacher_network, model_state_dict(self.network))
            self._freeze_teacher()
