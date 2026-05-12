import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import SimpleITK as sitk
import glob
import json
from pathlib import Path
from skimage.exposure import match_histograms

class Patch3DDataset_Teacher(Dataset):
    def __init__(self, image_dir="/home/jincan/long_seg/unet_liver/DATA/for_nnunet/R_P_GED4",
                 patch_size=(48, 192, 224),
                 normalization=True):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "**", "GED4.npy"), recursive=True))
        self.patch_size = patch_size

        # 构建 patch 索引表 [(volume_idx, z, y, x)]
        self.patch_index = []

        self.normalization = normalization

 
    def ZScoreNormalization(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(torch.float32)
        mean = image.mean()
        std = image.std()
        normalized = (image - mean) / max(std.item(), 1e-8)
        return normalized

    def get_random_start_pos(self, im_shape, patch_size):
        """
        给定图像大小和目标 patch 大小，返回一个合法的随机起始位置，用于从图像中裁剪 patch。

        :param im_shape: tuple[int]，图像的 shape，例如 (Z, Y, X)
        :param patch_size: tuple[int]，希望裁剪出的 patch shape，例如 (64, 128, 128)
        :return: tuple[int]，patch 的起始坐标（起点），shape 同维度数
        """
        assert len(im_shape) == len(patch_size), "维度数必须匹配"

        start_pos = []
        for img_dim, patch_dim in zip(im_shape, patch_size):
            if img_dim <= patch_dim:
                # 图像比 patch 小，直接从 0 开始并 pad（通常在调用者处理）
                start = 0
            else:
                # 随机选择一个合法的起点，使 patch 完全落在图像内
                start = np.random.randint(0, img_dim - patch_dim + 1)
            start_pos.append(start)

        return tuple(start_pos)

    def __len__(self):
        return len(self.image_paths)

    def load_np_array_and_metadata(self, im_dir, is_image=True):
        array_path = Path(im_dir)

        np_array = np.load(array_path)

        return np_array

    def _pad_if_needed(self, image, patch_size, bias=3):
        """
        对 image (Tensor) 进行对称 padding，使其大小至少为 patch_size
        :param image: Tensor，形状为 [D, H, W]
        :param patch_size: list or tuple, [D, H, W]
        :return: padded image (Tensor), shape >= patch_size
        """
        assert image.ndim == 3, "Input image must be [D, H, W]"
        pad = []
        for i in range(2, -1, -1):  # 从 W, H, D 方向构建 pad
            diff = max(0, patch_size[i] - image.shape[i])
            pad_left = diff // 2 + bias
            pad_right = diff - pad_left + bias
            pad.extend([pad_left, pad_right])

        # torch.nn.functional.pad 的 pad 顺序是从最后一维开始的
        padded = F.pad(image, pad, mode='constant', value=0)
        return padded

    def __getitem__(self, idx):
        image = self.load_np_array_and_metadata(self.image_paths[idx])
        image_tensor = torch.FloatTensor(image).float()
        if np.random.choice([True, False]):
            idx_ref = np.random.choice(len(self.image_paths))
            image_ref = self.load_np_array_and_metadata(self.image_paths[idx_ref])
            ref_tensor = torch.FloatTensor(image).float()

            image_tensor = convert_style(image_tensor, ref_tensor)

        if self.normalization:
            image_tensor = self.ZScoreNormalization(image_tensor)
            
        # 若尺寸小于 patch_size，则先 pad
        if any(s < p for s, p in zip(image_tensor.shape, self.patch_size)):
            image_tensor = self._pad_if_needed(image_tensor, self.patch_size)

        sta_pos = self.get_random_start_pos(image_tensor.shape, self.patch_size)

        image_patch = image_tensor[sta_pos[0]:sta_pos[0]+self.patch_size[0],
                            sta_pos[1]:sta_pos[1]+self.patch_size[1],
                            sta_pos[2]:sta_pos[2]+self.patch_size[2]]

        # image_patch = self._pad_if_needed(image_patch, self.patch_size)
        image_patch = image_patch.unsqueeze(0)  # [1, D, H, W]

        return {'data': image_patch}



    
def custom_collate_fn_Teacher(batch):
    """
    batch: List of tuples (image_choose, mask_choose, affine_mask2, save_path_nii, crop_offset_mask, tp_name)
    
    按照返回值的顺序组织 batch，并确保 batch 形式正确：
    - `image_choose` 和 `mask_choose` 会被 `stack` 成 Tensor
    - `affine_mask2`, `save_path_nii`, `crop_offset_mask`, `tp_name` 保持原格式，变成 list
    """
    batch_={}
    batch_['data'] = torch.stack([item['data'] for item in batch], dim=0)
    return batch_


def sample_random_value(center1=5000, center2=20000, delta_ratio=0.1):
    center = np.random.choice([center1, center2])
    delta = delta_ratio * center
    return np.random.uniform(center - delta, center + delta)

def scale_intensity(image_tensor, out_range=5000, delta_ratio=0.1):
    delta = delta_ratio * out_range
    upper = torch.empty(1).uniform_(out_range - delta, out_range + delta).item()

    cur_max = approx_quantile(image_tensor, q=0.99)
    if cur_max == 0:
        cur_max = torch.tensor(1.0, device=image_tensor.device)

    scale_factor = upper / cur_max
    scaled = image_tensor * scale_factor
    return scaled, upper


def approx_quantile(tensor, q=0.99, sample_size=100_000):
    """
    近似计算大张量的百分位数（适用于 3D 医学图像等）
    """
    flat = tensor.flatten()
    if flat.numel() > sample_size:
        indices = torch.randperm(flat.numel(), device=tensor.device)[:sample_size]
        sample = flat[indices]
    else:
        sample = flat
    return torch.quantile(sample, q)


def add_intensity_perturbation_conditional(image, noise_std=50, gamma_range=(0.8, 1.2), threshold=2000):
    mask = image <= threshold
    noise = torch.normal(0, noise_std, size=image.shape, device=image.device)
    noisy = image.clone()
    noisy[mask] += noise[mask]
    noisy = torch.clamp(noisy, min=0)

    gamma = torch.empty(1).uniform_(*gamma_range).item()
    noisy_gamma = noisy.clone()
    if mask.sum() > 0:
        max_val = noisy[mask].max()
        if max_val > 0:
            normalized = noisy[mask] / max_val
            noisy_gamma[mask] = torch.pow(normalized, gamma) * max_val
    return noisy_gamma


def simulate_gray(image_tensor, threshold=5000, delta_ratio=0.3):
    scaled, upper = scale_intensity(image_tensor, out_range=threshold)
    perturbed = add_intensity_perturbation_conditional(
        scaled,
        noise_std=upper * 0.04,
        gamma_range=(1.0, 1.5),
        threshold=upper * delta_ratio
    )
    max_val = perturbed.max()
    final = perturbed / max_val * upper if max_val > 0 else perturbed
    return final.to(dtype=torch.float32)

def ZScoreNormalization_local(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    mean = image.mean()
    std = image.std()
    image = (image - mean) / max(std, 1e-8)
    return image

def fast_histogram_match(source: torch.Tensor, reference: torch.Tensor, num_bins=256):
    src_flat = source.flatten()
    ref_flat = reference.flatten()

    s_min, s_max = src_flat.min(), src_flat.max()
    r_min, r_max = ref_flat.min(), ref_flat.max()

    s_hist = torch.histc(src_flat, bins=num_bins, min=s_min.item(), max=s_max.item())
    r_hist = torch.histc(ref_flat, bins=num_bins, min=r_min.item(), max=r_max.item())

    s_cdf = torch.cumsum(s_hist, dim=0)
    r_cdf = torch.cumsum(r_hist, dim=0)
    s_cdf = s_cdf / s_cdf[-1].clone()
    r_cdf = r_cdf / r_cdf[-1].clone()

    r_bin_edges = torch.linspace(r_min, r_max, steps=num_bins, device=source.device)
    lookup_table = torch.zeros_like(s_cdf)

    for i in range(num_bins):
        idx = torch.argmin(torch.abs(r_cdf - s_cdf[i]))
        lookup_table[i] = r_bin_edges[idx]

    s_bin_edges = torch.linspace(s_min, s_max, steps=num_bins, device=source.device)
    digitized = torch.bucketize(source, s_bin_edges, right=True)
    matched = lookup_table[digitized.clamp(0, num_bins - 1)]
    return matched


def convert_style(image_tensor, ref_tensor, t1=5000, t2=20000):

    # Load original image
    matched_tensor = fast_histogram_match(image_tensor, ref_tensor)
    threshold = sample_random_value(center1=t1, center2=t2, delta_ratio=0.1)
    modified = simulate_gray(matched_tensor, threshold)
    result = ZScoreNormalization_local(modified)
    return result