from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset

from common.utils import (
    build_image_items,
    find_gt_path,
    manifest_to_map,
    nper_pseudo_bank_manifest_path,
    read_jsonl,
    torch_load,
)
from nper.native_cue import sobel_tensor


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


NPER_TENSOR_FIELDS = {
    "p_fixed": torch.float32,
    "p_despl": torch.float32,
    "p_gcm": torch.float32,
    "p_init": torch.float32,
    "anchor_fg": torch.bool,
    "anchor_bg": torch.bool,
    "pixel_weight": torch.float32,
}


def load_image_tensor(image_path, size):
    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)
    image = image.resize((int(size), int(size)), RESAMPLE_BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, original_size


def load_gt_tensor(gt_path):
    gt = Image.open(gt_path).convert("L")
    array = np.asarray(gt, dtype=np.float32) / 255.0
    array = (array > 0.5).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


def _load_bank_payload(row, dataset, stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"NPER pseudo bank payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"NPER pseudo bank key mismatch: {row['cache_path']}")
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"NPER pseudo bank backbone mismatch: {payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    out = dict(payload)
    for name, dtype in NPER_TENSOR_FIELDS.items():
        if name not in payload:
            raise KeyError(f"NPER pseudo bank missing {name}: {row['cache_path']}")
        tensor = payload[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"NPER pseudo bank {name} must be tensor: {row['cache_path']}")
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"NPER pseudo bank {name} shape mismatch: {list(tensor.shape)} != {expected_shape}"
            )
        out[name] = tensor.to(dtype=dtype)
    return out


class NPERTrainDataset(Dataset):
    def __init__(self, cfg, max_samples=-1):
        self.cfg = cfg
        self.items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
        if max_samples is not None and int(max_samples) >= 0:
            self.items = self.items[: int(max_samples)]
        if not self.items:
            raise RuntimeError("NPER training dataset is empty.")
        self.keys = [(item["dataset"], item["stem"]) for item in self.items]
        manifest_path = nper_pseudo_bank_manifest_path(cfg)
        rows = read_jsonl(manifest_path)
        self.bank_map = manifest_to_map(rows, manifest_path)
        missing = sorted(set(self.keys) - set(self.bank_map))
        if missing:
            raise RuntimeError(f"NPER pseudo bank missing first 10: {missing[:10]}")
        first_dataset, first_stem = self.keys[0]
        self.first_cache_path = self.bank_map[(first_dataset, first_stem)]["cache_path"]
        first_payload = _load_bank_payload(self.bank_map[(first_dataset, first_stem)], first_dataset, first_stem, cfg)
        self.pseudo_shape = list(first_payload["p_init"].shape)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        dataset = item["dataset"]
        stem = item["stem"]
        image, original_size = load_image_tensor(item["image_path"], int(self.cfg.INPUT_SIZE))
        payload = _load_bank_payload(self.bank_map[(dataset, stem)], dataset, stem, self.cfg)
        sample = {
            "image": image,
            "image_path": item["image_path"],
            "dataset": dataset,
            "stem": stem,
            "original_size": original_size,
            "sobel": sobel_tensor(image),
            "quality_score": torch.tensor(float(payload.get("quality_score", 0.5)), dtype=torch.float32),
            "hard_score": torch.tensor(float(payload.get("hard_score", 0.5)), dtype=torch.float32),
            "mnp_score_fixed": torch.tensor(float(payload.get("mnp_score_fixed", 0.0)), dtype=torch.float32),
            "mnp_score_despl": torch.tensor(float(payload.get("mnp_score_despl", 0.0)), dtype=torch.float32),
            "mnp_score_gcm": torch.tensor(float(payload.get("mnp_score_gcm", 0.0)), dtype=torch.float32),
        }
        for name in NPER_TENSOR_FIELDS:
            sample[name] = payload[name]
        return sample


class NPEREvalDataset(Dataset):
    def __init__(self, cfg, split, datasets=None, max_samples=-1):
        if split not in {"val", "test"}:
            raise ValueError(f"split must be val or test, got {split}")
        self.cfg = cfg
        self.split = split
        names = list(datasets) if datasets is not None else (
            cfg.VAL_DATASETS if split == "val" else cfg.TEST_DATASETS
        )
        self.items = build_image_items(cfg.DATA_ROOT, names, require_gt=True)
        if max_samples is not None and int(max_samples) >= 0:
            self.items = self.items[: int(max_samples)]
        if not self.items:
            raise RuntimeError(f"NPER {split} dataset is empty.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image, original_size = load_image_tensor(item["image_path"], int(self.cfg.INPUT_SIZE))
        gt_path = item.get("gt_path") or str(find_gt_path(self.cfg.DATA_ROOT, item["dataset"], item["stem"]))
        gt = load_gt_tensor(gt_path)
        return {
            "image": image,
            "gt": gt,
            "dataset": item["dataset"],
            "stem": item["stem"],
            "image_path": item["image_path"],
            "gt_path": gt_path,
            "original_size": original_size,
        }


def build_train_loader(cfg, max_samples=-1):
    dataset = NPERTrainDataset(cfg, max_samples=max_samples)
    generator = torch.Generator()
    generator.manual_seed(int(cfg.SEED))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.BATCH_SIZE),
        shuffle=True,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    return dataset, loader


def build_eval_loader(cfg, split, datasets=None, max_samples=-1):
    dataset = NPEREvalDataset(cfg, split=split, datasets=datasets, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.VAL_BATCH_SIZE),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def make_psta_view(images, scale=1.0, color_jitter=False, gaussian_blur=False):
    size = images.shape[-1]
    scaled = max(8, int(round(size * float(scale))))
    view = F.interpolate(images, size=(scaled, scaled), mode="bilinear", align_corners=False)
    if scaled >= size:
        top = (scaled - size) // 2
        left = (scaled - size) // 2
        view = view[:, :, top : top + size, left : left + size]
    else:
        pad_total = size - scaled
        left = pad_total // 2
        right = pad_total - left
        view = F.pad(view, (left, right, left, right), mode="reflect")
    if color_jitter:
        factor = 0.9 + 0.2 * torch.rand(images.shape[0], 1, 1, 1, device=images.device)
        view = (view * factor).clamp(0.0, 1.0)
    if gaussian_blur:
        view = F.avg_pool2d(view, kernel_size=3, stride=1, padding=1)
    return view
