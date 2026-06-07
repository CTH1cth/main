from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common.utils import (
    build_image_items,
    check_exact_keys,
    manifest_to_map,
    read_jsonl,
    torch_load,
)


def _feature_manifest_path(cfg, split):
    # feature cache manifest 按 split 管理，train/val/test 不混用。
    return Path(cfg.CACHE_ROOT) / "features_cache" / cfg.BACKBONE_KEY / f"manifest_{split}.jsonl"


def _pseudo_manifest_path(cfg):
    # fixed pseudo 只为训练集生成，因此 manifest 名固定为 train。
    return Path(cfg.CACHE_ROOT) / "pseudo_label_cache" / cfg.BACKBONE_KEY / "manifest_train.jsonl"


def _load_cache_payload(row, expected_dataset, expected_stem, tensor_name):
    # 读取 .pt 后再次校验 dataset/stem，防止 manifest 指向错误文件。
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"{tensor_name} cache must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"{tensor_name} dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"{tensor_name} stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if "tensor" not in payload:
        raise KeyError(f"{tensor_name} cache missing tensor: {row['cache_path']}")
    return payload


def _load_feature(row, expected_dataset, expected_stem):
    # feature tensor 必须是 [C,H,W]，channel 数后续用于初始化 head。
    payload = _load_cache_payload(row, expected_dataset, expected_stem, "feature")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3:
        raise RuntimeError(f"Feature tensor must be [C,H,W], got {list(tensor.shape)}")
    return tensor, payload


def _load_pseudo(row, expected_dataset, expected_stem):
    # pseudo tensor 必须是单通道 [1,H,W]，训练时再插值到 loss size。
    payload = _load_cache_payload(row, expected_dataset, expected_stem, "pseudo")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"Pseudo tensor must be [1,H,W], got {list(tensor.shape)}")
    return tensor, payload


def _load_gt(gt_path):
    # GT 读成 0/1 float tensor，指标计算前保持 [1,H,W]。
    gt = Image.open(gt_path).convert("L")
    array = np.asarray(gt, dtype=np.float32) / 255.0
    array = (array > 0.5).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class CachedTrainDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        # 训练集只建立 image/cache 索引，不读取 GT，避免把训练 GT 引入监督。
        self.items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
        self.keys = [(item["dataset"], item["stem"]) for item in self.items]

        feature_rows = read_jsonl(_feature_manifest_path(cfg, "train"))
        pseudo_rows = read_jsonl(_pseudo_manifest_path(cfg))
        self.feature_map = manifest_to_map(feature_rows, _feature_manifest_path(cfg, "train"))
        self.pseudo_map = manifest_to_map(pseudo_rows, _pseudo_manifest_path(cfg))
        # train cache 必须与训练图片精确一一对应；错配直接报错，不按 list 顺序兜底。
        check_exact_keys("feature train cache", self.feature_map.keys(), self.keys)
        check_exact_keys("pseudo train cache", self.pseudo_map.keys(), self.keys)

        first_dataset, first_stem = self.keys[0]
        feature, _ = _load_feature(self.feature_map[(first_dataset, first_stem)], first_dataset, first_stem)
        pseudo, _ = _load_pseudo(self.pseudo_map[(first_dataset, first_stem)], first_dataset, first_stem)
        self.feature_shape = list(feature.shape)
        self.pseudo_shape = list(pseudo.shape)
        self.in_channels = int(feature.shape[0])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)

        feature, feature_payload = _load_feature(self.feature_map[key], dataset, stem)
        pseudo, pseudo_payload = _load_pseudo(self.pseudo_map[key], dataset, stem)
        if feature_payload.get("dataset") != pseudo_payload.get("dataset"):
            raise RuntimeError(f"Feature/pseudo dataset mismatch for {dataset}/{stem}")
        if feature_payload.get("stem") != pseudo_payload.get("stem"):
            raise RuntimeError(f"Feature/pseudo stem mismatch for {dataset}/{stem}")

        return {
            "feature": feature,
            "pseudo": pseudo,
            "dataset": dataset,
            "stem": stem,
            "image_path": item["image_path"],
        }


class CachedEvalDataset(Dataset):
    def __init__(self, cfg, split, datasets=None):
        if split not in {"val", "test"}:
            raise ValueError(f"Eval split must be val or test, got {split}")
        self.cfg = cfg
        self.split = split
        dataset_names = list(datasets) if datasets is not None else (
            cfg.VAL_DATASETS if split == "val" else cfg.TEST_DATASETS
        )
        # GT 只在 val/test dataset 中读取，用于指标计算。
        self.items = build_image_items(cfg.DATA_ROOT, dataset_names, require_gt=True)
        self.keys = [(item["dataset"], item["stem"]) for item in self.items]

        feature_rows = read_jsonl(_feature_manifest_path(cfg, split))
        self.feature_map = manifest_to_map(feature_rows, _feature_manifest_path(cfg, split))
        missing = sorted(set(self.keys) - set(self.feature_map.keys()))
        if missing:
            raise RuntimeError(f"feature {split} cache missing first 10: {missing[:10]}")

        first_dataset, first_stem = self.keys[0]
        feature, _ = _load_feature(self.feature_map[(first_dataset, first_stem)], first_dataset, first_stem)
        self.feature_shape = list(feature.shape)
        self.in_channels = int(feature.shape[0])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)
        feature, payload = _load_feature(self.feature_map[key], dataset, stem)
        gt = _load_gt(item["gt_path"])
        original_size = tuple(payload.get("original_size", gt.shape[-2:]))
        return {
            "feature": feature,
            "gt": gt,
            "dataset": dataset,
            "stem": stem,
            "image_path": item["image_path"],
            "gt_path": item["gt_path"],
            "original_size": original_size,
        }
