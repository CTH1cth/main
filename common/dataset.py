from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common.utils import (
    build_image_items,
    ccr_manifest_path,
    check_exact_keys,
    manifest_to_map,
    qra_manifest_path,
    read_jsonl,
    torch_load,
)


QRA_TENSOR_FIELDS = {
    "p_fixed": torch.float32,
    "p_despl": torch.float32,
    "p_fused": torch.float32,
    "anchor_fg": torch.bool,
    "anchor_bg": torch.bool,
    "pixel_weight": torch.float32,
}

CCR_TENSOR_FIELDS = {
    "p_fixed": torch.float32,
    "p_despl": torch.float32,
    "p_corr": torch.float32,
    "agree_fg": torch.bool,
    "agree_bg": torch.bool,
    "raw_expand": torch.bool,
    "raw_shrink": torch.bool,
    "trusted_expand": torch.bool,
    "trusted_shrink": torch.bool,
    "anchor_fg": torch.bool,
    "anchor_bg": torch.bool,
}


def _feature_manifest_path(cfg, split):
    # feature cache manifest 按 split 管理，train/val/test 不混用。
    return Path(cfg.CACHE_ROOT) / "features_cache" / cfg.BACKBONE_KEY / f"manifest_{split}.jsonl"


def _pseudo_manifest_path(cfg):
    # fixed pseudo 只为训练集生成，因此 manifest 名固定为 train。
    return Path(cfg.CACHE_ROOT) / "pseudo_label_cache" / cfg.BACKBONE_KEY / "manifest_train.jsonl"


def _pseudo_override_path(override_root, dataset, stem):
    # GCM candidate 输出按 dataset/candidate_cache/stem.pt 组织。
    return Path(override_root) / dataset / "candidate_cache" / f"{stem}.pt"


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


def _load_qra(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"QRA cache must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"QRA dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"QRA stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"QRA backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    out = dict(payload)
    for name, dtype in QRA_TENSOR_FIELDS.items():
        if name not in payload:
            raise KeyError(f"QRA cache missing {name}: {row['cache_path']}")
        tensor = payload[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"QRA {name} must be a tensor: {row['cache_path']}")
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"QRA {name} shape mismatch for {row['cache_path']}: "
                f"{list(tensor.shape)} != {expected_shape}"
            )
        out[name] = tensor.to(dtype=dtype)

    quality = int(payload.get("quality", -1))
    if quality not in {0, 1, 2}:
        raise RuntimeError(f"QRA quality must be 0/1/2, got {quality}: {row['cache_path']}")
    out["quality"] = quality
    return out


def _load_ccr(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"CCR cache must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"CCR dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"CCR stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"CCR backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )

    size = int(getattr(cfg, "CCR_LOSS_SIZE", cfg.LOSS_SIZE))
    expected_shape = [1, size, size]
    out = dict(payload)
    for name, dtype in CCR_TENSOR_FIELDS.items():
        if name not in payload:
            raise KeyError(f"CCR cache missing {name}: {row['cache_path']}")
        tensor = payload[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"CCR {name} must be a tensor: {row['cache_path']}")
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"CCR {name} shape mismatch for {row['cache_path']}: "
                f"{list(tensor.shape)} != {expected_shape}"
            )
        out[name] = tensor.to(dtype=dtype)

    quality = int(payload.get("quality", -1))
    if quality not in {0, 1, 2}:
        raise RuntimeError(f"CCR quality must be 0/1/2, got {quality}: {row['cache_path']}")
    out["quality"] = quality
    return out


def _load_gt(gt_path):
    # GT 读成 0/1 float tensor，指标计算前保持 [1,H,W]。
    gt = Image.open(gt_path).convert("L")
    array = np.asarray(gt, dtype=np.float32) / 255.0
    array = (array > 0.5).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class CachedTrainDataset(Dataset):
    def __init__(self, cfg, max_samples=-1):
        self.cfg = cfg
        self.use_qra = bool(getattr(cfg, "USE_QRA", False))
        self.use_ccr = bool(getattr(cfg, "USE_CCR", False))
        if self.use_qra and self.use_ccr:
            raise RuntimeError("USE_QRA=True and USE_CCR=True cannot be combined.")
        # 训练集只建立 image/cache 索引，不读取 GT，避免把训练 GT 引入监督。
        self.items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
        if max_samples >= 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise RuntimeError("Training dataset is empty.")
        self.keys = [(item["dataset"], item["stem"]) for item in self.items]

        feature_rows = read_jsonl(_feature_manifest_path(cfg, "train"))
        self.feature_map = manifest_to_map(feature_rows, _feature_manifest_path(cfg, "train"))
        if max_samples < 0:
            check_exact_keys(
                "feature train cache", self.feature_map.keys(), self.keys
            )
        else:
            missing_features = sorted(set(self.keys) - set(self.feature_map))
            if missing_features:
                raise RuntimeError(
                    f"feature train cache missing first 10: {missing_features[:10]}"
                )

        override = getattr(cfg, "PSEUDO_CACHE_OVERRIDE", None)
        self.pseudo_cache_override = (
            str(Path(override).expanduser().resolve()) if override else None
        )
        if self.use_qra and self.pseudo_cache_override is not None:
            raise RuntimeError("USE_QRA=True cannot be combined with PSEUDO_CACHE_OVERRIDE.")
        if self.use_ccr and self.pseudo_cache_override is not None:
            raise RuntimeError("USE_CCR=True cannot be combined with PSEUDO_CACHE_OVERRIDE.")
        self.original_pseudo_cache_root = str(
            (
                Path(cfg.CACHE_ROOT)
                / "pseudo_label_cache"
                / cfg.BACKBONE_KEY
            ).resolve()
        )
        if self.pseudo_cache_override is None:
            pseudo_manifest = _pseudo_manifest_path(cfg)
            pseudo_rows = read_jsonl(pseudo_manifest)
            self.pseudo_map = manifest_to_map(pseudo_rows, pseudo_manifest)
            if max_samples < 0:
                check_exact_keys(
                    "pseudo train cache", self.pseudo_map.keys(), self.keys
                )
            else:
                missing_pseudo = sorted(set(self.keys) - set(self.pseudo_map))
                if missing_pseudo:
                    raise RuntimeError(
                        f"pseudo train cache missing first 10: {missing_pseudo[:10]}"
                    )
            self.actual_pseudo_cache_root = self.original_pseudo_cache_root
            self.actual_pseudo_cache_pattern = (
                f"{self.original_pseudo_cache_root}/<dataset>/<stem>.pt"
            )
        else:
            override_root = Path(self.pseudo_cache_override)
            if not override_root.is_dir():
                raise FileNotFoundError(
                    f"Pseudo cache override directory not found: {override_root}"
                )
            self.pseudo_map = {}
            missing_pseudo = []
            for item in self.items:
                key = (item["dataset"], item["stem"])
                cache_path = _pseudo_override_path(
                    override_root, item["dataset"], item["stem"]
                )
                if not cache_path.is_file():
                    missing_pseudo.append(str(cache_path))
                    continue
                self.pseudo_map[key] = {
                    "dataset": item["dataset"],
                    "stem": item["stem"],
                    "cache_path": str(cache_path.resolve()),
                }
            if missing_pseudo:
                raise RuntimeError(
                    "pseudo cache override missing first 10: "
                    f"{missing_pseudo[:10]}"
                )
            self.actual_pseudo_cache_root = str(override_root.resolve())
            self.actual_pseudo_cache_pattern = (
                f"{self.actual_pseudo_cache_root}"
                "/<dataset>/candidate_cache/<stem>.pt"
            )

        first_dataset, first_stem = self.keys[0]
        feature, _ = _load_feature(self.feature_map[(first_dataset, first_stem)], first_dataset, first_stem)
        pseudo, pseudo_payload = _load_pseudo(
            self.pseudo_map[(first_dataset, first_stem)],
            first_dataset,
            first_stem,
        )
        self.feature_shape = list(feature.shape)
        self.pseudo_shape = list(pseudo.shape)
        self.in_channels = int(feature.shape[0])
        self.first_pseudo_cache_path = self.pseudo_map[
            (first_dataset, first_stem)
        ]["cache_path"]
        self.pseudo_source = pseudo_payload.get("source", "original_fixed")
        self.pseudo_final_candidate = pseudo_payload.get("final_candidate", "")
        if self.pseudo_cache_override is not None:
            input_size = int(cfg.DINO["pseudo_input_size"])
            patch_size = int(cfg.DINO["patch_size"])
            grid = input_size // patch_size
            expected_shape = [1, grid, grid]
            if self.pseudo_shape != expected_shape:
                raise RuntimeError(
                    "Override pseudo tensor shape mismatch: "
                    f"{self.pseudo_shape} != {expected_shape} | "
                    f"{self.first_pseudo_cache_path}"
                )
            self.override_expected_shape = expected_shape
        else:
            self.override_expected_shape = None

        self.qra_map = None
        self.qra_cache_root = None
        self.qra_first_cache_path = None
        self.qra_first_quality = None
        self.qra_first_sim = None
        self.qra_first_anchor_ratio = None
        if self.use_qra:
            qra_manifest = qra_manifest_path(cfg)
            qra_rows = read_jsonl(qra_manifest)
            self.qra_map = manifest_to_map(qra_rows, qra_manifest)
            if max_samples < 0:
                check_exact_keys("QRA train cache", self.qra_map.keys(), self.keys)
            else:
                missing_qra = sorted(set(self.keys) - set(self.qra_map))
                if missing_qra:
                    raise RuntimeError(f"QRA train cache missing first 10: {missing_qra[:10]}")
            qra_payload = _load_qra(
                self.qra_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.qra_cache_root = str(qra_manifest.parent.resolve())
            self.qra_first_cache_path = self.qra_map[(first_dataset, first_stem)]["cache_path"]
            self.qra_first_quality = int(qra_payload["quality"])
            self.qra_first_sim = float(qra_payload.get("sim", 0.0))
            self.qra_first_anchor_ratio = float(qra_payload.get("anchor_ratio", 0.0))

        self.ccr_map = None
        self.ccr_cache_root = None
        self.ccr_first_cache_path = None
        self.ccr_first_quality = None
        self.ccr_first_iou = None
        self.ccr_first_anchor_ratio = None
        if self.use_ccr:
            ccr_manifest = ccr_manifest_path(cfg)
            ccr_rows = read_jsonl(ccr_manifest)
            self.ccr_map = manifest_to_map(ccr_rows, ccr_manifest)
            if max_samples < 0:
                check_exact_keys("CCR train cache", self.ccr_map.keys(), self.keys)
            else:
                missing_ccr = sorted(set(self.keys) - set(self.ccr_map))
                if missing_ccr:
                    raise RuntimeError(f"CCR train cache missing first 10: {missing_ccr[:10]}")
            ccr_payload = _load_ccr(
                self.ccr_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.ccr_cache_root = str(ccr_manifest.parent.resolve())
            self.ccr_first_cache_path = self.ccr_map[(first_dataset, first_stem)]["cache_path"]
            self.ccr_first_quality = int(ccr_payload["quality"])
            self.ccr_first_iou = float(ccr_payload.get("iou_fixed_despl", 0.0))
            self.ccr_first_anchor_ratio = float(ccr_payload.get("anchor_ratio", 0.0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)

        feature, feature_payload = _load_feature(self.feature_map[key], dataset, stem)
        pseudo, pseudo_payload = _load_pseudo(self.pseudo_map[key], dataset, stem)
        if (
            self.override_expected_shape is not None
            and list(pseudo.shape) != self.override_expected_shape
        ):
            raise RuntimeError(
                "Override pseudo tensor shape mismatch: "
                f"{list(pseudo.shape)} != {self.override_expected_shape} | "
                f"{self.pseudo_map[key]['cache_path']}"
            )
        if feature_payload.get("dataset") != pseudo_payload.get("dataset"):
            raise RuntimeError(f"Feature/pseudo dataset mismatch for {dataset}/{stem}")
        if feature_payload.get("stem") != pseudo_payload.get("stem"):
            raise RuntimeError(f"Feature/pseudo stem mismatch for {dataset}/{stem}")

        sample = {
            "feature": feature,
            "pseudo": pseudo,
            "dataset": dataset,
            "stem": stem,
            "image_path": item["image_path"],
        }
        if self.use_qra:
            qra_payload = _load_qra(self.qra_map[key], dataset, stem, self.cfg)
            sample.update(
                {
                    "qra_p_fixed": qra_payload["p_fixed"].float(),
                    "qra_p_despl": qra_payload["p_despl"].float(),
                    "qra_p_fused": qra_payload["p_fused"].float(),
                    "qra_anchor_fg": qra_payload["anchor_fg"].bool(),
                    "qra_anchor_bg": qra_payload["anchor_bg"].bool(),
                    "qra_pixel_weight": qra_payload["pixel_weight"].float(),
                    "qra_quality": int(qra_payload["quality"]),
                    "qra_sim": float(qra_payload.get("sim", 0.0)),
                    "qra_area": float(qra_payload.get("area", 0.0)),
                    "qra_num_cc": int(qra_payload.get("num_cc", 0)),
                    "qra_edge_touch": int(qra_payload.get("edge_touch", 0)),
                    "qra_anchor_ratio": float(qra_payload.get("anchor_ratio", 0.0)),
                }
            )
        if self.use_ccr:
            ccr_payload = _load_ccr(self.ccr_map[key], dataset, stem, self.cfg)
            sample.update(
                {
                    "ccr_p_fixed": ccr_payload["p_fixed"].float(),
                    "ccr_p_despl": ccr_payload["p_despl"].float(),
                    "ccr_p_corr": ccr_payload["p_corr"].float(),
                    "ccr_trusted_expand": ccr_payload["trusted_expand"].bool(),
                    "ccr_trusted_shrink": ccr_payload["trusted_shrink"].bool(),
                    "ccr_anchor_fg": ccr_payload["anchor_fg"].bool(),
                    "ccr_anchor_bg": ccr_payload["anchor_bg"].bool(),
                    "ccr_quality": int(ccr_payload["quality"]),
                    "ccr_iou_fixed_despl": float(ccr_payload.get("iou_fixed_despl", 0.0)),
                    "ccr_fixed_area": float(ccr_payload.get("fixed_area", 0.0)),
                    "ccr_despl_area": float(ccr_payload.get("despl_area", 0.0)),
                    "ccr_corr_area": float(ccr_payload.get("corr_area", 0.0)),
                    "ccr_trusted_expand_area": float(ccr_payload.get("trusted_expand_area", 0.0)),
                    "ccr_trusted_shrink_area": float(ccr_payload.get("trusted_shrink_area", 0.0)),
                    "ccr_anchor_ratio": float(ccr_payload.get("anchor_ratio", 0.0)),
                }
            )
        return sample


class CachedEvalDataset(Dataset):
    def __init__(self, cfg, split, datasets=None, max_samples=-1):
        if split not in {"val", "test"}:
            raise ValueError(f"Eval split must be val or test, got {split}")
        self.cfg = cfg
        self.split = split
        dataset_names = list(datasets) if datasets is not None else (
            cfg.VAL_DATASETS if split == "val" else cfg.TEST_DATASETS
        )
        # GT 只在 val/test dataset 中读取，用于指标计算。
        self.items = build_image_items(cfg.DATA_ROOT, dataset_names, require_gt=True)
        if max_samples >= 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise RuntimeError(f"{split} dataset is empty.")
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
