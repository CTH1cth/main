from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from common.dre_safe_prior import build_dre_safe_prior
from common.utils import (
    build_image_items,
    ccr_manifest_path,
    check_exact_keys,
    despl_light_cache_manifest_path,
    despl_pseudo_bank_manifest_path,
    drepp_manifest_path,
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

DREPP_TENSOR_FIELDS = {
    "p_despl": torch.float32,
    "p_fixed": torch.float32,
    "core_fg": torch.bool,
    "core_bg": torch.bool,
    "uncertain": torch.bool,
    "fixed_local_recall": torch.bool,
    "boundary_band": torch.bool,
    "memory_init": torch.float32,
    "feature_sim": torch.float32,
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


def _as_single_channel_tensor(payload, name, cache_path, required=True):
    if name not in payload:
        if required:
            raise KeyError(f"DESPL pseudo bank missing {name}: {cache_path}")
        return None
    tensor = payload[name]
    if not torch.is_tensor(tensor):
        raise TypeError(f"DESPL pseudo bank {name} must be tensor: {cache_path}")
    tensor = tensor.float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(
            f"DESPL pseudo bank {name} must be [1,H,W], got {list(tensor.shape)}: {cache_path}"
        )
    return tensor


def _validate_unit_range(tensor, name, cache_path):
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise RuntimeError(
            f"{name} values must be in [0,1], got min={min_value:.6f}, "
            f"max={max_value:.6f}: {cache_path}"
        )


def _load_despl_pseudo(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DESPL pseudo bank payload must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"DESPL pseudo bank dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"DESPL pseudo bank stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"DESPL pseudo bank backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    if "p_despl" not in payload:
        raise KeyError(
            f"DESPL pseudo missing for {expected_dataset}/{expected_stem}: {row['cache_path']}"
        )
    p_despl = _as_single_channel_tensor(payload, "p_despl", row["cache_path"], required=True)
    _validate_unit_range(p_despl, "p_despl", row["cache_path"])
    return {
        "p_despl": p_despl,
        "p_fixed": _as_single_channel_tensor(payload, "p_fixed", row["cache_path"], required=False),
    }


def _load_despl_light_pseudo(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DESPL light cache payload must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"DESPL light cache dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"DESPL light cache stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"DESPL light cache backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    tensor = payload.get("tensor", payload.get("p_init"))
    if not torch.is_tensor(tensor):
        raise TypeError(f"DESPL light cache tensor/p_init must be tensor: {row['cache_path']}")
    tensor = tensor.float()
    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    if list(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"DESPL light cache tensor shape mismatch: {list(tensor.shape)} != {expected_shape} | "
            f"{row['cache_path']}"
        )
    pseudo_fixed = payload.get("p_fixed_68")
    pseudo_despl = payload.get("p_despl_68")
    has_pseudo_fixed = torch.is_tensor(pseudo_fixed)
    has_pseudo_despl = torch.is_tensor(pseudo_despl)
    if not torch.is_tensor(pseudo_fixed):
        pseudo_fixed = torch.zeros_like(tensor)
    if not torch.is_tensor(pseudo_despl):
        pseudo_despl = torch.zeros_like(tensor)
    else:
        pseudo_despl = pseudo_despl.float()
        if list(pseudo_despl.shape) != expected_shape:
            raise RuntimeError(
                f"DESPL light cache p_despl_68 shape mismatch: "
                f"{list(pseudo_despl.shape)} != {expected_shape} | {row['cache_path']}"
            )
        _validate_unit_range(pseudo_despl, "p_despl_68", row["cache_path"])
    if torch.is_tensor(pseudo_fixed):
        pseudo_fixed = pseudo_fixed.float()
        if list(pseudo_fixed.shape) != expected_shape:
            raise RuntimeError(
                f"DESPL light cache p_fixed_68 shape mismatch: "
                f"{list(pseudo_fixed.shape)} != {expected_shape} | {row['cache_path']}"
            )
    return {
        "pseudo": tensor,
        "pseudo_fixed": pseudo_fixed.float(),
        "pseudo_despl": pseudo_despl.float(),
        "p_init_area": float(payload.get("p_init_area", tensor.mean().item())),
        "p_fixed_area": float(payload.get("p_fixed_area", pseudo_fixed.float().mean().item())),
        "p_despl_area": float(payload.get("p_despl_area", pseudo_despl.float().mean().item())),
        "_has_p_fixed_68": bool(has_pseudo_fixed),
        "_has_p_despl_68": bool(has_pseudo_despl),
    }


def _resize_like(tensor, reference):
    if list(tensor.shape[-2:]) == list(reference.shape[-2:]):
        return tensor.float()
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


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


def _load_drepp(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DRE++ cache must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"DRE++ dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"DRE++ stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"DRE++ backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    if bool(payload.get("global_blend", False)):
        raise RuntimeError(f"DRE++ cache must not use global blending: {row['cache_path']}")

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    out = dict(payload)
    for name, dtype in DREPP_TENSOR_FIELDS.items():
        if name not in payload:
            raise KeyError(f"DRE++ cache missing {name}: {row['cache_path']}")
        tensor = payload[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"DRE++ {name} must be a tensor: {row['cache_path']}")
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"DRE++ {name} shape mismatch for {row['cache_path']}: "
                f"{list(tensor.shape)} != {expected_shape}"
            )
        out[name] = tensor.to(dtype=dtype)
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
        self.use_drepp = bool(getattr(cfg, "USE_DREPP", False))
        self.use_despl_pseudo = bool(getattr(cfg, "USE_DESPL_PSEUDO", False))
        self.p_init_mode = str(getattr(cfg, "P_INIT_MODE", ""))
        self.use_despl_only = self.use_despl_pseudo and self.p_init_mode == "despl_only"
        self.use_dre_safe_prior = self.use_despl_pseudo and bool(getattr(cfg, "USE_DRE_SAFE_PRIOR", False))
        self.use_despl_light_cache = self.use_despl_pseudo and bool(getattr(cfg, "USE_DESPL_LIGHT_CACHE", False))
        if self.use_qra and self.use_ccr:
            raise RuntimeError("USE_QRA=True and USE_CCR=True cannot be combined.")
        if self.use_drepp and (self.use_qra or self.use_ccr or self.use_despl_pseudo):
            raise RuntimeError("USE_DREPP=True cannot be combined with USE_QRA, USE_CCR, or USE_DESPL_PSEUDO.")
        if self.use_despl_pseudo and (self.use_qra or self.use_ccr):
            raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with USE_QRA=True or USE_CCR=True.")
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
        if self.use_drepp and self.pseudo_cache_override is not None:
            raise RuntimeError("USE_DREPP=True cannot be combined with PSEUDO_CACHE_OVERRIDE.")
        if self.use_despl_pseudo and self.pseudo_cache_override is not None:
            raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with PSEUDO_CACHE_OVERRIDE.")
        self.original_pseudo_cache_root = str(
            (
                Path(cfg.CACHE_ROOT)
                / "pseudo_label_cache"
                / cfg.BACKBONE_KEY
            ).resolve()
        )
        self.despl_map = None
        self.despl_cache_root = None
        self.despl_first_cache_path = None
        self.drepp_map = None
        self.drepp_cache_root = None
        self.drepp_first_cache_path = None
        self.despl_blend_despl_weight = float(getattr(cfg, "P_INIT_DESPL_WEIGHT", 0.8))
        self.despl_blend_fixed_weight = float(getattr(cfg, "P_INIT_FIXED_WEIGHT", 0.2))
        if self.use_drepp:
            drepp_manifest = drepp_manifest_path(cfg)
            drepp_rows = read_jsonl(drepp_manifest)
            self.drepp_map = manifest_to_map(drepp_rows, drepp_manifest)
            if max_samples < 0:
                check_exact_keys("DRE++ cache", self.drepp_map.keys(), self.keys)
            else:
                missing_drepp = sorted(set(self.keys) - set(self.drepp_map))
                if missing_drepp:
                    raise RuntimeError(f"DRE++ cache missing first 10: {missing_drepp[:10]}")
            self.drepp_cache_root = str(drepp_manifest.parent.resolve())
            self.actual_pseudo_cache_root = self.drepp_cache_root
            self.actual_pseudo_cache_pattern = f"{self.drepp_cache_root}/<dataset>/<stem>.pt"
            self.pseudo_map = {}
        elif self.use_despl_pseudo:
            despl_manifest = (
                despl_light_cache_manifest_path(cfg)
                if self.use_despl_light_cache
                else despl_pseudo_bank_manifest_path(cfg)
            )
            despl_rows = read_jsonl(despl_manifest)
            self.despl_map = manifest_to_map(despl_rows, despl_manifest)
            if max_samples < 0:
                cache_name = "DESPL light cache" if self.use_despl_light_cache else "DESPL pseudo bank"
                check_exact_keys(cache_name, self.despl_map.keys(), self.keys)
            else:
                missing_despl = sorted(set(self.keys) - set(self.despl_map))
                if missing_despl:
                    cache_name = "DESPL light cache" if self.use_despl_light_cache else "DESPL pseudo bank"
                    raise RuntimeError(f"{cache_name} missing first 10: {missing_despl[:10]}")
            self.despl_cache_root = str(despl_manifest.parent.resolve())
            self.actual_pseudo_cache_root = self.despl_cache_root
            self.actual_pseudo_cache_pattern = f"{self.despl_cache_root}/<dataset>/<stem>.pt"
            self.pseudo_map = {}
            if not self.use_despl_light_cache:
                pseudo_manifest = _pseudo_manifest_path(cfg)
                if pseudo_manifest.exists():
                    pseudo_rows = read_jsonl(pseudo_manifest)
                    self.pseudo_map = manifest_to_map(pseudo_rows, pseudo_manifest)
        elif self.pseudo_cache_override is None:
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
        self.feature_shape = list(feature.shape)
        self.in_channels = int(feature.shape[0])
        if self.use_drepp:
            drepp_payload = _load_drepp(
                self.drepp_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.pseudo_shape = list(drepp_payload["p_despl"].shape)
            self.pseudo_source = "drepp_v1"
            self.drepp_first_cache_path = self.drepp_map[(first_dataset, first_stem)]["cache_path"]
            self.first_pseudo_cache_path = self.drepp_first_cache_path
            self.pseudo_final_candidate = "p_despl memory anchor + fixed-local recall (global_blend=false)"
        elif self.use_despl_pseudo:
            if self.use_despl_light_cache:
                despl_payload = _load_despl_light_pseudo(
                    self.despl_map[(first_dataset, first_stem)],
                    first_dataset,
                    first_stem,
                    cfg,
                )
                if self.use_despl_only and not despl_payload.get("_has_p_despl_68", False):
                    raise RuntimeError(
                        "DESPL-only requires p_despl_68 in DESPL light cache: "
                        f"{self.despl_map[(first_dataset, first_stem)]['cache_path']}"
                    )
                shape_source = "pseudo_despl" if self.use_despl_only else "pseudo"
                self.pseudo_shape = list(despl_payload[shape_source].shape)
                self.pseudo_source = "despl_light_cache"
            else:
                despl_payload = _load_despl_pseudo(
                    self.despl_map[(first_dataset, first_stem)],
                    first_dataset,
                    first_stem,
                    cfg,
                )
                self.pseudo_shape = list(despl_payload["p_despl"].shape)
                self.pseudo_source = getattr(cfg, "DESPL_PSEUDO_SOURCE", "nper_pseudo_bank")
            self.despl_first_cache_path = self.despl_map[(first_dataset, first_stem)]["cache_path"]
            self.first_pseudo_cache_path = self.despl_first_cache_path
            if self.use_dre_safe_prior:
                self.pseudo_final_candidate = "safe_despl_residual"
            elif self.use_despl_only:
                self.pseudo_final_candidate = "p_despl"
            else:
                self.pseudo_final_candidate = (
                    f"{self.despl_blend_despl_weight:.3f}*p_despl+"
                    f"{self.despl_blend_fixed_weight:.3f}*p_fixed"
                )
        else:
            pseudo, pseudo_payload = _load_pseudo(
                self.pseudo_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
            )
            self.pseudo_shape = list(pseudo.shape)
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
        if self.use_drepp:
            drepp_payload = _load_drepp(self.drepp_map[key], dataset, stem, self.cfg)
            pseudo = drepp_payload["p_despl"].float()
        elif self.use_despl_pseudo:
            if self.use_despl_light_cache:
                light_payload = _load_despl_light_pseudo(self.despl_map[key], dataset, stem, self.cfg)
                pseudo_fixed = light_payload["pseudo_fixed"]
                pseudo_despl = light_payload["pseudo_despl"]
                if self.use_despl_only:
                    if not light_payload.get("_has_p_despl_68", False):
                        raise RuntimeError(
                            "DESPL pseudo missing for "
                            f"{dataset}/{stem}: p_despl_68 not found in {self.despl_map[key]['cache_path']}"
                        )
                    pseudo = pseudo_despl.float()
                    pseudo_base = pseudo
                    pseudo_safe = pseudo
                    dre_safe = None
                    use_fixed_in_pseudo = False
                else:
                    use_fixed_in_pseudo = bool(abs(self.despl_blend_fixed_weight) > 0.0)
                    pseudo_base = (
                        self.despl_blend_despl_weight * pseudo_despl
                        + self.despl_blend_fixed_weight * pseudo_fixed
                    ).clamp(0.0, 1.0)
                if self.use_dre_safe_prior and (
                    not light_payload.get("_has_p_fixed_68", False)
                    or not light_payload.get("_has_p_despl_68", False)
                ):
                    raise RuntimeError(
                        "DRE-SAFE requires p_fixed_68 and p_despl_68 in DESPL light cache: "
                        f"{self.despl_map[key]['cache_path']}"
                    )
            else:
                despl_payload = _load_despl_pseudo(self.despl_map[key], dataset, stem, self.cfg)
                pseudo_despl = despl_payload["p_despl"].float()
                pseudo_fixed = despl_payload["p_fixed"]
                if self.use_despl_only:
                    if pseudo_fixed is None:
                        pseudo_fixed = torch.zeros_like(pseudo_despl)
                    else:
                        pseudo_fixed = _resize_like(pseudo_fixed.float(), pseudo_despl)
                    pseudo = pseudo_despl
                    pseudo_base = pseudo
                    pseudo_safe = pseudo
                    dre_safe = None
                    use_fixed_in_pseudo = False
                elif pseudo_fixed is None:
                    if key not in self.pseudo_map:
                        raise RuntimeError(
                            "DESPL pseudo bank payload missing p_fixed and original fixed pseudo "
                            f"cache is unavailable for {dataset}/{stem}"
                        )
                    pseudo_fixed, _ = _load_pseudo(self.pseudo_map[key], dataset, stem)
                    pseudo_fixed = _resize_like(pseudo_fixed.float(), pseudo_despl)
                    use_fixed_in_pseudo = bool(abs(self.despl_blend_fixed_weight) > 0.0)
                    pseudo_base = (
                        self.despl_blend_despl_weight * pseudo_despl
                        + self.despl_blend_fixed_weight * pseudo_fixed
                    ).clamp(0.0, 1.0)
                else:
                    pseudo_fixed = _resize_like(pseudo_fixed.float(), pseudo_despl)
                    use_fixed_in_pseudo = bool(abs(self.despl_blend_fixed_weight) > 0.0)
                    pseudo_base = (
                        self.despl_blend_despl_weight * pseudo_despl
                        + self.despl_blend_fixed_weight * pseudo_fixed
                    ).clamp(0.0, 1.0)
            if self.use_dre_safe_prior:
                dre_safe = build_dre_safe_prior(pseudo_despl, pseudo_fixed, self.cfg)
                pseudo = dre_safe["p_safe"]
                pseudo_base = dre_safe["p_base"]
                pseudo_safe = dre_safe["p_safe"]
                use_fixed_in_pseudo = True
            elif not self.use_despl_only:
                dre_safe = None
                pseudo = pseudo_base
                pseudo_safe = pseudo_base
            p_init_area = float(pseudo.mean().item())
            p_fixed_area = float(pseudo_fixed.mean().item())
            p_despl_area = float(pseudo_despl.mean().item())
        else:
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
        if self.use_despl_pseudo:
            sample.update(
                {
                    "pseudo_fixed": pseudo_fixed.float(),
                    "pseudo_despl": pseudo_despl.float(),
                    "p_init_area": p_init_area,
                    "p_fixed_area": p_fixed_area,
                    "p_despl_area": p_despl_area,
                    "use_fixed_in_pseudo": bool(use_fixed_in_pseudo),
                    "fixed_used_for_training": bool(use_fixed_in_pseudo),
                }
            )
            if self.use_dre_safe_prior:
                sample.update(
                    {
                        "pseudo_base": pseudo_base.float(),
                        "pseudo_safe": pseudo_safe.float(),
                        "dre_safe_candidate": dre_safe["candidate"].bool(),
                        "dre_safe_fallback": bool(dre_safe["fallback"]),
                        "dre_safe_area_base": float(dre_safe["area_base"]),
                        "dre_safe_area_safe": float(dre_safe["area_safe"]),
                        "dre_safe_candidate_ratio": float(dre_safe["candidate_ratio"]),
                        "dre_safe_cc_base": int(dre_safe["cc_base"]),
                        "dre_safe_cc_safe": int(dre_safe["cc_safe"]),
                        "dre_safe_positive_delta_mean": float(dre_safe["safe_positive_delta_mean"]),
                        "dre_safe_changed_ratio": float(dre_safe["safe_changed_ratio"]),
                    }
                )
        if self.use_drepp:
            sample.update(
                {
                    "drepp_p_fixed": drepp_payload["p_fixed"].float(),
                    "drepp_core_fg": drepp_payload["core_fg"].bool(),
                    "drepp_core_bg": drepp_payload["core_bg"].bool(),
                    "drepp_uncertain": drepp_payload["uncertain"].bool(),
                    "drepp_fixed_local_recall": drepp_payload["fixed_local_recall"].bool(),
                    "drepp_boundary_band": drepp_payload["boundary_band"].bool(),
                    "drepp_memory_init": drepp_payload["memory_init"].float(),
                    "drepp_feature_sim": drepp_payload["feature_sim"].float(),
                    "drepp_p_despl_area": float(drepp_payload.get("p_despl_area", 0.0)),
                    "drepp_p_fixed_area": float(drepp_payload.get("p_fixed_area", 0.0)),
                    "drepp_core_fg_area": float(drepp_payload.get("core_fg_area", 0.0)),
                    "drepp_core_bg_area": float(drepp_payload.get("core_bg_area", 0.0)),
                    "drepp_uncertain_area": float(drepp_payload.get("uncertain_area", 0.0)),
                    "drepp_fixed_local_area": float(drepp_payload.get("fixed_local_area", 0.0)),
                    "drepp_boundary_band_area": float(drepp_payload.get("boundary_band_area", 0.0)),
                    "drepp_fixed_local_ratio": float(drepp_payload.get("fixed_local_ratio", 0.0)),
                    "drepp_global_blend": bool(drepp_payload.get("global_blend", False)),
                }
            )
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
