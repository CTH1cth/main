from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from common.dre_safe_prior import build_dre_safe_prior
from common.dabe_clean import (
    DABE_CLEAN_CONTREC_VERSION,
    DABE_CLEAN_OFFLINE_VERSION,
    DABE_CLEAN_TARGET_MODES,
    expected_dabe_clean_payload_version,
    select_clean_target,
)
from common.dabe_clean_offline import DABE_CLEAN_OFFLINE_MODES
from common.found_static import (
    FOUND_STATIC_RESIZE_MODE,
    FOUND_STATIC_SOURCE,
    build_found_static_target,
    found_static_enabled,
    found_static_manifest_path,
)
from common.teacher_only import (
    TEACHER_ONLY_NO_OFFLINE_PSEUDO_SOURCE,
    teacher_only_no_offline_pseudo_enabled,
)
from common.utils import (
    build_image_items,
    cacd_feature_manifest_path,
    ccr_manifest_path,
    check_exact_keys,
    cssd_hr_feature_manifest_path,
    dabe_clean_legacy_region_manifest_path,
    dabe_clean_manifest_path,
    dabe_pu_manifest_path,
    dabe_pseudo_manifest_path,
    despl_light_cache_manifest_path,
    despl_paper_manifest_path,
    despl_pseudo_bank_manifest_path,
    drepp_manifest_path,
    hflip_feature_cache_manifest_path,
    lceg_cover_manifest_path,
    manifest_to_map,
    ml_feature_cache_manifest_path,
    qra_manifest_path,
    read_jsonl,
    tce_cover_manifest_path,
    torch_load,
)
from common.bitc import BITC_CACHE_SCHEMA, BITC_GRID


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

DABE_PU_REQUIRED_68_FIELDS = [
    "target_soft_68",
    "weight_map_68",
    "fg_core_pu_68",
    "fg_core_fallback_68",
    "bg_core_pu_68",
    "extent_candidate_68",
    "unknown_68",
]

DABE_PU_REQUIRED_37_FIELDS = [
    "fg_core_pu_37",
    "fg_core_fallback_37",
    "bg_core_pu_37",
    "extent_candidate_37",
    "unknown_37",
]

DABE_CLEAN_REQUIRED_FIELDS = (
    "foreground_evidence_37",
    "foreground_evidence_68",
    "background_evidence_37",
    "background_evidence_68",
    "target_dp_37",
    "target_dp_68",
    "target_diff_37",
    "target_diff_68",
)

DABE_CLEAN_CONTREC_REQUIRED_FIELDS = (
    "foreground_evidence_37",
    "foreground_evidence_68",
    "background_evidence_37",
    "background_evidence_68",
    "target_dp_37",
    "target_dp_68",
    "p_rw_37",
    "evidence_gate_37",
    "semantic_fg_tendency_37",
    "latent_rw_37",
    "recoverability_37",
    "recoverability_68",
)

LEGACY_ECST_REGION_FIELDS = {
    "legacy_ecst_fg_core": "fg_core_pu_68",
    "legacy_ecst_fg_fallback": "fg_core_fallback_68",
    "legacy_ecst_bg_core": "bg_core_pu_68",
    "legacy_ecst_extent": "extent_candidate_68",
    "legacy_ecst_unknown": "unknown_68",
}

DABE_PU_DESPL_SCHED_MODES = {
    "dabe_pu_v11_desplsched",
    "dabe_pu_v11_desplsched_exactreset",
    "dabe_pu_v11_desplsched_A1_keepteacher_lowlr",
    "dabe_pu_v11_desplsched_A2_resetteacher_highlr",
    "dabe_pu_v11_desplsched_softteacher",
    "dabe_pu_v11_desplsched_dabehard",
    "dabe_pu_v11_dagp_csd_v1r_desplsched",
    "dabe_pu_v12_shape_desplsched",
}

DABE_PU_ALLOWED_MODES = {
    "dabe_pu_v11",
    "dabe_pu_v11_oem",
    *DABE_PU_DESPL_SCHED_MODES,
}

DABE_PU_ALLOWED_VERSIONS = {"pu_v11", "pu_v12_shape_complete"}


def _cvsa_config(cfg):
    value = getattr(cfg, "CVSA", None)
    if not isinstance(value, dict):
        raise RuntimeError("USE_CVSA=True requires a CVSA dict configuration.")
    return value


def _cvsa_manifest_path(cfg, cache_key):
    root = _cvsa_config(cfg).get(cache_key)
    if not root:
        raise RuntimeError(f"CVSA.{cache_key} must be configured.")
    return Path(root) / "manifest_train.jsonl"


def _load_cvsa_hflip_fixed(row, expected_dataset, expected_stem, cfg):
    cache_path = row.get("fixed_path", row.get("cache_path"))
    if not cache_path:
        raise RuntimeError(
            f"CVSA hflip fixed manifest has no cache path for "
            f"{expected_dataset}/{expected_stem}."
        )
    payload = torch_load(cache_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"CVSA hflip fixed payload must be a dict: {cache_path}")
    checks = {
        "dataset": expected_dataset,
        "stem": expected_stem,
        "view": "hflip",
        "backbone_key": cfg.BACKBONE_KEY,
        "dabe_version": "pu_v11",
        "source_feature_view": "hflip",
        "independently_generated": True,
    }
    for field, expected in checks.items():
        if payload.get(field) != expected:
            raise RuntimeError(
                f"CVSA hflip fixed provenance mismatch for "
                f"{expected_dataset}/{expected_stem}: "
                f"{field}={payload.get(field)!r} != {expected!r} | {cache_path}"
            )
    target = payload.get("target_soft_68")
    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    if not torch.is_tensor(target) or list(target.shape) != expected_shape:
        raise RuntimeError(
            f"CVSA hflip fixed target shape mismatch for "
            f"{expected_dataset}/{expected_stem}: "
            f"{list(target.shape) if torch.is_tensor(target) else type(target)!r} "
            f"!= {expected_shape} | {cache_path}"
        )
    target = target.float()
    _validate_unit_range(target, "CVSA target_soft_68", cache_path)
    return target


def _feature_manifest_path(cfg, split):
    # feature cache manifest 按 split 管理，train/val/test 不混用。
    return Path(cfg.CACHE_ROOT) / "features_cache" / cfg.BACKBONE_KEY / f"manifest_{split}.jsonl"


def _pseudo_manifest_path(cfg):
    # fixed pseudo 只为训练集生成，因此 manifest 名固定为 train。
    return Path(cfg.CACHE_ROOT) / "pseudo_label_cache" / cfg.BACKBONE_KEY / "manifest_train.jsonl"


def _bitc_manifest_path(cfg):
    return Path(getattr(cfg, "BITC_CACHE_ROOT")) / "manifest_train.jsonl"


def _load_bitc_cache(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"BITC cache payload must be a dict: {row['cache_path']}")
    expected = {
        "schema": BITC_CACHE_SCHEMA,
        "dataset": expected_dataset,
        "stem": expected_stem,
        "backbone_key": cfg.BACKBONE_KEY,
        "dabe_version": str(getattr(cfg, "DABE_PU_VERSION", "pu_v11")),
        "seed": int(getattr(cfg, "BITC_CACHE_SEED", 20260724)),
    }
    mismatched = {
        name: (payload.get(name), value)
        for name, value in expected.items()
        if payload.get(name) != value
    }
    if mismatched:
        raise RuntimeError(
            f"BITC cache metadata mismatch for {row['cache_path']}: {mismatched}"
        )
    required = (
        "topk_indices",
        "topk_weights",
        "background_anchor_indices",
        "nearest_bg_index",
        "fixed_random_bg_index",
    )
    missing = [name for name in required if not torch.is_tensor(payload.get(name))]
    if missing:
        raise RuntimeError(
            f"BITC cache is missing tensor fields for {row['cache_path']}: {missing}"
        )
    nodes = BITC_GRID * BITC_GRID
    topk_indices = payload["topk_indices"].long()
    topk_weights = payload["topk_weights"].float()
    anchor_indices = payload["background_anchor_indices"].long()
    nearest = payload["nearest_bg_index"].long()
    random_index = payload["fixed_random_bg_index"].long()
    if topk_indices.ndim != 2 or int(topk_indices.shape[0]) != nodes:
        raise RuntimeError(
            f"BITC topk_indices must be [1369,K], got {list(topk_indices.shape)}"
        )
    if list(topk_weights.shape) != list(topk_indices.shape):
        raise RuntimeError(
            f"BITC topk weight/index mismatch: {list(topk_weights.shape)}/"
            f"{list(topk_indices.shape)}"
        )
    if list(nearest.shape) != [nodes] or list(random_index.shape) != [nodes]:
        raise RuntimeError(
            f"BITC nearest/random shape mismatch: {list(nearest.shape)}/{list(random_index.shape)}"
        )
    tensors = (topk_indices, topk_weights, anchor_indices, nearest, random_index)
    if not all(bool(torch.isfinite(tensor.float()).all().item()) for tensor in tensors):
        raise RuntimeError(f"BITC cache contains NaN/Inf: {row['cache_path']}")
    index_tensors = (topk_indices, anchor_indices, nearest, random_index)
    if any(
        int(tensor.min().item()) < 0 or int(tensor.max().item()) >= nodes
        for tensor in index_tensors
        if int(tensor.numel()) > 0
    ):
        raise RuntimeError(f"BITC cache contains an out-of-range index: {row['cache_path']}")
    if int(anchor_indices.numel()) < 1:
        raise RuntimeError(f"BITC background anchor set is empty: {row['cache_path']}")
    anchor_mask = torch.zeros(nodes, dtype=torch.bool)
    anchor_mask[anchor_indices] = True
    if not bool(anchor_mask[topk_indices].all().item()):
        raise RuntimeError(f"BITC top-K contains non-anchor indices: {row['cache_path']}")
    if not bool(anchor_mask[nearest].all().item()):
        raise RuntimeError(f"BITC nearest contains non-anchor indices: {row['cache_path']}")
    if not bool(anchor_mask[random_index].all().item()):
        raise RuntimeError(f"BITC random contains non-anchor indices: {row['cache_path']}")
    sum_error = float((topk_weights.sum(dim=1) - 1.0).abs().max().item())
    if sum_error > 2e-3:
        raise RuntimeError(
            f"BITC float16 top-K weight sum error is too large: {sum_error:.9g}"
        )
    fingerprint = str(payload.get("cache_fingerprint", ""))
    if not fingerprint:
        raise RuntimeError(f"BITC cache fingerprint is missing: {row['cache_path']}")
    manifest_fingerprint = str(row.get("cache_fingerprint", ""))
    if manifest_fingerprint != fingerprint:
        raise RuntimeError(
            "BITC manifest/payload fingerprint mismatch: "
            f"{manifest_fingerprint!r} != {fingerprint!r} | {row['cache_path']}"
        )
    return {
        "bitc_topk_indices": topk_indices.to(torch.int16),
        "bitc_topk_weights": topk_weights.to(torch.float16),
        "bitc_background_anchor_mask": anchor_mask,
        "bitc_nearest_bg_index": nearest.to(torch.int16),
        "bitc_fixed_random_bg_index": random_index.to(torch.int16),
        "bitc_background_anchor_count": int(anchor_indices.numel()),
        "bitc_cache_fingerprint": fingerprint,
        "bitc_cache_path": str(row["cache_path"]),
    }


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


def _load_cacd_features(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"CACD feature payload must be dict: {row['cache_path']}")
    for field, expected in (
        ("dataset", expected_dataset),
        ("stem", expected_stem),
        ("backbone", cfg.BACKBONE_KEY),
        ("backbone_key", cfg.BACKBONE_KEY),
        ("model_key", cfg.DINO["model_name"]),
        ("version", "cacd_last3_extra_v1"),
        ("feature_type", "attention_key_projection"),
        ("dtype", "float32"),
        ("input_size", int(cfg.DINO["feature_input_size"])),
        ("patch_size", int(cfg.DINO["patch_size"])),
        ("layer_indices_0based", [9, 10]),
        ("layer_indices_1based", [10, 11]),
    ):
        if payload.get(field) != expected:
            raise RuntimeError(
                f"CACD feature metadata mismatch for {expected_dataset}/{expected_stem}: "
                f"{field}={payload.get(field)!r} != {expected!r} | {row['cache_path']}"
            )
    if payload.get("stored_layer_indices") != [9, 10]:
        raise RuntimeError(
            f"CACD stored layers mismatch for {expected_dataset}/{expected_stem}: "
            f"{payload.get('stored_layer_indices')!r} != [9, 10]"
        )
    if any(field in payload for field in ("feature_l12", "tensor", "feature")):
        raise RuntimeError(
            f"CACD extra cache must not duplicate F12: {row['cache_path']}"
        )
    expected_shape = [
        int(getattr(cfg, "CACD_IN_CHANNELS", 384)),
        int(getattr(cfg, "CACD_FEATURE_SIZE", 37)),
        int(getattr(cfg, "CACD_FEATURE_SIZE", 37)),
    ]
    result = {}
    for field in ("feature_l10", "feature_l11"):
        tensor = payload.get(field)
        if not torch.is_tensor(tensor):
            raise TypeError(f"CACD feature payload missing {field}: {row['cache_path']}")
        if tensor.dtype != torch.float32 or list(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"CACD {field} shape/dtype mismatch for {expected_dataset}/{expected_stem}: "
                f"shape={list(tensor.shape)}, dtype={tensor.dtype}, expected={expected_shape}/float32"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"CACD {field} contains NaN/Inf: {row['cache_path']}")
        result[field] = tensor.float()
    result["payload"] = payload
    return result


def _load_cssd_hr_feature(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"CSSD HR feature cache must be dict: {row['cache_path']}")
    for field, expected in (
        ("dataset", expected_dataset),
        ("stem", expected_stem),
        ("backbone_key", cfg.BACKBONE_KEY),
        ("model_key", cfg.DINO["model_name"]),
        ("feature_layer", "final_attention_key"),
        ("input_size", int(getattr(cfg, "CSSD_HR_INPUT_SIZE", 384))),
        ("version", "cssd_hr_feature_v1"),
        ("dtype", "float32"),
    ):
        if payload.get(field) != expected:
            raise RuntimeError(
                f"CSSD HR feature metadata mismatch for {expected_dataset}/{expected_stem}: "
                f"{field}={payload.get(field)!r} != {expected!r} | {row['cache_path']}"
            )
    if not isinstance(payload.get("key_projection_path"), str) or not payload["key_projection_path"]:
        raise RuntimeError(
            f"CSSD HR feature missing key_projection_path for {expected_dataset}/{expected_stem}: "
            f"{row['cache_path']}"
        )
    field = str(getattr(cfg, "CSSD_HR_CACHE_KEY", "feature"))
    feature = payload.get(field)
    expected_shape = [
        int(getattr(cfg, "CSSD_HR_FEATURE_CHANNELS", 384)),
        int(getattr(cfg, "CSSD_HR_FEATURE_SIZE", 48)),
        int(getattr(cfg, "CSSD_HR_FEATURE_SIZE", 48)),
    ]
    if not torch.is_tensor(feature) or list(feature.shape) != expected_shape:
        shape = list(feature.shape) if torch.is_tensor(feature) else None
        raise RuntimeError(
            f"CSSD HR feature shape mismatch for {expected_dataset}/{expected_stem}: "
            f"{shape} != {expected_shape} | {row['cache_path']}"
        )
    if feature.dtype != torch.float32:
        raise RuntimeError(
            f"CSSD HR feature dtype mismatch for {expected_dataset}/{expected_stem}: "
            f"{feature.dtype} != torch.float32 | {row['cache_path']}"
        )
    if not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError(f"CSSD HR feature contains NaN/Inf: {row['cache_path']}")
    return feature, payload


def _expected_feature_channels(cfg):
    if cfg.BACKBONE_KEY == "dinov1-s8":
        return 384
    if cfg.BACKBONE_KEY in {"dinov1-b8", "dinov2-b14"}:
        return 768
    return None


def _load_multi_level_feature(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Multi-level feature cache must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"Multi-level feature dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"Multi-level feature stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Multi-level feature backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    features = payload.get("features")
    if not isinstance(features, dict):
        raise KeyError(f"Multi-level feature payload missing features dict: {row['cache_path']}")
    labels = [f"l{int(layer)}" for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12])]
    expected_channels = _expected_feature_channels(cfg)
    out = {}
    spatial = None
    for label in labels:
        tensor = features.get(label)
        if not torch.is_tensor(tensor):
            raise KeyError(f"Multi-level feature payload missing features/{label}: {row['cache_path']}")
        if tensor.ndim != 3:
            raise RuntimeError(f"Multi-level feature {label} must be [C,H,W], got {list(tensor.shape)}")
        if expected_channels is not None and int(tensor.shape[0]) != expected_channels:
            raise RuntimeError(
                f"Multi-level feature {label} channel mismatch: "
                f"{int(tensor.shape[0])} != {expected_channels} | {row['cache_path']}"
            )
        if spatial is None:
            spatial = tuple(tensor.shape[-2:])
        elif tuple(tensor.shape[-2:]) != spatial:
            raise RuntimeError(f"Multi-level feature spatial mismatch in {row['cache_path']}")
        out[label] = tensor
    result = {f"feature_{label}": out[label] for label in labels}
    result["feature"] = out[labels[-1]]
    result["payload"] = payload
    return result


def _load_pseudo(row, expected_dataset, expected_stem):
    # pseudo tensor 必须是单通道 [1,H,W]，训练时再插值到 loss size。
    payload = _load_cache_payload(row, expected_dataset, expected_stem, "pseudo")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"Pseudo tensor must be [1,H,W], got {list(tensor.shape)}")
    return tensor, payload


def _load_found_static(row, expected_dataset, expected_stem, cfg):
    native, payload = _load_pseudo(row, expected_dataset, expected_stem)
    target = build_found_static_target(native, cfg.LOSS_SIZE)
    return {
        "target_offline_68": target,
        "dabe_clean_version": "found_fixed_v1_protocol",
        "source": FOUND_STATIC_SOURCE,
        "native_shape": list(native.shape),
        "resize_mode": FOUND_STATIC_RESIZE_MODE,
        "cache_path": str(row["cache_path"]),
        "payload": payload,
    }


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


def _dabe_source_name(cfg):
    version = str(getattr(cfg, "DABE_VERSION", "v2")).lower()
    if version == "gc":
        return "dabe_gc_cache"
    return f"dabe_{version}_cache"


def _dabe_pseudo_tensor_keys(cfg):
    version = str(getattr(cfg, "DABE_VERSION", "v2")).lower()
    keys = ["p_dabe_68"]
    if version == "gc":
        keys.append("p_dabe_gc_68")
    keys.append("p_dabe_37")
    if version == "gc":
        keys.append("p_dabe_gc_37")
    return keys


def _select_dabe_pseudo_tensor(payload, row, cfg):
    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    raw_shape = [1, 37, 37]
    found_bad_shapes = []
    for key in _dabe_pseudo_tensor_keys(cfg):
        value = payload.get(key)
        if not torch.is_tensor(value):
            continue
        tensor = value.float()
        shape = list(tensor.shape)
        if shape == expected_shape:
            _validate_unit_range(tensor, key, row["cache_path"])
            return {
                "tensor": tensor,
                "source_key": key,
                "resized_from_37": False,
                "resized_from": shape,
                "resized_to": shape,
            }
        if shape == raw_shape:
            _validate_unit_range(tensor, key, row["cache_path"])
            resized = F.interpolate(
                tensor.unsqueeze(0),
                size=(int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)
            _validate_unit_range(resized, f"{key}->p_dabe_68", row["cache_path"])
            return {
                "tensor": resized,
                "source_key": key,
                "resized_from_37": True,
                "resized_from": shape,
                "resized_to": expected_shape,
            }
        found_bad_shapes.append((key, shape))
    if found_bad_shapes:
        raise RuntimeError(
            "DABE pseudo tensor shape mismatch | "
            f"cache_path={row['cache_path']} | expected={expected_shape} or {raw_shape} | "
            f"found={found_bad_shapes}"
        )
    raise TypeError(
        "DABE pseudo payload missing usable pseudo tensor | "
        f"cache_path={row['cache_path']} | tried={_dabe_pseudo_tensor_keys(cfg)}"
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


def _load_dabe_pseudo(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DABE pseudo payload must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"DABE pseudo dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"DABE pseudo stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"DABE pseudo backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    expected_version = str(getattr(cfg, "DABE_VERSION", "v2")).lower()
    if str(payload.get("dabe_version", "")).lower() != expected_version:
        raise RuntimeError(
            f"DABE pseudo version mismatch for {row['cache_path']}: "
            f"{payload.get('dabe_version')} != {expected_version}"
        )
    selection = _select_dabe_pseudo_tensor(payload, row, cfg)
    tensor = selection["tensor"]
    out = {
        "pseudo_dabe": tensor,
        "p_dabe_area": float(payload.get("area", tensor.mean().item())),
        "dabe_fallback_flag": bool(payload.get("fallback_flag", False)),
        "dabe_large_area_flag": bool(payload.get("large_area_flag", False)),
        "dabe_num_components": int(payload.get("num_components", 0)),
        "dabe_source_key": selection["source_key"],
        "dabe_resized_from_37": bool(selection["resized_from_37"]),
        "dabe_resized_from": selection["resized_from"],
        "dabe_resized_to": selection["resized_to"],
    }
    if bool(getattr(cfg, "USE_DABE_AWARE_LOSS", False)):
        out.update(_load_dabe_aware_fields(payload, row, expected_dataset, expected_stem, cfg))
    return out


def _dabe_payload_first_tensor(payload, keys):
    for key in keys:
        value = payload.get(key)
        if torch.is_tensor(value):
            return key, value.float()
    return None, None


def _resize_dabe_aware_field(tensor, field_name, row, loss_size, mode, threshold=None):
    if list(tensor.shape) != [1, 37, 37]:
        raise RuntimeError(
            f"DABE aware {field_name} shape mismatch: {list(tensor.shape)} != [1, 37, 37] | "
            f"{row['cache_path']}"
        )
    tensor4 = tensor.unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(tensor4, size=(int(loss_size), int(loss_size)), mode="nearest")
    else:
        resized = F.interpolate(
            tensor4,
            size=(int(loss_size), int(loss_size)),
            mode="bilinear",
            align_corners=False,
        )
    resized = resized.squeeze(0).float()
    if threshold is not None:
        resized = (resized > float(threshold)).float()
    else:
        resized = resized.clamp(0.0, 1.0)
    return resized


def _load_dabe_aware_fields(payload, row, expected_dataset, expected_stem, cfg):
    field_keys = {
        "fg_core": ["fg_core_37", "fg_core"],
        "bg_core": ["bg_core_37", "bg_core"],
        "evidence": ["evidence_37", "evidence"],
    }
    missing = []
    tensors = {}
    for field_name, keys in field_keys.items():
        _, tensor = _dabe_payload_first_tensor(payload, keys)
        if tensor is None:
            missing.append(field_name)
        else:
            tensors[field_name] = tensor
    if missing:
        raise RuntimeError(
            "DABE aware payload missing required fields | "
            f"dataset={expected_dataset} | stem={expected_stem} | "
            f"cache_path={row['cache_path']} | missing_keys={missing}"
        )

    loss_size = int(cfg.LOSS_SIZE)
    core_thresh = float(getattr(cfg, "DABE_CORE_THRESH", 0.5))
    fg_core_68 = _resize_dabe_aware_field(
        tensors["fg_core"],
        "fg_core",
        row,
        loss_size,
        mode="nearest",
        threshold=core_thresh,
    )
    bg_core_68 = _resize_dabe_aware_field(
        tensors["bg_core"],
        "bg_core",
        row,
        loss_size,
        mode="nearest",
        threshold=core_thresh,
    )
    evidence_68 = _resize_dabe_aware_field(
        tensors["evidence"],
        "evidence",
        row,
        loss_size,
        mode="bilinear",
    )
    uncertain_68 = (1.0 - torch.clamp(fg_core_68 + bg_core_68, 0.0, 1.0)).float()
    return {
        "dabe_fg_core_68": fg_core_68,
        "dabe_bg_core_68": bg_core_68,
        "dabe_evidence_68": evidence_68,
        "dabe_uncertain_68": uncertain_68,
    }


def _load_dabe_pu_v11(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DABE-PU payload must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"DABE-PU dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"DABE-PU stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"DABE-PU backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    expected_version = str(getattr(cfg, "DABE_PU_VERSION", "pu_v11")).lower()
    if str(payload.get("dabe_version", "")).lower() != expected_version:
        raise RuntimeError(
            f"DABE-PU version mismatch for {row['cache_path']}: "
            f"{payload.get('dabe_version')} != {expected_version}"
        )

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    expected_shape_37 = [1, 37, 37]
    use_oem = bool(getattr(cfg, "USE_DABE_OEM", False)) or str(
        getattr(cfg, "P_INIT_MODE", "")
    ) == "dabe_pu_v11_oem"
    use_ap_stcr = bool(getattr(cfg, "USE_AP_STCR", False))
    out = {}
    missing = []
    required_fields = list(DABE_PU_REQUIRED_68_FIELDS)
    static_source = str(
        getattr(cfg, "DABE_PU_STATIC_SOURCE", "target_soft_68")
    ).strip().lower()
    if static_source not in {"target_soft_68", "p_base_68"}:
        raise RuntimeError(
            "Unsupported DABE_PU_STATIC_SOURCE="
            f"{static_source!r}; expected 'target_soft_68' or 'p_base_68'."
        )
    if static_source == "p_base_68":
        required_fields.append("p_base_68")
    if use_oem:
        required_fields.extend(DABE_PU_REQUIRED_37_FIELDS)
    if use_ap_stcr:
        required_fields.extend(["target_soft_37", "bg_anchor_37"])
    for field in required_fields:
        tensor = payload.get(field)
        if not torch.is_tensor(tensor):
            missing.append(field)
            continue
        tensor = tensor.float()
        shape = expected_shape_37 if field.endswith("_37") else expected_shape
        if list(tensor.shape) != shape:
            raise RuntimeError(
                "DABE-PU tensor shape mismatch | "
                f"dataset={expected_dataset} | stem={expected_stem} | "
                f"cache_path={row['cache_path']} | {field} {list(tensor.shape)} != {shape}"
            )
        _validate_unit_range(tensor, field, row["cache_path"])
        if field == "p_base_68" and not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(
                "DABE-PU p_base_68 contains NaN/Inf | "
                f"dataset={expected_dataset} | stem={expected_stem} | "
                f"cache_path={row['cache_path']}"
            )
        out[field] = tensor
    if missing:
        raise RuntimeError(
            "DABE-PU payload missing required fields | "
            f"dataset={expected_dataset} | stem={expected_stem} | "
            f"cache_path={row['cache_path']} | missing_key={missing[0]} | missing_keys={missing}"
        )
    target_base = payload.get("target_soft_base_68", payload.get("target_soft_base"))
    weight_base = payload.get("weight_map_base_68", payload.get("weight_map_base"))
    if torch.is_tensor(target_base):
        target_base = target_base.float()
    else:
        target_base = out["target_soft_68"]
    if torch.is_tensor(weight_base):
        weight_base = weight_base.float()
    else:
        weight_base = out["weight_map_68"]
    sc_bg_lock = payload.get("sc_bg_lock_68", payload.get("sc_bg_lock"))
    sc_extent_agree_fg = payload.get("sc_extent_agree_fg_68", payload.get("sc_extent_agree_fg"))
    sc_lost_extent = payload.get("sc_lost_extent_68", payload.get("sc_lost_extent"))
    sc_new_boundary = payload.get("sc_new_boundary_68", payload.get("sc_new_boundary"))
    zero_diag = torch.zeros_like(out["target_soft_68"])
    sc_bg_lock = sc_bg_lock.float() if torch.is_tensor(sc_bg_lock) else zero_diag
    sc_extent_agree_fg = sc_extent_agree_fg.float() if torch.is_tensor(sc_extent_agree_fg) else zero_diag
    sc_lost_extent = sc_lost_extent.float() if torch.is_tensor(sc_lost_extent) else zero_diag
    sc_new_boundary = sc_new_boundary.float() if torch.is_tensor(sc_new_boundary) else zero_diag

    return {
        "pu_target_soft": out["target_soft_68"],
        "pu_p_base_soft": out.get("p_base_68"),
        "pu_weight_map": out["weight_map_68"],
        "pu_fg_core": out["fg_core_pu_68"],
        "pu_fg_fallback": out["fg_core_fallback_68"],
        "pu_bg_core": out["bg_core_pu_68"],
        "pu_extent": out["extent_candidate_68"],
        "pu_unknown": out["unknown_68"],
        "pu_fg_core_37": out.get("fg_core_pu_37"),
        "pu_fg_fallback_37": out.get("fg_core_fallback_37"),
        "pu_bg_core_37": out.get("bg_core_pu_37"),
        "pu_extent_37": out.get("extent_candidate_37"),
        "pu_unknown_37": out.get("unknown_37"),
        "pu_target_soft_37": out.get("target_soft_37"),
        "pu_bg_anchor_37": out.get("bg_anchor_37"),
        "pu_target_area": float(payload.get("target_soft_area", out["target_soft_68"].mean().item())),
        "pu_weight_mean": float(payload.get("weight_mean", out["weight_map_68"].mean().item())),
        "pu_fg_core_area": float(payload.get("fg_core_pu_area", out["fg_core_pu_68"].mean().item())),
        "pu_fg_fallback_area": float(payload.get("fg_core_fallback_area", out["fg_core_fallback_68"].mean().item())),
        "pu_bg_core_area": float(payload.get("bg_core_pu_area", out["bg_core_pu_68"].mean().item())),
        "pu_extent_area": float(payload.get("extent_area", out["extent_candidate_68"].mean().item())),
        "pu_unknown_area": float(payload.get("unknown_area", out["unknown_68"].mean().item())),
        "pu_target_base": target_base,
        "pu_weight_base": weight_base,
        "pu_sc_bg_lock": sc_bg_lock,
        "pu_sc_extent_agree_fg": sc_extent_agree_fg,
        "pu_sc_lost_extent": sc_lost_extent,
        "pu_sc_new_boundary": sc_new_boundary,
        "pu_target_base_mean": float(payload.get("target_base_mean", target_base.mean().item())),
        "pu_weight_base_mean": float(payload.get("weight_base_mean", weight_base.mean().item())),
        "pu_target_delta_mean": float(payload.get("target_delta_mean", out["target_soft_68"].mean().item() - target_base.mean().item())),
        "pu_sc_bg_lock_ratio": float(payload.get("sc_bg_lock_ratio", sc_bg_lock.mean().item())),
        "pu_sc_extent_agree_fg_ratio": float(payload.get("sc_extent_agree_fg_ratio", sc_extent_agree_fg.mean().item())),
        "pu_sc_lost_extent_ratio": float(payload.get("sc_lost_extent_ratio", sc_lost_extent.mean().item())),
        "pu_sc_new_boundary_ratio": float(payload.get("sc_new_boundary_ratio", sc_new_boundary.mean().item())),
    }


def _load_dabe_clean(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DABE-Clean payload must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset or payload.get("stem") != expected_stem:
        raise RuntimeError(
            "DABE-Clean identity mismatch | "
            f"expected={expected_dataset}/{expected_stem} | "
            f"actual={payload.get('dataset')}/{payload.get('stem')} | "
            f"cache={row['cache_path']}"
        )
    if str(payload.get("backbone_key")) != str(cfg.BACKBONE_KEY):
        raise RuntimeError(f"DABE-Clean backbone mismatch: {row['cache_path']}")
    clean_version = str(getattr(cfg, "DABE_CLEAN_VERSION", "")).strip().lower()
    if clean_version == "v3_offline_consolidation":
        expected_version = DABE_CLEAN_OFFLINE_VERSION
        mode = str(getattr(cfg, "DABE_CLEAN_OFFLINE_MODE", "")).strip().lower()
        if mode not in DABE_CLEAN_OFFLINE_MODES:
            raise RuntimeError(f"Unsupported DABE_CLEAN_OFFLINE_MODE={mode!r}.")
        if (
            str(payload.get("version")) != expected_version
            or str(payload.get("payload_version")) != expected_version
        ):
            raise RuntimeError(
                "DABE-Clean offline payload version mismatch: "
                f"{payload.get('version')}/{payload.get('payload_version')} != "
                f"{expected_version} | {row['cache_path']}"
            )
        if str(payload.get("offline_mode")) != mode:
            raise RuntimeError(
                "DABE-Clean offline mode mismatch: "
                f"{payload.get('offline_mode')} != {mode} | {row['cache_path']}"
            )
        target = payload.get("target_offline_68")
        if not torch.is_tensor(target):
            raise RuntimeError(
                f"DABE-Clean offline payload is missing target_offline_68: {row['cache_path']}"
            )
        target = target.detach().cpu().float()
        expected = (1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
        if tuple(target.shape) != expected:
            raise RuntimeError(
                f"DABE-Clean offline target shape mismatch: {list(target.shape)} "
                f"!= {list(expected)} | {row['cache_path']}"
            )
        _validate_unit_range(target, "target_offline_68", row["cache_path"])
        if target.requires_grad:
            raise RuntimeError(
                f"DABE-Clean offline target must be detached: {row['cache_path']}"
            )
        return {
            "target_offline_68": target,
            "dabe_clean_target_mode": mode,
            "dabe_clean_version": expected_version,
        }

    mode = str(getattr(cfg, "DABE_CLEAN_TARGET_MODE", "dp")).strip().lower()
    if mode not in DABE_CLEAN_TARGET_MODES:
        raise RuntimeError(f"Unsupported DABE_CLEAN_TARGET_MODE={mode!r}.")
    expected_version = expected_dabe_clean_payload_version(cfg, mode)
    if str(payload.get("version")) != expected_version:
        raise RuntimeError(
            f"DABE-Clean version mismatch: {payload.get('version')} != "
            f"{expected_version} | {row['cache_path']}"
        )
    target_37 = select_clean_target(payload, mode, suffix="37")
    target_68 = select_clean_target(payload, mode, suffix="68")
    expected_37 = (1, 37, 37)
    expected_68 = (1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
    if tuple(target_37.shape) != expected_37 or tuple(target_68.shape) != expected_68:
        raise RuntimeError(
            f"DABE-Clean target shape mismatch: {list(target_37.shape)}/"
            f"{list(target_68.shape)}"
        )
    out = {
        "dabe_clean_target_37": target_37,
        "dabe_clean_target_68": target_68,
        "dabe_clean_target_mode": mode,
        "dabe_clean_version": expected_version,
    }
    if mode == "bridge":
        foreground_37 = payload.get("source_p_base_37")
        if not torch.is_tensor(foreground_37):
            raise RuntimeError(
                f"Bridge cache is missing source_p_base_37: {row['cache_path']}"
            )
        foreground_37 = foreground_37.detach().cpu().float()
        _validate_unit_range(foreground_37, "source_p_base_37", row["cache_path"])
        out["dabe_clean_fg_evidence_37"] = foreground_37
        out["dabe_clean_fg_evidence_68"] = torch.empty(0)
        out["dabe_clean_bg_evidence_37"] = torch.empty(0)
        out["dabe_clean_bg_evidence_68"] = torch.empty(0)
    else:
        required_fields = (
            DABE_CLEAN_CONTREC_REQUIRED_FIELDS
            if expected_version == DABE_CLEAN_CONTREC_VERSION
            else DABE_CLEAN_REQUIRED_FIELDS
        )
        for field in required_fields:
            value = payload.get(field)
            if not torch.is_tensor(value):
                raise RuntimeError(
                    f"DABE-Clean payload is missing {field}: {row['cache_path']}"
                )
            value = value.detach().cpu().float()
            expected = expected_37 if field.endswith("_37") else expected_68
            if tuple(value.shape) != expected:
                raise RuntimeError(
                    f"DABE-Clean {field} shape mismatch: {list(value.shape)} != "
                    f"{list(expected)} | {row['cache_path']}"
                )
            _validate_unit_range(value, field, row["cache_path"])
        out.update(
            {
                "dabe_clean_fg_evidence_37": payload[
                    "foreground_evidence_37"
                ].detach().cpu().float(),
                "dabe_clean_fg_evidence_68": payload[
                    "foreground_evidence_68"
                ].detach().cpu().float(),
                "dabe_clean_bg_evidence_37": payload[
                    "background_evidence_37"
                ].detach().cpu().float(),
                "dabe_clean_bg_evidence_68": payload[
                    "background_evidence_68"
                ].detach().cpu().float(),
            }
        )
        if expected_version == DABE_CLEAN_CONTREC_VERSION:
            if bool(payload.get("training_gt_read", True)):
                raise RuntimeError(
                    f"DABE-Clean contrec cache reports GT access: {row['cache_path']}"
                )
            forbidden = {
                "weight_map",
                "static_weight_map",
                "hard_ring",
                "extent",
                "unknown",
                "fg_core",
                "bg_core",
            }
            leaked = sorted(forbidden.intersection(payload))
            if leaked:
                raise RuntimeError(
                    "DABE-Clean contrec cache leaked forbidden fields: "
                    f"{leaked} | {row['cache_path']}"
                )
            out.update(
                {
                    "dabe_clean_p_rw_37": payload["p_rw_37"].detach().cpu().float(),
                    "dabe_clean_evidence_gate_37": payload[
                        "evidence_gate_37"
                    ].detach().cpu().float(),
                    "dabe_clean_semantic_fg_tendency_37": payload[
                        "semantic_fg_tendency_37"
                    ].detach().cpu().float(),
                    "dabe_clean_latent_rw_37": payload[
                        "latent_rw_37"
                    ].detach().cpu().float(),
                    "dabe_clean_recoverability_37": payload[
                        "recoverability_37"
                    ].detach().cpu().float(),
                    "dabe_clean_recoverability_68": payload[
                        "recoverability_68"
                    ].detach().cpu().float(),
                }
            )
    return out


def _load_dabe_clean_legacy_regions(row, expected_dataset, expected_stem, cfg):
    """Materialize only the five PU tensors needed by legacy ECST routing."""

    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Legacy ECST source must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset or payload.get("stem") != expected_stem:
        raise RuntimeError(f"Legacy ECST source identity mismatch: {row['cache_path']}")
    if str(payload.get("backbone_key")) != str(cfg.BACKBONE_KEY):
        raise RuntimeError(f"Legacy ECST source backbone mismatch: {row['cache_path']}")
    expected = (1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
    out = {}
    for output_key, source_key in LEGACY_ECST_REGION_FIELDS.items():
        value = payload.get(source_key)
        if not torch.is_tensor(value):
            raise RuntimeError(
                f"Legacy ECST source is missing {source_key}: {row['cache_path']}"
            )
        value = value.detach().cpu().float()
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"Legacy ECST {source_key} shape mismatch: {list(value.shape)} != "
                f"{list(expected)} | {row['cache_path']}"
            )
        _validate_unit_range(value, source_key, row["cache_path"])
        out[output_key] = value
    return out


def _load_tce_cover(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"TCE cover payload must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"TCE cover dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"TCE cover stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"TCE cover backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    expected_epoch = int(getattr(cfg, "TCE_COVER_EPOCH", -1))
    if expected_epoch >= 0 and int(payload.get("source_epoch", expected_epoch)) != expected_epoch:
        raise RuntimeError(
            f"TCE cover source_epoch mismatch for {row['cache_path']}: "
            f"{payload.get('source_epoch')} != {expected_epoch}"
        )
    expected_model = str(getattr(cfg, "TCE_COVER_MODEL", "")).lower()
    if expected_model and str(payload.get("model_for_cache", expected_model)).lower() != expected_model:
        raise RuntimeError(
            f"TCE cover model_for_cache mismatch for {row['cache_path']}: "
            f"{payload.get('model_for_cache')} != {expected_model}"
        )

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    out = {}
    for field in ("cover_prob_68", "cover_binary_68", "cover_conf_68"):
        tensor = payload.get(field)
        if not torch.is_tensor(tensor):
            raise RuntimeError(
                "TCE cover payload missing required field | "
                f"dataset={expected_dataset} | stem={expected_stem} | "
                f"cache_path={row['cache_path']} | missing_key={field}"
            )
        tensor = tensor.float()
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(
                "TCE cover tensor shape mismatch | "
                f"dataset={expected_dataset} | stem={expected_stem} | "
                f"cache_path={row['cache_path']} | {field} {list(tensor.shape)} != {expected_shape}"
            )
        _validate_unit_range(tensor, field, row["cache_path"])
        out[field] = tensor
    return {
        "tce_cover_prob_68": out["cover_prob_68"],
        "tce_cover_binary_68": out["cover_binary_68"],
        "tce_cover_conf_68": out["cover_conf_68"],
        "tce_cover_area": float(payload.get("cover_area", out["cover_binary_68"].mean().item())),
    }


def _load_lceg_cover(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"LCEG cover payload must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"LCEG cover dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"LCEG cover stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"LCEG cover backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    expected_epoch = int(getattr(cfg, "LCEG_COVER_EPOCH", -1))
    if expected_epoch >= 0 and int(payload.get("source_epoch", expected_epoch)) != expected_epoch:
        raise RuntimeError(
            f"LCEG cover source_epoch mismatch for {row['cache_path']}: "
            f"{payload.get('source_epoch')} != {expected_epoch}"
        )
    expected_model = str(getattr(cfg, "LCEG_COVER_MODEL", "")).lower()
    if expected_model and str(payload.get("model_for_cache", expected_model)).lower() != expected_model:
        raise RuntimeError(
            f"LCEG cover model_for_cache mismatch for {row['cache_path']}: "
            f"{payload.get('model_for_cache')} != {expected_model}"
        )

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    out = {}
    for field in ("cover_prob_68", "cover_binary_68", "cover_conf_68"):
        tensor = payload.get(field)
        if not torch.is_tensor(tensor):
            raise RuntimeError(
                "LCEG cover payload missing required field | "
                f"dataset={expected_dataset} | stem={expected_stem} | "
                f"cache_path={row['cache_path']} | missing_key={field}"
            )
        tensor = tensor.float()
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(
                "LCEG cover tensor shape mismatch | "
                f"dataset={expected_dataset} | stem={expected_stem} | "
                f"cache_path={row['cache_path']} | {field} {list(tensor.shape)} != {expected_shape}"
            )
        _validate_unit_range(tensor, field, row["cache_path"])
        out[field] = tensor
    return {
        "lceg_cover_prob_68": out["cover_prob_68"],
        "lceg_cover_binary_68": out["cover_binary_68"],
        "lceg_cover_conf_68": out["cover_conf_68"],
        "lceg_cover_area": float(payload.get("cover_area", out["cover_binary_68"].mean().item())),
    }


def _load_despl_paper(row, expected_dataset, expected_stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DESPL-paper cache payload must be a dict: {row['cache_path']}")
    if payload.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"DESPL-paper dataset mismatch for {row['cache_path']}: "
            f"{payload.get('dataset')} != {expected_dataset}"
        )
    if payload.get("stem") != expected_stem:
        raise RuntimeError(
            f"DESPL-paper stem mismatch for {row['cache_path']}: "
            f"{payload.get('stem')} != {expected_stem}"
        )
    expected_backbone = getattr(cfg, "DESPL_PAPER_BACKBONE_KEY", cfg.BACKBONE_KEY)
    if payload.get("backbone_key") != expected_backbone:
        raise RuntimeError(
            f"DESPL-paper backbone mismatch for {row['cache_path']}: "
            f"{payload.get('backbone_key')} != {expected_backbone}"
        )
    expected_sign = getattr(cfg, "DESPL_PAPER_SIGN_MODE", "paper")
    if payload.get("sign_mode") != expected_sign:
        raise RuntimeError(
            f"DESPL-paper sign_mode mismatch for {row['cache_path']}: "
            f"{payload.get('sign_mode')} != {expected_sign}"
        )
    if "p_despl_paper_soft" not in payload:
        raise KeyError(f"DESPL-paper cache missing p_despl_paper_soft: {row['cache_path']}")
    pseudo = payload["p_despl_paper_soft"]
    if not torch.is_tensor(pseudo):
        raise TypeError(f"DESPL-paper p_despl_paper_soft must be tensor: {row['cache_path']}")
    pseudo = pseudo.float()
    grid = int(getattr(cfg, "DESPL_GRID", pseudo.shape[-1]))
    expected_shape = [1, grid, grid]
    if list(pseudo.shape) != expected_shape:
        raise RuntimeError(
            f"DESPL-paper p_despl_paper_soft shape mismatch: "
            f"{list(pseudo.shape)} != {expected_shape} | {row['cache_path']}"
        )
    _validate_unit_range(pseudo, "p_despl_paper_soft", row["cache_path"])
    binary = payload.get("p_despl_paper", (pseudo > 0.5).float())
    if not torch.is_tensor(binary):
        raise TypeError(f"DESPL-paper p_despl_paper must be tensor: {row['cache_path']}")
    binary = binary.float()
    if list(binary.shape) != expected_shape:
        raise RuntimeError(
            f"DESPL-paper p_despl_paper shape mismatch: "
            f"{list(binary.shape)} != {expected_shape} | {row['cache_path']}"
        )
    return {
        "pseudo": pseudo,
        "binary": binary,
        "area": float(payload.get("area", pseudo.mean().item())),
        "view_consistency": float(payload.get("view_consistency", 0.0)),
        "sign_mode": str(payload.get("sign_mode", expected_sign)),
        "cache_path": row["cache_path"],
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


def _load_image_resize(image_path, size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((int(size), int(size)), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_image_68(image_path, loss_size):
    return _load_image_resize(image_path, loss_size)


def _load_image_136(image_path, size):
    return _load_image_resize(image_path, size)


def _normalized_sobel_from_image(image):
    if image.ndim != 3 or image.shape[0] != 3:
        raise RuntimeError(f"Sobel image must be [3,H,W], got {list(image.shape)}")
    gray = (
        0.299 * image[0:1]
        + 0.587 * image[1:2]
        + 0.114 * image[2:3]
    ).unsqueeze(0)
    sobel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    sobel_y = image.new_tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
    ).unsqueeze(0)
    dx = F.conv2d(gray, sobel_x, padding=1)
    dy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(dx * dx + dy * dy + 1e-6)
    maximum = magnitude.flatten(1).max(dim=1).values.view(1, 1, 1, 1)
    return (magnitude / (maximum + 1e-6)).clamp(0.0, 1.0).squeeze(0)


class CachedTrainDataset(Dataset):
    def __init__(self, cfg, max_samples=-1):
        self.cfg = cfg
        self.use_qra = bool(getattr(cfg, "USE_QRA", False))
        self.use_ccr = bool(getattr(cfg, "USE_CCR", False))
        self.use_drepp = bool(getattr(cfg, "USE_DREPP", False))
        self.use_dabe_pseudo = bool(getattr(cfg, "USE_DABE_PSEUDO", False))
        self.use_dabe_pu = bool(getattr(cfg, "USE_DABE_PU", False))
        self.use_dabe_clean = bool(getattr(cfg, "USE_DABE_CLEAN", False))
        self.use_ecst_clean = bool(getattr(cfg, "USE_ECST_CLEAN", False))
        self.use_dabe_clean_offline = self.use_dabe_clean and str(
            getattr(cfg, "DABE_CLEAN_VERSION", "")
        ).strip().lower() == "v3_offline_consolidation"
        self.use_teacher_only_no_offline_pseudo = (
            teacher_only_no_offline_pseudo_enabled(cfg)
        )
        self.use_found_static = (
            self.use_dabe_clean_offline and found_static_enabled(cfg)
        )
        self.dabe_clean_training_target_source = str(
            getattr(cfg, "DABE_CLEAN_TRAINING_TARGET_SOURCE", "target_offline_68")
        ).strip().lower()
        self.dabe_clean_target_mode = str(
            getattr(
                cfg,
                "DABE_CLEAN_OFFLINE_MODE"
                if self.use_dabe_clean_offline
                else "DABE_CLEAN_TARGET_MODE",
                "" if self.use_dabe_clean_offline else "dp",
            )
        ).strip().lower()
        self.dabe_clean_use_legacy_regions = self.use_dabe_clean and bool(
            getattr(cfg, "DABE_CLEAN_USE_LEGACY_ECST_REGIONS", False)
        )
        self.use_tce = bool(getattr(cfg, "USE_TCE", False))
        self.use_lceg = bool(getattr(cfg, "USE_LCEG", False))
        self.use_despl_pseudo = bool(getattr(cfg, "USE_DESPL_PSEUDO", False))
        self.p_init_mode = str(getattr(cfg, "P_INIT_MODE", ""))
        self.use_dabe_oem = self.use_dabe_pu and (
            bool(getattr(cfg, "USE_DABE_OEM", False)) or self.p_init_mode == "dabe_pu_v11_oem"
        )
        self.use_dabe_pu_despl_sched = self.use_dabe_pu and self.p_init_mode in DABE_PU_DESPL_SCHED_MODES
        self.use_dabe_only = self.use_dabe_pseudo and self.p_init_mode in {"dabe_only", "dabe_gc_only"}
        self.use_dabe_pu_v11 = self.use_dabe_pu and self.p_init_mode in DABE_PU_ALLOWED_MODES
        self.use_despl_only = self.use_despl_pseudo and self.p_init_mode == "despl_only"
        self.use_despl_paper = self.use_despl_pseudo and self.p_init_mode == "despl_paper_only"
        self.use_dre_safe_prior = self.use_despl_pseudo and bool(getattr(cfg, "USE_DRE_SAFE_PRIOR", False))
        self.use_despl_light_cache = self.use_despl_pseudo and bool(getattr(cfg, "USE_DESPL_LIGHT_CACHE", False))
        self.use_multi_level_feature = bool(getattr(cfg, "USE_MULTI_LEVEL_FEATURE", False))
        self.use_ndr_branch = bool(getattr(cfg, "USE_NDR_BRANCH", False))
        self.use_csd_decoder = bool(getattr(cfg, "USE_CSD_DECODER", False))
        self.use_csd_v1r = bool(getattr(cfg, "USE_CSD_V1R", False)) or str(
            getattr(cfg, "HEAD_TYPE", "")
        ).lower() == "dagp_safe_csd_v1r"
        self.use_hr_bfr = bool(getattr(cfg, "USE_HR_BFR", False))
        self.use_cssd = bool(getattr(cfg, "USE_CSSD", False))
        self.use_cacd = bool(getattr(cfg, "USE_CACD", False)) or str(
            getattr(cfg, "HEAD_TYPE", "")
        ).lower() == "cacd_v1_base"
        self.use_multi_view_feature = bool(getattr(cfg, "USE_MULTI_VIEW_FEATURE", False))
        self.multi_view_types = [str(view).lower() for view in getattr(cfg, "MULTI_VIEW_TYPES", [])]
        self.use_source_arbiter = bool(getattr(cfg, "USE_SOURCE_ARBITER", False))
        self.use_ap_stcr = bool(getattr(cfg, "USE_AP_STCR", False))
        self.use_bitc = bool(getattr(cfg, "USE_BITC", False))
        self.use_cvsa = bool(getattr(cfg, "USE_CVSA", False)) or str(
            getattr(cfg, "SUPERVISION_MODE", "")
        ).lower() == "cvsa"
        self.use_arbiter_hflip = self.use_source_arbiter and bool(
            getattr(cfg, "SOURCE_ARBITER_UTILITY_USE_HFLIP", True)
        )
        self.use_hflip_view = (
            self.use_multi_view_feature and "hflip" in self.multi_view_types
        ) or self.use_arbiter_hflip or self.use_cvsa
        self.multi_level_layers = [int(layer) for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12])]
        if self.use_hflip_view and self.use_multi_level_feature:
            raise RuntimeError("HFlip multi-view feature currently supports single-level cached DINO features only.")
        if self.use_qra and self.use_ccr:
            raise RuntimeError("USE_QRA=True and USE_CCR=True cannot be combined.")
        if self.use_drepp and (self.use_qra or self.use_ccr or self.use_despl_pseudo or self.use_dabe_pseudo or self.use_dabe_pu or self.use_dabe_clean):
            raise RuntimeError("USE_DREPP=True cannot be combined with QRA/CCR/DESPL/DABE-PU/DABE-Clean.")
        if self.use_despl_pseudo and (self.use_qra or self.use_ccr):
            raise RuntimeError("USE_DESPL_PSEUDO=True cannot be combined with USE_QRA=True or USE_CCR=True.")
        if self.use_dabe_pseudo:
            if not self.use_dabe_only:
                raise RuntimeError(
                    "USE_DABE_PSEUDO=True currently requires P_INIT_MODE in "
                    "{'dabe_only', 'dabe_gc_only'}."
                )
            if not (self.use_despl_pseudo and self.use_despl_light_cache):
                raise RuntimeError("DABE-only training requires DESPL light cache for pseudo_fixed/pseudo_despl diagnostics.")
        if self.use_dabe_pu:
            if self.use_dabe_pseudo:
                raise RuntimeError("USE_DABE_PU=True cannot be combined with USE_DABE_PSEUDO=True.")
            if not self.use_dabe_pu_v11:
                raise RuntimeError(
                    "USE_DABE_PU=True currently requires P_INIT_MODE in "
                    "{'dabe_pu_v11', 'dabe_pu_v11_oem', "
                    "'dabe_pu_v11_desplsched', 'dabe_pu_v11_desplsched_exactreset', "
                    "'dabe_pu_v11_desplsched_A1_keepteacher_lowlr', "
                    "'dabe_pu_v11_desplsched_A2_resetteacher_highlr', "
                    "'dabe_pu_v11_desplsched_softteacher', "
                    "'dabe_pu_v11_desplsched_dabehard', "
                    "'dabe_pu_v12_shape_desplsched'}."
                )
            if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() not in DABE_PU_ALLOWED_VERSIONS:
                raise RuntimeError(
                    "USE_DABE_PU=True currently requires DABE_PU_VERSION in "
                    "{'pu_v11', 'pu_v12_shape_complete'}."
                )
            if (
                not self.use_dabe_oem
                and not self.use_dabe_pu_despl_sched
                and not (self.use_despl_pseudo and self.use_despl_light_cache)
            ):
                raise RuntimeError("DABE-PU training requires DESPL light cache for pseudo_fixed/pseudo_despl diagnostics.")
        if self.use_dabe_clean:
            if self.use_dabe_pu or self.use_dabe_pseudo or self.use_despl_pseudo:
                raise RuntimeError(
                    "USE_DABE_CLEAN=True is an independent supervision path and "
                    "cannot be combined with DABE-PU/DABE-pseudo/DESPL."
                )
            allowed_clean_modes = (
                DABE_CLEAN_OFFLINE_MODES
                if self.use_dabe_clean_offline
                else DABE_CLEAN_TARGET_MODES
            )
            if self.dabe_clean_target_mode not in allowed_clean_modes:
                raise RuntimeError(
                    f"Unsupported DABE-Clean mode={self.dabe_clean_target_mode!r}."
                )
            if str(getattr(cfg, "DABE_CLEAN_STATIC_WEIGHT_MODE", "ones")).lower() != "ones":
                raise RuntimeError("DABE-Clean requires DABE_CLEAN_STATIC_WEIGHT_MODE='ones'.")
            if self.use_ecst_clean:
                if self.dabe_clean_use_legacy_regions:
                    raise RuntimeError(
                        "Clean-ECST Dataset must not load legacy ECST regions."
                    )
                if bool(getattr(cfg, "DABE_CLEAN_LEGACY_ROUTING_ONLY", False)):
                    raise RuntimeError(
                        "Clean-ECST Dataset requires DABE_CLEAN_LEGACY_ROUTING_ONLY=False."
                    )
                if str(getattr(cfg, "DABE_CLEAN_LEGACY_REGION_ROOT", "")).strip():
                    raise RuntimeError(
                        "Clean-ECST Dataset requires an empty legacy region root."
                    )
                if str(getattr(cfg, "DABE_PU_ROOT", "")).strip():
                    raise RuntimeError(
                        "Clean-ECST Dataset requires the inactive DABE-PU root to be empty."
                    )
            if self.use_dabe_clean_offline:
                if self.dabe_clean_training_target_source not in {
                    "target_offline_68",
                    FOUND_STATIC_SOURCE,
                    TEACHER_ONLY_NO_OFFLINE_PSEUDO_SOURCE,
                }:
                    raise RuntimeError(
                        "Unsupported DABE_CLEAN_TRAINING_TARGET_SOURCE="
                        f"{self.dabe_clean_training_target_source!r}."
                    )
                if self.use_found_static:
                    if str(
                        getattr(cfg, "FOUND_STATIC_RESIZE_MODE", "")
                    ).strip().lower() != FOUND_STATIC_RESIZE_MODE:
                        raise RuntimeError(
                            "FOUND static ablation requires bilinear resize."
                        )
                forbidden_flags = {
                    "USE_ECST": bool(getattr(cfg, "USE_ECST", False)),
                    "USE_ECST_MINIMAL": bool(
                        getattr(cfg, "USE_ECST_MINIMAL", False)
                    ),
                    "USE_ECST_CLEAN": bool(getattr(cfg, "USE_ECST_CLEAN", False)),
                    "DABE_CLEAN_USE_LEGACY_ECST_REGIONS": self.dabe_clean_use_legacy_regions,
                    "DABE_CLEAN_LEGACY_ROUTING_ONLY": bool(
                        getattr(cfg, "DABE_CLEAN_LEGACY_ROUTING_ONLY", False)
                    ),
                }
                enabled = [name for name, value in forbidden_flags.items() if value]
                if enabled:
                    raise RuntimeError(
                        f"Pure-offline DABE-Clean forbids online routing: {enabled}."
                    )
                if str(getattr(cfg, "TEACHER_ROUTING_MODE", "none")).lower() != "none":
                    raise RuntimeError(
                        "Pure-offline DABE-Clean requires TEACHER_ROUTING_MODE='none'."
                    )
        # 训练集只建立 image/cache 索引，不读取 GT，避免把训练 GT 引入监督。
        self.items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
        if max_samples >= 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise RuntimeError("Training dataset is empty.")
        self.keys = [(item["dataset"], item["stem"]) for item in self.items]

        feature_manifest = (
            ml_feature_cache_manifest_path(cfg, "train")
            if self.use_multi_level_feature
            else _feature_manifest_path(cfg, "train")
        )
        feature_rows = read_jsonl(feature_manifest)
        self.feature_map = manifest_to_map(feature_rows, feature_manifest)
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

        self.cacd_feature_map = None
        self.cacd_first_cache_path = None
        if self.use_cacd:
            cacd_manifest = cacd_feature_manifest_path(cfg, "train")
            cacd_rows = read_jsonl(cacd_manifest)
            self.cacd_feature_map = manifest_to_map(cacd_rows, cacd_manifest)
            if max_samples < 0:
                check_exact_keys("CACD feature train cache", self.cacd_feature_map.keys(), self.keys)
            else:
                missing_cacd = sorted(set(self.keys) - set(self.cacd_feature_map))
                if missing_cacd:
                    raise RuntimeError(f"CACD feature train cache missing first 10: {missing_cacd[:10]}")
            self.cacd_first_cache_path = self.cacd_feature_map[self.keys[0]]["cache_path"]

        self.cssd_hr_feature_map = None
        self.cssd_hr_first_cache_path = None
        self.cssd_hr_feature_shape = None
        if self.use_cssd:
            cssd_manifest = cssd_hr_feature_manifest_path(cfg)
            cssd_rows = read_jsonl(cssd_manifest)
            self.cssd_hr_feature_map = manifest_to_map(cssd_rows, cssd_manifest)
            if max_samples < 0:
                check_exact_keys("CSSD HR feature train cache", self.cssd_hr_feature_map.keys(), self.keys)
            else:
                missing_cssd = sorted(set(self.keys) - set(self.cssd_hr_feature_map))
                if missing_cssd:
                    raise RuntimeError(f"CSSD HR feature cache missing first 10: {missing_cssd[:10]}")
            first_dataset, first_stem = self.keys[0]
            first_cssd_feature, _ = _load_cssd_hr_feature(
                self.cssd_hr_feature_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.cssd_hr_feature_shape = list(first_cssd_feature.shape)
            self.cssd_hr_first_cache_path = self.cssd_hr_feature_map[(first_dataset, first_stem)]["cache_path"]

        self.hflip_feature_map = None
        self.hflip_feature_root = None
        self.hflip_first_cache_path = None
        if self.use_hflip_view:
            hflip_manifest = (
                _cvsa_manifest_path(cfg, "feature_cache_hflip_root")
                if self.use_cvsa
                else hflip_feature_cache_manifest_path(cfg)
            )
            hflip_rows = read_jsonl(hflip_manifest)
            self.hflip_feature_map = manifest_to_map(hflip_rows, hflip_manifest)
            if max_samples < 0:
                check_exact_keys("hflip feature train cache", self.hflip_feature_map.keys(), self.keys)
            else:
                missing_hflip = sorted(set(self.keys) - set(self.hflip_feature_map))
                if missing_hflip:
                    raise RuntimeError(f"hflip feature train cache missing first 10: {missing_hflip[:10]}")
            self.hflip_feature_root = str(hflip_manifest.parent.resolve())

        self.cvsa_fixed_hflip_map = None
        self.cvsa_fixed_hflip_root = None
        self.cvsa_fixed_hflip_first_cache_path = None
        if self.use_cvsa:
            fixed_hflip_manifest = _cvsa_manifest_path(
                cfg, "fixed_cache_hflip_root"
            )
            fixed_hflip_rows = read_jsonl(fixed_hflip_manifest)
            self.cvsa_fixed_hflip_map = manifest_to_map(
                fixed_hflip_rows, fixed_hflip_manifest
            )
            if max_samples < 0:
                check_exact_keys(
                    "CVSA hflip DABE-PU cache",
                    self.cvsa_fixed_hflip_map.keys(),
                    self.keys,
                )
            else:
                missing_fixed_hflip = sorted(
                    set(self.keys) - set(self.cvsa_fixed_hflip_map)
                )
                if missing_fixed_hflip:
                    raise RuntimeError(
                        "CVSA hflip DABE-PU cache missing first 10: "
                        f"{missing_fixed_hflip[:10]}"
                    )
            self.cvsa_fixed_hflip_root = str(
                fixed_hflip_manifest.parent.resolve()
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
        if self.use_dabe_pseudo and self.pseudo_cache_override is not None:
            raise RuntimeError("USE_DABE_PSEUDO=True cannot be combined with PSEUDO_CACHE_OVERRIDE.")
        if self.use_dabe_pu and self.pseudo_cache_override is not None:
            raise RuntimeError("USE_DABE_PU=True cannot be combined with PSEUDO_CACHE_OVERRIDE.")
        if self.use_dabe_clean and self.pseudo_cache_override is not None:
            raise RuntimeError("USE_DABE_CLEAN=True cannot be combined with PSEUDO_CACHE_OVERRIDE.")
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
        self.despl_paper_map = None
        self.dabe_map = None
        self.dabe_cache_root = None
        self.dabe_first_cache_path = None
        self.dabe_first_source_key = None
        self.dabe_first_resized_from_37 = False
        self.dabe_first_resized_from = None
        self.dabe_first_resized_to = None
        self.dabe_pu_map = None
        self.dabe_pu_cache_root = None
        self.dabe_pu_first_cache_path = None
        self.dabe_clean_map = None
        self.dabe_clean_cache_root = None
        self.dabe_clean_first_cache_path = None
        self.found_static_map = None
        self.found_static_cache_root = None
        self.dabe_clean_legacy_region_map = None
        self.dabe_clean_legacy_region_root = None
        self.bitc_map = None
        self.bitc_cache_root = None
        self.bitc_first_cache_path = None
        self.bitc_cache_fingerprint = None
        self.tce_cover_map = None
        self.tce_cover_cache_root = None
        self.tce_cover_first_cache_path = None
        self.lceg_cover_map = None
        self.lceg_cover_cache_root = None
        self.lceg_cover_first_cache_path = None
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
            if self.use_despl_paper:
                despl_manifest = despl_paper_manifest_path(
                    cfg,
                    sign_mode=getattr(cfg, "DESPL_PAPER_SIGN_MODE", "paper"),
                )
            else:
                despl_manifest = (
                    despl_light_cache_manifest_path(cfg)
                    if self.use_despl_light_cache
                    else despl_pseudo_bank_manifest_path(cfg)
                )
            despl_rows = read_jsonl(despl_manifest)
            self.despl_map = manifest_to_map(despl_rows, despl_manifest)
            if max_samples < 0:
                if self.use_despl_paper:
                    cache_name = "DESPL-paper cache"
                else:
                    cache_name = "DESPL light cache" if self.use_despl_light_cache else "DESPL pseudo bank"
                check_exact_keys(cache_name, self.despl_map.keys(), self.keys)
            else:
                missing_despl = sorted(set(self.keys) - set(self.despl_map))
                if missing_despl:
                    if self.use_despl_paper:
                        cache_name = "DESPL-paper cache"
                    else:
                        cache_name = "DESPL light cache" if self.use_despl_light_cache else "DESPL pseudo bank"
                    raise RuntimeError(f"{cache_name} missing first 10: {missing_despl[:10]}")
            self.despl_cache_root = str(despl_manifest.parent.resolve())
            self.actual_pseudo_cache_root = self.despl_cache_root
            self.actual_pseudo_cache_pattern = f"{self.despl_cache_root}/<dataset>/<stem>.pt"
            self.pseudo_map = {}
            if not self.use_despl_light_cache and not self.use_despl_paper:
                pseudo_manifest = _pseudo_manifest_path(cfg)
                if pseudo_manifest.exists():
                    pseudo_rows = read_jsonl(pseudo_manifest)
                    self.pseudo_map = manifest_to_map(pseudo_rows, pseudo_manifest)
        elif self.use_dabe_pu or self.use_dabe_clean:
            self.pseudo_map = {}
            self.actual_pseudo_cache_root = self.original_pseudo_cache_root
            self.actual_pseudo_cache_pattern = f"{self.original_pseudo_cache_root}/<dataset>/<stem>.pt"
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

        if self.use_dabe_pseudo:
            dabe_manifest = dabe_pseudo_manifest_path(cfg)
            dabe_rows = read_jsonl(dabe_manifest)
            self.dabe_map = manifest_to_map(dabe_rows, dabe_manifest)
            if max_samples < 0:
                check_exact_keys("DABE pseudo cache", self.dabe_map.keys(), self.keys)
            else:
                missing_dabe = sorted(set(self.keys) - set(self.dabe_map))
                if missing_dabe:
                    raise RuntimeError(f"DABE pseudo cache missing first 10: {missing_dabe[:10]}")
            self.dabe_cache_root = str(dabe_manifest.parent.resolve())
            self.actual_pseudo_cache_root = self.dabe_cache_root
            self.actual_pseudo_cache_pattern = f"{self.dabe_cache_root}/<dataset>/<stem>.pt"
        if self.use_dabe_pu:
            dabe_pu_manifest = dabe_pu_manifest_path(cfg)
            dabe_pu_rows = read_jsonl(dabe_pu_manifest)
            self.dabe_pu_map = manifest_to_map(dabe_pu_rows, dabe_pu_manifest)
            if max_samples < 0:
                check_exact_keys("DABE-PU cache", self.dabe_pu_map.keys(), self.keys)
            else:
                missing_dabe_pu = sorted(set(self.keys) - set(self.dabe_pu_map))
                if missing_dabe_pu:
                    raise RuntimeError(f"DABE-PU cache missing first 10: {missing_dabe_pu[:10]}")
            self.dabe_pu_cache_root = str(dabe_pu_manifest.parent.resolve())
            self.actual_pseudo_cache_root = self.dabe_pu_cache_root
            self.actual_pseudo_cache_pattern = f"{self.dabe_pu_cache_root}/<dataset>/<stem>.pt"
        if self.use_dabe_clean and not self.use_teacher_only_no_offline_pseudo:
            dabe_clean_manifest = (
                found_static_manifest_path(cfg)
                if self.use_found_static
                else dabe_clean_manifest_path(cfg)
            )
            dabe_clean_rows = read_jsonl(dabe_clean_manifest)
            clean_map = manifest_to_map(dabe_clean_rows, dabe_clean_manifest)
            if self.use_found_static:
                self.found_static_map = clean_map
                self.dabe_clean_map = None
            else:
                self.dabe_clean_map = clean_map
            if max_samples < 0:
                check_exact_keys(
                    "FOUND static cache" if self.use_found_static else "DABE-Clean cache",
                    clean_map.keys(),
                    self.keys,
                )
            else:
                missing_clean = sorted(set(self.keys) - set(clean_map))
                if missing_clean:
                    raise RuntimeError(
                        f"{'FOUND static' if self.use_found_static else 'DABE-Clean'} "
                        f"cache missing first 10: {missing_clean[:10]}"
                    )
            self.dabe_clean_cache_root = str(dabe_clean_manifest.parent.resolve())
            if self.use_found_static:
                self.found_static_cache_root = self.dabe_clean_cache_root
            self.actual_pseudo_cache_root = self.dabe_clean_cache_root
            self.actual_pseudo_cache_pattern = (
                f"{self.dabe_clean_cache_root}/<dataset>/<stem>.pt"
            )
        elif self.use_teacher_only_no_offline_pseudo:
            self.dabe_clean_map = None
            self.found_static_map = None
            self.dabe_clean_cache_root = ""
            self.actual_pseudo_cache_root = ""
            self.actual_pseudo_cache_pattern = "not_used_teacher_only"
        if self.dabe_clean_use_legacy_regions:
            legacy_manifest = dabe_clean_legacy_region_manifest_path(cfg)
            legacy_rows = read_jsonl(legacy_manifest)
            self.dabe_clean_legacy_region_map = manifest_to_map(
                legacy_rows, legacy_manifest
            )
            if max_samples < 0:
                check_exact_keys(
                    "DABE-Clean legacy ECST region cache",
                    self.dabe_clean_legacy_region_map.keys(),
                    self.keys,
                )
            else:
                missing_legacy = sorted(
                    set(self.keys) - set(self.dabe_clean_legacy_region_map)
                )
                if missing_legacy:
                    raise RuntimeError(
                        "DABE-Clean legacy ECST region cache missing first 10: "
                        f"{missing_legacy[:10]}"
                    )
            self.dabe_clean_legacy_region_root = str(
                legacy_manifest.parent.resolve()
            )
        if self.use_bitc:
            bitc_manifest = _bitc_manifest_path(cfg)
            bitc_rows = read_jsonl(bitc_manifest)
            self.bitc_map = manifest_to_map(bitc_rows, bitc_manifest)
            if max_samples < 0:
                check_exact_keys("BITC background intervention cache", self.bitc_map.keys(), self.keys)
            else:
                missing_bitc = sorted(set(self.keys) - set(self.bitc_map))
                if missing_bitc:
                    raise RuntimeError(
                        f"BITC background intervention cache missing first 10: {missing_bitc[:10]}"
                    )
            self.bitc_cache_root = str(bitc_manifest.parent.resolve())
        if self.use_tce:
            tce_manifest = tce_cover_manifest_path(cfg)
            tce_rows = read_jsonl(tce_manifest)
            self.tce_cover_map = manifest_to_map(tce_rows, tce_manifest)
            if max_samples < 0:
                check_exact_keys("TCE cover cache", self.tce_cover_map.keys(), self.keys)
            else:
                missing_tce = sorted(set(self.keys) - set(self.tce_cover_map))
                if missing_tce:
                    raise RuntimeError(f"TCE cover cache missing first 10: {missing_tce[:10]}")
            self.tce_cover_cache_root = str(tce_manifest.parent.resolve())
        if self.use_lceg:
            lceg_manifest = lceg_cover_manifest_path(cfg)
            lceg_rows = read_jsonl(lceg_manifest)
            self.lceg_cover_map = manifest_to_map(lceg_rows, lceg_manifest)
            if max_samples < 0:
                check_exact_keys("LCEG cover cache", self.lceg_cover_map.keys(), self.keys)
            else:
                missing_lceg = sorted(set(self.keys) - set(self.lceg_cover_map))
                if missing_lceg:
                    raise RuntimeError(f"LCEG cover cache missing first 10: {missing_lceg[:10]}")
            self.lceg_cover_cache_root = str(lceg_manifest.parent.resolve())

        first_dataset, first_stem = self.keys[0]
        if self.use_multi_level_feature:
            ml_payload = _load_multi_level_feature(
                self.feature_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            feature = ml_payload["feature"]
        else:
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
            if self.use_despl_paper:
                despl_payload = _load_despl_paper(
                    self.despl_map[(first_dataset, first_stem)],
                    first_dataset,
                    first_stem,
                    cfg,
                )
                self.pseudo_shape = list(despl_payload["pseudo"].shape)
                self.pseudo_source = "despl_paper_cache"
            elif self.use_despl_light_cache:
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
            elif self.use_despl_paper:
                self.pseudo_final_candidate = "p_despl_paper_soft"
            elif self.use_despl_only:
                self.pseudo_final_candidate = "p_despl"
            else:
                self.pseudo_final_candidate = (
                    f"{self.despl_blend_despl_weight:.3f}*p_despl+"
                    f"{self.despl_blend_fixed_weight:.3f}*p_fixed"
                )
        elif not (self.use_dabe_pu or self.use_dabe_clean):
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
        if self.use_dabe_pseudo:
            dabe_payload = _load_dabe_pseudo(
                self.dabe_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.pseudo_shape = list(dabe_payload["pseudo_dabe"].shape)
            self.pseudo_source = _dabe_source_name(cfg)
            self.pseudo_final_candidate = "p_dabe_68"
            self.dabe_first_cache_path = self.dabe_map[(first_dataset, first_stem)]["cache_path"]
            self.dabe_first_source_key = dabe_payload["dabe_source_key"]
            self.dabe_first_resized_from_37 = bool(dabe_payload["dabe_resized_from_37"])
            self.dabe_first_resized_from = list(dabe_payload["dabe_resized_from"])
            self.dabe_first_resized_to = list(dabe_payload["dabe_resized_to"])
            self.first_pseudo_cache_path = self.dabe_first_cache_path
        if self.use_dabe_pu:
            dabe_pu_payload = _load_dabe_pu_v11(
                self.dabe_pu_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.pseudo_shape = list(dabe_pu_payload["pu_target_soft"].shape)
            self.pseudo_source = (
                "dabe_pu_v12_shape_cache"
                if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() == "pu_v12_shape_complete"
                else "dabe_pu_v11_cache"
            )
            teacher_target_desc = (
                "soft_prob"
                if (
                    bool(getattr(cfg, "USE_TEACHER_SOFT_FULL_LOSS", False))
                    or str(getattr(cfg, "TEACHER_TARGET_MODE", "binary")).lower() == "soft_prob"
                )
                else "binary"
            )
            static_target_desc = (
                "target_hard_from_target_soft_68"
                if (
                    bool(getattr(cfg, "USE_DABE_PU_HARD_STATIC_TARGET", False))
                    or str(getattr(cfg, "DABE_PU_STATIC_TARGET_MODE", "soft")).lower()
                    == "hard_from_target_soft"
                )
                else str(
                    getattr(cfg, "DABE_PU_STATIC_SOURCE", "target_soft_68")
                ).strip().lower()
            )
            self.pseudo_final_candidate = (
                "DABE-PU seed masks + OEM dynamic extent"
                if self.use_dabe_oem
                else (
                    f"{static_target_desc} + DESPL-style full teacher {teacher_target_desc}"
                    if self.use_dabe_pu_despl_sched
                    else "target_soft_68"
                )
            )
            self.dabe_pu_first_cache_path = self.dabe_pu_map[(first_dataset, first_stem)]["cache_path"]
            self.first_pseudo_cache_path = self.dabe_pu_first_cache_path
        if self.use_teacher_only_no_offline_pseudo:
            self.pseudo_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
            self.pseudo_source = TEACHER_ONLY_NO_OFFLINE_PSEUDO_SOURCE
            self.pseudo_final_candidate = "binary EMA Teacher only"
            self.dabe_clean_first_cache_path = "not_used_teacher_only"
            self.first_pseudo_cache_path = "not_used_teacher_only"
        elif self.use_dabe_clean:
            first_clean_map = (
                self.found_static_map if self.use_found_static else self.dabe_clean_map
            )
            dabe_clean_payload = (
                _load_found_static(
                    first_clean_map[(first_dataset, first_stem)],
                    first_dataset,
                    first_stem,
                    cfg,
                )
                if self.use_found_static
                else _load_dabe_clean(
                    first_clean_map[(first_dataset, first_stem)],
                    first_dataset,
                    first_stem,
                    cfg,
                )
            )
            clean_target_key = (
                "target_offline_68"
                if self.use_dabe_clean_offline
                else "dabe_clean_target_68"
            )
            self.pseudo_shape = list(dabe_clean_payload[clean_target_key].shape)
            self.pseudo_source = str(dabe_clean_payload["dabe_clean_version"])
            self.pseudo_final_candidate = (
                "FOUND fixed 28x28 -> bilinear 68x68 + DESPL-style full "
                "binary EMA Teacher"
                if self.use_found_static
                else (
                    f"single continuous DABE target ({self.dabe_clean_target_mode}) "
                    "+ DESPL-style full binary EMA Teacher"
                )
            )
            self.dabe_clean_first_cache_path = first_clean_map[
                (first_dataset, first_stem)
            ]["cache_path"]
            self.first_pseudo_cache_path = self.dabe_clean_first_cache_path
            if self.dabe_clean_use_legacy_regions:
                _load_dabe_clean_legacy_regions(
                    self.dabe_clean_legacy_region_map[(first_dataset, first_stem)],
                    first_dataset,
                    first_stem,
                    cfg,
                )
        if self.use_bitc:
            bitc_payload = _load_bitc_cache(
                self.bitc_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.bitc_first_cache_path = bitc_payload["bitc_cache_path"]
            self.bitc_cache_fingerprint = bitc_payload["bitc_cache_fingerprint"]
        if self.use_tce:
            _load_tce_cover(
                self.tce_cover_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.tce_cover_first_cache_path = self.tce_cover_map[(first_dataset, first_stem)]["cache_path"]
        if self.use_lceg:
            _load_lceg_cover(
                self.lceg_cover_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            self.lceg_cover_first_cache_path = self.lceg_cover_map[(first_dataset, first_stem)]["cache_path"]
        if self.use_hflip_view:
            hflip_feature, _ = _load_feature(
                self.hflip_feature_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
            )
            self.hflip_feature_shape = list(hflip_feature.shape)
            if self.hflip_feature_shape != self.feature_shape:
                raise RuntimeError(
                    f"HFlip feature shape mismatch: {self.hflip_feature_shape} != {self.feature_shape} | "
                    f"{self.hflip_feature_map[(first_dataset, first_stem)]['cache_path']}"
                )
            self.hflip_first_cache_path = self.hflip_feature_map[(first_dataset, first_stem)]["cache_path"]
        if self.use_cvsa:
            _load_cvsa_hflip_fixed(
                self.cvsa_fixed_hflip_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            first_row = self.cvsa_fixed_hflip_map[(first_dataset, first_stem)]
            self.cvsa_fixed_hflip_first_cache_path = first_row.get(
                "fixed_path", first_row.get("cache_path")
            )
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

    def load_dabe_pu_target_soft(self, index):
        """Load only the validated DABE-PU soft target for state-bank setup."""
        if not self.use_dabe_pu:
            raise RuntimeError(
                "load_dabe_pu_target_soft() requires USE_DABE_PU=True."
            )
        index = int(index)
        if index < 0 or index >= len(self.items):
            raise IndexError(f"DABE-PU target index out of range: {index}.")
        item = self.items[index]
        dataset = item["dataset"]
        stem = item["stem"]
        payload = _load_dabe_pu_v11(
            self.dabe_pu_map[(dataset, stem)],
            dataset,
            stem,
            self.cfg,
        )
        target = payload["pu_target_soft"].detach().cpu().float()
        expected = (1, int(self.cfg.LOSS_SIZE), int(self.cfg.LOSS_SIZE))
        if tuple(target.shape) != expected:
            raise RuntimeError(
                f"DABE-PU target shape mismatch for {dataset}/{stem}: "
                f"{list(target.shape)} != {list(expected)}."
            )
        return target

    def __getitem__(self, index):
        item = self.items[index]
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)

        if self.use_multi_level_feature:
            ml_feature_payload = _load_multi_level_feature(self.feature_map[key], dataset, stem, self.cfg)
            feature = ml_feature_payload["feature"]
            feature_payload = ml_feature_payload["payload"]
        else:
            feature, feature_payload = _load_feature(self.feature_map[key], dataset, stem)
        cacd_features = None
        if self.use_cacd:
            cacd_features = _load_cacd_features(
                self.cacd_feature_map[key], dataset, stem, self.cfg
            )
        cssd_hr_feature = None
        if self.use_cssd:
            cssd_hr_feature, _ = _load_cssd_hr_feature(
                self.cssd_hr_feature_map[key], dataset, stem, self.cfg
            )
        hflip_feature = None
        cvsa_fixed_hflip = None
        if self.use_hflip_view:
            hflip_feature, hflip_payload = _load_feature(self.hflip_feature_map[key], dataset, stem)
            if list(hflip_feature.shape) != list(feature.shape):
                raise RuntimeError(
                    f"HFlip feature shape mismatch for {dataset}/{stem}: "
                    f"{list(hflip_feature.shape)} != {list(feature.shape)}"
                )
            if hflip_payload.get("view") not in {None, "hflip"}:
                raise RuntimeError(
                    f"HFlip feature payload view mismatch for {dataset}/{stem}: {hflip_payload.get('view')}"
                )
        if self.use_cvsa:
            cvsa_fixed_hflip = _load_cvsa_hflip_fixed(
                self.cvsa_fixed_hflip_map[key], dataset, stem, self.cfg
            )
        if self.use_drepp:
            drepp_payload = _load_drepp(self.drepp_map[key], dataset, stem, self.cfg)
            pseudo = drepp_payload["p_despl"].float()
        elif self.use_despl_pseudo:
            if self.use_despl_paper:
                paper_payload = _load_despl_paper(self.despl_map[key], dataset, stem, self.cfg)
                pseudo = paper_payload["pseudo"].float()
                pseudo_fixed = torch.zeros_like(pseudo)
                pseudo_despl = pseudo
                pseudo_base = pseudo
                pseudo_safe = pseudo
                dre_safe = None
                use_fixed_in_pseudo = False
                p_init_area = float(pseudo.mean().item())
                p_fixed_area = 0.0
                p_despl_area = p_init_area
                p_despl_paper_area = float(paper_payload["area"])
                p_despl_paper_view_consistency = float(paper_payload["view_consistency"])
                p_despl_paper_binary = paper_payload["binary"].float()
                p_despl_paper_sign_mode = paper_payload["sign_mode"]
            elif self.use_despl_light_cache:
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
            if not self.use_despl_paper:
                p_despl_paper_area = 0.0
                p_despl_paper_view_consistency = 0.0
                p_despl_paper_binary = torch.zeros_like(pseudo)
                p_despl_paper_sign_mode = ""
        elif not self.use_dabe_pu and not self.use_dabe_clean:
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
        else:
            pseudo = torch.zeros((1, int(self.cfg.LOSS_SIZE), int(self.cfg.LOSS_SIZE)), dtype=torch.float32)

        if self.use_dabe_pseudo:
            dabe_payload = _load_dabe_pseudo(self.dabe_map[key], dataset, stem, self.cfg)
            pseudo_dabe = dabe_payload["pseudo_dabe"].float()
            pseudo = pseudo_dabe
            p_dabe_area = float(dabe_payload["p_dabe_area"])
            p_init_area = p_dabe_area
            use_fixed_in_pseudo = False
            pseudo_base = pseudo_dabe
            pseudo_safe = pseudo_dabe
            dabe_fallback_flag = bool(dabe_payload["dabe_fallback_flag"])
            dabe_large_area_flag = bool(dabe_payload["dabe_large_area_flag"])
            dabe_num_components = int(dabe_payload["dabe_num_components"])
            dabe_source_key = str(dabe_payload["dabe_source_key"])
            dabe_resized_from_37 = bool(dabe_payload["dabe_resized_from_37"])
            dabe_fg_core_68 = dabe_payload.get("dabe_fg_core_68")
            dabe_bg_core_68 = dabe_payload.get("dabe_bg_core_68")
            dabe_evidence_68 = dabe_payload.get("dabe_evidence_68")
            dabe_uncertain_68 = dabe_payload.get("dabe_uncertain_68")
        if self.use_dabe_pu:
            dabe_pu_payload = _load_dabe_pu_v11(self.dabe_pu_map[key], dataset, stem, self.cfg)
            pseudo = dabe_pu_payload["pu_target_soft"].float()
            pseudo_base = pseudo
            pseudo_safe = pseudo
            p_init_area = float(dabe_pu_payload["pu_target_area"])
            use_fixed_in_pseudo = False
        if self.use_teacher_only_no_offline_pseudo:
            # Shape-only sentinel for legacy batching/debug interfaces. It is
            # constructed in memory, has zero loss weight, and is never used as
            # a supervision target.
            pseudo = torch.zeros(
                (1, int(self.cfg.LOSS_SIZE), int(self.cfg.LOSS_SIZE)),
                dtype=torch.float32,
            )
            pseudo_base = pseudo
            pseudo_safe = pseudo
            p_init_area = 0.0
            use_fixed_in_pseudo = False
            dabe_clean_payload = None
            dabe_clean_legacy_regions = None
        elif self.use_dabe_clean:
            dabe_clean_payload = (
                _load_found_static(
                    self.found_static_map[key], dataset, stem, self.cfg
                )
                if self.use_found_static
                else _load_dabe_clean(
                    self.dabe_clean_map[key], dataset, stem, self.cfg
                )
            )
            clean_target_key = (
                "target_offline_68"
                if self.use_dabe_clean_offline
                else "dabe_clean_target_68"
            )
            pseudo = dabe_clean_payload[clean_target_key].float()
            pseudo_base = pseudo
            pseudo_safe = pseudo
            p_init_area = float(pseudo.mean().item())
            use_fixed_in_pseudo = False
            dabe_clean_legacy_regions = (
                _load_dabe_clean_legacy_regions(
                    self.dabe_clean_legacy_region_map[key],
                    dataset,
                    stem,
                    self.cfg,
                )
                if self.dabe_clean_use_legacy_regions
                else None
            )
        if self.use_bitc:
            bitc_payload = _load_bitc_cache(
                self.bitc_map[key], dataset, stem, self.cfg
            )
        if self.use_tce:
            tce_cover_payload = _load_tce_cover(self.tce_cover_map[key], dataset, stem, self.cfg)
        if self.use_lceg:
            lceg_cover_payload = _load_lceg_cover(self.lceg_cover_map[key], dataset, stem, self.cfg)

        sample = {
            "feature": feature,
            "pseudo": pseudo,
            "dataset": dataset,
            "dataset_name": dataset,
            "stem": stem,
            "image_path": item["image_path"],
            "sample_index": int(index),
        }
        if self.use_cacd:
            sample["feature_l10"] = cacd_features["feature_l10"]
            sample["feature_l11"] = cacd_features["feature_l11"]
        if self.use_cssd:
            sample[str(getattr(self.cfg, "CSSD_HR_FEATURE_FIELD", "feature_cssd_hr"))] = cssd_hr_feature.float()
        if self.use_ndr_branch or self.use_csd_decoder or self.use_csd_v1r or self.use_cacd:
            image_68 = _load_image_68(item["image_path"], int(self.cfg.LOSS_SIZE))
            sample["image_68"] = image_68
            if self.use_cacd:
                sample["sobel_68"] = _normalized_sobel_from_image(image_68)
            if self.use_hflip_view:
                sample["image_hflip_68"] = torch.flip(image_68, dims=[-1])
        if self.use_hr_bfr:
            sample["image_136"] = _load_image_136(
                item["image_path"],
                int(getattr(self.cfg, "HR_BFR_SIZE", 136)),
            )
        if self.use_hflip_view:
            sample["feature_hflip"] = hflip_feature.float()
        if self.use_cvsa:
            sample["cvsa_fixed_hflip_68"] = cvsa_fixed_hflip.float()
        if self.use_multi_level_feature:
            sample.update(
                {
                    f"feature_l{int(layer)}": ml_feature_payload[f"feature_l{int(layer)}"]
                    for layer in self.multi_level_layers
                }
            )
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
            if self.use_despl_paper:
                sample.update(
                    {
                        "pseudo_despl_paper": pseudo.float(),
                        "pseudo_despl_paper_binary": p_despl_paper_binary.float(),
                        "p_despl_paper_area": p_despl_paper_area,
                        "p_despl_paper_view_consistency": p_despl_paper_view_consistency,
                        "despl_paper_sign_mode": p_despl_paper_sign_mode,
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
        if self.use_dabe_pseudo:
            sample.update(
                {
                    "pseudo_dabe": pseudo_dabe.float(),
                    "p_dabe_area": p_dabe_area,
                    "dabe_fallback_flag": bool(dabe_fallback_flag),
                    "dabe_large_area_flag": bool(dabe_large_area_flag),
                    "dabe_num_components": int(dabe_num_components),
                    "dabe_source_key": dabe_source_key,
                    "dabe_resized_from_37": bool(dabe_resized_from_37),
                    "use_fixed_in_pseudo": False,
                    "fixed_used_for_training": False,
                }
            )
            if bool(getattr(self.cfg, "USE_DABE_AWARE_LOSS", False)):
                sample.update(
                    {
                        "dabe_fg_core_68": dabe_fg_core_68.float(),
                        "dabe_bg_core_68": dabe_bg_core_68.float(),
                        "dabe_evidence_68": dabe_evidence_68.float(),
                        "dabe_uncertain_68": dabe_uncertain_68.float(),
                    }
                )
        if self.use_dabe_pu:
            if not self.use_despl_pseudo:
                zero_pseudo = torch.zeros_like(pseudo)
                sample.update(
                    {
                        "pseudo_fixed": zero_pseudo.float(),
                        "pseudo_despl": zero_pseudo.float(),
                        "p_fixed_area": 0.0,
                        "p_despl_area": 0.0,
                    }
                )
            sample.update(
                {
                    "pu_target_soft": dabe_pu_payload["pu_target_soft"].float(),
                    "pu_weight_map": dabe_pu_payload["pu_weight_map"].float(),
                    "pu_fg_core": dabe_pu_payload["pu_fg_core"].float(),
                    "pu_fg_fallback": dabe_pu_payload["pu_fg_fallback"].float(),
                    "pu_bg_core": dabe_pu_payload["pu_bg_core"].float(),
                    "pu_extent": dabe_pu_payload["pu_extent"].float(),
                    "pu_unknown": dabe_pu_payload["pu_unknown"].float(),
                    "pu_fg_core_37": dabe_pu_payload["pu_fg_core_37"].float()
                    if dabe_pu_payload.get("pu_fg_core_37") is not None
                    else torch.empty(0),
                    "pu_fg_fallback_37": dabe_pu_payload["pu_fg_fallback_37"].float()
                    if dabe_pu_payload.get("pu_fg_fallback_37") is not None
                    else torch.empty(0),
                    "pu_bg_core_37": dabe_pu_payload["pu_bg_core_37"].float()
                    if dabe_pu_payload.get("pu_bg_core_37") is not None
                    else torch.empty(0),
                    "pu_extent_37": dabe_pu_payload["pu_extent_37"].float()
                    if dabe_pu_payload.get("pu_extent_37") is not None
                    else torch.empty(0),
                    "pu_unknown_37": dabe_pu_payload["pu_unknown_37"].float()
                    if dabe_pu_payload.get("pu_unknown_37") is not None
                    else torch.empty(0),
                    "pu_target_soft_37": dabe_pu_payload["pu_target_soft_37"].float()
                    if self.use_ap_stcr
                    else torch.empty(0),
                    "pu_bg_anchor_37": dabe_pu_payload["pu_bg_anchor_37"].float()
                    if self.use_ap_stcr
                    else torch.empty(0),
                    "pu_target_area": float(dabe_pu_payload["pu_target_area"]),
                    "pu_weight_mean": float(dabe_pu_payload["pu_weight_mean"]),
                    "pu_fg_core_area": float(dabe_pu_payload["pu_fg_core_area"]),
                    "pu_fg_fallback_area": float(dabe_pu_payload["pu_fg_fallback_area"]),
                    "pu_bg_core_area": float(dabe_pu_payload["pu_bg_core_area"]),
                    "pu_extent_area": float(dabe_pu_payload["pu_extent_area"]),
                    "pu_unknown_area": float(dabe_pu_payload["pu_unknown_area"]),
                    "pu_target_base": dabe_pu_payload["pu_target_base"].float(),
                    "pu_weight_base": dabe_pu_payload["pu_weight_base"].float(),
                    "pu_sc_bg_lock": dabe_pu_payload["pu_sc_bg_lock"].float(),
                    "pu_sc_extent_agree_fg": dabe_pu_payload["pu_sc_extent_agree_fg"].float(),
                    "pu_sc_lost_extent": dabe_pu_payload["pu_sc_lost_extent"].float(),
                    "pu_sc_new_boundary": dabe_pu_payload["pu_sc_new_boundary"].float(),
                    "pu_target_base_mean": float(dabe_pu_payload["pu_target_base_mean"]),
                    "pu_weight_base_mean": float(dabe_pu_payload["pu_weight_base_mean"]),
                    "pu_target_delta_mean": float(dabe_pu_payload["pu_target_delta_mean"]),
                    "pu_sc_bg_lock_ratio": float(dabe_pu_payload["pu_sc_bg_lock_ratio"]),
                    "pu_sc_extent_agree_fg_ratio": float(dabe_pu_payload["pu_sc_extent_agree_fg_ratio"]),
                    "pu_sc_lost_extent_ratio": float(dabe_pu_payload["pu_sc_lost_extent_ratio"]),
                    "pu_sc_new_boundary_ratio": float(dabe_pu_payload["pu_sc_new_boundary_ratio"]),
                    "use_fixed_in_pseudo": False,
                    "fixed_used_for_training": False,
                }
            )
            if dabe_pu_payload.get("pu_p_base_soft") is not None:
                sample["pu_p_base_soft"] = dabe_pu_payload["pu_p_base_soft"].float()
        if self.use_teacher_only_no_offline_pseudo:
            sample.update(
                {
                    "teacher_only_no_offline_pseudo": True,
                    "use_fixed_in_pseudo": False,
                    "fixed_used_for_training": False,
                }
            )
        elif self.use_dabe_clean:
            if self.use_dabe_clean_offline:
                sample.update(
                    {
                        "target_offline_68": dabe_clean_payload[
                            "target_offline_68"
                        ].float(),
                    }
                )
            else:
                zero_pseudo = torch.zeros_like(pseudo)
                sample.update(
                    {
                        "pseudo_fixed": zero_pseudo,
                        "pseudo_despl": zero_pseudo,
                        "p_fixed_area": 0.0,
                        "p_despl_area": 0.0,
                        "dabe_clean_target_37": dabe_clean_payload[
                            "dabe_clean_target_37"
                        ].float(),
                        "dabe_clean_target_68": dabe_clean_payload[
                            "dabe_clean_target_68"
                        ].float(),
                        "dabe_clean_fg_evidence_37": dabe_clean_payload[
                            "dabe_clean_fg_evidence_37"
                        ].float(),
                        "dabe_clean_fg_evidence_68": dabe_clean_payload[
                            "dabe_clean_fg_evidence_68"
                        ].float(),
                        "dabe_clean_bg_evidence_37": dabe_clean_payload[
                            "dabe_clean_bg_evidence_37"
                        ].float(),
                        "dabe_clean_bg_evidence_68": dabe_clean_payload[
                            "dabe_clean_bg_evidence_68"
                        ].float(),
                        "dabe_clean_target_mode": str(
                            dabe_clean_payload["dabe_clean_target_mode"]
                        ),
                        "dabe_clean_version": str(
                            dabe_clean_payload["dabe_clean_version"]
                        ),
                        "use_fixed_in_pseudo": False,
                        "fixed_used_for_training": False,
                    }
                )
                for field in (
                    "dabe_clean_p_rw_37",
                    "dabe_clean_evidence_gate_37",
                    "dabe_clean_semantic_fg_tendency_37",
                    "dabe_clean_latent_rw_37",
                    "dabe_clean_recoverability_37",
                    "dabe_clean_recoverability_68",
                ):
                    if field in dabe_clean_payload:
                        sample[field] = dabe_clean_payload[field].float()
            if dabe_clean_legacy_regions is not None:
                sample.update(
                    {
                        key: value.float()
                        for key, value in dabe_clean_legacy_regions.items()
                    }
                )
        if self.use_bitc:
            sample.update(
                {
                    "bitc_topk_indices": bitc_payload["bitc_topk_indices"],
                    "bitc_topk_weights": bitc_payload["bitc_topk_weights"],
                    "bitc_background_anchor_mask": bitc_payload[
                        "bitc_background_anchor_mask"
                    ],
                    "bitc_nearest_bg_index": bitc_payload[
                        "bitc_nearest_bg_index"
                    ],
                    "bitc_fixed_random_bg_index": bitc_payload[
                        "bitc_fixed_random_bg_index"
                    ],
                    "bitc_background_anchor_count": int(
                        bitc_payload["bitc_background_anchor_count"]
                    ),
                    "bitc_cache_fingerprint": bitc_payload[
                        "bitc_cache_fingerprint"
                    ],
                }
            )
        if self.use_tce:
            sample.update(
                {
                    "tce_cover_prob_68": tce_cover_payload["tce_cover_prob_68"].float(),
                    "tce_cover_binary_68": tce_cover_payload["tce_cover_binary_68"].float(),
                    "tce_cover_conf_68": tce_cover_payload["tce_cover_conf_68"].float(),
                    "tce_cover_area": float(tce_cover_payload["tce_cover_area"]),
                }
            )
        if self.use_lceg:
            sample.update(
                {
                    "lceg_cover_prob_68": lceg_cover_payload["lceg_cover_prob_68"].float(),
                    "lceg_cover_binary_68": lceg_cover_payload["lceg_cover_binary_68"].float(),
                    "lceg_cover_conf_68": lceg_cover_payload["lceg_cover_conf_68"].float(),
                    "lceg_cover_area": float(lceg_cover_payload["lceg_cover_area"]),
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
        if (
            self.use_dabe_clean_offline
            and not self.use_teacher_only_no_offline_pseudo
        ):
            if not torch.equal(sample["pseudo"], sample["target_offline_68"]):
                raise RuntimeError(
                    "Pure-offline Dataset requires pseudo == target_offline_68."
                )
            forbidden_offline_outputs = {
                "dabe_clean_target_37",
                "dabe_clean_target_68",
                "dabe_clean_fg_evidence_37",
                "dabe_clean_fg_evidence_68",
                "dabe_clean_bg_evidence_37",
                "dabe_clean_bg_evidence_68",
                "dabe_clean_p_rw_37",
                "dabe_clean_evidence_gate_37",
                "dabe_clean_semantic_fg_tendency_37",
                "dabe_clean_latent_rw_37",
                "dabe_clean_recoverability_37",
                "dabe_clean_recoverability_68",
                "legacy_ecst_fg_core",
                "legacy_ecst_fg_fallback",
                "legacy_ecst_bg_core",
                "legacy_ecst_extent",
                "legacy_ecst_unknown",
                "weight_map",
                "static_weight_map",
                "routing_map",
            }.intersection(sample)
            if forbidden_offline_outputs:
                raise RuntimeError(
                    "Pure-offline Dataset leaked diagnostic/routing fields: "
                    f"{sorted(forbidden_offline_outputs)}."
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
        self.use_multi_level_feature = bool(getattr(cfg, "USE_MULTI_LEVEL_FEATURE", False))
        self.use_ndr_branch = bool(getattr(cfg, "USE_NDR_BRANCH", False))
        self.use_csd_decoder = bool(getattr(cfg, "USE_CSD_DECODER", False))
        self.use_csd_v1r = bool(getattr(cfg, "USE_CSD_V1R", False)) or str(
            getattr(cfg, "HEAD_TYPE", "")
        ).lower() == "dagp_safe_csd_v1r"
        self.use_hr_bfr = bool(getattr(cfg, "USE_HR_BFR", False))
        self.use_cacd = bool(getattr(cfg, "USE_CACD", False)) or str(
            getattr(cfg, "HEAD_TYPE", "")
        ).lower() == "cacd_v1_base"
        self.multi_level_layers = [int(layer) for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12])]

        feature_manifest = (
            ml_feature_cache_manifest_path(cfg, split)
            if self.use_multi_level_feature
            else _feature_manifest_path(cfg, split)
        )
        feature_rows = read_jsonl(feature_manifest)
        self.feature_map = manifest_to_map(feature_rows, feature_manifest)
        missing = sorted(set(self.keys) - set(self.feature_map.keys()))
        if missing:
            raise RuntimeError(f"feature {split} cache missing first 10: {missing[:10]}")

        self.cacd_feature_map = None
        self.cacd_first_cache_path = None
        if self.use_cacd:
            cacd_manifest = cacd_feature_manifest_path(cfg, split)
            cacd_rows = read_jsonl(cacd_manifest)
            self.cacd_feature_map = manifest_to_map(cacd_rows, cacd_manifest)
            missing_cacd = sorted(set(self.keys) - set(self.cacd_feature_map))
            if missing_cacd:
                raise RuntimeError(f"CACD feature {split} cache missing first 10: {missing_cacd[:10]}")
            self.cacd_first_cache_path = self.cacd_feature_map[self.keys[0]]["cache_path"]

        first_dataset, first_stem = self.keys[0]
        if self.use_multi_level_feature:
            ml_payload = _load_multi_level_feature(
                self.feature_map[(first_dataset, first_stem)],
                first_dataset,
                first_stem,
                cfg,
            )
            feature = ml_payload["feature"]
        else:
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
        if self.use_multi_level_feature:
            ml_feature_payload = _load_multi_level_feature(self.feature_map[key], dataset, stem, self.cfg)
            feature = ml_feature_payload["feature"]
            payload = ml_feature_payload["payload"]
        else:
            feature, payload = _load_feature(self.feature_map[key], dataset, stem)
        cacd_features = None
        if self.use_cacd:
            cacd_features = _load_cacd_features(
                self.cacd_feature_map[key], dataset, stem, self.cfg
            )
        gt = _load_gt(item["gt_path"])
        original_size = tuple(payload.get("original_size", gt.shape[-2:]))
        sample = {
            "feature": feature,
            "gt": gt,
            "dataset": dataset,
            "stem": stem,
            "image_path": item["image_path"],
            "gt_path": item["gt_path"],
            "original_size": original_size,
        }
        if self.use_cacd:
            sample["feature_l10"] = cacd_features["feature_l10"]
            sample["feature_l11"] = cacd_features["feature_l11"]
        if self.use_ndr_branch or self.use_csd_decoder or self.use_csd_v1r or self.use_cacd:
            image_68 = _load_image_68(item["image_path"], int(self.cfg.LOSS_SIZE))
            sample["image_68"] = image_68
            if self.use_cacd:
                sample["sobel_68"] = _normalized_sobel_from_image(image_68)
        if self.use_hr_bfr:
            sample["image_136"] = _load_image_136(
                item["image_path"],
                int(getattr(self.cfg, "HR_BFR_SIZE", 136)),
            )
        if self.use_multi_level_feature:
            sample.update(
                {
                    f"feature_l{int(layer)}": ml_feature_payload[f"feature_l{int(layer)}"]
                    for layer in self.multi_level_layers
                }
            )
        return sample
