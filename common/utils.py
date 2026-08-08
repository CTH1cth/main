import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from common.dabe_clean_offline import (
    DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT,
    DABE_CLEAN_OFFLINE_MODES,
    DABE_CLEAN_OFFLINE_PAYLOAD_VERSION,
)
from common.found_static import (
    FOUND_STATIC_NATIVE_SHAPE,
    FOUND_STATIC_RESIZE_MODE,
    FOUND_STATIC_SOURCE,
    build_found_static_target,
    found_static_enabled,
    found_static_manifest_path,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(config_path):
    # 将单文件 Python config 动态加载成模块对象。
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_to_dict(cfg):
    # 只导出大写配置项，作为运行时 config.yaml 快照。
    out = {}
    for key in dir(cfg):
        if not key.isupper():
            continue
        value = getattr(cfg, key)
        out[key] = make_jsonable(value)
    return out


def make_jsonable(value):
    # 把 Path、numpy 等对象转成 yaml/json 可安全写入的基础类型。
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def set_seed(seed):
    # 固定 Python、numpy、torch 的随机种子，保证 DataLoader shuffle 可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    # 创建输出目录；已有目录时不报错。
    Path(path).mkdir(parents=True, exist_ok=True)


def list_images(data_root, dataset_name):
    # 读取一个数据集 im/ 下的图片列表，并按文件名排序。
    image_dir = Path(data_root) / dataset_name / "im"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return sorted(paths)


def find_gt_path(data_root, dataset_name, stem):
    # 根据 image stem 找对应 GT，兼容常见图片后缀。
    gt_dir = Path(data_root) / dataset_name / "gt"
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        path = gt_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"GT not found for {dataset_name}/{stem} under {gt_dir}")


def build_image_items(data_root, dataset_names, require_gt=False):
    # 将多个数据集合并成统一 item 列表，并用 dataset+stem 去重。
    items = []
    seen = set()
    for dataset_name in dataset_names:
        for image_path in list_images(data_root, dataset_name):
            stem = image_path.stem
            key = (dataset_name, stem)
            if key in seen:
                raise RuntimeError(f"Duplicate image key: {dataset_name}/{stem}")
            seen.add(key)
            item = {
                "dataset": dataset_name,
                "stem": stem,
                "image_path": str(image_path.resolve()),
            }
            if require_gt:
                item["gt_path"] = str(find_gt_path(data_root, dataset_name, stem).resolve())
            items.append(item)
    return items


def read_jsonl(path):
    # 逐行读取 manifest jsonl，空行会被忽略。
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    # 写 manifest jsonl，每个样本一行，便于后续按 dataset+stem 校验。
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_jsonable(row), ensure_ascii=False) + "\n")


def write_json(path, data):
    # 保留通用 JSON 写入函数，当前主要供小型诊断/扩展使用。
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_jsonable(data), f, ensure_ascii=False, indent=2)


def write_yaml(path, data):
    # 写运行配置快照，记录本次实际解析出的参数。
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(make_jsonable(data), f, allow_unicode=True, sort_keys=True)


def manifest_to_map(rows, manifest_path):
    # 将 manifest 转为 dataset+stem 索引，并检查重复 key 和文件存在性。
    out = {}
    for row in rows:
        try:
            key = (row["dataset"], row["stem"])
            cache_path = row["cache_path"]
        except KeyError as exc:
            raise KeyError(f"Bad manifest row in {manifest_path}: missing {exc}") from exc
        if key in out:
            raise RuntimeError(f"Duplicate manifest key in {manifest_path}: {key}")
        if not Path(cache_path).exists():
            raise FileNotFoundError(f"Cache file listed in manifest does not exist: {cache_path}")
        out[key] = row
    return out


def check_exact_keys(name, actual, expected):
    # 要求两组 dataset+stem 完全一致，防止 cache 与图片列表错位。
    actual_keys = set(actual)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        msg = [f"{name} key mismatch"]
        if missing:
            msg.append(f"missing first 10: {missing[:10]}")
        if extra:
            msg.append(f"extra first 10: {extra[:10]}")
        raise RuntimeError("; ".join(msg))


def metric_value(metrics, key):
    # 将表格字段名映射回内部 metric key，用于 best checkpoint 判断。
    mapping = {
        "S_m": "SMeasure",
        "SMeasure": "SMeasure",
        "F_beta^w": "WFM",
        "WFM": "WFM",
        "F_beta^m": "F_MEAN",
        "F_MEAN": "F_MEAN",
        "E_phi^m": "E_MEAN",
        "E_MEAN": "E_MEAN",
        "M": "MAE",
        "MAE": "MAE",
    }
    metric_key = mapping.get(key, key)
    if metric_key not in metrics:
        raise KeyError(f"Metric {key} not found in {sorted(metrics)}")
    return float(metrics[metric_key])


def format_metric_table(metrics):
    # 按固定顺序格式化 COD 指标，保持 train/eval 日志可读。
    headers = ["S_m ↑", "F_beta^w↑", "F_beta^m↑", "E_phi^m↑", "M ↓"]
    values = [
        float(metrics["SMeasure"]),
        float(metrics["WFM"]),
        float(metrics["F_MEAN"]),
        float(metrics["E_MEAN"]),
        float(metrics["MAE"]),
    ]
    widths = [7, 10, 10, 9, 7]
    sep = "+" + "+".join("-" * w for w in widths) + "+"
    header_line = "|" + "|".join(h.center(w) for h, w in zip(headers, widths)) + "|"
    value_line = "|" + "|".join(f"{v:.4f}".center(w) for v, w in zip(values, widths)) + "|"
    return "\n".join([sep, header_line, sep, value_line, sep])


def current_lr(optimizer):
    # 只读取第一个 param group 的学习率，本实验只有一个参数组。
    return float(optimizer.param_groups[0]["lr"])


def torch_load(path, map_location="cpu"):
    # 兼容新旧 PyTorch 的 weights_only 参数差异。
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def feature_manifest_path(cfg, split):
    # feature manifest 按 backbone 和 split 分开，避免不同 DINO 版本混用。
    return Path(cfg.CACHE_ROOT) / "features_cache" / cfg.BACKBONE_KEY / f"manifest_{split}.jsonl"


def cacd_feature_manifest_path(cfg, split):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"CACD feature split must be train/val/test, got {split!r}.")
    root = getattr(cfg, "CACD_EXTRA_FEATURE_CACHE_ROOT", None)
    if root is None:
        root = getattr(cfg, "CACD_FEATURE_CACHE_ROOT")
    return Path(root) / f"manifest_{split}.jsonl"


def check_cacd_feature_cache(cfg, split, max_samples=None):
    """Strictly validate the extra F10/F11 cache used by CACD-v1-Base."""
    if not bool(getattr(cfg, "USE_CACD", False)):
        return True, "disabled"
    manifest_path = cacd_feature_manifest_path(cfg, split)
    if not manifest_path.exists():
        raise RuntimeError(f"CACD feature manifest missing: {manifest_path}")
    rows = read_jsonl(manifest_path)
    row_map = manifest_to_map(rows, manifest_path)
    dataset_names = {
        "train": cfg.TRAIN_DATASETS,
        "val": cfg.VAL_DATASETS,
        "test": cfg.TEST_DATASETS,
    }[split]
    items = build_image_items(cfg.DATA_ROOT, dataset_names, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in items]
    expected_item_map = {(item["dataset"], item["stem"]): item for item in items}
    expected_set = set(expected_keys)
    actual_set = set(row_map)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set) if max_samples is None or int(max_samples) < 0 else []
    if missing or extra:
        raise RuntimeError(
            f"CACD {split} cache key mismatch: missing first 10={missing[:10]}, "
            f"extra first 10={extra[:10]}"
        )

    expected_shape = [
        int(getattr(cfg, "CACD_IN_CHANNELS", 384)),
        int(getattr(cfg, "CACD_FEATURE_SIZE", 37)),
        int(getattr(cfg, "CACD_FEATURE_SIZE", 37)),
    ]
    for dataset, stem in expected_keys:
        row = row_map[(dataset, stem)]
        payload = torch_load(row["cache_path"], map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"CACD feature payload must be dict: {row['cache_path']}")
        metadata = (
            ("dataset", dataset),
            ("stem", stem),
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
            ("feature_shape", expected_shape),
        )
        for field, expected in metadata:
            if payload.get(field) != expected:
                raise RuntimeError(
                    f"CACD metadata mismatch for {dataset}/{stem}: "
                    f"{field}={payload.get(field)!r} != {expected!r} | {row['cache_path']}"
                )
        if payload.get("stored_layer_indices") != [9, 10]:
            raise RuntimeError(
                f"CACD stored layers mismatch for {dataset}/{stem}: "
                f"{payload.get('stored_layer_indices')!r} != [9, 10]"
            )
        if Path(payload.get("image_path", "")).resolve() != Path(
            expected_item_map[(dataset, stem)]["image_path"]
        ).resolve():
            raise RuntimeError(
                f"CACD image_path mismatch for {dataset}/{stem}: {payload.get('image_path')!r}"
            )
        key_paths = payload.get("key_projection_paths")
        expected_paths = {
            "9": "encoder.layer[9].attention.attention.key",
            "10": "encoder.layer[10].attention.attention.key",
            "11": "encoder.layer[11].attention.attention.key",
        }
        if key_paths != expected_paths:
            raise RuntimeError(
                f"CACD key projection paths mismatch for {dataset}/{stem}: {key_paths!r}"
            )
        if any(field in payload for field in ("feature_l12", "tensor", "feature")):
            raise RuntimeError(
                f"CACD extra cache must not duplicate F12 for {dataset}/{stem}: {row['cache_path']}"
            )
        for field in ("feature_l10", "feature_l11"):
            tensor = payload.get(field)
            if not torch.is_tensor(tensor):
                raise TypeError(f"CACD payload missing tensor {field}: {row['cache_path']}")
            if tensor.dtype != torch.float32 or list(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"CACD {field} mismatch for {dataset}/{stem}: "
                    f"shape={list(tensor.shape)}, dtype={tensor.dtype}, expected={expected_shape}/float32"
                )
            if not bool(torch.isfinite(tensor).all().item()):
                raise RuntimeError(f"CACD {field} contains NaN/Inf: {row['cache_path']}")
    return True, f"complete: {manifest_path} | rows_checked={len(expected_keys)}"


def cssd_hr_feature_manifest_path(cfg):
    return Path(cfg.CSSD_HR_CACHE_ROOT) / "manifest_train.jsonl"


def check_cssd_hr_feature_cache(cfg, max_samples=None):
    if not bool(getattr(cfg, "USE_CSSD", False)):
        return True, "disabled"
    manifest_path = cssd_hr_feature_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(f"CSSD HR feature manifest missing: {manifest_path}")
    rows = read_jsonl(manifest_path)
    row_map = manifest_to_map(rows, manifest_path)
    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_item_map = {(item["dataset"], item["stem"]): item for item in expected_items}
    expected_set = set(expected_keys)
    actual_set = set(row_map)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    full_check = max_samples is None or int(max_samples) < 0
    if missing or (full_check and extra):
        raise RuntimeError(
            "CSSD HR feature manifest coverage mismatch | "
            f"missing first 10={missing[:10]} | extra first 10={extra[:10]}"
        )
    if full_check:
        counts = {}
        for dataset, _stem in expected_keys:
            counts[dataset] = counts.get(dataset, 0) + 1
        expected_counts = {"TR-CAMO": 1000, "TR-COD10K": 3040}
        if len(expected_keys) != 4040 or counts != expected_counts:
            raise RuntimeError(
                f"CSSD expected train split 4040 with {expected_counts}, got total={len(expected_keys)}, counts={counts}"
            )

    expected_shape = [
        int(getattr(cfg, "CSSD_HR_FEATURE_CHANNELS", 384)),
        int(getattr(cfg, "CSSD_HR_FEATURE_SIZE", 48)),
        int(getattr(cfg, "CSSD_HR_FEATURE_SIZE", 48)),
    ]
    expected_key = str(getattr(cfg, "CSSD_HR_CACHE_KEY", "feature"))
    for key in expected_keys:
        row = row_map[key]
        if row.get("shape") != expected_shape:
            raise RuntimeError(f"CSSD HR manifest shape mismatch for {key}: {row.get('shape')} != {expected_shape}")
        payload = torch_load(row["cache_path"], map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"CSSD HR cache must be dict for {key}: {row['cache_path']}")
        checks = {
            "dataset": key[0],
            "stem": key[1],
            "backbone_key": cfg.BACKBONE_KEY,
            "model_key": cfg.DINO["model_name"],
            "feature_layer": "final_attention_key",
            "input_size": int(getattr(cfg, "CSSD_HR_INPUT_SIZE", 384)),
            "patch_size": 8,
            "feature_shape": expected_shape,
            "version": "cssd_hr_feature_v1",
        }
        for field, expected in checks.items():
            if payload.get(field) != expected:
                raise RuntimeError(
                    f"CSSD HR cache metadata mismatch for {key}: {field}={payload.get(field)!r} != {expected!r}"
                )
        if payload.get("dtype") != "float32":
            raise RuntimeError(f"CSSD HR cache dtype metadata mismatch for {key}: {payload.get('dtype')!r}")
        if not isinstance(payload.get("key_projection_path"), str) or not payload["key_projection_path"]:
            raise RuntimeError(f"CSSD HR cache missing truthful key_projection_path for {key}")
        if Path(payload.get("image_path", "")).resolve() != Path(expected_item_map[key]["image_path"]).resolve():
            raise RuntimeError(f"CSSD HR cache image_path mismatch for {key}: {payload.get('image_path')!r}")
        feature = payload.get(expected_key)
        if not torch.is_tensor(feature):
            raise RuntimeError(f"CSSD HR cache missing tensor field {expected_key!r} for {key}")
        if list(feature.shape) != expected_shape or feature.dtype != torch.float32:
            raise RuntimeError(
                f"CSSD HR feature tensor mismatch for {key}: shape={list(feature.shape)}, dtype={feature.dtype}"
            )
        if not bool(torch.isfinite(feature).all().item()):
            raise RuntimeError(f"CSSD HR feature contains NaN/Inf for {key}: {row['cache_path']}")
    return True, f"complete: {manifest_path} | num_samples={len(expected_keys)}"


def pseudo_manifest_path(cfg):
    # fixed pseudo 只用于训练集，因此只有 train manifest。
    return Path(cfg.CACHE_ROOT) / "pseudo_label_cache" / cfg.BACKBONE_KEY / "manifest_train.jsonl"


def ml_feature_cache_dir(cfg):
    if str(getattr(cfg, "DINO_FEATURE_MODE", "")).strip().lower() == "last4":
        return Path(
            getattr(
                cfg,
                "LAST4_FEATURE_CACHE_ROOT",
                "../datasets/cache/dinov1_s8_last4_296",
            )
        )
    root = getattr(cfg, "MULTI_LEVEL_FEATURE_ROOT", "../datasets/cache/features_cache_ml")
    return Path(root) / cfg.BACKBONE_KEY


def ml_feature_cache_manifest_path(cfg, split):
    return ml_feature_cache_dir(cfg) / f"manifest_{split}.jsonl"


def ml_feature_labels(cfg):
    if str(getattr(cfg, "DINO_FEATURE_MODE", "")).strip().lower() == "last4":
        return [str(key) for key in getattr(cfg, "DINO_FEATURE_KEYS", ("f9", "f10", "f11", "f12"))]
    return [
        f"l{int(layer)}"
        for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12])
    ]


def hflip_feature_cache_dir(cfg):
    root = getattr(cfg, "HFLIP_FEATURE_CACHE_ROOT", "../datasets/cache/features_cache_hflip")
    return Path(root) / cfg.BACKBONE_KEY


def hflip_feature_cache_manifest_path(cfg):
    return hflip_feature_cache_dir(cfg) / "manifest_train.jsonl"


def qra_cache_dir(cfg):
    return Path(cfg.QRA_CACHE_ROOT) / cfg.BACKBONE_KEY


def qra_manifest_path(cfg):
    return qra_cache_dir(cfg) / "manifest_train.jsonl"


def ccr_cache_dir(cfg):
    return Path(cfg.CCR_CACHE_ROOT) / cfg.BACKBONE_KEY


def ccr_manifest_path(cfg):
    return ccr_cache_dir(cfg) / "manifest_train.jsonl"


def drepp_cache_dir(cfg):
    return Path(cfg.DREPP_CACHE_ROOT) / cfg.BACKBONE_KEY


def drepp_manifest_path(cfg):
    return drepp_cache_dir(cfg) / "manifest_train.jsonl"


def nper_pseudo_bank_dir(cfg):
    return Path(cfg.PSEUDO_BANK_ROOT) / cfg.BACKBONE_KEY


def nper_pseudo_bank_manifest_path(cfg):
    return nper_pseudo_bank_dir(cfg) / "manifest_train.jsonl"


def despl_pseudo_bank_dir(cfg):
    root = getattr(
        cfg,
        "NPER_PSEUDO_BANK_ROOT",
        getattr(cfg, "PSEUDO_BANK_ROOT", "../datasets/cache/nper_pseudo_bank"),
    )
    return Path(root) / cfg.BACKBONE_KEY


def despl_pseudo_bank_manifest_path(cfg):
    return despl_pseudo_bank_dir(cfg) / "manifest_train.jsonl"


def despl_light_cache_dir(cfg):
    return Path(cfg.DESPL_LIGHT_CACHE_ROOT) / cfg.BACKBONE_KEY


def despl_light_cache_manifest_path(cfg):
    return despl_light_cache_dir(cfg) / "manifest_train.jsonl"


def dabe_pseudo_manifest_path(cfg):
    root = getattr(cfg, "DABE_PSEUDO_ROOT", "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8")
    return Path(root) / "manifest_train.jsonl"


def dabe_pu_manifest_path(cfg):
    root = getattr(cfg, "DABE_PU_ROOT", "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8")
    return Path(root) / "manifest_train.jsonl"


def dabe_clean_manifest_path(cfg):
    root = getattr(
        cfg,
        "DABE_CLEAN_ROOT",
        "../datasets/cache/dabe_clean_v1_pseudo_cache/dinov1-s8",
    )
    return Path(root) / "manifest_train.jsonl"


def dabe_clean_legacy_region_manifest_path(cfg):
    root = getattr(
        cfg,
        "DABE_CLEAN_LEGACY_REGION_ROOT",
        "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8",
    )
    return Path(root) / "manifest_train.jsonl"


def tce_cover_manifest_path(cfg):
    root = getattr(cfg, "TCE_COVER_CACHE_ROOT", "../datasets/cache/tce_cover_cache/dinov1-s8/long35_epoch030")
    return Path(root) / "manifest_train.jsonl"


def lceg_cover_manifest_path(cfg):
    root = getattr(cfg, "LCEG_COVER_CACHE_ROOT", "../datasets/cache/lceg_cover_cache/dinov1-s8/long35_epoch025")
    return Path(root) / "manifest_train.jsonl"


def despl_paper_cache_dir(cfg, sign_mode=None):
    root = Path(cfg.DESPL_PAPER_CACHE_ROOT)
    backbone = getattr(cfg, "DESPL_PAPER_BACKBONE_KEY", cfg.BACKBONE_KEY)
    mode = sign_mode
    if mode is None:
        mode = getattr(cfg, "DESPL_PAPER_SIGN_MODE", getattr(cfg, "DESPL_SIGN_MODE", "paper"))
    suffix = backbone if str(mode) == "paper" else f"{backbone}__{mode}"
    return root / suffix


def despl_paper_manifest_path(cfg, sign_mode=None):
    return despl_paper_cache_dir(cfg, sign_mode=sign_mode) / "manifest_train.jsonl"


def split_dataset_names(cfg, split):
    # 根据 split 选择配置里的数据集列表。
    if split == "train":
        return cfg.TRAIN_DATASETS
    if split == "val":
        return cfg.VAL_DATASETS
    if split == "test":
        return cfg.TEST_DATASETS
    raise ValueError(f"Unknown split: {split}")


def _cache_shape_ok(kind, shape):
    # manifest 中的 shape 只做结构检查，详细 tensor 内容由 Dataset 再校验。
    if not isinstance(shape, list):
        return False
    if kind == "feature":
        return len(shape) == 3 and all(isinstance(x, int) and x > 0 for x in shape)
    if kind == "pseudo":
        return len(shape) == 3 and shape[0] == 1 and all(isinstance(x, int) and x > 0 for x in shape)
    raise ValueError(f"Unknown cache kind: {kind}")


def _ml_feature_error(split, reason):
    return (
        f"Multi-level feature cache {split} missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_features_ml.py --config <your_multi_level_config.py> "
        f"--split {split}"
    )


def _hflip_feature_error(reason):
    return (
        f"HFlip feature cache train missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_features_hflip.py --config <your_config.py> "
        "--split train"
    )


def _feature_shape3_ok(shape):
    return (
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(value, int) and value > 0 for value in shape)
    )


def _sample_ordered_keys(keys, max_samples):
    keys = list(keys)
    if max_samples is None or int(max_samples) < 0 or int(max_samples) >= len(keys):
        return keys
    max_samples = int(max_samples)
    if max_samples <= 0:
        return []
    if max_samples == 1:
        return [keys[0]]
    last = len(keys) - 1
    indices = sorted({round(i * last / (max_samples - 1)) for i in range(max_samples)})
    return [keys[index] for index in indices]


def check_ml_feature_cache(cfg, split, max_samples=None):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    manifest_path = ml_feature_cache_manifest_path(cfg, split)
    if not manifest_path.exists():
        raise RuntimeError(_ml_feature_error(split, f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, split_dataset_names(cfg, split), require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_ml_feature_error(split, f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_ml_feature_error(split, f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_ml_feature_error(split, f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_ml_feature_error(split, f"missing manifest rows first 10: {missing[:10]}"))

    layers = [int(layer) for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12])]
    labels = ml_feature_labels(cfg)
    if len(labels) != len(layers):
        raise RuntimeError(
            _ml_feature_error(
                split,
                f"feature key/layer count mismatch: {labels} vs {layers}",
            )
        )
    feature_type = str(getattr(cfg, "MULTI_LEVEL_FEATURE_TYPE", "key"))
    expected_dtype = getattr(cfg, "MULTI_LEVEL_FEATURE_DTYPE", None)
    if expected_dtype is not None:
        expected_dtype = str(expected_dtype).lower()
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _ml_feature_error(
                    split,
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}",
                )
            )
        if str(row.get("feature_type", feature_type)) != feature_type:
            raise RuntimeError(
                _ml_feature_error(split, f"feature_type mismatch for {key}: {row.get('feature_type')} != {feature_type}")
            )
        if expected_dtype is not None and str(row.get("dtype", "float32")).lower() != expected_dtype:
            raise RuntimeError(
                _ml_feature_error(split, f"dtype mismatch for {key}: {row.get('dtype', 'float32')} != {expected_dtype}")
            )
        row_layers = [int(layer) for layer in row.get("layers", layers)]
        if row_layers != layers:
            raise RuntimeError(_ml_feature_error(split, f"layers mismatch for {key}: {row_layers} != {layers}"))
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_ml_feature_error(split, f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_ml_feature_error(split, f"missing cache file: {cache_path}"))
        row_shape = row.get("shape")
        if not isinstance(row_shape, dict):
            raise RuntimeError(_ml_feature_error(split, f"missing or invalid shape dict for {key}"))
        row_spatial = None
        row_channels = None
        for label in labels:
            shape = row_shape.get(label)
            if not _feature_shape3_ok(shape):
                raise RuntimeError(
                    _ml_feature_error(split, f"invalid manifest shape for {key} {label}: {shape}")
                )
            if row_spatial is None:
                row_spatial = tuple(shape[-2:])
                row_channels = int(shape[0])
            elif tuple(shape[-2:]) != row_spatial or int(shape[0]) != row_channels:
                raise RuntimeError(_ml_feature_error(split, f"manifest multi-level shapes are inconsistent for {key}"))

    preflight_mode = str(getattr(cfg, "ML_FEATURE_PREFLIGHT_MODE", "sample")).lower()
    if preflight_mode not in {"sample", "full"}:
        raise RuntimeError(_ml_feature_error(split, f"unknown ML_FEATURE_PREFLIGHT_MODE: {preflight_mode}"))
    if preflight_mode == "full":
        payload_keys = expected_keys
    else:
        payload_keys = _sample_ordered_keys(
            expected_keys,
            int(getattr(cfg, "ML_FEATURE_PREFLIGHT_SAMPLES", 32)),
        )

    for key in payload_keys:
        row = row_map[key]
        row_shape = row.get("shape")
        cache_path = row.get("cache_path")
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(_ml_feature_error(split, f"payload must be dict for {key}: {cache_path}"))
        if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
            raise RuntimeError(_ml_feature_error(split, f"payload key mismatch for {key}: {cache_path}"))
        if payload.get("backbone_key") != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _ml_feature_error(
                    split,
                    f"payload backbone mismatch for {key}: {payload.get('backbone_key')} != {cfg.BACKBONE_KEY}",
                )
            )
        payload_layers = [int(layer) for layer in payload.get("layers", row.get("layers", layers))]
        if payload_layers != layers:
            raise RuntimeError(_ml_feature_error(split, f"payload layers mismatch for {key}: {payload_layers} != {layers}"))
        if str(payload.get("feature_type", row.get("feature_type", feature_type))) != feature_type:
            raise RuntimeError(
                _ml_feature_error(
                    split,
                    f"payload feature_type mismatch for {key}: "
                    f"{payload.get('feature_type', row.get('feature_type'))} != {feature_type}",
                )
            )
        if expected_dtype is not None and str(payload.get("dtype", row.get("dtype", "float32"))).lower() != expected_dtype:
            raise RuntimeError(
                _ml_feature_error(
                    split,
                    f"payload dtype mismatch for {key}: "
                    f"{payload.get('dtype', row.get('dtype', 'float32'))} != {expected_dtype}",
                )
            )
        features = payload.get("features")
        if not isinstance(features, dict):
            raise RuntimeError(_ml_feature_error(split, f"payload missing features dict for {key}: {cache_path}"))
        spatial = None
        channels = None
        for label in labels:
            tensor = features.get(label)
            if not torch.is_tensor(tensor):
                raise RuntimeError(_ml_feature_error(split, f"payload missing tensor features/{label} for {key}: {cache_path}"))
            if tensor.ndim != 3:
                raise RuntimeError(_ml_feature_error(split, f"features/{label} must be [C,H,W], got {list(tensor.shape)}"))
            shape = list(tensor.shape)
            if row_shape.get(label) != shape:
                raise RuntimeError(
                    _ml_feature_error(
                        split,
                        f"shape mismatch for {key} {label}: manifest {row_shape.get(label)} != payload {shape}",
                    )
                )
            if expected_dtype == "float16" and tensor.dtype != torch.float16:
                raise RuntimeError(
                    _ml_feature_error(split, f"features/{label} dtype must be float16 for {key}: {tensor.dtype}")
                )
            if expected_dtype == "float32" and tensor.dtype != torch.float32:
                raise RuntimeError(
                    _ml_feature_error(split, f"features/{label} dtype must be float32 for {key}: {tensor.dtype}")
                )
            if spatial is None:
                spatial = tuple(shape[-2:])
                channels = int(shape[0])
            elif tuple(shape[-2:]) != spatial or int(shape[0]) != channels:
                raise RuntimeError(_ml_feature_error(split, f"multi-level feature shapes are inconsistent for {key}"))
        tensor = payload.get("tensor")
        if not torch.is_tensor(tensor):
            raise RuntimeError(_ml_feature_error(split, f"payload missing tensor field for {key}: {cache_path}"))
        if list(tensor.shape) != list(features[labels[-1]].shape):
            raise RuntimeError(_ml_feature_error(split, f"payload tensor must match {labels[-1]} for {key}: {cache_path}"))
        if expected_dtype == "float16" and tensor.dtype != torch.float16:
            raise RuntimeError(_ml_feature_error(split, f"payload tensor dtype must be float16 for {key}: {tensor.dtype}"))
        if expected_dtype == "float32" and tensor.dtype != torch.float32:
            raise RuntimeError(_ml_feature_error(split, f"payload tensor dtype must be float32 for {key}: {tensor.dtype}"))

    return True, f"complete: {manifest_path} | payload_check={preflight_mode}:{len(payload_keys)}/{len(expected_keys)}"


def check_hflip_feature_cache(cfg, max_samples=None):
    manifest_path = hflip_feature_cache_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_hflip_feature_error(f"missing manifest: {manifest_path}"))
    normal_manifest_path = feature_manifest_path(cfg, "train")
    if not normal_manifest_path.exists():
        raise RuntimeError(
            _hflip_feature_error(
                f"normal feature manifest is missing: {normal_manifest_path}"
            )
        )

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_hflip_feature_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_hflip_feature_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_hflip_feature_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_hflip_feature_error(f"missing manifest rows first 10: {missing[:10]}"))

    normal_rows = read_jsonl(normal_manifest_path)
    normal_row_map = {}
    for row in normal_rows:
        key = (row.get("dataset"), row.get("stem"))
        if None in key:
            raise RuntimeError(
                _hflip_feature_error(
                    f"bad normal feature manifest row in {normal_manifest_path}"
                )
            )
        if key in normal_row_map:
            raise RuntimeError(
                _hflip_feature_error(
                    f"duplicate normal feature manifest key: {key}"
                )
            )
        normal_row_map[key] = row
    missing_normal = sorted(expected_key_set - set(normal_row_map))
    if missing_normal:
        raise RuntimeError(
            _hflip_feature_error(
                "normal feature manifest is missing matching rows first 10: "
                f"{missing_normal[:10]}"
            )
        )

    full_payload_audit = bool(getattr(cfg, "USE_SOURCE_ARBITER", False))
    payload_keys = (
        expected_keys
        if full_payload_audit
        else _sample_ordered_keys(
            expected_keys,
            int(getattr(cfg, "HFLIP_FEATURE_PREFLIGHT_SAMPLES", 16)),
        )
    )
    if max_samples is not None and int(max_samples) >= 0:
        payload_keys = expected_keys
    for key in expected_keys:
        row = row_map[key]
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_hflip_feature_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_hflip_feature_error(f"missing cache file: {cache_path}"))
        shape = row.get("shape")
        if not _feature_shape3_ok(shape):
            raise RuntimeError(_hflip_feature_error(f"invalid manifest shape for {key}: {shape}"))
        normal_shape = normal_row_map[key].get("shape")
        if list(shape) != list(normal_shape or []):
            raise RuntimeError(
                _hflip_feature_error(
                    f"normal/hflip manifest shape mismatch for {key}: "
                    f"{normal_shape} != {shape}"
                )
            )
    for key in payload_keys:
        row = row_map[key]
        cache_path = row["cache_path"]
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(_hflip_feature_error(f"payload must be dict for {key}: {cache_path}"))
        if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
            raise RuntimeError(_hflip_feature_error(f"payload key mismatch for {key}: {cache_path}"))
        if payload.get("view") not in {None, "hflip"}:
            raise RuntimeError(_hflip_feature_error(f"payload view mismatch for {key}: {payload.get('view')}"))
        tensor = payload.get("tensor")
        if not torch.is_tensor(tensor):
            raise RuntimeError(_hflip_feature_error(f"payload missing tensor for {key}: {cache_path}"))
        if tensor.ndim != 3:
            raise RuntimeError(_hflip_feature_error(f"payload tensor must be [C,H,W], got {list(tensor.shape)}"))
        if list(tensor.shape) != list(row.get("shape")):
            raise RuntimeError(
                _hflip_feature_error(
                    f"shape mismatch for {key}: manifest {row.get('shape')} != payload {list(tensor.shape)}"
                )
            )
        if not tensor.is_floating_point():
            raise RuntimeError(
                _hflip_feature_error(
                    f"payload tensor must be floating point for {key}: {tensor.dtype}"
                )
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(
                _hflip_feature_error(
                    f"payload tensor contains NaN/Inf for {key}: {cache_path}"
                )
            )

    audit_mode = "full" if full_payload_audit else "sampled"
    return True, (
        f"complete: {manifest_path} | payload_check={audit_mode}:"
        f"{len(payload_keys)}/{len(expected_keys)} | "
        f"normal_manifest={normal_manifest_path}"
    )


def _qra_required_shapes(cfg):
    size = int(cfg.LOSS_SIZE)
    return {
        "p_fixed": [1, size, size],
        "p_despl": [1, size, size],
        "p_fused": [1, size, size],
        "anchor_fg": [1, size, size],
        "anchor_bg": [1, size, size],
        "pixel_weight": [1, size, size],
    }


def _qra_error(reason):
    return (
        f"QRA cache missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_qra_pseudo.py --config configs/dinov1_s8_qra.py --overwrite"
    )


def check_qra_cache(cfg, max_samples=None):
    manifest_path = qra_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_qra_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_qra_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_qra_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_qra_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_qra_error(f"missing manifest rows first 10: {missing[:10]}"))

    required_shapes = _qra_required_shapes(cfg)
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _qra_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_qra_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_qra_error(f"missing cache file: {cache_path}"))
        shape = row.get("shape")
        if not isinstance(shape, dict):
            raise RuntimeError(_qra_error(f"missing or invalid shape dict for {key}"))
        for name, expected_shape in required_shapes.items():
            if shape.get(name) != expected_shape:
                raise RuntimeError(
                    _qra_error(
                        f"invalid shape for {key} {name}: {shape.get(name)} != {expected_shape}"
                    )
                )

    return True, f"complete: {manifest_path}"


def _ccr_required_shapes(cfg):
    size = int(getattr(cfg, "CCR_LOSS_SIZE", cfg.LOSS_SIZE))
    return {
        "p_fixed": [1, size, size],
        "p_despl": [1, size, size],
        "p_corr": [1, size, size],
        "agree_fg": [1, size, size],
        "agree_bg": [1, size, size],
        "raw_expand": [1, size, size],
        "raw_shrink": [1, size, size],
        "trusted_expand": [1, size, size],
        "trusted_shrink": [1, size, size],
        "anchor_fg": [1, size, size],
        "anchor_bg": [1, size, size],
    }


def _ccr_error(reason):
    return (
        f"CCR cache missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_ccr_pseudo.py --config configs/dinov1_s8_ccr.py --overwrite"
    )


def check_ccr_cache(cfg, max_samples=None):
    manifest_path = ccr_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_ccr_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_ccr_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_ccr_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_ccr_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_ccr_error(f"missing manifest rows first 10: {missing[:10]}"))

    required_shapes = _ccr_required_shapes(cfg)
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _ccr_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_ccr_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_ccr_error(f"missing cache file: {cache_path}"))
        shape = row.get("shape")
        if not isinstance(shape, dict):
            raise RuntimeError(_ccr_error(f"missing or invalid shape dict for {key}"))
        for name, expected_shape in required_shapes.items():
            if shape.get(name) != expected_shape:
                raise RuntimeError(
                    _ccr_error(
                        f"invalid shape for {key} {name}: {shape.get(name)} != {expected_shape}"
                    )
                )

    return True, f"complete: {manifest_path}"


def _drepp_required_shapes(cfg):
    size = int(cfg.LOSS_SIZE)
    return {
        "p_despl": [1, size, size],
        "p_fixed": [1, size, size],
        "core_fg": [1, size, size],
        "core_bg": [1, size, size],
        "uncertain": [1, size, size],
        "fixed_local_recall": [1, size, size],
        "boundary_band": [1, size, size],
        "memory_init": [1, size, size],
        "feature_sim": [1, size, size],
    }


def _drepp_error(reason):
    return (
        f"DRE++ cache missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_drepp_pseudo.py --config configs/drepp_v1.py --overwrite"
    )


def check_drepp_cache(cfg, max_samples=None):
    manifest_path = drepp_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_drepp_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_drepp_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_drepp_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_drepp_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_drepp_error(f"missing manifest rows first 10: {missing[:10]}"))

    required_shapes = _drepp_required_shapes(cfg)
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _drepp_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        if row.get("global_blend", False):
            raise RuntimeError(_drepp_error(f"global_blend must be false for {key}"))
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_drepp_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_drepp_error(f"missing cache file: {cache_path}"))
        shape = row.get("shape")
        if not isinstance(shape, dict):
            raise RuntimeError(_drepp_error(f"missing or invalid shape dict for {key}"))
        for name, expected_shape in required_shapes.items():
            if shape.get(name) != expected_shape:
                raise RuntimeError(
                    _drepp_error(
                        f"invalid shape for {key} {name}: {shape.get(name)} != {expected_shape}"
                    )
                )

    return True, f"complete: {manifest_path}"


def _nper_required_shapes(cfg):
    size = int(cfg.LOSS_SIZE)
    return {
        "p_fixed": [1, size, size],
        "p_despl": [1, size, size],
        "p_gcm": [1, size, size],
        "p_init": [1, size, size],
        "anchor_fg": [1, size, size],
        "anchor_bg": [1, size, size],
        "pixel_weight": [1, size, size],
    }


def _nper_error(reason):
    return (
        f"NPER pseudo bank missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_nper_pseudo_bank.py --config configs/nper_ucod_v1.py --overwrite"
    )


def check_nper_pseudo_bank(cfg, max_samples=None):
    manifest_path = nper_pseudo_bank_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_nper_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_nper_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_nper_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_nper_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_nper_error(f"missing manifest rows first 10: {missing[:10]}"))

    required_shapes = _nper_required_shapes(cfg)
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _nper_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_nper_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_nper_error(f"missing cache file: {cache_path}"))
        shape = row.get("shape")
        if not isinstance(shape, dict):
            raise RuntimeError(_nper_error(f"missing or invalid shape dict for {key}"))
        for name, expected_shape in required_shapes.items():
            if shape.get(name) != expected_shape:
                raise RuntimeError(
                    _nper_error(
                        f"invalid shape for {key} {name}: {shape.get(name)} != {expected_shape}"
                    )
                )

    return True, f"complete: {manifest_path}"


def _despl_error(reason):
    return (
        f"DESPL pseudo bank missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_nper_pseudo_bank.py --config configs/nper_ucod_v1.py --overwrite"
    )


def _single_channel_shape_ok(shape):
    return (
        isinstance(shape, list)
        and len(shape) == 3
        and shape[0] == 1
        and all(isinstance(x, int) and x > 0 for x in shape)
    )


def check_despl_pseudo_bank(cfg, max_samples=None):
    manifest_path = despl_pseudo_bank_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_despl_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_despl_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_despl_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_despl_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_despl_error(f"missing manifest rows first 10: {missing[:10]}"))

    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _despl_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_despl_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_despl_error(f"missing cache file: {cache_path}"))
        shape = row.get("shape")
        if not isinstance(shape, dict):
            raise RuntimeError(_despl_error(f"missing or invalid shape dict for {key}"))
        if not _single_channel_shape_ok(shape.get("p_despl")):
            raise RuntimeError(_despl_error(f"invalid p_despl shape for {key}: {shape.get('p_despl')}"))
        if "p_fixed" in shape and not _single_channel_shape_ok(shape.get("p_fixed")):
            raise RuntimeError(_despl_error(f"invalid p_fixed shape for {key}: {shape.get('p_fixed')}"))

    return True, f"complete: {manifest_path}"


def _despl_light_error(reason):
    return (
        f"DESPL light cache missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_despl_blend_pseudo.py --config configs/dinov1_s8_despl_teacher_cache.py --overwrite"
    )


def check_despl_light_cache(cfg, max_samples=None):
    manifest_path = despl_light_cache_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_despl_light_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_despl_light_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_despl_light_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_despl_light_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_despl_light_error(f"missing manifest rows first 10: {missing[:10]}"))

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    require_p_despl_68 = (
        bool(getattr(cfg, "USE_DESPL_PSEUDO", False))
        and str(getattr(cfg, "P_INIT_MODE", "")) == "despl_only"
    )
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _despl_light_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_despl_light_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_despl_light_error(f"missing cache file: {cache_path}"))
        shape = row.get("shape")
        if shape != expected_shape:
            raise RuntimeError(_despl_light_error(f"invalid shape for {key}: {shape} != {expected_shape}"))
        if require_p_despl_68:
            payload = torch_load(cache_path, map_location="cpu")
            tensor = payload.get("p_despl_68") if isinstance(payload, dict) else None
            if not torch.is_tensor(tensor):
                raise RuntimeError(_despl_light_error(f"missing p_despl_68 for {key}: {cache_path}"))
            if list(tensor.shape) != expected_shape:
                raise RuntimeError(
                    _despl_light_error(
                        f"invalid p_despl_68 shape for {key}: {list(tensor.shape)} != {expected_shape}"
                    )
                )

    return True, f"complete: {manifest_path}"


def _dabe_pseudo_error(reason):
    return (
        f"DABE pseudo cache missing or incomplete: {reason}\n"
        "Please generate it first, for example:\n"
        "python common/cache_dabe_pseudo.py --config <config.py> "
        "--out_root <DABE_PSEUDO_ROOT> --split train "
        "--augs identity,hflip,vflip,rot180 --dabe_version <DABE_VERSION> --overwrite"
    )


def _dabe_first_tensor(payload, keys):
    for key in keys:
        value = payload.get(key)
        if torch.is_tensor(value):
            return key, value.float()
    return None, None


def _dabe_pseudo_tensor_keys(cfg):
    version = str(getattr(cfg, "DABE_VERSION", "v2")).lower()
    keys = ["p_dabe_68"]
    if version == "gc":
        keys.append("p_dabe_gc_68")
    keys.append("p_dabe_37")
    if version == "gc":
        keys.append("p_dabe_gc_37")
    return keys


def _check_dabe_pseudo_tensor(payload, cfg, key, cache_path):
    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    raw_shape = [1, 37, 37]
    found_bad_shapes = []
    for tensor_key in _dabe_pseudo_tensor_keys(cfg):
        tensor = payload.get(tensor_key)
        if not torch.is_tensor(tensor):
            continue
        tensor = tensor.float()
        shape = list(tensor.shape)
        if shape not in (expected_shape, raw_shape):
            found_bad_shapes.append((tensor_key, shape))
            continue
        min_value = float(tensor.min().item())
        max_value = float(tensor.max().item())
        if min_value < -1e-6 or max_value > 1.0 + 1e-6:
            raise RuntimeError(
                _dabe_pseudo_error(
                    f"{tensor_key} values out of [0,1] for {key}: "
                    f"min={min_value:.6f}, max={max_value:.6f} | {cache_path}"
                )
            )
        return tensor_key, shape
    if found_bad_shapes:
        raise RuntimeError(
            _dabe_pseudo_error(
                f"invalid DABE pseudo tensor shape for {key}: "
                f"expected {expected_shape} or {raw_shape}, found={found_bad_shapes} | {cache_path}"
            )
        )
    raise RuntimeError(
        _dabe_pseudo_error(
            f"missing usable DABE pseudo tensor for {key}: tried={_dabe_pseudo_tensor_keys(cfg)} | {cache_path}"
        )
    )


def check_dabe_pseudo_cache(cfg, max_samples=None):
    manifest_path = dabe_pseudo_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_dabe_pseudo_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_dabe_pseudo_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_dabe_pseudo_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_dabe_pseudo_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_dabe_pseudo_error(f"missing manifest rows first 10: {missing[:10]}"))

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    expected_version = str(getattr(cfg, "DABE_VERSION", "v2")).lower()
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _dabe_pseudo_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        if str(row.get("dabe_version", expected_version)).lower() != expected_version:
            raise RuntimeError(
                _dabe_pseudo_error(
                    f"version mismatch for {key}: {row.get('dabe_version')} != {expected_version}"
                )
            )
        if row.get("shape_68") is not None and row.get("shape_68") != expected_shape:
            raise RuntimeError(_dabe_pseudo_error(f"invalid shape_68 for {key}: {row.get('shape_68')} != {expected_shape}"))
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_dabe_pseudo_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_dabe_pseudo_error(f"missing cache file: {cache_path}"))
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(_dabe_pseudo_error(f"payload must be dict for {key}: {cache_path}"))
        if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
            raise RuntimeError(_dabe_pseudo_error(f"payload key mismatch for {key}: {cache_path}"))
        if payload.get("backbone_key") != cfg.BACKBONE_KEY:
            raise RuntimeError(_dabe_pseudo_error(f"payload backbone mismatch for {key}: {cache_path}"))
        if str(payload.get("dabe_version", "")).lower() != expected_version:
            raise RuntimeError(_dabe_pseudo_error(f"payload version mismatch for {key}: {cache_path}"))
        _check_dabe_pseudo_tensor(payload, cfg, key, cache_path)
        if bool(getattr(cfg, "USE_DABE_AWARE_LOSS", False)):
            aware_fields = {
                "fg_core": ["fg_core_37", "fg_core"],
                "bg_core": ["bg_core_37", "bg_core"],
                "evidence": ["evidence_37", "evidence"],
            }
            missing = []
            for field_name, field_keys in aware_fields.items():
                _, aware_tensor = _dabe_first_tensor(payload, field_keys)
                if aware_tensor is None:
                    missing.append(field_name)
                    continue
                if list(aware_tensor.shape) != [1, 37, 37]:
                    raise RuntimeError(
                        _dabe_pseudo_error(
                            f"invalid aware field shape for {key}: {field_name} "
                            f"{list(aware_tensor.shape)} != [1, 37, 37] | {cache_path}"
                        )
                    )
                min_aware = float(aware_tensor.min().item())
                max_aware = float(aware_tensor.max().item())
                if min_aware < -1e-6 or max_aware > 1.0 + 1e-6:
                    raise RuntimeError(
                        _dabe_pseudo_error(
                            f"aware field values out of [0,1] for {key}: {field_name} "
                            f"min={min_aware:.6f}, max={max_aware:.6f} | {cache_path}"
                        )
                    )
            if missing:
                raise RuntimeError(
                    _dabe_pseudo_error(
                        f"missing aware fields for {key}: cache_path={cache_path} missing_keys={missing}"
                    )
                )

    return True, f"complete: {manifest_path} | rows_checked={len(expected_keys)} | version={expected_version}"


def _dabe_pu_error(reason):
    return (
        f"DABE-PU cache missing or incomplete: {reason}\n"
        "Please generate the offline DABE-PU cache first; training will not auto-generate it."
    )


def _check_dabe_pu_tensor(payload, key, cache_path, tensor_key, expected_shape):
    tensor = payload.get(tensor_key)
    if not torch.is_tensor(tensor):
        raise RuntimeError(
            _dabe_pu_error(
                f"missing_key={tensor_key} for {key}: cache_path={cache_path}"
            )
        )
    tensor = tensor.float()
    if list(tensor.shape) != expected_shape:
        raise RuntimeError(
            _dabe_pu_error(
                f"invalid shape for {key}: {tensor_key} {list(tensor.shape)} != "
                f"{expected_shape} | cache_path={cache_path}"
            )
        )
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise RuntimeError(
            _dabe_pu_error(
                f"{tensor_key} values out of [0,1] for {key}: "
                f"min={min_value:.6f}, max={max_value:.6f} | cache_path={cache_path}"
            )
        )


def check_dabe_pu_cache(cfg, max_samples=None):
    manifest_path = dabe_pu_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_dabe_pu_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_dabe_pu_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_dabe_pu_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_dabe_pu_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_dabe_pu_error(f"missing manifest rows first 10: {missing[:10]}"))

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    expected_shape_37 = [1, 37, 37]
    expected_version = str(getattr(cfg, "DABE_PU_VERSION", "pu_v11")).lower()
    if expected_version not in {"pu_v11", "pu_v12_shape_complete"}:
        raise RuntimeError(_dabe_pu_error(f"unsupported DABE_PU_VERSION={expected_version}"))
    use_oem = bool(getattr(cfg, "USE_DABE_OEM", False)) or str(
        getattr(cfg, "P_INIT_MODE", "")
    ) == "dabe_pu_v11_oem"
    use_ap_stcr = bool(getattr(cfg, "USE_AP_STCR", False))
    required_fields = [
        "target_soft_68",
        "weight_map_68",
        "fg_core_pu_68",
        "fg_core_fallback_68",
        "bg_core_pu_68",
        "extent_candidate_68",
        "unknown_68",
    ]
    static_source = str(
        getattr(cfg, "DABE_PU_STATIC_SOURCE", "target_soft_68")
    ).strip().lower()
    if static_source not in {"target_soft_68", "p_base_68"}:
        raise RuntimeError(
            _dabe_pu_error(
                "unsupported DABE_PU_STATIC_SOURCE="
                f"{static_source!r}; expected 'target_soft_68' or 'p_base_68'"
            )
        )
    if static_source == "p_base_68":
        required_fields.append("p_base_68")
    if use_oem:
        required_fields.extend(
            [
                "fg_core_pu_37",
                "fg_core_fallback_37",
                "bg_core_pu_37",
                "extent_candidate_37",
                "unknown_37",
            ]
        )
    if use_ap_stcr:
        required_fields.extend(["target_soft_37", "bg_anchor_37"])
    audit_arbiter_weight = bool(getattr(cfg, "USE_SOURCE_ARBITER", False))
    weight_min = float("inf")
    weight_max = float("-inf")
    weight_sum = 0.0
    weight_nonzero = 0
    weight_count = 0
    per_image_max_values = []
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _dabe_pu_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        if str(row.get("dabe_version", expected_version)).lower() != expected_version:
            raise RuntimeError(
                _dabe_pu_error(
                    f"version mismatch for {key}: {row.get('dabe_version')} != {expected_version}"
                )
            )
        if row.get("shape_68") is not None and row.get("shape_68") != expected_shape:
            raise RuntimeError(_dabe_pu_error(f"invalid shape_68 for {key}: {row.get('shape_68')} != {expected_shape}"))
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_dabe_pu_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_dabe_pu_error(f"missing cache file: {cache_path}"))
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(_dabe_pu_error(f"payload must be dict for {key}: {cache_path}"))
        if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
            raise RuntimeError(_dabe_pu_error(f"payload key mismatch for {key}: {cache_path}"))
        if payload.get("backbone_key") != cfg.BACKBONE_KEY:
            raise RuntimeError(_dabe_pu_error(f"payload backbone mismatch for {key}: {cache_path}"))
        if str(payload.get("dabe_version", "")).lower() != expected_version:
            raise RuntimeError(_dabe_pu_error(f"payload version mismatch for {key}: {cache_path}"))
        for tensor_key in required_fields:
            shape = expected_shape_37 if tensor_key.endswith("_37") else expected_shape
            _check_dabe_pu_tensor(payload, key, cache_path, tensor_key, shape)
        if static_source == "p_base_68" and not bool(
            torch.isfinite(payload["p_base_68"]).all().item()
        ):
            raise RuntimeError(
                _dabe_pu_error(
                    f"p_base_68 contains NaN/Inf for {key}: {cache_path}"
                )
            )
        if audit_arbiter_weight:
            weight = payload["weight_map_68"].float()
            if not bool(torch.isfinite(weight).all().item()):
                raise RuntimeError(
                    _dabe_pu_error(
                        f"weight_map_68 contains NaN/Inf for {key}: {cache_path}"
                    )
                )
            image_min = float(weight.min().item())
            image_max = float(weight.max().item())
            if image_min < -1e-6 or image_max > 1.0 + 1e-6:
                raise RuntimeError(
                    _dabe_pu_error(
                        "EGSA requires audited raw weight_map_68 in [0,1], "
                        f"got min={image_min:.8f}, max={image_max:.8f} for "
                        f"{key}: {cache_path}"
                    )
                )
            weight_min = min(weight_min, image_min)
            weight_max = max(weight_max, image_max)
            weight_sum += float(weight.double().sum().item())
            weight_nonzero += int((weight > 0.0).sum().item())
            weight_count += int(weight.numel())
            per_image_max_values.append(image_max)

    reason = (
        f"complete: {manifest_path} | rows_checked={len(expected_keys)} | "
        f"version={expected_version}"
    )
    if audit_arbiter_weight:
        per_image_max = torch.tensor(per_image_max_values, dtype=torch.float64)
        reason += (
            " | EGSA weight audit: "
            f"min={weight_min:.8f}, max={weight_max:.8f}, "
            f"mean={weight_sum / max(weight_count, 1):.8f}, "
            f"nonzero_ratio={weight_nonzero / max(weight_count, 1):.8f}, "
            "per_image_max="
            f"{float(per_image_max.min()):.8f}/"
            f"{float(per_image_max.mean()):.8f}/"
            f"{float(per_image_max.max()):.8f}"
        )
    return True, reason


def check_r1_only_static_cache(cfg, max_samples=None, payload_samples=2):
    """Audit the only cache consumed by a static Hard-R1 pure Student.

    All manifest identities and file paths are checked, while only a small,
    deterministic sample of payloads is deserialized.  This intentionally
    avoids touching the legacy DABE-Clean cache, which is not a supervision
    input for this protocol.
    """

    if not bool(getattr(cfg, "R1_ONLY_CACHE_IO", False)):
        raise RuntimeError("R1-only cache audit requires R1_ONLY_CACHE_IO=True.")
    static_source = str(
        getattr(cfg, "DABE_CLEAN_STATIC_TARGET_SOURCE", "")
    ).strip().lower()
    source_keys = {
        "dabe_v2_r1_hard_68": "residual_pass1_37",
        "cvbr_v1_second_ring_hard_68": "v1_cvbr_second_ring_37",
        "rpr_p1_second_ring_hard_68": "p1_rpr_secondring_37",
        "gbsp_abs_minmax_hard_68": "gbsp_abs_minmax_37",
    }
    if static_source not in source_keys:
        raise RuntimeError(
            "R1-only cache I/O requires an independent 37x37 static source, "
            f"got {static_source!r}."
        )
    if not bool(getattr(cfg, "DABEV2HARD_PURE_STUDENT", False)) or not bool(
        getattr(cfg, "DABEV2HARD_STATIC_ONLY", False)
    ):
        raise RuntimeError(
            "R1-only cache I/O requires pure-Student static-only training."
        )

    root = Path(str(getattr(cfg, "DABE_CLEAN_DABE_V2_ROOT", "")).strip())
    manifest_path = root / "manifest_train.jsonl"
    if not manifest_path.is_file():
        raise RuntimeError(f"R1 static manifest is missing: {manifest_path}")
    expected_items = build_image_items(
        cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False
    )
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [
        (item["dataset"], item["stem"]) for item in expected_items
    ]
    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        key = (row.get("dataset"), row.get("stem"))
        if None in key:
            raise RuntimeError(f"Bad R1 static manifest row: {row}")
        if key in row_map:
            raise RuntimeError(f"Duplicate R1 static manifest key: {key}")
        row_map[key] = row
    missing = sorted(set(expected_keys) - set(row_map))
    if missing:
        raise RuntimeError(
            f"R1 static cache missing first 10 keys: {missing[:10]}"
        )
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(set(row_map) - set(expected_keys))
        if extra:
            raise RuntimeError(
                f"R1 static cache has unexpected first 10 keys: {extra[:10]}"
            )
        if len(expected_keys) != 4040:
            raise RuntimeError(
                "Full Hard-R1 training protocol requires exactly 4040 rows, "
                f"got {len(expected_keys)}."
            )

    expected_backbone = str(cfg.BACKBONE_KEY)
    expected_version = str(
        getattr(cfg, "DABE_CLEAN_DABE_V2_VERSION", "v2")
    ).strip().lower()
    for key in expected_keys:
        row = row_map[key]
        cache_path = Path(str(row.get("cache_path", "")))
        if not cache_path.is_file():
            raise RuntimeError(f"R1 static cache is missing: {cache_path}")
        if row.get("backbone_key") is not None and str(
            row.get("backbone_key")
        ) != expected_backbone:
            raise RuntimeError(
                f"R1 manifest backbone mismatch for {key}: "
                f"{row.get('backbone_key')} != {expected_backbone}."
            )
        if static_source == "dabe_v2_r1_hard_68":
            if row.get("dabe_version") is not None and str(
                row.get("dabe_version")
            ).strip().lower() != expected_version:
                raise RuntimeError(
                    f"R1 manifest version mismatch for {key}: "
                    f"{row.get('dabe_version')} != {expected_version}."
                )
            if row.get("shape_37") is not None and list(
                row.get("shape_37")
            ) != [1, 37, 37]:
                raise RuntimeError(
                    f"R1 manifest shape mismatch for {key}: "
                    f"{row.get('shape_37')}."
                )
        elif static_source == "gbsp_abs_minmax_hard_68":
            expected_gbsp_version = str(
                getattr(cfg, "DABE_CLEAN_GBSP_VERSION", "")
            ).strip().lower()
            if str(row.get("gbsp_version", "")).strip().lower() != expected_gbsp_version:
                raise RuntimeError(
                    f"GBSP manifest version mismatch for {key}: "
                    f"{row.get('gbsp_version')} != {expected_gbsp_version}."
                )
            if list(row.get("shape", [])) != [1, 37, 37]:
                raise RuntimeError(
                    f"GBSP manifest shape mismatch for {key}: {row.get('shape')}."
                )

    sample_count = min(max(int(payload_samples), 0), len(expected_keys))
    sample_indices = []
    if sample_count:
        sample_indices = sorted(
            set(
                round(index * (len(expected_keys) - 1) / max(sample_count - 1, 1))
                for index in range(sample_count)
            )
        )
    source_key = source_keys[static_source]
    for index in sample_indices:
        key = expected_keys[index]
        cache_path = Path(str(row_map[key]["cache_path"]))
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(f"R1 static payload must be a dict: {cache_path}")
        if str(payload.get("dataset")) != str(key[0]) or str(
            payload.get("stem")
        ) != str(key[1]):
            raise RuntimeError(f"R1 static payload identity mismatch: {cache_path}")
        if static_source == "gbsp_abs_minmax_hard_68":
            expected_gbsp_version = str(
                getattr(cfg, "DABE_CLEAN_GBSP_VERSION", "")
            ).strip().lower()
            if (
                str(payload.get("gbsp_version", "")).strip().lower()
                != expected_gbsp_version
            ):
                raise RuntimeError(
                    f"GBSP payload version mismatch: {cache_path}"
                )
            if bool(payload.get("gt_used_for_generation", True)):
                raise RuntimeError(
                    f"GBSP payload reports GT use during generation: {cache_path}"
                )
        source = payload.get(source_key)
        if not torch.is_tensor(source) or tuple(source.shape) != (1, 37, 37):
            raise RuntimeError(
                f"R1 source {source_key} must be [1,37,37]: {cache_path}"
            )
        source = source.detach().cpu().float()
        if not torch.isfinite(source).all() or float(source.min()) < 0.0 or float(
            source.max()
        ) > 1.0:
            raise RuntimeError(
                f"R1 source {source_key} is not finite in [0,1]: {cache_path}"
            )

    reason = (
        f"complete: {manifest_path} | rows_checked={len(expected_keys)} | "
        "manifest_only=True | "
        f"payload_samples_checked={len(sample_indices)} | "
        "dabe_clean_payloads_read=False | "
        f"source_key={source_key}"
    )
    return True, reason


def check_dabe_clean_cache(cfg, max_samples=None, return_stats=False):
    """Validate the independent Bridge/Clean cache without touching GT."""

    if found_static_enabled(cfg):
        manifest_path = found_static_manifest_path(cfg)
        if not manifest_path.exists():
            raise RuntimeError(
                f"FOUND static cache manifest is missing: {manifest_path}"
            )
        expected_items = build_image_items(
            cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False
        )
        if max_samples is not None and int(max_samples) >= 0:
            expected_items = expected_items[: int(max_samples)]
        expected_keys = [
            (item["dataset"], item["stem"]) for item in expected_items
        ]
        rows = read_jsonl(manifest_path)
        row_map = {}
        for row in rows:
            key = (row.get("dataset"), row.get("stem"))
            if None in key:
                raise RuntimeError(f"Bad FOUND static manifest row: {row}")
            if key in row_map:
                raise RuntimeError(f"Duplicate FOUND static manifest key: {key}")
            row_map[key] = row
        missing = sorted(set(expected_keys) - set(row_map))
        if missing:
            raise RuntimeError(
                f"FOUND static cache missing first 10 keys: {missing[:10]}"
            )
        if max_samples is None or int(max_samples) < 0:
            extra = sorted(set(row_map) - set(expected_keys))
            if extra:
                raise RuntimeError(
                    f"FOUND static cache has unexpected first 10 keys: {extra[:10]}"
                )
            if len(expected_keys) != 4040:
                raise RuntimeError(
                    "FOUND static full training protocol requires exactly "
                    f"4040 rows, got {len(expected_keys)}."
                )

        value_sum = 0.0
        value_sq_sum = 0.0
        hard_sum = 0.0
        value_count = 0
        native_fg_sum = 0.0
        native_count = 0
        for key in expected_keys:
            row = row_map[key]
            if list(row.get("shape", [])) != list(FOUND_STATIC_NATIVE_SHAPE):
                raise RuntimeError(
                    f"FOUND static manifest shape mismatch for {key}: "
                    f"{row.get('shape')} != {list(FOUND_STATIC_NATIVE_SHAPE)}."
                )
            cache_path = Path(str(row.get("cache_path", "")))
            if not cache_path.is_file():
                raise RuntimeError(f"FOUND static cache is missing: {cache_path}")
            payload = torch_load(cache_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"FOUND static payload must be a dict: {cache_path}"
                )
            if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
                raise RuntimeError(
                    f"FOUND static payload identity mismatch: {cache_path}"
                )
            native = payload.get("tensor")
            target = build_found_static_target(native, cfg.LOSS_SIZE)
            flat = target.reshape(-1).double()
            value_sum += float(flat.sum().item())
            value_sq_sum += float((flat * flat).sum().item())
            hard_sum += float((flat > 0.5).sum().item())
            value_count += int(flat.numel())
            native_fg_sum += float(native.double().sum().item())
            native_count += int(native.numel())
        target_mean = value_sum / max(value_count, 1)
        target_variance = max(
            value_sq_sum / max(value_count, 1) - target_mean * target_mean,
            0.0,
        )
        stats = {
            "target_offline": {
                "mean": target_mean,
                "std": target_variance ** 0.5,
                "hard_area": hard_sum / max(value_count, 1),
            },
            "offline_evidence": {
                "found_native_hard_area": native_fg_sum / max(native_count, 1),
            },
        }
        reason = (
            f"complete: {manifest_path} | rows_checked={len(expected_keys)} | "
            f"source={FOUND_STATIC_SOURCE} | native_shape="
            f"{list(FOUND_STATIC_NATIVE_SHAPE)} | resize={FOUND_STATIC_RESIZE_MODE} | "
            f"target_shape=[1,{int(cfg.LOSS_SIZE)},{int(cfg.LOSS_SIZE)}] | "
            "training_gt_read=False | Dataset_target=target_offline_68"
        )
        if return_stats:
            return True, reason, stats
        return True, reason

    manifest_path = dabe_clean_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(f"DABE-Clean cache manifest is missing: {manifest_path}")
    expected_items = build_image_items(
        cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False
    )
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        key = (row.get("dataset"), row.get("stem"))
        if None in key:
            raise RuntimeError(f"Bad DABE-Clean manifest row: {row}")
        if key in row_map:
            raise RuntimeError(f"Duplicate DABE-Clean manifest key: {key}")
        row_map[key] = row
    missing = sorted(set(expected_keys) - set(row_map))
    if missing:
        raise RuntimeError(f"DABE-Clean cache missing first 10 keys: {missing[:10]}")
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(set(row_map) - set(expected_keys))
        if extra:
            raise RuntimeError(
                f"DABE-Clean cache has unexpected first 10 keys: {extra[:10]}"
            )

    is_offline_v3 = str(
        getattr(cfg, "DABE_CLEAN_VERSION", "")
    ).strip().lower() == "v3_offline_consolidation"
    if is_offline_v3:
        mode = str(getattr(cfg, "DABE_CLEAN_OFFLINE_MODE", "")).strip().lower()
        if mode not in DABE_CLEAN_OFFLINE_MODES:
            raise RuntimeError(f"Unsupported DABE_CLEAN_OFFLINE_MODE={mode!r}.")
        if max_samples is None or int(max_samples) < 0:
            if len(expected_keys) != 4040:
                raise RuntimeError(
                    "DABE-Clean v3 full training protocol requires exactly "
                    f"4040 rows, got {len(expected_keys)}."
                )
        fields_37 = (
            "residual_37",
            "bc_map_37",
            "p_rw_37",
            "evidence_37",
            "primary_fg_37",
            "background_evidence_37",
            "semantic_fg_tendency_37",
            "latent_support_37",
            "complementary_fg_37",
            "complementary_increment_37",
            "consolidated_fg_37",
            "conflict_37",
            "commitment_37",
            "target_offline_37",
        )
        target_values = []
        offline_manifest_stat_fields = (
            "primary_fg_mean",
            "primary_fg_hard_area",
            "complementary_fg_mean",
            "consolidated_fg_mean",
            "consolidated_fg_hard_area",
            "background_mean",
            "conflict_mean",
            "commitment_mean",
        )
        offline_manifest_stats = {
            field: [] for field in offline_manifest_stat_fields
        }
        for key in expected_keys:
            row = row_map[key]
            for row_field, expected in (
                ("payload_version", DABE_CLEAN_OFFLINE_PAYLOAD_VERSION),
                ("version", DABE_CLEAN_OFFLINE_PAYLOAD_VERSION),
                ("offline_mode", mode),
                ("formula_fingerprint", DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT),
                ("cache_kind", "offline_consolidation"),
            ):
                if str(row.get(row_field)) != expected:
                    raise RuntimeError(
                        f"DABE-Clean offline manifest {row_field} mismatch for "
                        f"{key}: {row.get(row_field)!r} != {expected!r}."
                    )
            if list(row.get("target_shape", [])) != [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]:
                raise RuntimeError(
                    f"DABE-Clean offline manifest target_shape mismatch for {key}."
                )
            for field in offline_manifest_stat_fields:
                try:
                    value = float(row[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"DABE-Clean offline manifest has invalid {field} for {key}."
                    ) from exc
                if not np.isfinite(value) or value < -1e-6 or value > 1.0 + 1e-6:
                    raise RuntimeError(
                        f"DABE-Clean offline manifest {field} is outside [0,1] "
                        f"for {key}: {value}."
                    )
                offline_manifest_stats[field].append(value)
            for flag in (
                "training_gt_read",
                "teacher_prediction_read",
                "teacher_checkpoint_read",
                "student_checkpoint_read",
                "epoch_dependent",
                "history_read",
                "static_weight_map_written",
                "routing_map_written",
            ):
                if bool(row.get(flag, True)):
                    raise RuntimeError(
                        f"DABE-Clean offline manifest forbids {flag}=True for {key}."
                    )
            cache_path = Path(row.get("cache_path", ""))
            if not cache_path.is_file():
                raise RuntimeError(f"DABE-Clean offline cache is missing: {cache_path}")
            payload = torch_load(cache_path, map_location="cpu")
            if not isinstance(payload, dict):
                raise RuntimeError(f"DABE-Clean offline payload must be dict: {cache_path}")
            if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
                raise RuntimeError(f"DABE-Clean offline identity mismatch: {cache_path}")
            if str(payload.get("backbone_key")) != str(cfg.BACKBONE_KEY):
                raise RuntimeError(f"DABE-Clean offline backbone mismatch: {cache_path}")
            for payload_field, expected in (
                ("payload_version", DABE_CLEAN_OFFLINE_PAYLOAD_VERSION),
                ("version", DABE_CLEAN_OFFLINE_PAYLOAD_VERSION),
                ("offline_mode", mode),
                ("formula_fingerprint", DABE_CLEAN_OFFLINE_FORMULA_FINGERPRINT),
            ):
                if str(payload.get(payload_field)) != expected:
                    raise RuntimeError(
                        f"DABE-Clean offline payload {payload_field} mismatch: "
                        f"{cache_path}"
                    )
            for flag in (
                "training_gt_read",
                "teacher_prediction_read",
                "teacher_checkpoint_read",
                "student_checkpoint_read",
                "epoch_dependent",
                "history_read",
                "static_weight_map_written",
                "routing_map_written",
            ):
                if bool(payload.get(flag, True)):
                    raise RuntimeError(
                        f"DABE-Clean offline payload forbids {flag}=True: {cache_path}"
                    )
            forbidden_payload_fields = {
                "gt",
                "gt_path",
                "static_weight_map",
                "weight_map",
                "recovery_map",
                "routing_map",
                "teacher_region_map",
                "history",
                "epoch",
            }.intersection(payload)
            if forbidden_payload_fields:
                raise RuntimeError(
                    "DABE-Clean offline payload leaked forbidden fields "
                    f"{sorted(forbidden_payload_fields)}: {cache_path}"
                )
            for field in (*fields_37, "target_offline_68"):
                value = payload.get(field)
                if not torch.is_tensor(value):
                    raise RuntimeError(
                        f"DABE-Clean offline payload is missing {field}: {cache_path}"
                    )
                expected_shape = (
                    (1, 37, 37)
                    if field.endswith("_37")
                    else (1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE))
                )
                if tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        f"DABE-Clean offline {field} shape mismatch: "
                        f"{list(value.shape)} != {list(expected_shape)} | {cache_path}"
                    )
                if value.dtype != torch.float32 or value.requires_grad:
                    raise RuntimeError(
                        f"DABE-Clean offline {field} must be detached float32: {cache_path}"
                    )
                if not bool(torch.isfinite(value).all().item()):
                    raise RuntimeError(
                        f"DABE-Clean offline {field} contains NaN/Inf: {cache_path}"
                    )
                if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
                    raise RuntimeError(
                        f"DABE-Clean offline {field} is outside [0,1]: {cache_path}"
                    )
            target_values.append(payload["target_offline_68"].reshape(-1))
        targets = torch.cat(target_values)
        stats = {
            "target_offline": {
                "mean": float(targets.mean().item()),
                "std": float(targets.std(unbiased=False).item()),
                "hard_area": float((targets > 0.5).float().mean().item()),
            },
            "offline_evidence": {
                field: float(np.mean(values))
                for field, values in offline_manifest_stats.items()
            },
        }
        reason = (
            f"complete: {manifest_path} | rows_checked={len(expected_keys)} | "
            f"version={DABE_CLEAN_OFFLINE_PAYLOAD_VERSION} | offline_mode={mode} | "
            "training_gt_read=False | Dataset_target=target_offline_68"
        )
        if return_stats:
            return True, reason, stats
        return True, reason

    mode = str(getattr(cfg, "DABE_CLEAN_TARGET_MODE", "dp")).strip().lower()
    if mode not in {"dp", "diff", "bridge"}:
        raise RuntimeError(f"Unsupported DABE_CLEAN_TARGET_MODE={mode!r}.")
    configured_payload_version = str(
        getattr(cfg, "DABE_CLEAN_EXPECTED_PAYLOAD_VERSION", "")
    ).strip()
    if mode == "bridge":
        expected_version = "dabe_bridge_v1"
    elif configured_payload_version:
        expected_version = configured_payload_version
    elif str(getattr(cfg, "DABE_CLEAN_VERSION", "v1")).lower() == "v2_contrec":
        expected_version = "dabe_clean_v2_contrec"
    else:
        expected_version = "dabe_clean_v1"
    is_contrec = expected_version == "dabe_clean_v2_contrec"
    if is_contrec and (max_samples is None or int(max_samples) < 0):
        if len(expected_keys) != 4040:
            raise RuntimeError(
                "DABE-Clean v2-contrec full training protocol requires exactly "
                f"4040 rows, got {len(expected_keys)}."
            )
    expected_fields = (
        [
            "bridge_target_37",
            "bridge_target_68",
            "source_target_soft_37",
            "source_target_soft_68",
            "source_weight_map_37",
            "source_weight_map_68",
            "source_p_base_37",
        ]
        if mode == "bridge"
        else ([
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
        ] if is_contrec else [
            "foreground_evidence_37",
            "foreground_evidence_68",
            "background_evidence_37",
            "background_evidence_68",
            "target_dp_37",
            "target_dp_68",
            "target_diff_37",
            "target_diff_68",
            "source_p_base_37",
            "source_p_base_68",
            "source_bc_map_37",
            "source_residual_37",
        ])
    )
    distribution_fields = {
        "p_rw": "p_rw_37",
        "foreground_evidence": "foreground_evidence_37",
        "latent_rw": "latent_rw_37",
        "background_evidence": "background_evidence_37",
        "semantic_fg_tendency": "semantic_fg_tendency_37",
        "recoverability": "recoverability_37",
    }
    distribution_values = {name: [] for name in distribution_fields}
    for key in expected_keys:
        row = row_map[key]
        cache_path = Path(row.get("cache_path", ""))
        if not cache_path.is_file():
            raise RuntimeError(f"DABE-Clean cache file is missing: {cache_path}")
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(f"DABE-Clean payload must be dict: {cache_path}")
        if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
            raise RuntimeError(f"DABE-Clean payload identity mismatch: {cache_path}")
        if str(payload.get("version")) != expected_version:
            raise RuntimeError(
                f"DABE-Clean version mismatch: {payload.get('version')} != "
                f"{expected_version} | {cache_path}"
            )
        if str(payload.get("backbone_key")) != str(cfg.BACKBONE_KEY):
            raise RuntimeError(f"DABE-Clean backbone mismatch: {cache_path}")
        if is_contrec:
            if bool(payload.get("training_gt_read", True)):
                raise RuntimeError(
                    f"DABE-Clean contrec reports training_gt_read=True: {cache_path}"
                )
            if bool(payload.get("teacher_prediction_read", True)):
                raise RuntimeError(
                    f"DABE-Clean contrec reports Teacher access: {cache_path}"
                )
            forbidden = {
                "gt",
                "gt_path",
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
                    f"DABE-Clean contrec payload leaked forbidden fields {leaked}: "
                    f"{cache_path}"
                )
        for field in expected_fields:
            value = payload.get(field)
            if not torch.is_tensor(value):
                raise RuntimeError(
                    f"DABE-Clean payload is missing {field}: {cache_path}"
                )
            expected_shape = (1, 37, 37) if field.endswith("_37") else (
                1,
                int(cfg.LOSS_SIZE),
                int(cfg.LOSS_SIZE),
            )
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"DABE-Clean {field} shape mismatch: {list(value.shape)} != "
                    f"{list(expected_shape)} | {cache_path}"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"DABE-Clean {field} contains NaN/Inf: {cache_path}")
            if float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6:
                raise RuntimeError(f"DABE-Clean {field} is outside [0,1]: {cache_path}")
        if is_contrec:
            reconstruction_error = float(
                (
                    payload["p_rw_37"].float()
                    * payload["evidence_gate_37"].float()
                    - payload["foreground_evidence_37"].float()
                ).abs().max().item()
            )
            if reconstruction_error >= 1e-5:
                raise RuntimeError(
                    "DABE-Clean contrec foreground regression failed: "
                    f"error={reconstruction_error:.9g} | {cache_path}"
                )
            for name, field in distribution_fields.items():
                distribution_values[name].append(
                    payload[field].detach().cpu().float().reshape(-1)
                )
    stats = {}
    if is_contrec:
        for name, chunks in distribution_values.items():
            values = torch.cat(chunks)
            quantiles = torch.quantile(
                values, torch.tensor([0.50, 0.90, 0.95], dtype=values.dtype)
            )
            stats[name] = {
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "p50": float(quantiles[0].item()),
                "p90": float(quantiles[1].item()),
                "p95": float(quantiles[2].item()),
                "max": float(values.max().item()),
            }
            if name == "recoverability":
                stats[name].update(
                    {
                        "zero_ratio": float((values == 0.0).float().mean().item()),
                        "gt_0_1_ratio": float((values > 0.1).float().mean().item()),
                        "gt_0_3_ratio": float((values > 0.3).float().mean().item()),
                        "gt_0_5_ratio": float((values > 0.5).float().mean().item()),
                    }
                )
    reason = (
        f"complete: {manifest_path} | rows_checked={len(expected_keys)} | "
        f"version={expected_version} | mode={mode} | training_gt_read=False"
    )
    if return_stats:
        return True, reason, stats
    return True, reason


def _tce_cover_error(reason):
    return (
        f"TCE coverage cache missing or incomplete: {reason}\n"
        "Please build it first with tools/build_tce_cover_cache.py; training will not auto-generate it."
    )


def _check_tce_cover_tensor(payload, key, cache_path, tensor_key, expected_shape):
    tensor = payload.get(tensor_key)
    if not torch.is_tensor(tensor):
        raise RuntimeError(
            _tce_cover_error(
                f"missing_key={tensor_key} for {key}: cache_path={cache_path}"
            )
        )
    tensor = tensor.float()
    if list(tensor.shape) != expected_shape:
        raise RuntimeError(
            _tce_cover_error(
                f"invalid shape for {key}: {tensor_key} {list(tensor.shape)} != "
                f"{expected_shape} | cache_path={cache_path}"
            )
        )
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise RuntimeError(
            _tce_cover_error(
                f"{tensor_key} values out of [0,1] for {key}: "
                f"min={min_value:.6f}, max={max_value:.6f} | cache_path={cache_path}"
            )
        )


def check_tce_cover_cache(cfg, max_samples=None):
    manifest_path = tce_cover_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_tce_cover_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_tce_cover_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_tce_cover_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_tce_cover_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_tce_cover_error(f"missing manifest rows first 10: {missing[:10]}"))

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    expected_epoch = int(getattr(cfg, "TCE_COVER_EPOCH", -1))
    expected_model = str(getattr(cfg, "TCE_COVER_MODEL", "")).lower()
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _tce_cover_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        if row.get("shape_68") is not None and row.get("shape_68") != expected_shape:
            raise RuntimeError(_tce_cover_error(f"invalid shape_68 for {key}: {row.get('shape_68')} != {expected_shape}"))
        if expected_epoch >= 0 and int(row.get("source_epoch", expected_epoch)) != expected_epoch:
            raise RuntimeError(
                _tce_cover_error(
                    f"source_epoch mismatch for {key}: {row.get('source_epoch')} != {expected_epoch}"
                )
            )
        if expected_model and str(row.get("model_for_cache", expected_model)).lower() != expected_model:
            raise RuntimeError(
                _tce_cover_error(
                    f"model_for_cache mismatch for {key}: {row.get('model_for_cache')} != {expected_model}"
                )
            )
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_tce_cover_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_tce_cover_error(f"missing cache file: {cache_path}"))
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(_tce_cover_error(f"payload must be dict for {key}: {cache_path}"))
        if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
            raise RuntimeError(_tce_cover_error(f"payload key mismatch for {key}: {cache_path}"))
        if payload.get("backbone_key") != cfg.BACKBONE_KEY:
            raise RuntimeError(_tce_cover_error(f"payload backbone mismatch for {key}: {cache_path}"))
        if expected_epoch >= 0 and int(payload.get("source_epoch", expected_epoch)) != expected_epoch:
            raise RuntimeError(
                _tce_cover_error(
                    f"payload source_epoch mismatch for {key}: {payload.get('source_epoch')} != {expected_epoch}"
                )
            )
        if expected_model and str(payload.get("model_for_cache", expected_model)).lower() != expected_model:
            raise RuntimeError(
                _tce_cover_error(
                    f"payload model_for_cache mismatch for {key}: {payload.get('model_for_cache')} != {expected_model}"
                )
            )
        for tensor_key in ("cover_prob_68", "cover_binary_68", "cover_conf_68"):
            _check_tce_cover_tensor(payload, key, cache_path, tensor_key, expected_shape)

    return True, f"complete: {manifest_path} | rows_checked={len(expected_keys)} | source_epoch={expected_epoch}"


def _lceg_cover_error(reason):
    return (
        f"LCEG coverage cache missing or incomplete: {reason}\n"
        "Please build it first with tools/build_lceg_cover_cache.py; training will not auto-generate it."
    )


def _check_lceg_cover_tensor(payload, key, cache_path, tensor_key, expected_shape):
    tensor = payload.get(tensor_key)
    if not torch.is_tensor(tensor):
        raise RuntimeError(
            _lceg_cover_error(
                f"missing_key={tensor_key} for {key}: cache_path={cache_path}"
            )
        )
    tensor = tensor.float()
    if list(tensor.shape) != expected_shape:
        raise RuntimeError(
            _lceg_cover_error(
                f"invalid shape for {key}: {tensor_key} {list(tensor.shape)} != "
                f"{expected_shape} | cache_path={cache_path}"
            )
        )
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise RuntimeError(
            _lceg_cover_error(
                f"{tensor_key} values out of [0,1] for {key}: "
                f"min={min_value:.6f}, max={max_value:.6f} | cache_path={cache_path}"
            )
        )


def check_lceg_cover_cache(cfg, max_samples=None):
    manifest_path = lceg_cover_manifest_path(cfg)
    if not manifest_path.exists():
        raise RuntimeError(_lceg_cover_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_lceg_cover_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_lceg_cover_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_lceg_cover_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_lceg_cover_error(f"missing manifest rows first 10: {missing[:10]}"))

    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]
    expected_epoch = int(getattr(cfg, "LCEG_COVER_EPOCH", -1))
    expected_model = str(getattr(cfg, "LCEG_COVER_MODEL", "")).lower()
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", cfg.BACKBONE_KEY) != cfg.BACKBONE_KEY:
            raise RuntimeError(
                _lceg_cover_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {cfg.BACKBONE_KEY}"
                )
            )
        if row.get("shape_68") is not None and row.get("shape_68") != expected_shape:
            raise RuntimeError(
                _lceg_cover_error(f"invalid shape_68 for {key}: {row.get('shape_68')} != {expected_shape}")
            )
        if expected_epoch >= 0 and int(row.get("source_epoch", expected_epoch)) != expected_epoch:
            raise RuntimeError(
                _lceg_cover_error(
                    f"source_epoch mismatch for {key}: {row.get('source_epoch')} != {expected_epoch}"
                )
            )
        if expected_model and str(row.get("model_for_cache", expected_model)).lower() != expected_model:
            raise RuntimeError(
                _lceg_cover_error(
                    f"model_for_cache mismatch for {key}: {row.get('model_for_cache')} != {expected_model}"
                )
            )
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_lceg_cover_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_lceg_cover_error(f"missing cache file: {cache_path}"))
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(_lceg_cover_error(f"payload must be dict for {key}: {cache_path}"))
        if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
            raise RuntimeError(_lceg_cover_error(f"payload key mismatch for {key}: {cache_path}"))
        if payload.get("backbone_key") != cfg.BACKBONE_KEY:
            raise RuntimeError(_lceg_cover_error(f"payload backbone mismatch for {key}: {cache_path}"))
        if expected_epoch >= 0 and int(payload.get("source_epoch", expected_epoch)) != expected_epoch:
            raise RuntimeError(
                _lceg_cover_error(
                    f"payload source_epoch mismatch for {key}: {payload.get('source_epoch')} != {expected_epoch}"
                )
            )
        if expected_model and str(payload.get("model_for_cache", expected_model)).lower() != expected_model:
            raise RuntimeError(
                _lceg_cover_error(
                    f"payload model_for_cache mismatch for {key}: {payload.get('model_for_cache')} != {expected_model}"
                )
            )
        for tensor_key in ("cover_prob_68", "cover_binary_68", "cover_conf_68"):
            _check_lceg_cover_tensor(payload, key, cache_path, tensor_key, expected_shape)

    return True, f"complete: {manifest_path} | rows_checked={len(expected_keys)} | source_epoch={expected_epoch}"


def _despl_paper_error(reason):
    return (
        f"DESPL-paper cache missing or incomplete: {reason}\n"
        "Please run:\n"
        "python common/cache_despl_paper.py --config configs/despl_paper_dinov1_s8.py --overwrite"
    )


def check_despl_paper_cache(cfg, max_samples=None):
    sign_mode = getattr(cfg, "DESPL_PAPER_SIGN_MODE", getattr(cfg, "DESPL_SIGN_MODE", "paper"))
    manifest_path = despl_paper_manifest_path(cfg, sign_mode=sign_mode)
    if not manifest_path.exists():
        raise RuntimeError(_despl_paper_error(f"missing manifest: {manifest_path}"))

    expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        expected_items = expected_items[: int(max_samples)]
    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(_despl_paper_error(f"bad manifest row without dataset/stem in {manifest_path}"))
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(_despl_paper_error(f"duplicate manifest key in {manifest_path}: {key}"))
        row_map[key] = row

    actual_key_set = set(row_map)
    if max_samples is None or int(max_samples) < 0:
        extra = sorted(actual_key_set - expected_key_set)
        if extra:
            raise RuntimeError(_despl_paper_error(f"manifest has unexpected keys first 10: {extra[:10]}"))
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        raise RuntimeError(_despl_paper_error(f"missing manifest rows first 10: {missing[:10]}"))

    grid = int(getattr(cfg, "DESPL_GRID", 34))
    expected_shape = [1, grid, grid]
    expected_backbone = getattr(cfg, "DESPL_PAPER_BACKBONE_KEY", cfg.BACKBONE_KEY)
    for key in expected_keys:
        row = row_map[key]
        if row.get("backbone_key", expected_backbone) != expected_backbone:
            raise RuntimeError(
                _despl_paper_error(
                    f"backbone mismatch for {key}: {row.get('backbone_key')} != {expected_backbone}"
                )
            )
        if row.get("sign_mode", sign_mode) != sign_mode:
            raise RuntimeError(
                _despl_paper_error(f"sign_mode mismatch for {key}: {row.get('sign_mode')} != {sign_mode}")
            )
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(_despl_paper_error(f"missing cache_path for {key}"))
        if not Path(cache_path).exists():
            raise RuntimeError(_despl_paper_error(f"missing cache file: {cache_path}"))
        shape = row.get("shape")
        if shape != expected_shape:
            raise RuntimeError(_despl_paper_error(f"invalid shape for {key}: {shape} != {expected_shape}"))
        payload = torch_load(cache_path, map_location="cpu")
        tensor = payload.get("p_despl_paper_soft") if isinstance(payload, dict) else None
        if not torch.is_tensor(tensor):
            raise RuntimeError(_despl_paper_error(f"missing p_despl_paper_soft for {key}: {cache_path}"))
        if list(tensor.shape) != expected_shape:
            raise RuntimeError(
                _despl_paper_error(
                    f"invalid p_despl_paper_soft shape for {key}: {list(tensor.shape)} != {expected_shape}"
                )
            )

    return True, f"complete: {manifest_path}"


def cache_status(cfg, kind, split=None):
    # preflight 只判断当前 config 所需 cache；语义错配报错，缺文件才自动生成。
    if kind == "feature":
        if split is None:
            raise ValueError("Feature cache status requires split")
        manifest_path = feature_manifest_path(cfg, split)
        expected_items = build_image_items(cfg.DATA_ROOT, split_dataset_names(cfg, split), require_gt=False)
    elif kind == "pseudo":
        manifest_path = pseudo_manifest_path(cfg)
        expected_items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    else:
        raise ValueError(f"Unknown cache kind: {kind}")

    if not manifest_path.exists():
        return False, f"missing manifest: {manifest_path}"

    rows = read_jsonl(manifest_path)
    row_map = {}
    for row in rows:
        if "dataset" not in row or "stem" not in row:
            raise RuntimeError(f"Bad manifest row without dataset/stem in {manifest_path}")
        key = (row["dataset"], row["stem"])
        if key in row_map:
            raise RuntimeError(f"Duplicate manifest key in {manifest_path}: {key}")
        row_map[key] = row

    expected_keys = [(item["dataset"], item["stem"]) for item in expected_items]
    expected_key_set = set(expected_keys)
    actual_key_set = set(row_map)
    extra = sorted(actual_key_set - expected_key_set)
    if extra:
        raise RuntimeError(f"{kind} cache manifest has unexpected keys in {manifest_path}: {extra[:10]}")
    missing = sorted(expected_key_set - actual_key_set)
    if missing:
        return False, f"missing manifest rows first 10: {missing[:10]}"

    for key in expected_keys:
        row = row_map[key]
        cache_path = row.get("cache_path")
        if not cache_path:
            return False, f"missing cache_path for {key}"
        if not Path(cache_path).exists():
            return False, f"missing cache file: {cache_path}"
        shape = row.get("shape")
        if shape is None:
            return False, f"missing shape for {key}"
        if not _cache_shape_ok(kind, shape):
            raise RuntimeError(f"Invalid {kind} cache shape for {key}: {shape}")

    return True, f"complete: {manifest_path}"


def ensure_cache_available(cfg, kind, split=None, logger=print):
    complete, reason = cache_status(cfg, kind, split=split)
    cache_name = f"{kind}:{split}" if split else kind
    if complete:
        logger(f"[Cache] {cache_name} ready | {reason}")
        return

    logger(f"[Cache] {cache_name} incomplete | {reason}")
    logger(f"[Cache] auto_generate = true | backbone_key = {cfg.BACKBONE_KEY}")
    # 自动生成复用手动 cache 脚本的同一套实现，避免两条路径行为分叉。
    if kind == "feature":
        from common.cache_features import generate_feature_cache
        generate_feature_cache(cfg, split=split, overwrite=True, logger=logger)
    elif kind == "pseudo":
        from common.cache_pseudo import generate_pseudo_cache
        generate_pseudo_cache(cfg, overwrite=True, logger=logger)
    else:
        raise ValueError(f"Unknown cache kind: {kind}")

    complete, reason = cache_status(cfg, kind, split=split)
    if not complete:
        raise RuntimeError(f"Cache generation finished but cache is still incomplete: {cache_name} | {reason}")
    logger(f"[Cache] {cache_name} ready | {reason}")


class Logger:
    def __init__(self, path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.file = self.path.open("w", encoding="utf-8")

    def log(self, text=""):
        print(text)
        self.file.write(str(text) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
