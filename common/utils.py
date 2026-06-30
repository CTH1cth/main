import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


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


def pseudo_manifest_path(cfg):
    # fixed pseudo 只用于训练集，因此只有 train manifest。
    return Path(cfg.CACHE_ROOT) / "pseudo_label_cache" / cfg.BACKBONE_KEY / "manifest_train.jsonl"


def ml_feature_cache_dir(cfg):
    root = getattr(cfg, "MULTI_LEVEL_FEATURE_ROOT", "../datasets/cache/features_cache_ml")
    return Path(root) / cfg.BACKBONE_KEY


def ml_feature_cache_manifest_path(cfg, split):
    return ml_feature_cache_dir(cfg) / f"manifest_{split}.jsonl"


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
    labels = [f"l{layer}" for layer in layers]
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
