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


def qra_cache_dir(cfg):
    return Path(cfg.QRA_CACHE_ROOT) / cfg.BACKBONE_KEY


def qra_manifest_path(cfg):
    return qra_cache_dir(cfg) / "manifest_train.jsonl"


def ccr_cache_dir(cfg):
    return Path(cfg.CCR_CACHE_ROOT) / cfg.BACKBONE_KEY


def ccr_manifest_path(cfg):
    return ccr_cache_dir(cfg) / "manifest_train.jsonl"


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
