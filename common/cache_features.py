import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

if __package__ in {None, ""}:
    # 支持从 main/common 目录直接执行：python cache_features.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import build_image_items, ensure_dir, load_config, write_jsonl


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(image_path, size, interpolation="bicubic"):
    # DINO 特征提取使用固定方形输入和 ImageNet normalize。
    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)
    interpolation = str(interpolation).strip().lower()
    try:
        resampling = Image.Resampling
    except AttributeError:
        resampling = Image
    modes = {
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
    }
    if interpolation not in modes:
        raise ValueError(f"Unsupported feature resize interpolation: {interpolation}")
    image = image.resize((size, size), modes[interpolation])
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0), original_size


def resolve_key_projection_at_layer(model, layer_index):
    """Resolve one ViT attention-key projection by zero-based encoder index."""
    layers = getattr(getattr(model, "encoder", None), "layer", None)
    if layers is None:
        raise AttributeError("DINO model has no encoder.layer sequence.")
    layer_index = int(layer_index)
    if layer_index < 0:
        layer_index += len(layers)
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(
            f"DINO key projection layer index out of range: {layer_index} for {len(layers)} layers."
        )
    try:
        module = layers[layer_index].attention.attention.key
    except AttributeError as exc:
        raise AttributeError(
            f"Could not resolve attention key projection for encoder layer {layer_index}."
        ) from exc
    return module, f"encoder.layer[{layer_index}].attention.attention.key"


def resolve_key_projection(model):
    # DINOv1/DINOv2 在 transformers 中都优先取最后一层 attention key 投影。
    try:
        return resolve_key_projection_at_layer(model, -1)
    except AttributeError:
        pass

    candidates = []
    for name, module in model.named_modules():
        lname = name.lower()
        if lname.endswith(".attention.attention.key") or lname.endswith(".attention.key") or lname.endswith(".key"):
            candidates.append((name, module))
    if not candidates:
        raise AttributeError("Could not find DINO key projection module.")
    name, module = candidates[-1]
    return module, name


def call_dino(model, inputs):
    # 兼容不同 transformers ViT forward 参数，必要时开启位置编码插值。
    params = inspect.signature(model.forward).parameters
    kwargs = {}
    if "interpolate_pos_encoding" in params:
        kwargs["interpolate_pos_encoding"] = True
    if "output_attentions" in params:
        kwargs["output_attentions"] = False
    return model(inputs, **kwargs)


def load_dino(cfg, device):
    # 只从 CTH 内配置好的本地 HuggingFace 权重目录加载 DINO。
    dino_cfg = cfg.DINO
    model_path = Path(dino_cfg["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Local DINO weight path not found: {model_path}")
    try:
        model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            add_pooling_layer=False,
        )
    except TypeError:
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    return model


def split_datasets(cfg, split):
    # cache_features 支持 train/val/test 三种 split。
    if split == "train":
        return cfg.TRAIN_DATASETS
    if split == "val":
        return cfg.VAL_DATASETS
    if split == "test":
        return cfg.TEST_DATASETS
    raise ValueError(f"Unknown split: {split}")


def _load_sample_keys(path):
    """Read an optional deterministic subset without changing dataset order."""
    if path is None:
        return None
    sample_path = Path(path).expanduser().resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(f"Sample list not found: {sample_path}")
    keys = []
    for line_number, line in enumerate(sample_path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            row = json.loads(line)
            key = (str(row["dataset"]), str(row["stem"]))
        else:
            fields = line.replace("/", " ").split()
            if len(fields) < 2:
                raise ValueError(
                    f"Expected JSONL or 'dataset stem' at {sample_path}:{line_number}"
                )
            key = (fields[0], fields[1])
        if key in keys:
            raise RuntimeError(f"Duplicate sample key at {sample_path}:{line_number}: {key}")
        keys.append(key)
    if not keys:
        raise RuntimeError(f"Sample list is empty: {sample_path}")
    return keys


def _select_items(items, sample_list=None, max_samples=-1):
    keys = _load_sample_keys(sample_list)
    if keys is not None:
        item_map = {(str(item["dataset"]), str(item["stem"])): item for item in items}
        missing = [key for key in keys if key not in item_map]
        if missing:
            raise KeyError(f"Sample list keys missing from split, first 10: {missing[:10]}")
        items = [item_map[key] for key in keys]
    if int(max_samples) >= 0:
        items = items[: int(max_samples)]
    return items


def key_to_feature_tensor(key):
    _, n_tokens, channels = key.shape
    grid = int(math.sqrt(n_tokens - 1))
    if grid * grid != n_tokens - 1:
        raise RuntimeError(f"Non-square DINO patch token count: {n_tokens - 1}")
    # 只保存 patch-level key feature；不做 L2 normalize、PCA 或后处理。
    patch_tokens = key[:, 1:, :].reshape(1, grid, grid, channels)
    feature = patch_tokens.permute(0, 3, 1, 2).squeeze(0).detach().cpu().float()
    return feature


def generate_feature_cache(
    cfg,
    split,
    overwrite=False,
    resume=False,
    logger=print,
    sample_list=None,
    max_samples=-1,
):
    # 生成 frozen DINO key feature cache，并写对应 split 的 manifest。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_cfg = cfg.DINO
    input_size = int(dino_cfg["feature_input_size"])
    patch_size = int(dino_cfg["patch_size"])
    embed_dim = int(dino_cfg["embed_dim"])
    if input_size % patch_size:
        raise RuntimeError(
            f"feature_input_size={input_size} is not divisible by patch_size={patch_size}"
        )
    expected_grid = input_size // patch_size
    expected_shape = (embed_dim, expected_grid, expected_grid)
    cache_key = str(getattr(cfg, "FEATURE_CACHE_KEY", cfg.BACKBONE_KEY))
    interpolation = str(getattr(cfg, "FEATURE_RESIZE_INTERPOLATION", "bicubic"))
    cache_root = Path(cfg.CACHE_ROOT) / "features_cache" / cache_key
    split_root = cache_root / split
    manifest_path = cache_root / f"manifest_{split}.jsonl"
    ensure_dir(split_root)
    previous_rows = {}
    if manifest_path.exists() and resume and not overwrite:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            previous_rows[(str(row["dataset"]), str(row["stem"]))] = row

    if manifest_path.exists() and not overwrite and not resume:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    model = load_dino(cfg, device)
    key_holder = {"tensor": None}

    def hook_fn_key(_module, _input, output):
        key_holder["tensor"] = output.detach()

    key_module, key_path = resolve_key_projection(model)
    handle = key_module.register_forward_hook(hook_fn_key)
    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"model_path = {dino_cfg['model_path']}")
    logger(f"key_hook = {key_path}")
    logger(f"feature_input_size = {input_size}")
    logger(f"expected_patch_tokens = {expected_grid * expected_grid}")
    logger(f"expected_feature_grid = {expected_grid}x{expected_grid}")
    logger(f"feature_resize_interpolation = {interpolation}")
    logger("feature_postprocess = none")

    rows = []
    items = build_image_items(cfg.DATA_ROOT, split_datasets(cfg, split), require_gt=False)
    items = _select_items(items, sample_list=sample_list, max_samples=max_samples)
    if not items:
        raise RuntimeError("Feature cache selection is empty")
    extraction_started = time.perf_counter()
    try:
        for item_index, item in enumerate(tqdm(items, desc=f"cache features {split}"), 1):
            dataset = item["dataset"]
            stem = item["stem"]
            out_dir = split_root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite and not resume:
                raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")

            if out_path.exists() and resume and not overwrite:
                try:
                    payload = torch.load(out_path, map_location="cpu")
                    feature = payload.get("tensor")
                    if (
                        str(payload.get("dataset")) != str(dataset)
                        or str(payload.get("stem")) != str(stem)
                        or not torch.is_tensor(feature)
                        or tuple(feature.shape) != expected_shape
                        or not bool(torch.isfinite(feature).all())
                    ):
                        raise RuntimeError("identity, shape, or numerical mismatch")
                    prior = previous_rows.get((str(dataset), str(stem)))
                    if prior is not None:
                        rows.append(prior)
                        continue
                    rows.append({
                        "dataset": dataset,
                        "stem": stem,
                        "image_path": item["image_path"],
                        "cache_path": str(out_path.resolve()),
                        "backbone_key": str(cfg.BACKBONE_KEY),
                        "feature_cache_key": cache_key,
                        "resize_interpolation": interpolation,
                        "input_size": input_size,
                        "patch_size": patch_size,
                        "grid_size": expected_grid,
                        "token_count": expected_grid * expected_grid,
                        "feature_dim": embed_dim,
                        "shape": list(feature.shape),
                        "preprocess_seconds": 0.0,
                        "dino_seconds": 0.0,
                        "save_seconds": 0.0,
                        "total_seconds": 0.0,
                        "gpu_peak_mb": 0.0,
                        "cache_bytes": int(out_path.stat().st_size),
                        "status": "reused",
                    })
                    continue
                except Exception as exc:
                    logger(f"invalid_existing_cache = {out_path}: {exc!r}; regenerating")

            item_started = time.perf_counter()
            inputs, original_size = preprocess_image(
                item["image_path"], input_size, interpolation=interpolation
            )
            preprocess_seconds = time.perf_counter() - item_started
            inputs = inputs.to(device)
            key_holder["tensor"] = None
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            dino_started = time.perf_counter()
            with torch.no_grad():
                call_dino(model, inputs)
                key = key_holder["tensor"]
                if key is None:
                    raise RuntimeError("DINO key hook did not capture a tensor.")
                feature = key_to_feature_tensor(key)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dino_seconds = time.perf_counter() - dino_started
            if tuple(feature.shape) != expected_shape:
                raise RuntimeError(
                    f"DINO feature shape mismatch: {tuple(feature.shape)} != {expected_shape}"
                )
            token_count = int(feature.shape[-2] * feature.shape[-1])
            gpu_peak_mb = (
                float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                if device.type == "cuda"
                else 0.0
            )

            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "original_size": original_size,
                "backbone_key": str(cfg.BACKBONE_KEY),
                "feature_cache_key": cache_key,
                "resize_interpolation": interpolation,
                "input_size": input_size,
                "patch_size": patch_size,
                "grid_size": expected_grid,
                "token_count": token_count,
                "feature_dim": embed_dim,
                "tensor": feature,
            }
            save_started = time.perf_counter()
            torch.save(payload, out_path)
            save_seconds = time.perf_counter() - save_started
            total_seconds = time.perf_counter() - item_started
            if item_index <= 5:
                logger(
                    "shape_check "
                    f"{item_index}/5 | image={original_size} | "
                    f"dino_input={[1, 3, input_size, input_size]} | "
                    f"tokens={token_count} | grid={expected_grid}x{expected_grid} | "
                    f"dim={embed_dim}"
                )
            rows.append({
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "backbone_key": str(cfg.BACKBONE_KEY),
                "feature_cache_key": cache_key,
                "resize_interpolation": interpolation,
                "input_size": input_size,
                "patch_size": patch_size,
                "grid_size": expected_grid,
                "token_count": token_count,
                "feature_dim": embed_dim,
                "shape": list(feature.shape),
                "preprocess_seconds": preprocess_seconds,
                "dino_seconds": dino_seconds,
                "save_seconds": save_seconds,
                "total_seconds": total_seconds,
                "gpu_peak_mb": gpu_peak_mb,
                "cache_bytes": int(out_path.stat().st_size),
            })
    finally:
        handle.remove()

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_items = {len(rows)}")
    logger(f"wall_seconds = {time.perf_counter() - extraction_started:.6f}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate frozen DINO key feature cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument(
        "--cache_root",
        default=None,
        help="Optional output cache root override (keeps generated data outside the source repo).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid existing feature files and rebuild the complete manifest.",
    )
    parser.add_argument(
        "--sample_list",
        default=None,
        help="Optional JSONL or 'dataset stem' list for a deterministic subset.",
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cache_root:
        cfg.CACHE_ROOT = str(Path(args.cache_root).expanduser().resolve())
    generate_feature_cache(
        cfg,
        split=args.split,
        overwrite=args.overwrite,
        resume=args.resume,
        logger=print,
        sample_list=args.sample_list,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
