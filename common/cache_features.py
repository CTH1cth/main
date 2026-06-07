import argparse
import inspect
import math
import sys
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


def preprocess_image(image_path, size):
    # DINO 特征提取使用固定方形输入和 ImageNet normalize。
    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)
    image = image.resize((size, size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0), original_size


def resolve_key_projection(model):
    # DINOv1/DINOv2 在 transformers 中都优先取最后一层 attention key 投影。
    try:
        return model.encoder.layer[-1].attention.attention.key, "encoder.layer[-1].attention.attention.key"
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


def key_to_feature_tensor(key):
    _, n_tokens, channels = key.shape
    grid = int(math.sqrt(n_tokens - 1))
    if grid * grid != n_tokens - 1:
        raise RuntimeError(f"Non-square DINO patch token count: {n_tokens - 1}")
    # 只保存 patch-level key feature；不做 L2 normalize、PCA 或后处理。
    patch_tokens = key[:, 1:, :].reshape(1, grid, grid, channels)
    feature = patch_tokens.permute(0, 3, 1, 2).squeeze(0).detach().cpu().float()
    return feature


def generate_feature_cache(cfg, split, overwrite=False, logger=print):
    # 生成 frozen DINO key feature cache，并写对应 split 的 manifest。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_cfg = cfg.DINO
    input_size = int(dino_cfg["feature_input_size"])
    cache_root = Path(cfg.CACHE_ROOT) / "features_cache" / cfg.BACKBONE_KEY
    split_root = cache_root / split
    manifest_path = cache_root / f"manifest_{split}.jsonl"
    ensure_dir(split_root)

    if manifest_path.exists() and not overwrite:
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
    logger("feature_postprocess = none")

    rows = []
    items = build_image_items(cfg.DATA_ROOT, split_datasets(cfg, split), require_gt=False)
    try:
        for item in tqdm(items, desc=f"cache features {split}"):
            dataset = item["dataset"]
            stem = item["stem"]
            out_dir = split_root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")

            inputs, original_size = preprocess_image(item["image_path"], input_size)
            inputs = inputs.to(device)
            key_holder["tensor"] = None
            with torch.no_grad():
                call_dino(model, inputs)
                key = key_holder["tensor"]
                if key is None:
                    raise RuntimeError("DINO key hook did not capture a tensor.")
                feature = key_to_feature_tensor(key)

            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "original_size": original_size,
                "tensor": feature,
            }
            torch.save(payload, out_path)
            rows.append({
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "shape": list(feature.shape),
            })
    finally:
        handle.remove()

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_items = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate frozen DINO key feature cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_feature_cache(cfg, split=args.split, overwrite=args.overwrite, logger=print)


if __name__ == "__main__":
    main()
