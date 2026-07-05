import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import build_image_items, ensure_dir, load_config, read_jsonl, torch_load, write_jsonl  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _hflip_image(image):
    transpose = getattr(Image, "Transpose", None)
    if transpose is not None:
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def preprocess_hflip_image(image_path, size):
    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)
    image = _hflip_image(image)
    image = image.resize((size, size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0), original_size


def generate_hflip_feature_cache(cfg, split="train", force=False, max_samples=-1, logger=print):
    from common.cache_features import (  # noqa: WPS433
        call_dino,
        key_to_feature_tensor,
        load_dino,
        resolve_key_projection,
        split_datasets,
    )

    if split != "train":
        raise ValueError(f"HFlip feature cache is train-only, got split={split}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_cfg = cfg.DINO
    input_size = int(dino_cfg["feature_input_size"])
    cache_root = Path(getattr(cfg, "HFLIP_FEATURE_CACHE_ROOT", "../datasets/cache/features_cache_hflip"))
    cache_root = cache_root / cfg.BACKBONE_KEY
    manifest_path = cache_root / "manifest_train.jsonl"
    ensure_dir(cache_root)

    existing_map = {}
    manifest_exists = manifest_path.exists()
    if manifest_exists and not force:
        for row in read_jsonl(manifest_path):
            key = (row.get("dataset"), row.get("stem"))
            if None in key:
                raise RuntimeError(f"Bad manifest row without dataset/stem in {manifest_path}")
            if key in existing_map:
                raise RuntimeError(f"Duplicate manifest key in {manifest_path}: {key}")
            existing_map[key] = row

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
    logger("feature_view = hflip_image_then_dino")
    logger("feature_postprocess = none")
    logger(f"hflip_feature_cache_root = {cache_root}")

    rows = []
    items = build_image_items(cfg.DATA_ROOT, split_datasets(cfg, split), require_gt=False)
    if int(max_samples) >= 0:
        items = items[: int(max_samples)]
    generated = 0
    skipped = 0
    try:
        for item in tqdm(items, desc="cache hflip features train"):
            dataset = item["dataset"]
            stem = item["stem"]
            key = (dataset, stem)
            out_dir = cache_root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            existing_row = existing_map.get(key)
            if existing_row is not None and Path(existing_row.get("cache_path", "")).exists():
                rows.append(existing_row)
                skipped += 1
                continue
            if out_path.exists() and not force:
                payload = torch_load(out_path, map_location="cpu")
                if not isinstance(payload, dict):
                    raise RuntimeError(f"Existing HFlip cache payload must be dict: {out_path}")
                if payload.get("dataset") != dataset or payload.get("stem") != stem:
                    raise RuntimeError(f"Existing HFlip cache key mismatch: {out_path}")
                tensor = payload.get("tensor")
                if not torch.is_tensor(tensor) or tensor.ndim != 3:
                    raise RuntimeError(f"Existing HFlip cache tensor must be [C,H,W]: {out_path}")
                rows.append(
                    {
                        "dataset": dataset,
                        "stem": stem,
                        "image_path": item["image_path"],
                        "cache_path": str(out_path.resolve()),
                        "view": payload.get("view", "hflip"),
                        "shape": list(tensor.shape),
                    }
                )
                skipped += 1
                continue

            inputs, original_size = preprocess_hflip_image(item["image_path"], input_size)
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
                "view": "hflip",
                "tensor": feature,
            }
            torch.save(payload, out_path)
            rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "image_path": item["image_path"],
                    "cache_path": str(out_path.resolve()),
                    "view": "hflip",
                    "shape": list(feature.shape),
                }
            )
            generated += 1
    finally:
        handle.remove()

    if force or generated > 0 or not manifest_exists:
        write_jsonl(manifest_path, rows)
        logger(f"wrote_manifest = {manifest_path}")
    else:
        logger(f"manifest_unchanged = {manifest_path}")
    logger(f"num_items = {len(rows)}")
    logger(f"num_generated = {generated}")
    logger(f"num_reused = {skipped}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate HFlip frozen DINO key feature cache for train split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--force", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")

    cfg = load_config(args.config)
    generate_hflip_feature_cache(
        cfg,
        split=args.split,
        force=bool(args.force),
        max_samples=int(args.max_samples),
        logger=print,
    )


if __name__ == "__main__":
    main()
