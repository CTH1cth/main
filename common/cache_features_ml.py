import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_features import (  # noqa: E402
    call_dino,
    key_to_feature_tensor,
    load_dino,
    preprocess_image,
    split_datasets,
)
from common.utils import build_image_items, ensure_dir, load_config, write_jsonl  # noqa: E402


def ml_cache_root(cfg, debug_root=None):
    root = Path(debug_root) if debug_root else Path(cfg.MULTI_LEVEL_FEATURE_ROOT)
    return root / cfg.BACKBONE_KEY


def resolve_layer_key_projection(model, layer_id):
    hook_idx = int(layer_id) - 1
    if hook_idx < 0:
        raise ValueError(f"MULTI_LEVEL_LAYERS must be 1-based positive integers, got {layer_id}")
    try:
        module = model.encoder.layer[hook_idx].attention.attention.key
    except (AttributeError, IndexError) as exc:
        raise AttributeError(
            f"Could not find DINO key projection for 1-based layer {layer_id} "
            f"(hook index {hook_idx})."
        ) from exc
    return module, f"encoder.layer[{hook_idx}].attention.attention.key"


def generate_ml_feature_cache(
    cfg,
    split,
    overwrite=False,
    max_samples=-1,
    debug_root=None,
    dtype=None,
    logger=print,
):
    if str(getattr(cfg, "MULTI_LEVEL_FEATURE_TYPE", "key")) != "key":
        raise RuntimeError("cache_features_ml.py currently supports MULTI_LEVEL_FEATURE_TYPE='key' only.")
    dtype_name = str(dtype or getattr(cfg, "MULTI_LEVEL_FEATURE_DTYPE", "float32")).lower()
    if dtype_name not in {"float16", "float32"}:
        raise ValueError(f"--dtype must be float16 or float32, got {dtype_name}")
    save_dtype = torch.float16 if dtype_name == "float16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_cfg = cfg.DINO
    input_size = int(dino_cfg["feature_input_size"])
    layers = [int(layer) for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [4, 8, 12])]
    labels = [f"l{layer}" for layer in layers]
    cache_root = ml_cache_root(cfg, debug_root=debug_root)
    split_root = cache_root / split
    manifest_path = cache_root / f"manifest_{split}.jsonl"
    ensure_dir(split_root)

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    model = load_dino(cfg, device)
    key_holders = {label: {"tensor": None} for label in labels}
    handles = []

    def make_hook(label):
        def hook_fn(_module, _input, output):
            key_holders[label]["tensor"] = output.detach()

        return hook_fn

    for layer, label in zip(layers, labels):
        module, key_path = resolve_layer_key_projection(model, layer)
        handles.append(module.register_forward_hook(make_hook(label)))
        logger(f"key_hook_{label} = {key_path}")

    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"model_path = {dino_cfg['model_path']}")
    logger(f"feature_input_size = {input_size}")
    logger(f"multi_level_layers = {layers}")
    logger("feature_type = key")
    logger(f"feature_dtype = {dtype_name}")
    logger("feature_postprocess = none")
    logger(f"output_root = {cache_root}")
    if debug_root:
        logger("debug_root_active = true")

    rows = []
    items = build_image_items(cfg.DATA_ROOT, split_datasets(cfg, split), require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    try:
        for item in tqdm(items, desc=f"cache ml features {split}"):
            dataset = item["dataset"]
            stem = item["stem"]
            out_dir = split_root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")

            inputs, original_size = preprocess_image(item["image_path"], input_size)
            inputs = inputs.to(device)
            for holder in key_holders.values():
                holder["tensor"] = None
            with torch.no_grad():
                call_dino(model, inputs)
                features = {}
                for label in labels:
                    key = key_holders[label]["tensor"]
                    if key is None:
                        raise RuntimeError(f"DINO key hook did not capture a tensor for {label}.")
                    features[label] = key_to_feature_tensor(key).to(dtype=save_dtype)

            last_label = labels[-1]
            shape = {label: list(features[label].shape) for label in labels}
            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "original_size": original_size,
                "backbone_key": cfg.BACKBONE_KEY,
                "tensor": features[last_label],
                "features": features,
                "layers": layers,
                "feature_type": "key",
                "dtype": dtype_name,
                "shape": shape,
            }
            torch.save(payload, out_path)
            rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "image_path": item["image_path"],
                    "cache_path": str(out_path.resolve()),
                    "backbone_key": cfg.BACKBONE_KEY,
                    "layers": layers,
                    "feature_type": "key",
                    "dtype": dtype_name,
                    "shape": shape,
                }
            )
    finally:
        for handle in handles:
            handle.remove()

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_items = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate multi-level frozen DINO key feature cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--debug_root", default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_ml_feature_cache(
        cfg,
        split=args.split,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        debug_root=args.debug_root,
        dtype=args.dtype,
        logger=print,
    )


if __name__ == "__main__":
    main()
