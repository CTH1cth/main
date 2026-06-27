import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import (  # noqa: E402
    build_image_items,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
)
from nper.online_dino_key import OnlineDINOKeyExtractor  # noqa: E402


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def select_items(cfg, dataset=None, stem=None, max_samples=-1):
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if dataset is not None:
        items = [item for item in items if item["dataset"] == dataset]
    if stem is not None:
        items = [item for item in items if item["stem"] == stem]
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    if not items:
        raise RuntimeError("No samples selected for DINO feature comparison.")
    return items


def load_cached_feature(cache_map, item, cfg):
    key = (item["dataset"], item["stem"])
    if key not in cache_map:
        raise RuntimeError(f"Feature cache missing for {key}")
    row = cache_map[key]
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != item["dataset"] or payload.get("stem") != item["stem"]:
        raise RuntimeError(f"Feature cache key mismatch: {row['cache_path']}")
    tensor = payload["tensor"].float()
    expected = [int(getattr(cfg, "DINO_HIDDEN_SIZE", 384)), 37, 37]
    if list(tensor.shape) != expected:
        raise RuntimeError(f"Cached feature shape mismatch for {key}: {list(tensor.shape)} != {expected}")
    return tensor


@torch.no_grad()
def compare_online_dino_with_cache(cfg, dataset=None, stem=None, max_samples=-1, device_name="auto", logger=print):
    device = resolve_device(device_name)
    manifest_path = feature_manifest_path(cfg, "train")
    cache_map = manifest_to_map(read_jsonl(manifest_path), manifest_path)
    items = select_items(cfg, dataset=dataset, stem=stem, max_samples=max_samples)
    extractor = OnlineDINOKeyExtractor(cfg).to(device)

    rows = []
    logger(f"device = {device}")
    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"feature_manifest = {manifest_path}")
    logger(f"online_key_hook = {extractor.key_path}")
    logger(f"feature_input_size = {int(cfg.DINO['feature_input_size'])}")
    logger("| dataset | stem | online_shape | cache_shape | cosine | mean_abs_diff | max_abs_diff |")
    logger("|---|---|---|---|---:|---:|---:|")
    for item in items:
        online = extractor([item["image_path"]], device=device).squeeze(0).detach().cpu().float()
        cached = load_cached_feature(cache_map, item, cfg)
        if list(online.shape) != list(cached.shape):
            raise RuntimeError(
                f"Online/cache shape mismatch for {item['dataset']}/{item['stem']}: "
                f"{list(online.shape)} != {list(cached.shape)}"
            )
        cosine = F.cosine_similarity(online.flatten(), cached.flatten(), dim=0).item()
        diff = (online - cached).abs()
        mean_abs = diff.mean().item()
        max_abs = diff.max().item()
        row = {
            "dataset": item["dataset"],
            "stem": item["stem"],
            "shape": list(online.shape),
            "cosine": cosine,
            "mean_abs_diff": mean_abs,
            "max_abs_diff": max_abs,
        }
        rows.append(row)
        logger(
            f"| {item['dataset']} | {item['stem']} | {list(online.shape)} | {list(cached.shape)} | "
            f"{cosine:.8f} | {mean_abs:.8e} | {max_abs:.8e} |"
        )

    cosines = torch.tensor([row["cosine"] for row in rows], dtype=torch.float64)
    mean_abs = torch.tensor([row["mean_abs_diff"] for row in rows], dtype=torch.float64)
    max_abs = torch.tensor([row["max_abs_diff"] for row in rows], dtype=torch.float64)
    logger(
        "summary: "
        f"num_samples={len(rows)} "
        f"cosine_mean={cosines.mean().item():.8f} "
        f"cosine_min={cosines.min().item():.8f} "
        f"mean_abs_diff_mean={mean_abs.mean().item():.8e} "
        f"max_abs_diff_max={max_abs.max().item():.8e}"
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare online DINO key features with cached feature tensors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--stem", default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    compare_online_dino_with_cache(
        cfg,
        dataset=args.dataset,
        stem=args.stem,
        max_samples=args.max_samples,
        device_name=args.device,
        logger=print,
    )


if __name__ == "__main__":
    main()
