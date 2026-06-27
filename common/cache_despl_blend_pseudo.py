import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import (  # noqa: E402
    build_image_items,
    despl_light_cache_dir,
    despl_light_cache_manifest_path,
    despl_pseudo_bank_manifest_path,
    ensure_dir,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
    write_jsonl,
)


FORMULA = "resize(clamp(0.8*p_despl + 0.2*p_fixed, 0, 1), 68)"


def _single_channel(payload, name, cache_path):
    if name not in payload:
        raise KeyError(f"Source DESPL payload missing {name}: {cache_path}")
    tensor = payload[name]
    if not torch.is_tensor(tensor):
        raise TypeError(f"Source DESPL {name} must be tensor: {cache_path}")
    tensor = tensor.float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"Source DESPL {name} must be [1,H,W], got {list(tensor.shape)}: {cache_path}")
    return tensor


def _resize_to_loss(tensor, loss_size):
    if list(tensor.shape[-2:]) == [int(loss_size), int(loss_size)]:
        return tensor.float().contiguous()
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=(int(loss_size), int(loss_size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).contiguous()


def load_source_payload(row, item, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Source DESPL payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != item["dataset"] or payload.get("stem") != item["stem"]:
        raise RuntimeError(f"Source DESPL key mismatch: {row['cache_path']}")
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Source DESPL backbone mismatch: {payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    return payload


def build_light_payload(cfg, item, source_row):
    source = load_source_payload(source_row, item, cfg)
    p_fixed = _single_channel(source, "p_fixed", source_row["cache_path"])
    p_despl = _single_channel(source, "p_despl", source_row["cache_path"])
    if list(p_fixed.shape) != list(p_despl.shape):
        p_fixed = F.interpolate(
            p_fixed.unsqueeze(0),
            size=p_despl.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    despl_weight = float(getattr(cfg, "P_INIT_DESPL_WEIGHT", 0.8))
    fixed_weight = float(getattr(cfg, "P_INIT_FIXED_WEIGHT", 0.2))
    p_init_source = (despl_weight * p_despl + fixed_weight * p_fixed).clamp(0.0, 1.0)
    p_init = _resize_to_loss(p_init_source, int(cfg.LOSS_SIZE))
    p_fixed_68 = _resize_to_loss(p_fixed, int(cfg.LOSS_SIZE))
    p_despl_68 = _resize_to_loss(p_despl, int(cfg.LOSS_SIZE))

    return {
        "dataset": item["dataset"],
        "stem": item["stem"],
        "image_path": item["image_path"],
        "backbone_key": cfg.BACKBONE_KEY,
        "tensor": p_init,
        "p_init": p_init,
        "p_fixed_68": p_fixed_68,
        "p_despl_68": p_despl_68,
        "shape": list(p_init.shape),
        "source_shape": list(p_despl.shape),
        "source_cache_path": str(Path(source_row["cache_path"]).resolve()),
        "formula": FORMULA,
        "p_init_mode": getattr(cfg, "P_INIT_MODE", "despl_fixed_blend"),
        "p_init_despl_weight": despl_weight,
        "p_init_fixed_weight": fixed_weight,
        "p_fixed_area": float(p_fixed.mean().item()),
        "p_despl_area": float(p_despl.mean().item()),
        "p_init_area": float(p_init.mean().item()),
        "source_p_init_area": float(p_init_source.mean().item()),
    }


def generate_despl_blend_cache(cfg, overwrite=False, max_samples=-1, logger=print):
    source_manifest = despl_pseudo_bank_manifest_path(cfg)
    source_map = manifest_to_map(read_jsonl(source_manifest), source_manifest)
    out_root = despl_light_cache_dir(cfg)
    manifest_path = despl_light_cache_manifest_path(cfg)
    ensure_dir(out_root)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    if not items:
        raise RuntimeError("No training images selected for DESPL light cache.")

    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"source_manifest = {source_manifest}")
    logger(f"output_root = {out_root}")
    logger(f"loss_size = {int(cfg.LOSS_SIZE)}")
    logger(f"formula = {FORMULA}")
    logger(f"num_items = {len(items)}")
    logger("train_gt_used = false")

    rows = []
    for item in tqdm(items, desc="cache DESPL blend pseudo"):
        key = (item["dataset"], item["stem"])
        if key not in source_map:
            raise RuntimeError(f"Source DESPL pseudo missing for {key}")
        payload = build_light_payload(cfg, item, source_map[key])
        out_dir = out_root / item["dataset"]
        ensure_dir(out_dir)
        out_path = out_dir / f"{item['stem']}.pt"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")
        torch.save(payload, out_path)
        rows.append(
            {
                "dataset": item["dataset"],
                "stem": item["stem"],
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "backbone_key": cfg.BACKBONE_KEY,
                "shape": list(payload["tensor"].shape),
                "source_shape": payload["source_shape"],
                "source_cache_path": payload["source_cache_path"],
                "formula": payload["formula"],
            }
        )

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_rows = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate lightweight DESPL-primary blend pseudo cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_despl_blend_cache(
        cfg,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        logger=print,
    )


if __name__ == "__main__":
    main()
