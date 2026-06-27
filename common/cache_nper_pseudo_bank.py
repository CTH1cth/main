import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import (  # noqa: E402
    build_image_items,
    ensure_dir,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    nper_pseudo_bank_dir,
    nper_pseudo_bank_manifest_path,
    read_jsonl,
    torch_load,
    write_jsonl,
)
from nper.pseudo_bank import (  # noqa: E402
    build_nper_payload,
    load_dino_for_pseudo,
    resolve_device,
    save_payload_vis,
)
from nper.native_cue import NativeCueExtractor  # noqa: E402


def fixed_manifest_path(cfg):
    return Path(cfg.FIXED_PSEUDO_ROOT) / cfg.BACKBONE_KEY / "manifest_train.jsonl"


def load_feature(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Feature cache key mismatch: {row['cache_path']}")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3:
        raise RuntimeError(f"Feature tensor must be [C,H,W], got {list(tensor.shape)}")
    return tensor


def load_fixed(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Fixed pseudo key mismatch: {row['cache_path']}")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"Fixed pseudo tensor must be [1,H,W], got {list(tensor.shape)}")
    return tensor, payload


def payload_shape(payload):
    names = ("p_fixed", "p_despl", "p_gcm", "p_init", "anchor_fg", "anchor_bg", "pixel_weight")
    return {name: list(payload[name].shape) for name in names}


def generate_nper_pseudo_bank(cfg, overwrite=False, max_samples=-1, vis_num=0, device_name="auto", logger=print):
    if cfg.BACKBONE_KEY != "dinov1-s8":
        raise RuntimeError("NPER pseudo bank v1 currently supports BACKBONE_KEY=dinov1-s8 only.")
    device = resolve_device(device_name)
    bank_root = nper_pseudo_bank_dir(cfg)
    manifest_path = nper_pseudo_bank_manifest_path(cfg)
    ensure_dir(bank_root)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    if not items:
        raise RuntimeError("No training images found for NPER pseudo bank.")

    fixed_manifest = fixed_manifest_path(cfg)
    p_init_mode = str(getattr(cfg, "P_INIT_MODE", "quality_fusion"))
    fixed_only = p_init_mode == "fixed_only"
    need_feature = bool(getattr(cfg, "PSEUDO_USE_DESPL", True)) and not fixed_only
    need_dino = bool(getattr(cfg, "PSEUDO_USE_GCM", True)) and not fixed_only
    need_mnp = bool(getattr(cfg, "USE_MNP", True)) and not fixed_only
    feature_manifest = feature_manifest_path(cfg, "train")
    feature_map = manifest_to_map(read_jsonl(feature_manifest), feature_manifest) if need_feature else {}
    fixed_map = manifest_to_map(read_jsonl(fixed_manifest), fixed_manifest)

    dino_model = load_dino_for_pseudo(cfg, device) if need_dino else None
    native_extractor = NativeCueExtractor(cfg) if need_mnp else None
    logger(f"device = {device}")
    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"dino_model_path = {cfg.DINO_MODEL_PATH}")
    logger(f"p_init_mode = {p_init_mode}")
    logger(f"use_despl = {bool(getattr(cfg, 'PSEUDO_USE_DESPL', True))}")
    logger(f"use_gcm = {bool(getattr(cfg, 'PSEUDO_USE_GCM', True))}")
    logger(f"use_teacher = {bool(getattr(cfg, 'PSEUDO_USE_TEACHER', True))}")
    logger(f"feature_manifest = {feature_manifest if need_feature else 'not_used'}")
    logger(f"fixed_manifest = {fixed_manifest}")
    logger(f"pseudo_bank_root = {bank_root}")
    logger(f"num_items = {len(items)}")
    logger("train_gt_used = false")

    rows = []
    for index, item in enumerate(tqdm(items, desc="cache nper pseudo bank")):
        key = (item["dataset"], item["stem"])
        if need_feature and key not in feature_map:
            raise RuntimeError(f"Feature cache missing for {key}")
        if key not in fixed_map:
            raise RuntimeError(f"Fixed pseudo cache missing for {key}")
        feature = load_feature(feature_map[key], item["dataset"], item["stem"]) if need_feature else None
        fixed, fixed_payload = load_fixed(fixed_map[key], item["dataset"], item["stem"])
        payload = build_nper_payload(
            cfg,
            item,
            fixed,
            feature,
            dino_model,
            device,
            native_extractor=native_extractor,
        )
        payload["original_size"] = tuple(int(v) for v in fixed_payload.get("original_size", (0, 0)))

        out_dir = bank_root / item["dataset"]
        ensure_dir(out_dir)
        out_path = out_dir / f"{item['stem']}.pt"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")
        torch.save(payload, out_path)
        if index < int(vis_num):
            save_payload_vis(bank_root / "vis" / item["dataset"] / f"{item['stem']}.png", payload)
        rows.append(
            {
                "dataset": item["dataset"],
                "stem": item["stem"],
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "backbone_key": cfg.BACKBONE_KEY,
                "shape": payload_shape(payload),
                "quality_score": payload["quality_score"],
                "hard_score": payload["hard_score"],
            }
        )

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_rows = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate NPER-UCOD-V1 pseudo bank.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--vis_num", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_nper_pseudo_bank(
        cfg,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        vis_num=args.vis_num,
        device_name=args.device,
        logger=print,
    )


if __name__ == "__main__":
    main()
