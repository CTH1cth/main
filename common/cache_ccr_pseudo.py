import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import (
    build_image_items,
    ccr_cache_dir,
    ccr_manifest_path,
    ensure_dir,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
    write_jsonl,
)


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


CCR_TENSOR_FIELDS = (
    "p_fixed",
    "p_despl",
    "p_corr",
    "agree_fg",
    "agree_bg",
    "raw_expand",
    "raw_shrink",
    "trusted_expand",
    "trusted_shrink",
    "anchor_fg",
    "anchor_bg",
)


def resize_chw(tensor, size, mode="bilinear"):
    tensor = tensor.float().unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(tensor, size=(size, size), mode=mode)
    else:
        out = F.interpolate(tensor, size=(size, size), mode=mode, align_corners=False)
    return out.squeeze(0)


def qra_manifest_path_from_ccr(cfg):
    return Path(cfg.CCR_SOURCE_QRA_ROOT) / cfg.BACKBONE_KEY / "manifest_train.jsonl"


def load_feature(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Feature cache key mismatch: {row['cache_path']}")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3:
        raise RuntimeError(f"Feature tensor must be [C,H,W], got {list(tensor.shape)}")
    return tensor, payload


def load_qra(row, dataset, stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"QRA cache key mismatch: {row['cache_path']}")
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"QRA backbone mismatch: {payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    for name in ("p_fixed", "p_despl"):
        if name not in payload:
            raise KeyError(f"QRA cache missing {name}: {row['cache_path']}")
        if list(payload[name].shape) != [1, int(cfg.CCR_LOSS_SIZE), int(cfg.CCR_LOSS_SIZE)]:
            raise RuntimeError(f"QRA {name} shape mismatch: {row['cache_path']}")
    return payload


def bool_area(mask):
    return float(mask.float().mean().item())


def binary_iou(first, second):
    first_np = first.detach().cpu().numpy().astype(bool)
    second_np = second.detach().cpu().numpy().astype(bool)
    union = np.logical_or(first_np, second_np).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_np, second_np).sum() / union)


def normalized_feature_68(feature, size):
    feature_68 = resize_chw(feature, size, mode="bilinear")
    flat = feature_68.permute(1, 2, 0).reshape(size * size, -1)
    return F.normalize(flat, dim=1, p=2)


def prototype_sim(feat, mask, size):
    flat_mask = mask.reshape(-1)
    if int(flat_mask.sum().item()) == 0:
        return None
    proto = feat[flat_mask].mean(dim=0)
    proto = F.normalize(proto, dim=0, p=2)
    return (feat @ proto).reshape(1, size, size)


def sobel_edge(image_path, size):
    image = Image.open(image_path).convert("L").resize((size, size), RESAMPLE_BICUBIC)
    gray = np.asarray(image, dtype=np.float32) / 255.0
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sx * sx + sy * sy)
    max_value = float(edge.max())
    if max_value > 0:
        edge = edge / max_value
    return torch.from_numpy(edge).unsqueeze(0).float()


def limit_by_score(mask, score, max_pixels):
    count = int(mask.sum().item())
    keep = int(max_pixels)
    if count == 0 or keep >= count:
        return mask
    if keep <= 0:
        return torch.zeros_like(mask)
    flat_mask = mask.reshape(-1)
    flat_score = score.reshape(-1)
    indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
    selected_scores = flat_score[indices]
    top = torch.topk(selected_scores, k=keep, largest=True).indices
    out = torch.zeros_like(flat_mask)
    out[indices[top]] = True
    return out.reshape_as(mask)


def dilated(mask, iterations):
    array = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
    kernel = np.ones((3, 3), dtype=np.uint8)
    out = cv2.dilate(array, kernel, iterations=int(iterations))
    return torch.from_numpy(out.astype(bool)).unsqueeze(0)


def eroded(mask, iterations):
    array = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
    kernel = np.ones((3, 3), dtype=np.uint8)
    out = cv2.erode(array, kernel, iterations=int(iterations))
    return torch.from_numpy(out.astype(bool)).unsqueeze(0)


def foreground_distance(mask):
    array = mask.detach().cpu().numpy().astype(bool).squeeze()
    # For foreground pixels, distance_transform_edt(array) is distance to nearest background.
    return torch.from_numpy(cv2.distanceTransform(array.astype(np.uint8), cv2.DIST_L2, 3)).unsqueeze(0)


def connected_components(mask):
    array = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
    num_labels, _ = cv2.connectedComponents(array, connectivity=8)
    return int(num_labels - 1)


def build_ccr_payload(cfg, item, qra_payload, feature):
    size = int(cfg.CCR_LOSS_SIZE)
    p_fixed = qra_payload["p_fixed"].float().clamp(0.0, 1.0)
    p_despl = qra_payload["p_despl"].float().clamp(0.0, 1.0)
    fixed_bin = p_fixed > float(cfg.CCR_FIXED_BIN_TH)
    despl_bin = p_despl > float(cfg.CCR_DESPL_BIN_TH)

    agree_fg = fixed_bin & despl_bin
    agree_bg = (~fixed_bin) & (~despl_bin)
    raw_expand = (~fixed_bin) & despl_bin
    raw_shrink = fixed_bin & (~despl_bin)

    feat = normalized_feature_68(feature, size)
    sim_fg = prototype_sim(feat, agree_fg.squeeze(0), size)
    sim_bg = prototype_sim(feat, agree_bg.squeeze(0), size)
    if sim_fg is None:
        sim_fg = torch.zeros_like(p_fixed)
        trusted_expand = torch.zeros_like(fixed_bin)
    else:
        if sim_bg is None:
            sim_bg = torch.zeros_like(p_fixed)
        edge = sobel_edge(item["image_path"], size)
        near_fixed = dilated(fixed_bin, int(cfg.CCR_EXPAND_MAX_DIST))
        trusted_expand = (
            raw_expand
            & near_fixed
            & (sim_fg >= float(cfg.CCR_PROTO_SIM_EXPAND_TH))
            & (sim_fg >= sim_bg + float(cfg.CCR_PROTO_SIM_SHRINK_MARGIN))
            & (edge < float(cfg.CCR_EDGE_BLOCK_TH))
        )
        max_expand = int(fixed_bin.sum().item() * float(cfg.CCR_EXPAND_MAX_RATIO))
        trusted_expand = limit_by_score(trusted_expand, sim_fg - sim_bg, max_expand)

    if sim_bg is None:
        sim_bg = torch.zeros_like(p_fixed)
        trusted_shrink = torch.zeros_like(fixed_bin)
    else:
        if sim_fg is None:
            sim_fg = torch.zeros_like(p_fixed)
        core = eroded(fixed_bin, int(cfg.CCR_CORE_ERODE_ITER))
        dist_to_bg = foreground_distance(fixed_bin)
        near_edge = dist_to_bg <= float(cfg.CCR_SHRINK_MAX_DIST)
        trusted_shrink = (
            raw_shrink
            & near_edge
            & (~core)
            & (sim_bg >= sim_fg + float(cfg.CCR_PROTO_SIM_SHRINK_MARGIN))
        )
        fixed_pixels = int(fixed_bin.sum().item())
        max_ratio_pixels = fixed_pixels * float(cfg.CCR_SHRINK_MAX_RATIO)
        max_remaining_pixels = fixed_pixels * (1.0 - float(cfg.CCR_MIN_REMAIN_RATIO))
        max_shrink = int(min(max_ratio_pixels, max_remaining_pixels))
        trusted_shrink = limit_by_score(trusted_shrink, sim_bg - sim_fg, max_shrink)

    p_local = p_fixed.clone()
    if bool(trusted_expand.any().item()):
        p_local[trusted_expand] = torch.maximum(p_fixed[trusted_expand], p_despl[trusted_expand])
    if bool(trusted_shrink.any().item()):
        p_local[trusted_shrink] = torch.minimum(p_fixed[trusted_shrink], p_despl[trusted_shrink])
    keep = float(cfg.CCR_CORR_KEEP_FIXED)
    p_corr = ((1.0 - keep) * p_local + keep * p_fixed).clamp(0.0, 1.0)

    anchor_fg = p_corr >= float(cfg.CCR_TH_FG)
    anchor_bg = p_corr <= float(cfg.CCR_TH_BG)
    anchor_fg = anchor_fg | trusted_expand
    anchor_bg = anchor_bg | trusted_shrink
    anchor_fg = anchor_fg & (~anchor_bg)
    anchor_ratio = bool_area(anchor_fg | anchor_bg)

    quality = int(qra_payload.get("quality", 1))
    if anchor_ratio < 0.05:
        quality = 0

    corr_bin = p_corr > 0.5
    payload = {
        "dataset": item["dataset"],
        "stem": item["stem"],
        "image_path": item["image_path"],
        "original_size": tuple(int(v) for v in qra_payload.get("original_size", (0, 0))),
        "backbone_key": cfg.BACKBONE_KEY,
        "p_fixed": p_fixed.cpu().float(),
        "p_despl": p_despl.cpu().float(),
        "p_corr": p_corr.cpu().float(),
        "agree_fg": agree_fg.cpu().bool(),
        "agree_bg": agree_bg.cpu().bool(),
        "raw_expand": raw_expand.cpu().bool(),
        "raw_shrink": raw_shrink.cpu().bool(),
        "trusted_expand": trusted_expand.cpu().bool(),
        "trusted_shrink": trusted_shrink.cpu().bool(),
        "anchor_fg": anchor_fg.cpu().bool(),
        "anchor_bg": anchor_bg.cpu().bool(),
        "quality": quality,
        "iou_fixed_despl": binary_iou(fixed_bin, despl_bin),
        "fixed_area": bool_area(fixed_bin),
        "despl_area": bool_area(despl_bin),
        "corr_area": bool_area(corr_bin),
        "raw_expand_area": bool_area(raw_expand),
        "raw_shrink_area": bool_area(raw_shrink),
        "trusted_expand_area": bool_area(trusted_expand),
        "trusted_shrink_area": bool_area(trusted_shrink),
        "anchor_ratio": anchor_ratio,
        "num_cc_corr": connected_components(corr_bin),
    }
    return payload


def tensor_panel(tensor, title, size=224, nearest=False):
    array = tensor.detach().cpu().float().squeeze().numpy()
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="L")
    image = image.resize((size, size), RESAMPLE_NEAREST if nearest else RESAMPLE_BICUBIC).convert("RGB")
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def image_panel(image_path, title, size=224):
    image = Image.open(image_path).convert("RGB").resize((size, size), RESAMPLE_BICUBIC)
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def save_visualization(path, item, payload):
    panels = [
        image_panel(item["image_path"], "image"),
        tensor_panel(payload["p_fixed"], "p_fixed"),
        tensor_panel(payload["p_despl"], "p_despl"),
        tensor_panel(payload["p_corr"], "p_corr"),
        tensor_panel(payload["trusted_expand"].float(), "trusted_expand", nearest=True),
        tensor_panel(payload["trusted_shrink"].float(), "trusted_shrink", nearest=True),
        tensor_panel(payload["anchor_fg"].float(), "anchor_fg", nearest=True),
        tensor_panel(payload["anchor_bg"].float(), "anchor_bg", nearest=True),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    ensure_dir(Path(path).parent)
    canvas.save(path)


def manifest_shape(payload):
    return {name: list(payload[name].shape) for name in CCR_TENSOR_FIELDS}


def generate_ccr_cache(cfg, overwrite=False, max_samples=-1, vis_num=0, logger=print):
    if cfg.BACKBONE_KEY != "dinov1-s8":
        raise RuntimeError("CCR-DPL v1 only supports BACKBONE_KEY=dinov1-s8.")
    qra_manifest = qra_manifest_path_from_ccr(cfg)
    if not qra_manifest.exists():
        raise RuntimeError(
            f"QRA source cache missing: {qra_manifest}\n"
            "Run common/cache_qra_pseudo.py first; CCR does not auto-generate QRA."
        )

    out_root = ccr_cache_dir(cfg)
    manifest_path = ccr_manifest_path(cfg)
    ensure_dir(out_root)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    feature_manifest = feature_manifest_path(cfg, "train")
    feature_map = manifest_to_map(read_jsonl(feature_manifest), feature_manifest)
    qra_map = manifest_to_map(read_jsonl(qra_manifest), qra_manifest)
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples >= 0:
        items = items[: int(max_samples)]

    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"ccr_cache_root = {out_root}")
    logger(f"qra_source_manifest = {qra_manifest}")
    logger("train_gt_in_ccr_cache = false")
    logger(f"num_items = {len(items)}")

    rows = []
    for index, item in enumerate(tqdm(items, desc="cache CCR pseudo")):
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)
        if key not in feature_map:
            raise RuntimeError(f"Missing feature cache for {dataset}/{stem}")
        if key not in qra_map:
            raise RuntimeError(f"Missing QRA cache for {dataset}/{stem}")

        out_dir = out_root / dataset
        ensure_dir(out_dir)
        out_path = out_dir / f"{stem}.pt"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")

        feature, _ = load_feature(feature_map[key], dataset, stem)
        qra_payload = load_qra(qra_map[key], dataset, stem, cfg)
        payload = build_ccr_payload(cfg, item, qra_payload, feature)
        torch.save(payload, out_path)

        if index < int(vis_num):
            vis_name = (
                f"{dataset}_{stem}_q{payload['quality']}_"
                f"iou{payload['iou_fixed_despl']:.3f}_"
                f"exp{payload['trusted_expand_area']:.3f}_"
                f"shr{payload['trusted_shrink_area']:.3f}.png"
            )
            save_visualization(out_root / "vis" / vis_name, item, payload)

        rows.append(
            {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "backbone_key": cfg.BACKBONE_KEY,
                "shape": manifest_shape(payload),
                "quality": payload["quality"],
                "iou_fixed_despl": payload["iou_fixed_despl"],
                "fixed_area": payload["fixed_area"],
                "despl_area": payload["despl_area"],
                "corr_area": payload["corr_area"],
                "raw_expand_area": payload["raw_expand_area"],
                "raw_shrink_area": payload["raw_shrink_area"],
                "trusted_expand_area": payload["trusted_expand_area"],
                "trusted_shrink_area": payload["trusted_shrink_area"],
                "anchor_ratio": payload["anchor_ratio"],
                "num_cc_corr": payload["num_cc_corr"],
            }
        )

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_items = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate CCR-DPL pseudo cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--vis_num", type=int, default=0)
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")
    if args.vis_num < 0:
        raise ValueError("--vis_num must be >= 0.")
    cfg = load_config(args.config)
    generate_ccr_cache(
        cfg,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        vis_num=args.vis_num,
        logger=print,
    )


if __name__ == "__main__":
    main()
