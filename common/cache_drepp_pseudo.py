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

from common.utils import (  # noqa: E402
    build_image_items,
    despl_pseudo_bank_manifest_path,
    ensure_dir,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
    write_jsonl,
)


FORMULA = "memory_init=p_despl; fixed_local_only; global_blend=false"

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


def resize_chw(tensor, size, mode="bilinear"):
    tensor = tensor.float().unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(tensor, size=(int(size), int(size)), mode=mode)
    else:
        out = F.interpolate(tensor, size=(int(size), int(size)), mode=mode, align_corners=False)
    return out.squeeze(0).contiguous()


def load_feature(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Feature cache key mismatch: {row['cache_path']}")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3:
        raise RuntimeError(f"Feature tensor must be [C,H,W], got {list(tensor.shape)}")
    return tensor


def load_source_pseudo(row, item, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DRE++ source pseudo payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != item["dataset"] or payload.get("stem") != item["stem"]:
        raise RuntimeError(f"DRE++ source pseudo key mismatch: {row['cache_path']}")
    if payload.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"DRE++ source pseudo backbone mismatch: {payload.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    out = {}
    for name in ("p_despl", "p_fixed"):
        tensor = payload.get(name)
        if not torch.is_tensor(tensor):
            raise TypeError(f"DRE++ source pseudo missing tensor {name}: {row['cache_path']}")
        tensor = tensor.float()
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise RuntimeError(f"DRE++ source {name} must be [1,H,W], got {list(tensor.shape)}")
        out[name] = tensor
    return out


def dilate_mask(mask, radius):
    if int(radius) <= 0:
        return mask.bool()
    kernel = int(radius) * 2 + 1
    pooled = F.max_pool2d(mask.float().unsqueeze(0), kernel_size=kernel, stride=1, padding=int(radius))
    return pooled.squeeze(0) > 0.5


def erode_mask(mask, radius):
    if int(radius) <= 0:
        return mask.bool()
    kernel = int(radius) * 2 + 1
    pooled = -F.max_pool2d(-mask.float().unsqueeze(0), kernel_size=kernel, stride=1, padding=int(radius))
    return pooled.squeeze(0) > 0.5


def feature_similarity(feature, fg_mask, size):
    feature = resize_chw(feature, size, mode="bilinear")
    feature = F.normalize(feature.float(), dim=0, p=2)
    mask = fg_mask.bool().squeeze(0)
    if int(mask.sum().item()) == 0:
        return torch.zeros((1, int(size), int(size)), dtype=torch.float32)
    proto = feature[:, mask].mean(dim=1)
    proto = F.normalize(proto, dim=0, p=2)
    sim = (feature * proto[:, None, None]).sum(dim=0, keepdim=True)
    return sim.float()


def connected_components(mask):
    array = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
    num_labels, labels = cv2.connectedComponents(array, connectivity=8)
    return int(num_labels - 1), labels


def limit_by_similarity(mask, sim, max_count):
    if int(max_count) <= 0:
        return torch.zeros_like(mask, dtype=torch.bool)
    if int(mask.sum().item()) <= int(max_count):
        return mask.bool()
    flat_mask = mask.flatten()
    candidate_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
    scores = sim.flatten()[candidate_idx]
    keep_local = torch.topk(scores, k=int(max_count), largest=True).indices
    keep_idx = candidate_idx[keep_local]
    limited = torch.zeros_like(flat_mask, dtype=torch.bool)
    limited[keep_idx] = True
    return limited.view_as(mask)


def filter_components(mask, sim, max_cc, min_area):
    num_cc, labels = connected_components(mask)
    if num_cc == 0:
        return mask.bool(), 0
    entries = []
    sim_np = sim.detach().cpu().numpy().squeeze()
    for label in range(1, num_cc + 1):
        comp = labels == label
        area = int(comp.sum())
        if area < int(min_area):
            continue
        score = float(sim_np[comp].mean()) if area > 0 else -1.0
        entries.append((score, area, label))
    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    keep = {label for _, _, label in entries[: int(max_cc)]}
    filtered = np.isin(labels, list(keep)) if keep else np.zeros_like(labels, dtype=bool)
    return torch.from_numpy(filtered.astype(np.bool_)).unsqueeze(0), len(keep)


def binary_iou(first, second):
    first = first.bool()
    second = second.bool()
    union = (first | second).float().sum().item()
    if union <= 0:
        return 1.0
    return float(((first & second).float().sum().item()) / union)


def build_fixed_local_recall(cfg, feature, p_despl, p_fixed, core_fg, core_bg):
    despl_bin = p_despl > 0.5
    seed = core_fg if bool(core_fg.any().item()) else despl_bin
    near_core = dilate_mask(seed, int(cfg.DREPP_FIXED_DILATION_RADIUS))
    feature_sim = feature_similarity(feature, seed, int(cfg.LOSS_SIZE))
    candidate = (p_fixed > 0.5) & (p_despl < 0.5) & (~core_bg)
    candidate = candidate & near_core & (feature_sim > float(cfg.DREPP_FIXED_SIM_TH))
    max_count = int(float(despl_bin.float().sum().item()) * float(cfg.DREPP_FIXED_MAX_EXPAND_RATIO))
    candidate = limit_by_similarity(candidate, feature_sim, max_count)
    candidate, kept_cc = filter_components(
        candidate,
        feature_sim,
        max_cc=int(cfg.DREPP_FIXED_MAX_CC),
        min_area=int(cfg.DREPP_FIXED_MIN_CC_AREA),
    )
    return candidate.bool(), feature_sim.float(), kept_cc


def build_boundary_band(cfg, p_despl):
    radius = int(cfg.DREPP_BOUNDARY_RADIUS)
    mask = p_despl > 0.5
    return (dilate_mask(mask, radius) & (~erode_mask(mask, radius))).bool()


def area(tensor):
    return float(tensor.float().mean().item())


def build_payload(cfg, item, source_row, feature):
    source = load_source_pseudo(source_row, item, cfg)
    loss_size = int(cfg.LOSS_SIZE)
    p_despl = resize_chw(source["p_despl"], loss_size, mode="bilinear").clamp(0.0, 1.0)
    p_fixed = resize_chw(source["p_fixed"], loss_size, mode="bilinear").clamp(0.0, 1.0)
    core_fg = p_despl >= float(cfg.DREPP_CORE_FG_TH)
    core_bg = (p_despl <= float(cfg.DREPP_CORE_BG_TH)) & (~core_fg)
    uncertain = ~(core_fg | core_bg)
    fixed_local, feature_sim, kept_cc = build_fixed_local_recall(cfg, feature, p_despl, p_fixed, core_fg, core_bg)
    boundary_band = build_boundary_band(cfg, p_despl)
    memory_init = p_despl.clone()
    memory_init = torch.where(core_fg, torch.ones_like(memory_init), memory_init)
    memory_init = torch.where(core_bg, torch.zeros_like(memory_init), memory_init)

    p_despl_bin = p_despl > 0.5
    p_fixed_bin = p_fixed > 0.5
    local_values = feature_sim[fixed_local]
    local_sim_mean = float(local_values.mean().item()) if int(local_values.numel()) > 0 else 0.0
    payload = {
        "dataset": item["dataset"],
        "stem": item["stem"],
        "image_path": item["image_path"],
        "backbone_key": cfg.BACKBONE_KEY,
        "p_despl": p_despl.cpu().float(),
        "p_fixed": p_fixed.cpu().float(),
        "core_fg": core_fg.cpu().bool(),
        "core_bg": core_bg.cpu().bool(),
        "uncertain": uncertain.cpu().bool(),
        "fixed_local_recall": fixed_local.cpu().bool(),
        "boundary_band": boundary_band.cpu().bool(),
        "memory_init": memory_init.cpu().float(),
        "feature_sim": feature_sim.cpu().float(),
        "formula": FORMULA,
        "global_blend": False,
        "p_despl_area": area(p_despl),
        "p_fixed_area": area(p_fixed),
        "core_fg_area": area(core_fg),
        "core_bg_area": area(core_bg),
        "uncertain_area": area(uncertain),
        "fixed_local_area": area(fixed_local),
        "boundary_band_area": area(boundary_band),
        "fixed_local_ratio": float(fixed_local.float().sum().item() / max(float(p_despl_bin.float().sum().item()), 1.0)),
        "fixed_despl_iou": binary_iou(p_fixed_bin, p_despl_bin),
        "local_sim_mean": local_sim_mean,
        "num_cc_fixed_local": int(kept_cc),
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


def save_visualization(path, payload):
    panels = [
        image_panel(payload["image_path"], "image"),
        tensor_panel(payload["p_despl"], "p_despl"),
        tensor_panel(payload["p_fixed"], "p_fixed"),
        tensor_panel(payload["core_fg"].float(), "core_fg", nearest=True),
        tensor_panel(payload["core_bg"].float(), "core_bg", nearest=True),
        tensor_panel(payload["fixed_local_recall"].float(), "fixed_local", nearest=True),
        tensor_panel(payload["boundary_band"].float(), "band", nearest=True),
        tensor_panel(payload["memory_init"], "memory_init"),
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
    fields = [
        "p_despl",
        "p_fixed",
        "core_fg",
        "core_bg",
        "uncertain",
        "fixed_local_recall",
        "boundary_band",
        "memory_init",
        "feature_sim",
    ]
    return {name: list(payload[name].shape) for name in fields}


def generate_drepp_cache(cfg, overwrite=False, max_samples=-1, vis_num=0, logger=print):
    if cfg.BACKBONE_KEY != "dinov1-s8":
        raise RuntimeError("DRE++ v1 only supports BACKBONE_KEY=dinov1-s8.")
    out_root = Path(cfg.DREPP_CACHE_ROOT) / cfg.BACKBONE_KEY
    manifest_path = out_root / "manifest_train.jsonl"
    ensure_dir(out_root)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    feature_manifest = feature_manifest_path(cfg, "train")
    source_manifest = despl_pseudo_bank_manifest_path(cfg)
    feature_map = manifest_to_map(read_jsonl(feature_manifest), feature_manifest)
    source_map = manifest_to_map(read_jsonl(source_manifest), source_manifest)
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples >= 0:
        items = items[: int(max_samples)]

    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"drepp_cache_root = {out_root}")
    logger(f"source_manifest = {source_manifest}")
    logger("train_gt_in_drepp_cache = false")
    logger("DINO_forward_in_drepp_cache = false")
    logger("use_gcm = false")
    logger(f"formula = {FORMULA}")
    logger(f"num_items = {len(items)}")

    rows = []
    for index, item in enumerate(tqdm(items, desc="cache DRE++ pseudo")):
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)
        if key not in feature_map:
            raise RuntimeError(f"Missing feature cache for {dataset}/{stem}")
        if key not in source_map:
            raise RuntimeError(f"Missing source DESPL pseudo for {dataset}/{stem}")
        out_dir = out_root / dataset
        ensure_dir(out_dir)
        out_path = out_dir / f"{stem}.pt"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")

        feature = load_feature(feature_map[key], dataset, stem)
        payload = build_payload(cfg, item, source_map[key], feature)
        torch.save(payload, out_path)
        if index < int(vis_num):
            vis_name = (
                f"{dataset}_{stem}_core{payload['core_fg_area']:.3f}_"
                f"local{payload['fixed_local_area']:.3f}_band{payload['boundary_band_area']:.3f}.png"
            )
            save_visualization(out_root / "vis" / vis_name, payload)
        rows.append(
            {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "backbone_key": cfg.BACKBONE_KEY,
                "shape": manifest_shape(payload),
                "formula": payload["formula"],
                "global_blend": False,
            }
        )

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_rows = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate DRE++ v1 DESPL-core memory pseudo cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--vis_num", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_drepp_cache(
        cfg,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        vis_num=args.vis_num,
        logger=print,
    )


if __name__ == "__main__":
    main()
