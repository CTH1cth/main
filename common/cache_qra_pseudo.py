import argparse
import math
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
    ensure_dir,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    pseudo_manifest_path,
    qra_cache_dir,
    qra_manifest_path,
    read_jsonl,
    torch_load,
    write_jsonl,
)


EPS = 1e-8

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def resize_chw(tensor, size, mode="bilinear"):
    tensor = tensor.float().unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(tensor, size=(size, size), mode=mode)
    else:
        out = F.interpolate(tensor, size=(size, size), mode=mode, align_corners=False)
    return out.squeeze(0)


def load_feature(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Feature cache key mismatch: {row['cache_path']}")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3:
        raise RuntimeError(f"Feature tensor must be [C,H,W], got {list(tensor.shape)}")
    return tensor, payload


def load_fixed(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Fixed pseudo cache key mismatch: {row['cache_path']}")
    tensor = payload["tensor"].float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"Fixed pseudo tensor must be [1,H,W], got {list(tensor.shape)}")
    return tensor, payload


def load_rgb_grid(image_path, grid):
    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)
    resized = image.resize((grid, grid), RESAMPLE_BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return image, tensor, original_size


def apply_aug(tensor, aug):
    if aug == "identity":
        return tensor
    if aug == "hflip":
        return torch.flip(tensor, dims=[-1])
    if aug == "vflip":
        return torch.flip(tensor, dims=[-2])
    if aug == "rot180":
        return torch.flip(tensor, dims=[-2, -1])
    raise ValueError(f"Unsupported QRA augmentation: {aug}")


def minmax_norm(vector):
    vmin = vector.min()
    vmax = vector.max()
    return (vector - vmin) / (vmax - vmin + EPS)


def histogram_entropy(values, bins):
    array = values.detach().cpu().numpy().astype(np.float64)
    if float(array.max() - array.min()) < EPS:
        return float("inf")
    hist, _ = np.histogram(array, bins=int(bins), range=(0.0, 1.0))
    prob = hist.astype(np.float64)
    prob = prob / max(float(prob.sum()), EPS)
    prob = prob[prob > 0]
    return float(-(prob * np.log(prob)).sum())


def otsu_threshold(values):
    array = values.detach().cpu().numpy().astype(np.float32)
    if float(array.max() - array.min()) < EPS:
        return float(array.mean())
    array_u8 = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(
        array_u8.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(threshold) / 255.0


def binary_iou(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def spectral_mask(feature_grid, rgb_grid, fixed_grid, cfg, device):
    grid = int(cfg.QRA_DESPL_GRID)
    feat = feature_grid.permute(1, 2, 0).reshape(grid * grid, -1).to(device).float()
    feat = F.normalize(feat, dim=1, p=2)
    rgb = rgb_grid.permute(1, 2, 0).reshape(grid * grid, 3).to(device).float()

    affinity_sem = (feat @ feat.t() + 1.0) / 2.0
    affinity_sem.fill_diagonal_(0.0)
    dist2 = torch.cdist(rgb, rgb, p=2).pow(2)
    sigma = float(cfg.QRA_COLOR_SIGMA)
    affinity_color = torch.exp(-dist2 / (2.0 * sigma * sigma))
    affinity_color.fill_diagonal_(0.0)

    affinity = affinity_sem + float(cfg.QRA_LAMBDA_COLOR) * affinity_color
    affinity = ((affinity + affinity.t()) / 2.0).clamp_min(0.0)
    degree = affinity.sum(dim=1)
    inv_sqrt = 1.0 / torch.sqrt(degree + EPS)
    laplacian = torch.eye(affinity.shape[0], device=device) - (
        inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
    )

    _, eigvecs = torch.linalg.eigh(laplacian)
    first = minmax_norm(eigvecs[:, 0])
    second = minmax_norm(eigvecs[:, 1])
    first_entropy = histogram_entropy(first, cfg.QRA_EIG_BINS)
    second_entropy = histogram_entropy(second, cfg.QRA_EIG_BINS)
    if first_entropy <= second_entropy:
        main_vec, aux_vec = first, second
    else:
        main_vec, aux_vec = second, first

    threshold = otsu_threshold(main_vec)
    mask = main_vec > threshold
    delta = float(cfg.QRA_LOWCONF_ALPHA) * float(main_vec.max() - main_vec.min())
    low_conf = torch.abs(main_vec - threshold) <= delta
    mask[low_conf] = aux_vec[low_conf] > 0.5

    mask_np = mask.detach().cpu().numpy().reshape(grid, grid)
    fixed_np = fixed_grid.detach().cpu().numpy().astype(bool)
    if binary_iou(~mask_np, fixed_np) > binary_iou(mask_np, fixed_np):
        mask_np = ~mask_np
    return torch.from_numpy(mask_np.astype(np.float32))


def make_despl_pseudo(feature, rgb_grid, fixed_grid, cfg, device):
    masks = []
    for aug in cfg.QRA_AUGS:
        feature_aug = apply_aug(feature, aug)
        rgb_aug = apply_aug(rgb_grid, aug)
        fixed_aug = apply_aug(fixed_grid.unsqueeze(0).float(), aug).squeeze(0) > 0.5
        mask_aug = spectral_mask(feature_aug, rgb_aug, fixed_aug, cfg, device)
        masks.append(apply_aug(mask_aug.unsqueeze(0), aug).squeeze(0))
    return torch.stack(masks, dim=0).mean(dim=0).clamp(0.0, 1.0)


def global_ssim(first, second):
    x = first.float().flatten()
    y = second.float().flatten()
    mux = x.mean()
    muy = y.mean()
    varx = ((x - mux) ** 2).mean()
    vary = ((y - muy) ** 2).mean()
    cov = ((x - mux) * (y - muy)).mean()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    value = ((2 * mux * muy + c1) * (2 * cov + c2)) / (
        (mux * mux + muy * muy + c1) * (varx + vary + c2) + EPS
    )
    return float(value.clamp(0.0, 1.0).item())


def connected_component_stats(binary):
    mask = binary.detach().cpu().numpy().astype(np.uint8).squeeze()
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    num_cc = int(num_labels - 1)
    edge_touch = 0
    for label in range(1, num_labels):
        component = labels == label
        if (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        ):
            edge_touch += 1
    return num_cc, int(edge_touch)


def quality_grade(cfg, sim, area, num_cc, edge_touch):
    if (
        sim >= float(cfg.QRA_Q2_SIM)
        and float(cfg.QRA_Q2_AREA_MIN) <= area <= float(cfg.QRA_Q2_AREA_MAX)
        and num_cc <= int(cfg.QRA_Q2_MAX_CC)
        and edge_touch <= int(cfg.QRA_Q2_MAX_EDGE_TOUCH)
    ):
        return 2
    if (
        sim >= float(cfg.QRA_Q1_SIM)
        and float(cfg.QRA_Q1_AREA_MIN) <= area <= float(cfg.QRA_Q1_AREA_MAX)
        and num_cc <= int(cfg.QRA_Q1_MAX_CC)
    ):
        return 1
    return 0


def pixel_reliability(p_fused):
    eps = 1e-6
    p = p_fused.clamp(eps, 1.0 - eps)
    entropy = -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)
    entropy = entropy / math.log(2.0)
    return (1.0 - entropy).clamp(0.0, 1.0)


def build_qra_payload(cfg, item, feature, fixed, original_size, device):
    grid = int(cfg.QRA_DESPL_GRID)
    loss_size = int(cfg.LOSS_SIZE)
    feature_grid = resize_chw(feature, grid, mode="bilinear")
    rgb_image, rgb_grid, image_original_size = load_rgb_grid(item["image_path"], grid)
    if tuple(original_size) != tuple(image_original_size):
        original_size = image_original_size
    fixed_grid = resize_chw(fixed, grid, mode="bilinear").squeeze(0) > 0.5

    p_despl_grid = make_despl_pseudo(feature_grid, rgb_grid, fixed_grid, cfg, device)
    p_despl = resize_chw(p_despl_grid.unsqueeze(0), loss_size, mode="bilinear").clamp(0.0, 1.0)
    p_fixed = resize_chw(fixed, loss_size, mode="bilinear").clamp(0.0, 1.0)
    p_fused = ((p_fixed + p_despl) / 2.0).clamp(0.0, 1.0)

    binary_fixed = p_fixed > 0.5
    binary_despl = p_despl > 0.5
    iou = binary_iou(binary_fixed.numpy(), binary_despl.numpy())
    sim = 0.5 * iou + 0.5 * global_ssim(p_fixed, p_despl)

    binary_fused = p_fused > 0.5
    area = float(binary_fused.float().mean().item())
    num_cc, edge_touch = connected_component_stats(binary_fused)
    anchor_fg = p_fused >= float(cfg.QRA_TH_FG)
    anchor_bg = p_fused <= float(cfg.QRA_TH_BG)
    anchor_bg = anchor_bg & ~anchor_fg
    anchor_ratio = float((anchor_fg | anchor_bg).float().mean().item())
    quality = quality_grade(cfg, sim, area, num_cc, edge_touch)
    if anchor_ratio < 0.05:
        quality = 0

    pixel_weight = pixel_reliability(p_fused)
    return {
        "payload": {
            "dataset": item["dataset"],
            "stem": item["stem"],
            "image_path": item["image_path"],
            "original_size": tuple(int(value) for value in original_size),
            "backbone_key": cfg.BACKBONE_KEY,
            "p_fixed": p_fixed.cpu().float(),
            "p_despl": p_despl.cpu().float(),
            "p_fused": p_fused.cpu().float(),
            "anchor_fg": anchor_fg.cpu().bool(),
            "anchor_bg": anchor_bg.cpu().bool(),
            "pixel_weight": pixel_weight.cpu().float(),
            "quality": int(quality),
            "sim": float(sim),
            "area": float(area),
            "num_cc": int(num_cc),
            "edge_touch": int(edge_touch),
            "anchor_ratio": float(anchor_ratio),
        },
        "image": rgb_image,
    }


def tensor_panel(tensor, title, size=224, nearest=False):
    array = tensor.detach().cpu().float().squeeze().numpy()
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="L")
    image = image.resize((size, size), RESAMPLE_NEAREST if nearest else RESAMPLE_BICUBIC).convert("RGB")
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def image_panel(image, title, size=224):
    image = image.resize((size, size), RESAMPLE_BICUBIC).convert("RGB")
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def save_visualization(path, image, payload):
    panels = [
        image_panel(image, "image"),
        tensor_panel(payload["p_fixed"], "p_fixed"),
        tensor_panel(payload["p_despl"], "p_despl"),
        tensor_panel(payload["p_fused"], "p_fused"),
        tensor_panel(payload["anchor_fg"].float(), "anchor_fg", nearest=True),
        tensor_panel(payload["anchor_bg"].float(), "anchor_bg", nearest=True),
        tensor_panel(payload["pixel_weight"], "pixel_weight"),
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
    return {
        "p_fixed": list(payload["p_fixed"].shape),
        "p_despl": list(payload["p_despl"].shape),
        "p_fused": list(payload["p_fused"].shape),
        "anchor_fg": list(payload["anchor_fg"].shape),
        "anchor_bg": list(payload["anchor_bg"].shape),
        "pixel_weight": list(payload["pixel_weight"].shape),
    }


def generate_qra_cache(cfg, overwrite=False, max_samples=-1, vis_num=0, device_name="auto", logger=print):
    if cfg.BACKBONE_KEY != "dinov1-s8":
        raise RuntimeError("QRA-DPL v1 only supports BACKBONE_KEY=dinov1-s8.")
    device = resolve_device(device_name)
    out_root = qra_cache_dir(cfg)
    manifest_path = qra_manifest_path(cfg)
    ensure_dir(out_root)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    feature_manifest = feature_manifest_path(cfg, "train")
    pseudo_manifest = pseudo_manifest_path(cfg)
    feature_map = manifest_to_map(read_jsonl(feature_manifest), feature_manifest)
    pseudo_map = manifest_to_map(read_jsonl(pseudo_manifest), pseudo_manifest)
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples >= 0:
        items = items[: int(max_samples)]

    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"qra_cache_root = {out_root}")
    logger(f"device = {device}")
    logger("train_gt_in_qra_cache = false")
    logger(f"num_items = {len(items)}")

    rows = []
    for index, item in enumerate(tqdm(items, desc="cache QRA pseudo")):
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)
        if key not in feature_map:
            raise RuntimeError(f"Missing feature cache for {dataset}/{stem}")
        if key not in pseudo_map:
            raise RuntimeError(f"Missing fixed pseudo cache for {dataset}/{stem}")
        out_dir = out_root / dataset
        ensure_dir(out_dir)
        out_path = out_dir / f"{stem}.pt"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")

        feature, feature_payload = load_feature(feature_map[key], dataset, stem)
        fixed, _ = load_fixed(pseudo_map[key], dataset, stem)
        result = build_qra_payload(
            cfg,
            item,
            feature,
            fixed,
            feature_payload.get("original_size", (0, 0)),
            device,
        )
        payload = result["payload"]
        torch.save(payload, out_path)
        if index < int(vis_num):
            vis_name = (
                f"{dataset}_{stem}_q{payload['quality']}_sim{payload['sim']:.3f}_"
                f"area{payload['area']:.3f}_cc{payload['num_cc']}_"
                f"edge{payload['edge_touch']}.png"
            )
            save_visualization(out_root / "vis" / vis_name, result["image"], payload)
        rows.append(
            {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "backbone_key": cfg.BACKBONE_KEY,
                "shape": manifest_shape(payload),
                "quality": payload["quality"],
                "sim": payload["sim"],
                "area": payload["area"],
                "num_cc": payload["num_cc"],
                "edge_touch": payload["edge_touch"],
                "anchor_ratio": payload["anchor_ratio"],
            }
        )

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_items = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate QRA-DPL pseudo cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--vis_num", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")
    if args.vis_num < 0:
        raise ValueError("--vis_num must be >= 0.")
    cfg = load_config(args.config)
    generate_qra_cache(
        cfg,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        vis_num=args.vis_num,
        device_name=args.device,
        logger=print,
    )


if __name__ == "__main__":
    main()
