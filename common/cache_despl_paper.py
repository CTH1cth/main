import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_features import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    call_dino,
    key_to_feature_tensor,
    load_dino,
    resolve_key_projection,
)
from common.utils import (  # noqa: E402
    build_image_items,
    despl_paper_cache_dir,
    despl_paper_manifest_path,
    despl_pseudo_bank_manifest_path,
    ensure_dir,
    load_config,
    manifest_to_map,
    pseudo_manifest_path,
    read_jsonl,
    torch_load,
    write_jsonl,
)


SIGN_MODES = ("paper", "fixed_iou", "area_small")


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def preprocess_pil(image, size):
    image = image.resize((size, size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0)


def apply_aug(image, aug, cfg):
    if aug == "identity":
        return image.copy()
    if aug == "hflip":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    if aug == "vflip":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    if aug == "rot180":
        return image.transpose(Image.ROTATE_180)
    if aug == "bright_up":
        return ImageEnhance.Brightness(image).enhance(float(cfg.BRIGHT_UP_FACTOR))
    if aug == "bright_down":
        return ImageEnhance.Brightness(image).enhance(float(cfg.BRIGHT_DOWN_FACTOR))
    if aug == "contrast_up":
        return ImageEnhance.Contrast(image).enhance(float(cfg.CONTRAST_UP_FACTOR))
    if aug == "contrast_down":
        return ImageEnhance.Contrast(image).enhance(float(cfg.CONTRAST_DOWN_FACTOR))
    raise ValueError(f"Unknown DESPL augmentation: {aug}")


def inverse_mask(mask, aug):
    if aug == "hflip":
        return torch.flip(mask, dims=[1])
    if aug == "vflip":
        return torch.flip(mask, dims=[0])
    if aug == "rot180":
        return torch.flip(mask, dims=[0, 1])
    return mask


def image_to_grid_rgb(image, grid, device):
    resized = image.resize((grid, grid), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).to(device=device, dtype=torch.float32).view(-1, 3)


@torch.no_grad()
def extract_feature_grid(model, key_module, image, cfg, device):
    holder = {"tensor": None}

    def hook_key(_module, _inputs, output):
        holder["tensor"] = output.detach()

    handle = key_module.register_forward_hook(hook_key)
    try:
        tensor = preprocess_pil(image, int(cfg.DINO["feature_input_size"])).to(device)
        holder["tensor"] = None
        call_dino(model, tensor)
        key = holder["tensor"]
        if key is None:
            raise RuntimeError("DINO key hook did not capture a tensor.")
        feature = key_to_feature_tensor(key).unsqueeze(0).to(device)
        feature = F.interpolate(
            feature,
            size=(int(cfg.DESPL_GRID), int(cfg.DESPL_GRID)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return feature.float()
    finally:
        handle.remove()


def minmax_norm(vector):
    v_min = vector.min()
    v_max = vector.max()
    return (vector - v_min) / (v_max - v_min + 1e-8)


def entropy_score(vector, bins):
    hist = torch.histc(vector.float(), bins=int(bins), min=0.0, max=1.0)
    prob = hist / hist.sum().clamp_min(1.0)
    prob = prob[prob > 0]
    if prob.numel() == 0:
        return torch.tensor(0.0, device=vector.device)
    return -(prob * torch.log(prob)).sum()


def otsu_threshold(vector, bins=256):
    hist = torch.histc(vector.float(), bins=int(bins), min=0.0, max=1.0)
    total = hist.sum()
    if float(total.item()) <= 0:
        return vector.mean()
    prob = hist / total
    centers = torch.linspace(0.0, 1.0, int(bins), device=vector.device)
    omega = torch.cumsum(prob, dim=0)
    mu = torch.cumsum(prob * centers, dim=0)
    mu_t = mu[-1]
    sigma_b = (mu_t * omega - mu).pow(2) / (omega * (1.0 - omega) + 1e-8)
    index = int(torch.argmax(sigma_b).item())
    return centers[index]


def build_affinity(feature, image, cfg, device):
    grid = int(cfg.DESPL_GRID)
    flat = feature.flatten(1).transpose(0, 1)
    flat = F.normalize(flat, dim=1)
    affinity_sem = flat @ flat.t()
    if bool(getattr(cfg, "AFFINITY_CLAMP_NONNEG", False)):
        affinity_sem = affinity_sem.clamp_min(0.0)

    colors = image_to_grid_rgb(image, grid, device)
    color_dist = torch.cdist(colors, colors).pow(2)
    sigma = float(cfg.DESPL_COLOR_SIGMA)
    affinity_color = torch.exp(-color_dist / (2.0 * sigma * sigma + 1e-8))
    affinity = affinity_sem + float(cfg.DESPL_LAMBDA_COLOR) * affinity_color
    affinity = 0.5 * (affinity + affinity.t())
    if not torch.isfinite(affinity).all():
        raise RuntimeError("DESPL affinity contains NaN/Inf.")
    return affinity


def paper_despl_mask(feature, image, cfg, device):
    grid = int(cfg.DESPL_GRID)
    affinity = build_affinity(feature, image, cfg, device)
    degree = affinity.sum(dim=1)
    degree = degree.clamp_min(1e-6)
    inv_sqrt = torch.rsqrt(degree)
    laplacian = torch.diag(degree) - affinity
    laplacian = inv_sqrt[:, None] * laplacian * inv_sqrt[None, :]
    laplacian = 0.5 * (laplacian + laplacian.t())
    if not torch.isfinite(laplacian).all():
        raise RuntimeError("DESPL Laplacian contains NaN/Inf.")

    eigvals, eigvecs = torch.linalg.eigh(laplacian)
    order = torch.argsort(eigvals)[: int(cfg.DESPL_K)]
    candidates = [minmax_norm(eigvecs[:, int(i.item())]) for i in order[:2]]
    if len(candidates) < 2:
        raise RuntimeError("DESPL requires two eigenvectors.")
    h0 = entropy_score(candidates[0], int(cfg.DESPL_ENTROPY_BINS))
    h1 = entropy_score(candidates[1], int(cfg.DESPL_ENTROPY_BINS))
    if h1 < h0:
        main, aux = candidates[1], candidates[0]
    else:
        main, aux = candidates[0], candidates[1]

    tau = otsu_threshold(main)
    mask = main > tau
    delta = float(cfg.DESPL_LOWCONF_ALPHA) * float((main.max() - main.min()).item())
    low_conf = torch.abs(main - tau) <= delta
    mask = torch.where(low_conf, aux > 0.5, mask)
    return mask.float().view(grid, grid), {
        "eigvals": [float(v.item()) for v in eigvals[: max(2, int(cfg.DESPL_K))]],
        "entropy_v1": float(h0.item()),
        "entropy_v2": float(h1.item()),
        "otsu": float(tau.item()),
        "lowconf_ratio": float(low_conf.float().mean().item()),
    }


def binary_iou(a, b):
    a = a.bool()
    b = b.bool()
    union = torch.logical_or(a, b).sum().item()
    if union == 0:
        return 1.0
    inter = torch.logical_and(a, b).sum().item()
    return float(inter / union)


def apply_sign_mode(mask, sign_mode, fixed_mask=None):
    if sign_mode == "paper":
        return mask
    if sign_mode == "area_small":
        return 1.0 - mask if float(mask.mean().item()) > 0.5 else mask
    if sign_mode == "fixed_iou":
        if fixed_mask is None:
            raise RuntimeError("fixed_iou sign mode requires fixed pseudo.")
        iou_a = binary_iou(mask > 0.5, fixed_mask > 0.5)
        iou_b = binary_iou((1.0 - mask) > 0.5, fixed_mask > 0.5)
        return 1.0 - mask if iou_b > iou_a else mask
    raise ValueError(f"Unknown sign mode: {sign_mode}")


def view_consistency(masks):
    if len(masks) < 2:
        return 1.0
    values = []
    for a, b in itertools.combinations(masks, 2):
        values.append(binary_iou(a > 0.5, b > 0.5))
    return float(np.mean(values)) if values else 1.0


def load_optional_maps(cfg, sign_mode):
    old_map = {}
    fixed_map = {}
    try:
        old_manifest = despl_pseudo_bank_manifest_path(cfg)
        if old_manifest.exists():
            old_map = manifest_to_map(read_jsonl(old_manifest), old_manifest)
    except Exception:
        old_map = {}
    if sign_mode == "fixed_iou":
        fixed_manifest = pseudo_manifest_path(cfg)
        fixed_map = manifest_to_map(read_jsonl(fixed_manifest), fixed_manifest)
    return old_map, fixed_map


def load_single_channel(row, name):
    payload = torch_load(row["cache_path"], map_location="cpu")
    tensor = payload[name].float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"{name} must be [1,H,W]: {row['cache_path']}")
    return tensor


def resize_mask(tensor, grid):
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=(grid, grid),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def make_panel(title, image, size=160):
    if isinstance(image, torch.Tensor):
        array = image.detach().cpu().float().clamp(0, 1).numpy()
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        array = (array * 255).astype(np.uint8)
        image = Image.fromarray(array)
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    panel = Image.new("RGB", (size, size + 24), "white")
    panel.paste(image, (0, 24))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 4), title, fill=(0, 0, 0))
    return panel


def save_visualization(out_path, image, old_despl, view0, soft, binary, consistency):
    panels = [make_panel("image", image)]
    if old_despl is not None:
        panels.append(make_panel("old_despl", old_despl))
    panels.extend([
        make_panel("paper_view0", view0),
        make_panel("paper_soft", soft),
        make_panel("paper_binary", binary),
        make_panel(f"cons={consistency:.3f}", binary),
    ])
    canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(out_path)


def build_sample_payload(cfg, item, model, key_module, device, sign_mode, old_map, fixed_map):
    grid = int(cfg.DESPL_GRID)
    image = Image.open(item["image_path"]).convert("RGB")
    original_size = (image.height, image.width)
    key = (item["dataset"], item["stem"])
    fixed_mask = None
    if sign_mode == "fixed_iou":
        if key not in fixed_map:
            raise RuntimeError(f"Fixed pseudo missing for fixed_iou sign mode: {key}")
        fixed_mask = resize_mask(load_single_channel(fixed_map[key], "tensor"), grid).to(device)

    aligned_masks = []
    view_areas = []
    view_infos = []
    for aug in list(cfg.DESPL_AUGS)[: int(cfg.DESPL_NUM_AUGS)]:
        aug_image = apply_aug(image, aug, cfg)
        feature = extract_feature_grid(model, key_module, aug_image, cfg, device)
        mask, info = paper_despl_mask(feature, aug_image, cfg, device)
        aligned = inverse_mask(mask, aug)
        aligned = apply_sign_mode(aligned, sign_mode, fixed_mask=fixed_mask)
        aligned = aligned.detach().float().cpu()
        aligned_masks.append(aligned)
        view_areas.append(float(aligned.mean().item()))
        view_infos.append(info)

    soft = torch.stack(aligned_masks, dim=0).mean(dim=0).clamp(0.0, 1.0)
    binary = (soft > 0.5).float()
    consistency = view_consistency(aligned_masks)
    old_despl = None
    if key in old_map:
        old_despl = resize_mask(load_single_channel(old_map[key], "p_despl"), grid)

    payload = {
        "dataset": item["dataset"],
        "stem": item["stem"],
        "image_path": item["image_path"],
        "original_size": original_size,
        "backbone_key": cfg.BACKBONE_KEY,
        "p_despl_paper": binary.unsqueeze(0).float(),
        "p_despl_paper_soft": soft.unsqueeze(0).float(),
        "despl_grid": grid,
        "lambda_color": float(cfg.DESPL_LAMBDA_COLOR),
        "sigma": float(cfg.DESPL_COLOR_SIGMA),
        "entropy_bins": int(cfg.DESPL_ENTROPY_BINS),
        "lowconf_alpha": float(cfg.DESPL_LOWCONF_ALPHA),
        "num_augs": int(cfg.DESPL_NUM_AUGS),
        "sign_mode": sign_mode,
        "area": float(soft.mean().item()),
        "binary_area": float(binary.mean().item()),
        "view_areas": view_areas,
        "view_consistency": consistency,
        "view_infos": view_infos,
    }
    return payload, old_despl, aligned_masks[0]


def generate_despl_paper_cache(cfg, overwrite=False, max_samples=-1, vis_num=0, device_name="auto", sign_mode=None, logger=print):
    sign_mode = sign_mode or str(getattr(cfg, "DESPL_SIGN_MODE", "paper"))
    if sign_mode not in SIGN_MODES:
        raise ValueError(f"--sign_mode must be one of {SIGN_MODES}, got {sign_mode}")
    if sign_mode == "fixed_iou" and not bool(getattr(cfg, "ALLOW_FIXED_SIGN_ABLATION", False)):
        raise RuntimeError("fixed_iou sign mode requested but ALLOW_FIXED_SIGN_ABLATION=False")
    device = resolve_device(device_name)
    out_root = despl_paper_cache_dir(cfg, sign_mode=sign_mode)
    manifest_path = despl_paper_manifest_path(cfg, sign_mode=sign_mode)
    ensure_dir(out_root)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    if not items:
        raise RuntimeError("No training images selected for DESPL-paper cache.")

    old_map, fixed_map = load_optional_maps(cfg, sign_mode)
    model = load_dino(cfg, device)
    key_module, key_path = resolve_key_projection(model)

    logger(f"device = {device}")
    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"model_path = {cfg.DINO['model_path']}")
    logger(f"key_hook = {key_path}")
    logger(f"output_root = {out_root}")
    logger(f"manifest_path = {manifest_path}")
    logger(f"despl_grid = {int(cfg.DESPL_GRID)}")
    logger(f"sign_mode = {sign_mode}")
    logger("train_gt_used = false")
    logger("forward_dino = true | offline_cache_generation = true")

    rows = []
    vis_root = out_root / "vis"
    if int(vis_num) > 0:
        ensure_dir(vis_root)
    for index, item in enumerate(tqdm(items, desc=f"cache DESPL-paper {sign_mode}")):
        payload, old_despl, view0 = build_sample_payload(
            cfg, item, model, key_module, device, sign_mode, old_map, fixed_map
        )
        out_dir = out_root / item["dataset"]
        ensure_dir(out_dir)
        out_path = out_dir / f"{item['stem']}.pt"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")
        torch.save(payload, out_path)
        rows.append({
            "dataset": item["dataset"],
            "stem": item["stem"],
            "image_path": item["image_path"],
            "cache_path": str(out_path.resolve()),
            "backbone_key": cfg.BACKBONE_KEY,
            "shape": list(payload["p_despl_paper_soft"].shape),
            "area": float(payload["area"]),
            "view_consistency": float(payload["view_consistency"]),
            "sign_mode": sign_mode,
        })
        if index < int(vis_num):
            vis_name = (
                f"{item['dataset']}_{item['stem']}_"
                f"area{payload['area']:.3f}_cons{payload['view_consistency']:.3f}.png"
            )
            image = Image.open(item["image_path"]).convert("RGB")
            save_visualization(
                vis_root / vis_name,
                image,
                old_despl,
                view0,
                payload["p_despl_paper_soft"].squeeze(0),
                payload["p_despl_paper"].squeeze(0),
                float(payload["view_consistency"]),
            )

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_rows = {len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate paper-like DESPL pseudo cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--vis_num", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sign_mode", default=None, choices=SIGN_MODES)
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_despl_paper_cache(
        cfg,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        vis_num=args.vis_num,
        device_name=args.device,
        sign_mode=args.sign_mode,
        logger=print,
    )


if __name__ == "__main__":
    main()
