import inspect
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoModel

from nper.native_cue import NativeCueExtractor


EPS = 1e-8
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(name)


def resize_chw(tensor, size, mode="bilinear"):
    tensor = tensor.float().unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(tensor, size=(int(size), int(size)), mode=mode)
    else:
        out = F.interpolate(tensor, size=(int(size), int(size)), mode=mode, align_corners=False)
    return out.squeeze(0)


def load_dino_for_pseudo(cfg, device):
    model_path = Path(cfg.DINO_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Local DINO weight path not found: {model_path}")
    try:
        model = AutoModel.from_pretrained(
            str(model_path),
            local_files_only=True,
            add_pooling_layer=False,
            output_attentions=True,
        )
    except TypeError:
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True, output_attentions=True)
    model.to(device)
    model.eval()
    return model


def call_dino(model, inputs):
    params = inspect.signature(model.forward).parameters
    kwargs = {"output_attentions": True, "return_dict": True}
    if "interpolate_pos_encoding" in params:
        kwargs["interpolate_pos_encoding"] = True
    return model(inputs, **kwargs)


def image_to_dino_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


def minmax_norm(array):
    array = np.asarray(array, dtype=np.float32)
    amin = float(array.min())
    amax = float(array.max())
    if not np.isfinite(amin) or not np.isfinite(amax) or amax - amin < EPS:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - amin) / (amax - amin)).astype(np.float32)


def otsu_threshold(array):
    array = np.asarray(array, dtype=np.float32)
    if float(array.max() - array.min()) < EPS:
        return float(array.mean())
    u8 = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold) / 255.0


def binary_iou(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def apply_aug_tensor(tensor, aug):
    if aug == "identity":
        return tensor
    if aug == "hflip":
        return torch.flip(tensor, dims=[-1])
    if aug == "vflip":
        return torch.flip(tensor, dims=[-2])
    if aug == "rot180":
        return torch.flip(tensor, dims=[-2, -1])
    raise ValueError(f"Unsupported DESPL augmentation: {aug}")


def spectral_mask(feature_grid, rgb_grid, fixed_grid, cfg, device):
    grid = int(cfg.DESPL_GRID)
    feat = feature_grid.permute(1, 2, 0).reshape(grid * grid, -1).to(device).float()
    feat = F.normalize(feat, dim=1, p=2)
    rgb = rgb_grid.permute(1, 2, 0).reshape(grid * grid, 3).to(device).float()
    affinity_sem = (feat @ feat.t() + 1.0) / 2.0
    affinity_sem.fill_diagonal_(0.0)
    dist2 = torch.cdist(rgb, rgb, p=2).pow(2)
    sigma = float(cfg.DESPL_COLOR_SIGMA)
    affinity_color = torch.exp(-dist2 / (2.0 * sigma * sigma))
    affinity_color.fill_diagonal_(0.0)
    affinity = affinity_sem + float(cfg.DESPL_LAMBDA_COLOR) * affinity_color
    affinity = ((affinity + affinity.t()) * 0.5).clamp_min(0.0)
    degree = affinity.sum(dim=1)
    inv_sqrt = 1.0 / torch.sqrt(degree + EPS)
    laplacian = torch.eye(affinity.shape[0], device=device) - inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
    _, eigvecs = torch.linalg.eigh(laplacian)
    candidates = []
    for index in (0, 1):
        vec = eigvecs[:, index]
        vec = (vec - vec.min()) / (vec.max() - vec.min() + EPS)
        hist = torch.histc(vec.detach().cpu(), bins=int(cfg.DESPL_EIG_BINS), min=0.0, max=1.0)
        prob = hist / hist.sum().clamp_min(EPS)
        prob = prob[prob > 0]
        entropy = float(-(prob * prob.log()).sum().item())
        candidates.append((entropy, vec))
    candidates.sort(key=lambda item: item[0])
    main_vec = candidates[0][1]
    aux_vec = candidates[1][1]
    threshold = otsu_threshold(main_vec.detach().cpu().numpy())
    mask = main_vec > threshold
    delta = float(cfg.DESPL_LOWCONF_ALPHA) * float(main_vec.max() - main_vec.min())
    low_conf = torch.abs(main_vec - threshold) <= delta
    mask[low_conf] = aux_vec[low_conf] > 0.5
    mask_np = mask.detach().cpu().numpy().reshape(grid, grid)
    fixed_np = fixed_grid.detach().cpu().numpy().astype(bool)
    if binary_iou(~mask_np, fixed_np) > binary_iou(mask_np, fixed_np):
        mask_np = ~mask_np
    return torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)


def make_despl(feature, image_path, fixed_grid, cfg, device):
    grid = int(cfg.DESPL_GRID)
    feature_grid = resize_chw(feature, grid, mode="bilinear")
    image = Image.open(image_path).convert("RGB").resize((grid, grid), RESAMPLE_BICUBIC)
    rgb_grid = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    masks = []
    fixed = fixed_grid.squeeze(0) > 0.5
    for aug in getattr(cfg, "DESPL_AUGS", ["identity"]):
        feat_aug = apply_aug_tensor(feature_grid, aug)
        rgb_aug = apply_aug_tensor(rgb_grid, aug)
        fixed_aug = apply_aug_tensor(fixed.unsqueeze(0).float(), aug).squeeze(0) > 0.5
        mask_aug = spectral_mask(feat_aug, rgb_aug, fixed_aug, cfg, device)
        masks.append(apply_aug_tensor(mask_aug, aug))
    return torch.stack(masks, dim=0).mean(dim=0).clamp(0.0, 1.0)


def build_view_image(image, input_size, aug):
    view = image
    if aug.startswith("scale_"):
        scale = float(aug.split("_", 1)[1])
        scaled = max(input_size, int(round(input_size * scale)))
        view = image.resize((scaled, scaled), RESAMPLE_BICUBIC)
        left = (scaled - input_size) // 2
        top = (scaled - input_size) // 2
        view = view.crop((left, top, left + input_size, top + input_size))
    else:
        view = image.resize((input_size, input_size), RESAMPLE_BICUBIC)
    if aug == "hflip":
        view = view.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return view


def undo_aug_mask(mask, aug):
    if aug == "hflip":
        return np.flip(mask, axis=1).copy()
    return mask


@torch.no_grad()
def gcm_mask_for_view(model, view_image, cfg, device):
    input_size = int(cfg.GCM_INPUT_SIZE)
    inputs = image_to_dino_tensor(view_image.resize((input_size, input_size), RESAMPLE_BICUBIC)).to(device)
    outputs = call_dino(model, inputs)
    if outputs.attentions is None:
        raise RuntimeError("DINO did not return attentions for GCM.")
    layer = max(0, min(int(cfg.GCM_ATTN_LAYER) - 1, len(outputs.attentions) - 1))
    attention = outputs.attentions[layer][0, :, 0, 1:].detach().float().cpu().numpy()
    grid = int(math.sqrt(attention.shape[-1]))
    if grid * grid != attention.shape[-1]:
        raise RuntimeError(f"Non-square GCM attention tokens: {attention.shape[-1]}")
    best_score = -1.0
    best_heat = None
    for head in range(attention.shape[0]):
        heat = minmax_norm(attention[head].reshape(grid, grid))
        threshold = max(float(heat.mean() + 0.25 * heat.std()), otsu_threshold(heat) * 0.7)
        mask = heat > threshold
        area = float(mask.mean())
        area_score = max(0.0, 1.0 - abs(area - 0.25) / 0.35)
        contrast = float(heat[mask].mean() - heat[~mask].mean()) if mask.any() and (~mask).any() else 0.0
        score = area_score + contrast
        if score > best_score:
            best_score = score
            best_heat = heat
    if best_heat is None:
        best_heat = np.zeros((grid, grid), dtype=np.float32)
    heat_t = torch.from_numpy(best_heat).view(1, 1, grid, grid)
    kernel = int(getattr(cfg, "GCM_SAR_KERNEL", 5))
    if kernel > 1:
        heat_t = F.avg_pool2d(heat_t, kernel_size=kernel, stride=1, padding=kernel // 2)
    heat = minmax_norm(heat_t.squeeze().numpy())
    threshold = otsu_threshold(heat)
    return (heat > threshold).astype(np.float32), float(best_score)


def make_gcm(model, image_path, fixed_grid, cfg, device):
    input_size = int(cfg.GCM_INPUT_SIZE)
    image = Image.open(image_path).convert("RGB")
    masks = []
    scores = []
    for aug in getattr(cfg, "GCM_MCF_AUGS", ["identity"]):
        view = build_view_image(image, input_size, aug)
        mask, score = gcm_mask_for_view(model, view, cfg, device)
        mask = undo_aug_mask(mask, aug)
        masks.append(torch.from_numpy(mask).float())
        scores.append(score)
    vote = torch.stack(masks, dim=0)
    p_gcm = vote.mean(dim=0, keepdim=True)
    stability = 1.0 - float((vote - p_gcm).abs().mean().item())
    if p_gcm.max().item() <= 0:
        p_gcm = fixed_grid.float()
        stability = 0.0
    return p_gcm.clamp(0.0, 1.0), stability, float(np.mean(scores) if scores else 0.0)


def connected_components(mask):
    array = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
    num_labels, _ = cv2.connectedComponents(array, connectivity=8)
    return int(num_labels - 1)


def edge_overflow(mask):
    array = mask.detach().cpu().numpy().astype(bool).squeeze()
    if array.size == 0 or not array.any():
        return 0.0
    border = np.zeros_like(array, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return float(np.logical_and(array, border).sum() / max(float(array.sum()), 1.0))


def boundary_mask(mask):
    pooled_max = F.max_pool2d(mask.float().unsqueeze(0), 3, 1, 1)
    pooled_min = -F.max_pool2d(-mask.float().unsqueeze(0), 3, 1, 1)
    return (pooled_max - pooled_min).squeeze(0).clamp(0.0, 1.0)


def image_cues(image_path, size, native_extractor=None):
    image = Image.open(image_path).convert("RGB").resize((size, size), RESAMPLE_BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sx * sx + sy * sy)
    sobel = sobel / max(float(sobel.max()), EPS)
    dog = np.abs(cv2.GaussianBlur(gray, (3, 3), 0) - cv2.GaussianBlur(gray, (9, 9), 0))
    dog = dog / max(float(dog.max()), EPS)
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    lbp = np.zeros_like(center, dtype=np.float32)
    for bit, (dy, dx) in enumerate(((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))):
        neighbor = padded[1 + dy : 1 + dy + center.shape[0], 1 + dx : 1 + dx + center.shape[1]]
        lbp += (neighbor >= center).astype(np.float32) * float(2**bit)
    lbp = lbp / 255.0
    cues = {
        "gray": torch.from_numpy(gray).unsqueeze(0).float(),
        "sobel": torch.from_numpy(sobel).unsqueeze(0).float(),
        "dog": torch.from_numpy(dog).unsqueeze(0).float(),
        "lbp": torch.from_numpy(lbp).unsqueeze(0).float(),
    }
    if native_extractor is not None and bool(getattr(native_extractor, "use_resnet", False)):
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float()
        extracted = native_extractor.extract(tensor)
        if "resnet18_mid" in extracted:
            cues["resnet18_mid"] = extracted["resnet18_mid"].float()
    return cues


def cue_contrast(mask, cues):
    binary = mask > 0.5
    if binary.sum().item() == 0 or (~binary).sum().item() == 0:
        return 0.0
    values = []
    for name in ("gray", "dog", "lbp", "sobel"):
        cue = cues[name]
        inside = cue[binary].mean()
        outside = cue[~binary].mean()
        values.append(float((inside - outside).abs().item()))
    resnet_mid = cues.get("resnet18_mid")
    if resnet_mid is not None:
        small_binary = F.interpolate(
            binary.float().unsqueeze(0),
            size=resnet_mid.shape[-2:],
            mode="nearest",
        ).squeeze(0).bool()
        if small_binary.sum().item() > 0 and (~small_binary).sum().item() > 0:
            inside_feat = resnet_mid[:, small_binary.squeeze(0)].mean(dim=1)
            outside_feat = resnet_mid[:, (~small_binary.squeeze(0))].mean(dim=1)
            values.append(float((inside_feat - outside_feat).abs().mean().item()))
    return float(np.clip(np.mean(values) * 2.0, 0.0, 1.0))


def mnp_score(mask, cues):
    bnd = boundary_mask(mask)
    if bnd.sum().item() > 0:
        boundary_score = float((cues["sobel"] * bnd).sum().item() / bnd.sum().item())
    else:
        boundary_score = 0.0
    return float(np.clip(0.6 * boundary_score + 0.4 * cue_contrast(mask, cues), 0.0, 1.0))


def candidate_stats(candidates, cues, stability):
    names = ["fixed", "despl", "gcm"]
    binaries = [(cand > 0.5).bool() for cand in candidates]
    mnp_scores = [mnp_score(mask.float(), cues) for mask in binaries]
    areas = [float(mask.float().mean().item()) for mask in binaries]
    qualities = []
    agreements = []
    fragmentations = []
    boundaries = []
    overflows = []
    for i, mask in enumerate(binaries):
        other_ious = [binary_iou(mask.numpy(), binaries[j].numpy()) for j in range(len(binaries)) if j != i]
        agreement = float(np.mean(other_ious)) if other_ious else 1.0
        agreements.append(agreement)
        num_cc = connected_components(mask)
        fragmentation = min(1.0, float(max(0, num_cc - 1)) / 8.0)
        fragmentations.append(fragmentation)
        bnd = boundary_mask(mask.float())
        boundary = float((cues["sobel"] * bnd).sum().item() / max(float(bnd.sum().item()), 1.0))
        boundaries.append(boundary)
        overflow = edge_overflow(mask)
        overflows.append(overflow)
        cand_stability = stability if names[i] == "gcm" else max(0.0, min(1.0, agreement))
        q = (
            0.30 * agreement
            + 0.30 * mnp_scores[i]
            + 0.20 * cand_stability
            + 0.10 * boundary
            - 0.05 * fragmentation
            - 0.05 * overflow
        )
        qualities.append(float(np.clip(q, 0.0, 1.0)))
    temp = 0.5
    q_np = np.asarray(qualities, dtype=np.float64)
    weights = np.exp((q_np - q_np.max()) / temp)
    weights = weights / max(float(weights.sum()), EPS)
    return {
        "names": names,
        "mnp_scores": mnp_scores,
        "areas": areas,
        "qualities": qualities,
        "weights": weights.astype(np.float32).tolist(),
        "agreement_score": float(np.mean(agreements)),
        "fragmentation_score": float(np.mean(fragmentations)),
        "boundary_score": float(np.mean(boundaries)),
        "edge_overflow_score": float(np.mean(overflows)),
    }


def make_anchors(p_init, cfg):
    anchor_fg = p_init >= float(cfg.ANCHOR_FG_TH)
    anchor_bg = p_init <= float(cfg.ANCHOR_BG_TH)
    min_ratio = float(getattr(cfg, "ANCHOR_MIN_RATIO", 0.03))
    ratio = float((anchor_fg | anchor_bg).float().mean().item())
    if ratio < min_ratio:
        flat = p_init.flatten()
        total = flat.numel()
        k = max(1, int(total * min_ratio * 0.5))
        fg_idx = torch.topk(flat, k=k, largest=True).indices
        bg_idx = torch.topk(flat, k=k, largest=False).indices
        anchor_fg = torch.zeros_like(flat, dtype=torch.bool)
        anchor_bg = torch.zeros_like(flat, dtype=torch.bool)
        anchor_fg[fg_idx] = True
        anchor_bg[bg_idx] = True
        anchor_fg = anchor_fg.view_as(p_init)
        anchor_bg = anchor_bg.view_as(p_init)
    anchor_fg = anchor_fg & (~anchor_bg)
    return anchor_fg.bool(), anchor_bg.bool()


def pixel_reliability(p_init, anchor_fg, anchor_bg):
    p = p_init.clamp(EPS, 1.0 - EPS)
    entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log()) / math.log(2.0)
    weight = (1.0 - entropy).clamp(0.05, 1.0)
    weight[anchor_fg | anchor_bg] = 1.0
    return weight


def build_nper_payload(cfg, item, fixed_tensor, feature_tensor, dino_model, device, native_extractor=None):
    loss_size = int(cfg.LOSS_SIZE)
    p_fixed = resize_chw(fixed_tensor.float(), loss_size, mode="bilinear").clamp(0.0, 1.0)
    p_init_mode = str(getattr(cfg, "P_INIT_MODE", "quality_fusion"))
    fixed_only = p_init_mode == "fixed_only"
    use_despl = bool(getattr(cfg, "PSEUDO_USE_DESPL", True)) and not fixed_only
    use_gcm = bool(getattr(cfg, "PSEUDO_USE_GCM", True)) and not fixed_only
    zeros = torch.zeros_like(p_fixed)

    if fixed_only:
        p_despl = zeros.clone()
        p_gcm = zeros.clone()
        p_init = p_fixed.clone()
        stability = 0.0
        gcm_score = 0.0
        stats = {
            "mnp_scores": [0.0, 0.0, 0.0],
            "qualities": [1.0, 0.0, 0.0],
            "weights": [1.0, 0.0, 0.0],
            "agreement_score": 1.0,
            "fragmentation_score": 0.0,
            "boundary_score": 0.0,
            "edge_overflow_score": edge_overflow(p_fixed > 0.5),
            "areas": [
                float((p_fixed > 0.5).float().mean().item()),
                0.0,
                0.0,
            ],
        }
    else:
        fixed_grid = resize_chw(fixed_tensor.float(), int(cfg.DESPL_GRID), mode="bilinear").clamp(0.0, 1.0)
        if use_despl:
            if feature_tensor is None:
                raise RuntimeError("DESPL pseudo requires feature cache.")
            p_despl_grid = make_despl(feature_tensor.float(), item["image_path"], fixed_grid, cfg, device)
            p_despl = resize_chw(p_despl_grid, loss_size, mode="bilinear").clamp(0.0, 1.0)
        else:
            p_despl = zeros.clone()
        if use_gcm:
            if dino_model is None:
                raise RuntimeError("GCM pseudo requires a loaded DINO model.")
            p_gcm_grid, stability, gcm_score = make_gcm(dino_model, item["image_path"], fixed_grid, cfg, device)
            p_gcm = resize_chw(p_gcm_grid, loss_size, mode="bilinear").clamp(0.0, 1.0)
        else:
            p_gcm = zeros.clone()
            stability = 0.0
            gcm_score = 0.0
        cues = image_cues(item["image_path"], loss_size, native_extractor=native_extractor)
        candidates = [p_fixed, p_despl, p_gcm]
        stats = candidate_stats(candidates, cues, stability)
        weights = torch.tensor(stats["weights"], dtype=torch.float32).view(3, 1, 1, 1)
        p_init = (weights[0] * p_fixed + weights[1] * p_despl + weights[2] * p_gcm).clamp(0.0, 1.0)
    anchor_fg, anchor_bg = make_anchors(p_init, cfg)
    pixel_weight = pixel_reliability(p_init, anchor_fg, anchor_bg)
    quality_score = float(np.clip(max(stats["qualities"]), 0.0, 1.0))
    hard_score = float(np.clip(1.0 - quality_score, 0.0, 1.0))
    anchor_ratio = float((anchor_fg | anchor_bg).float().mean().item())
    payload = {
        "dataset": item["dataset"],
        "stem": item["stem"],
        "image_path": item["image_path"],
        "backbone_key": cfg.BACKBONE_KEY,
        "loss_size": loss_size,
        "p_init_mode": p_init_mode,
        "despl_enabled": bool(use_despl),
        "gcm_enabled": bool(use_gcm),
        "teacher_enabled": bool(getattr(cfg, "PSEUDO_USE_TEACHER", False)),
        "p_fixed": p_fixed.cpu().float(),
        "p_despl": p_despl.cpu().float(),
        "p_gcm": p_gcm.cpu().float(),
        "p_init": p_init.cpu().float(),
        "anchor_fg": anchor_fg.cpu().bool(),
        "anchor_bg": anchor_bg.cpu().bool(),
        "pixel_weight": pixel_weight.cpu().float(),
        "quality_score": quality_score,
        "hard_score": hard_score,
        "mnp_score_fixed": float(stats["mnp_scores"][0]),
        "mnp_score_despl": float(stats["mnp_scores"][1]),
        "mnp_score_gcm": float(stats["mnp_scores"][2]),
        "quality_fixed": float(stats["qualities"][0]),
        "quality_despl": float(stats["qualities"][1]),
        "quality_gcm": float(stats["qualities"][2]),
        "weight_fixed": float(stats["weights"][0]),
        "weight_despl": float(stats["weights"][1]),
        "weight_gcm": float(stats["weights"][2]),
        "agreement_score": float(stats["agreement_score"]),
        "stability_score": float(stability),
        "gcm_head_score": float(gcm_score),
        "fragmentation_score": float(stats["fragmentation_score"]),
        "boundary_score": float(stats["boundary_score"]),
        "edge_overflow_score": float(stats["edge_overflow_score"]),
        "fixed_area": float(stats["areas"][0]),
        "despl_area": float(stats["areas"][1]),
        "gcm_area": float(stats["areas"][2]),
        "init_area": float((p_init > 0.5).float().mean().item()),
        "anchor_ratio": anchor_ratio,
    }
    return payload


def tensor_to_gray_image(tensor):
    array = tensor.detach().cpu().float().squeeze().numpy()
    array = np.clip(array, 0.0, 1.0)
    return Image.fromarray(np.uint8(array * 255.0), mode="L").convert("RGB")


def save_payload_vis(path, payload):
    panels = [
        ("fixed", payload["p_fixed"]),
        ("despl", payload["p_despl"]),
        ("gcm", payload["p_gcm"]),
        ("init", payload["p_init"]),
        ("fg", payload["anchor_fg"].float()),
        ("bg", payload["anchor_bg"].float()),
    ]
    cell = 128
    canvas = Image.new("RGB", (cell * len(panels), cell + 18), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (name, tensor) in enumerate(panels):
        image = tensor_to_gray_image(tensor).resize((cell, cell), RESAMPLE_BICUBIC)
        canvas.paste(image, (idx * cell, 18))
        draw.text((idx * cell + 4, 3), name, fill=(0, 0, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
