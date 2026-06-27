import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc_label


def _dilate(mask, radius):
    if int(radius) <= 0:
        return mask.bool()
    kernel = int(radius) * 2 + 1
    pooled = F.max_pool2d(mask.float().unsqueeze(0), kernel_size=kernel, stride=1, padding=int(radius))
    return pooled.squeeze(0) > 0.5


def _connected_components(mask):
    array = mask.detach().cpu().numpy().astype(np.uint8).squeeze()
    if array.size == 0:
        return 0
    _, num = cc_label(array)
    return int(num)


def _area(mask):
    return float(mask.float().mean().item())


def build_dre_safe_prior(p_despl, p_fixed, cfg):
    p_despl = p_despl.float().clamp(0.0, 1.0)
    p_fixed = p_fixed.float().clamp(0.0, 1.0)
    if p_despl.ndim != 3 or p_despl.shape[0] != 1:
        raise RuntimeError(f"p_despl must be [1,H,W], got {list(p_despl.shape)}")
    if p_fixed.ndim != 3 or p_fixed.shape[0] != 1:
        raise RuntimeError(f"p_fixed must be [1,H,W], got {list(p_fixed.shape)}")
    if list(p_despl.shape[-2:]) != list(p_fixed.shape[-2:]):
        p_fixed = F.interpolate(
            p_fixed.unsqueeze(0),
            size=p_despl.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    despl_weight = float(getattr(cfg, "P_INIT_DESPL_WEIGHT", 0.8))
    fixed_weight = float(getattr(cfg, "P_INIT_FIXED_WEIGHT", 0.2))
    p_base = (despl_weight * p_despl + fixed_weight * p_fixed).clamp(0.0, 1.0)

    uncertain = (
        (p_despl >= float(getattr(cfg, "DRE_SAFE_DESPL_UNCERT_LOW", 0.35)))
        & (p_despl <= float(getattr(cfg, "DRE_SAFE_DESPL_UNCERT_HIGH", 0.65)))
    )
    fixed_strong = p_fixed >= float(getattr(cfg, "DRE_SAFE_FIXED_FG_TH", 0.75))
    near_despl = _dilate(
        p_despl >= float(getattr(cfg, "DRE_SAFE_DESPL_FG_TH", 0.50)),
        int(getattr(cfg, "DRE_SAFE_DILATE_RADIUS", 2)),
    )
    candidate = uncertain & fixed_strong & near_despl

    positive_residual = (p_fixed - p_base).clamp(
        min=0.0,
        max=float(getattr(cfg, "DRE_SAFE_MAX_POS_RESIDUAL", 0.15)),
    )
    p_safe = p_base.clone()
    gamma = float(getattr(cfg, "DRE_SAFE_RESIDUAL_GAMMA", 0.5))
    p_safe = torch.where(candidate, p_safe + gamma * positive_residual, p_safe).clamp(0.0, 1.0)

    base_mask = p_base > 0.5
    safe_mask = p_safe > 0.5
    area_base = _area(base_mask)
    area_safe = _area(safe_mask)
    cc_base = _connected_components(base_mask)
    cc_safe = _connected_components(safe_mask)

    fallback = False
    if area_safe > area_base * float(getattr(cfg, "DRE_SAFE_AREA_GROWTH_MAX", 1.10)):
        fallback = True
    if area_safe < area_base * float(getattr(cfg, "DRE_SAFE_AREA_SHRINK_MIN", 0.90)):
        fallback = True
    if cc_safe > cc_base + int(getattr(cfg, "DRE_SAFE_CC_INCREASE_MAX", 3)):
        fallback = True
    if fallback:
        p_safe = p_base.clone()
        safe_mask = base_mask
        area_safe = area_base
        cc_safe = cc_base

    return {
        "p_base": p_base.contiguous(),
        "p_safe": p_safe.contiguous(),
        "candidate": candidate.bool().contiguous(),
        "fallback": bool(fallback),
        "area_base": float(area_base),
        "area_safe": float(area_safe),
        "candidate_ratio": _area(candidate),
        "cc_base": int(cc_base),
        "cc_safe": int(cc_safe),
        "safe_positive_delta_mean": float((p_safe - p_base).clamp_min(0.0).mean().item()),
        "safe_changed_ratio": float(((p_safe - p_base).abs() > 1e-6).float().mean().item()),
    }


def resize_single_channel(tensor, size):
    tensor = tensor.float()
    if list(tensor.shape[-2:]) == [int(size), int(size)]:
        return tensor.contiguous()
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).contiguous()
