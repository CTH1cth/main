import torch
import torch.nn.functional as F


def weighted_bce_with_logits(logits, target, weight=None):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        loss = loss * weight
        return loss.sum() / weight.sum().clamp_min(1.0)
    return loss.mean()


def weighted_iou_loss(prob, target, weight=None, eps=1e-6):
    if weight is None:
        weight = torch.ones_like(target)
    inter = (prob * target * weight).sum(dim=(1, 2, 3))
    union = ((prob + target - prob * target) * weight).sum(dim=(1, 2, 3))
    return (1.0 - (inter + eps) / (union + eps)).mean()


def dice_loss(prob, target, weight=None, eps=1e-6):
    if weight is None:
        weight = torch.ones_like(target)
    inter = (prob * target * weight).sum(dim=(1, 2, 3))
    denom = ((prob + target) * weight).sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def partial_bce(logits, target, mask):
    if mask.sum().item() == 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return loss[mask].mean()


def boundary_target(mask):
    pooled_max = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    pooled_min = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (pooled_max - pooled_min).clamp(0.0, 1.0)


def boundary_loss(boundary_logits, pseudo):
    return weighted_bce_with_logits(boundary_logits, boundary_target(pseudo))


def entropy_loss(prob, eps=1e-6):
    prob = prob.clamp(eps, 1.0 - eps)
    entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
    return entropy.mean()


def mnp_loss(prob, sobel, weight=None):
    edge = sobel.clamp(0.0, 1.0)
    pred_edge = boundary_target(prob)
    loss = F.l1_loss(pred_edge, edge, reduction="none")
    if weight is not None:
        loss = loss * weight
        return loss.sum() / weight.sum().clamp_min(1.0)
    return loss.mean()


def local_pseudo_loss(logits, local_pseudo, local_mask):
    if local_mask is None or local_mask.sum().item() == 0:
        return logits.sum() * 0.0
    return partial_bce(logits, local_pseudo, local_mask)


def _ssim_loss(first, second, eps=1e-6):
    mux = first.mean(dim=(-2, -1), keepdim=True)
    muy = second.mean(dim=(-2, -1), keepdim=True)
    vx = ((first - mux) ** 2).mean(dim=(-2, -1), keepdim=True)
    vy = ((second - muy) ** 2).mean(dim=(-2, -1), keepdim=True)
    cov = ((first - mux) * (second - muy)).mean(dim=(-2, -1), keepdim=True)
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2 * mux * muy + c1) * (2 * cov + c2)) / (
        (mux * mux + muy * muy + c1) * (vx + vy + c2) + eps
    )
    return (1.0 - ssim.clamp(0.0, 1.0)).mean()


def psta_loss(prob_high, prob_mid, prob_low):
    high_to_mid = F.interpolate(prob_high, size=prob_mid.shape[-2:], mode="bilinear", align_corners=False)
    mid_to_low = F.interpolate(prob_mid, size=prob_low.shape[-2:], mode="bilinear", align_corners=False)
    return (
        F.smooth_l1_loss(high_to_mid, prob_mid.detach())
        + F.smooth_l1_loss(mid_to_low, prob_low.detach())
        + _ssim_loss(high_to_mid, prob_mid.detach())
        + _ssim_loss(mid_to_low, prob_low.detach())
    )


def contrast_loss(features, target):
    fused = features.get("fused") if isinstance(features, dict) else None
    if fused is None:
        return target.sum() * 0.0
    small_target = F.interpolate(target, size=fused.shape[-2:], mode="bilinear", align_corners=False)
    fg = small_target > 0.7
    bg = small_target < 0.2
    if fg.sum().item() == 0 or bg.sum().item() == 0:
        return fused.sum() * 0.0
    feat = F.normalize(fused, dim=1)
    fg_proto = feat.permute(0, 2, 3, 1)[fg.squeeze(1)].mean(dim=0)
    bg_proto = feat.permute(0, 2, 3, 1)[bg.squeeze(1)].mean(dim=0)
    return F.cosine_similarity(fg_proto, bg_proto, dim=0).clamp_min(0.0)
