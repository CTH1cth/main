try:
    import cv2
except Exception:
    cv2 = None

import math

import numpy as np
import torch
import torch.nn.functional as F


def _safe_get_cfg(cfg, name, default):
    return getattr(cfg, name, default)


def get_gkd_mode(cfg) -> str:
    mode = str(getattr(cfg, "GKD_MODE", "")).lower().strip()
    if mode in {"off", "audit", "reweight", "branch"}:
        return mode
    if bool(getattr(cfg, "USE_GKD_LITE", False)):
        return "reweight"
    return "off"


def is_gkd_enabled(cfg) -> bool:
    return get_gkd_mode(cfg) in {"audit", "reweight", "branch"}


def is_gkd_v3_enabled(cfg) -> bool:
    return (
        bool(getattr(cfg, "GKD_ENABLE_DYNAMIC_HIGH_CAP", False))
        or int(getattr(cfg, "GKD_DISABLE_AFTER_EPOCH", -1)) > 0
    )


def _binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _component_stats(mask: np.ndarray):
    fg_area = int(mask.sum())
    if fg_area <= 0:
        return 0, 0.0

    if cv2 is not None:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return 0, 0.0
        areas = stats[1:, cv2.CC_STAT_AREA]
        num_cc = len(areas)
        largest = float(areas.max()) if len(areas) > 0 else 0.0
    else:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        areas = []
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                area = 0
                while stack:
                    cy, cx = stack.pop()
                    area += 1
                    for ny in range(max(0, cy - 1), min(height, cy + 2)):
                        for nx in range(max(0, cx - 1), min(width, cx + 2)):
                            if visited[ny, nx] or not mask[ny, nx]:
                                continue
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                areas.append(area)
        num_cc = len(areas)
        largest = float(max(areas)) if areas else 0.0
    largest_ratio = largest / max(float(fg_area), 1.0)
    return int(num_cc), float(largest_ratio)


def _edge_touch_ratio(mask: np.ndarray) -> float:
    fg_area = int(mask.sum())
    if fg_area <= 0:
        return 1.0

    edge = np.zeros_like(mask, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True

    edge_fg = np.logical_and(mask, edge).sum()
    return float(edge_fg) / float(fg_area)


def _score_area(area, cfg):
    if area < float(_safe_get_cfg(cfg, "GKD_AREA_MIN_LOW", 0.005)):
        return 0.0
    if area > float(_safe_get_cfg(cfg, "GKD_AREA_MAX_LOW", 0.80)):
        return 0.0
    if (
        float(_safe_get_cfg(cfg, "GKD_AREA_MIN_HIGH", 0.01))
        <= area
        <= float(_safe_get_cfg(cfg, "GKD_AREA_MAX_HIGH", 0.60))
    ):
        return 1.0
    return 0.5


def _score_component(num_cc, largest_cc_ratio, cfg):
    if (
        num_cc <= int(_safe_get_cfg(cfg, "GKD_CC_HIGH_MAX", 4))
        and largest_cc_ratio >= float(_safe_get_cfg(cfg, "GKD_LCC_HIGH_MIN", 0.55))
    ):
        return 1.0
    if (
        num_cc <= int(_safe_get_cfg(cfg, "GKD_CC_MID_MAX", 8))
        and largest_cc_ratio >= float(_safe_get_cfg(cfg, "GKD_LCC_MID_MIN", 0.35))
    ):
        return 0.5
    return 0.0


def _score_edge(edge_touch_ratio, cfg):
    if edge_touch_ratio <= float(_safe_get_cfg(cfg, "GKD_EDGE_HIGH_MAX", 0.15)):
        return 1.0
    if edge_touch_ratio <= float(_safe_get_cfg(cfg, "GKD_EDGE_MID_MAX", 0.30)):
        return 0.5
    return 0.0


def _score_agreement(iou, cfg):
    if iou < 0:
        return 0.5
    if iou >= float(_safe_get_cfg(cfg, "GKD_AGREE_HIGH_MIN", 0.50)):
        return 1.0
    if iou >= float(_safe_get_cfg(cfg, "GKD_AGREE_MID_MIN", 0.25)):
        return 0.5
    return 0.0


def compute_gkd_quality(
    p_despl: torch.Tensor,
    p_fixed: torch.Tensor,
    cfg,
    teacher_prob=None,
    epoch=None,
    entropy_mean_per_sample=None,
):
    if p_despl.ndim != 4 or p_despl.shape[1] != 1:
        raise RuntimeError(f"p_despl must be [B,1,H,W], got {list(p_despl.shape)}")

    p_despl_cpu = p_despl.detach().float().clamp(0.0, 1.0).cpu()
    p_fixed_cpu = None
    if p_fixed is not None and p_fixed.shape == p_despl.shape:
        p_fixed_cpu = p_fixed.detach().float().clamp(0.0, 1.0).cpu()
    teacher_cpu = None
    if teacher_prob is not None and teacher_prob.shape == p_despl.shape:
        teacher_cpu = teacher_prob.detach().float().clamp(0.0, 1.0).cpu()
    entropy_cpu = None
    if entropy_mean_per_sample is not None:
        entropy_cpu = entropy_mean_per_sample.detach().float().cpu().view(-1)

    batch_size = int(p_despl_cpu.shape[0])
    grades = []
    raw_grades = []
    q_scores = []
    sample_weights = []
    per_sample = []
    grade_reasons = []

    stats = {
        "num_samples": batch_size,
        "num_high": 0,
        "num_normal": 0,
        "num_low": 0,
        "num_raw_high": 0,
        "num_raw_normal": 0,
        "num_raw_low": 0,
        "num_strict_high_block": 0,
        "num_high_downgrade": 0,
        "num_static_low": 0,
        "num_teacher_low": 0,
        "num_teacher_normal_cap": 0,
        "num_dynamic_low": 0,
        "num_dynamic_high_cap": 0,
        "num_entropy_low": 0,
        "num_entropy_normal_cap": 0,
        "q_score_mean": 0.0,
        "sample_weight_mean": 0.0,
        "area_mean": 0.0,
        "num_cc_mean": 0.0,
        "largest_cc_ratio_mean": 0.0,
        "edge_touch_ratio_mean": 0.0,
        "despl_fixed_iou_mean": 0.0,
        "despl_fixed_iou_count": 0,
        "teacher_despl_iou_mean": 0.0,
        "teacher_despl_iou_count": 0,
        "teacher_area_mean": 0.0,
        "teacher_area_count": 0,
        "teacher_despl_iou_values": [],
        "teacher_area_values": [],
    }
    use_v3 = is_gkd_v3_enabled(cfg)
    use_v2 = (not use_v3) and any(
        bool(_safe_get_cfg(cfg, name, False))
        for name in (
            "GKD_ENABLE_STRICT_HIGH_GATE",
            "GKD_ENABLE_HARD_DOWNGRADE",
            "GKD_ENABLE_DYNAMIC_LOW",
            "GKD_ENABLE_ENTROPY_GRADE",
        )
    )
    if use_v3:
        q_high = float(
            _safe_get_cfg(
                cfg,
                "GKD_V3_Q_HIGH",
                float(_safe_get_cfg(cfg, "GKD_V2_Q_HIGH", _safe_get_cfg(cfg, "GKD_Q_HIGH", 0.75))),
            )
        )
        q_normal = float(
            _safe_get_cfg(
                cfg,
                "GKD_V3_Q_NORMAL",
                float(_safe_get_cfg(cfg, "GKD_V2_Q_NORMAL", _safe_get_cfg(cfg, "GKD_Q_NORMAL", 0.45))),
            )
        )
    elif use_v2:
        q_high = float(_safe_get_cfg(cfg, "GKD_V2_Q_HIGH", float(_safe_get_cfg(cfg, "GKD_Q_HIGH", 0.75))))
        q_normal = float(_safe_get_cfg(cfg, "GKD_V2_Q_NORMAL", float(_safe_get_cfg(cfg, "GKD_Q_NORMAL", 0.45))))
    else:
        q_high = float(_safe_get_cfg(cfg, "GKD_Q_HIGH", 0.75))
        q_normal = float(_safe_get_cfg(cfg, "GKD_Q_NORMAL", 0.45))

    for idx in range(batch_size):
        despl_np = p_despl_cpu[idx, 0].numpy()
        mask = despl_np > 0.5
        area = float(mask.mean())
        num_cc, largest_cc_ratio = _component_stats(mask)
        edge_touch = _edge_touch_ratio(mask)

        if p_fixed_cpu is None:
            iou = -1.0
        else:
            fixed_mask = p_fixed_cpu[idx, 0].numpy() > 0.5
            iou = _binary_iou(mask, fixed_mask)

        area_score = _score_area(area, cfg)
        component_score = _score_component(num_cc, largest_cc_ratio, cfg)
        edge_score = _score_edge(edge_touch, cfg)
        agreement_score = _score_agreement(iou, cfg)
        q_score = (
            0.35 * area_score
            + 0.30 * component_score
            + 0.20 * edge_score
            + 0.15 * agreement_score
        )

        if q_score >= q_high:
            raw_grade = 2
            stats["num_raw_high"] += 1
        elif q_score >= q_normal:
            raw_grade = 1
            stats["num_raw_normal"] += 1
        else:
            raw_grade = 0
            stats["num_raw_low"] += 1

        grade = raw_grade
        reason = []

        teacher_area = -1.0
        teacher_iou = -1.0
        if teacher_cpu is not None:
            teacher_mask = teacher_cpu[idx, 0].numpy() > 0.5
            teacher_area = float(teacher_mask.mean())
            teacher_iou = _binary_iou(teacher_mask, mask)
            stats["teacher_area_mean"] += teacher_area
            stats["teacher_area_count"] += 1
            stats["teacher_area_values"].append(teacher_area)
            stats["teacher_despl_iou_mean"] += teacher_iou
            stats["teacher_despl_iou_count"] += 1
            stats["teacher_despl_iou_values"].append(teacher_iou)

        if bool(_safe_get_cfg(cfg, "GKD_ENABLE_STRICT_HIGH_GATE", False)):
            high_allowed = (
                area >= float(_safe_get_cfg(cfg, "GKD_HIGH_AREA_MIN", 0.01))
                and area <= float(_safe_get_cfg(cfg, "GKD_HIGH_AREA_MAX", 0.60))
                and num_cc <= int(_safe_get_cfg(cfg, "GKD_HIGH_CC_MAX", 4))
                and largest_cc_ratio >= float(_safe_get_cfg(cfg, "GKD_HIGH_LCC_MIN", 0.70))
                and edge_touch <= float(_safe_get_cfg(cfg, "GKD_HIGH_EDGE_MAX", 0.15))
                and iou >= float(_safe_get_cfg(cfg, "GKD_HIGH_AGREE_MIN", 0.50))
            )
            if grade == 2 and not high_allowed:
                grade = 1
                stats["num_strict_high_block"] += 1
                reason.append("strict_high_block")

        if bool(_safe_get_cfg(cfg, "GKD_ENABLE_HARD_DOWNGRADE", False)):
            hard_downgrade = (
                num_cc > int(_safe_get_cfg(cfg, "GKD_DOWNGRADE_CC_MAX", 6))
                or iou < float(_safe_get_cfg(cfg, "GKD_DOWNGRADE_AGREE_MIN", 0.45))
                or area > float(_safe_get_cfg(cfg, "GKD_DOWNGRADE_AREA_MAX", 0.70))
                or edge_touch > float(_safe_get_cfg(cfg, "GKD_DOWNGRADE_EDGE_MAX", 0.20))
            )
            if grade == 2 and hard_downgrade:
                grade = 1
                stats["num_high_downgrade"] += 1
                reason.append("hard_downgrade")

        static_low = (
            area < float(_safe_get_cfg(cfg, "GKD_STATIC_LOW_AREA_MIN", 0.003))
            or area > float(_safe_get_cfg(cfg, "GKD_STATIC_LOW_AREA_MAX", 0.85))
            or num_cc > int(_safe_get_cfg(cfg, "GKD_STATIC_LOW_CC_MAX", 15))
            or largest_cc_ratio < float(_safe_get_cfg(cfg, "GKD_STATIC_LOW_LCC_MIN", 0.20))
            or edge_touch > float(_safe_get_cfg(cfg, "GKD_STATIC_LOW_EDGE_MAX", 0.45))
            or iou < float(_safe_get_cfg(cfg, "GKD_STATIC_LOW_AGREE_MIN", 0.15))
        )
        if (use_v2 or use_v3) and static_low:
            grade = 0
            stats["num_static_low"] += 1
            if not use_v3:
                stats["num_dynamic_low"] += 1
            reason.append("static_low")

        disable_dynamic_low = use_v3 and bool(_safe_get_cfg(cfg, "GKD_V3_LOW_ONLY_STATIC", True))
        dynamic_enabled = (
            (not disable_dynamic_low)
            and bool(_safe_get_cfg(cfg, "GKD_ENABLE_DYNAMIC_LOW", False))
            and teacher_cpu is not None
            and epoch is not None
            and int(epoch) >= int(_safe_get_cfg(cfg, "GKD_DYNAMIC_LOW_START_EPOCH", 6))
        )
        if dynamic_enabled:
            teacher_valid = (
                teacher_area >= float(_safe_get_cfg(cfg, "GKD_TEACHER_AREA_MIN", 0.005))
                and teacher_area <= float(_safe_get_cfg(cfg, "GKD_TEACHER_AREA_MAX", 0.80))
            )
            if teacher_valid:
                if teacher_iou < float(_safe_get_cfg(cfg, "GKD_TEACHER_DESPL_LOW_IOU", 0.35)):
                    if grade != 0:
                        stats["num_dynamic_low"] += 1
                    grade = 0
                    stats["num_teacher_low"] += 1
                    reason.append("teacher_despl_low")
                elif teacher_iou < float(_safe_get_cfg(cfg, "GKD_TEACHER_DESPL_NORMAL_IOU", 0.55)):
                    if grade == 2:
                        grade = 1
                        stats["num_teacher_normal_cap"] += 1
                        reason.append("teacher_despl_normal_cap")

        dynamic_high_cap_enabled = (
            use_v3
            and bool(_safe_get_cfg(cfg, "GKD_ENABLE_DYNAMIC_HIGH_CAP", False))
            and teacher_cpu is not None
            and epoch is not None
            and int(epoch) >= int(_safe_get_cfg(cfg, "GKD_DYNAMIC_HIGH_CAP_START_EPOCH", 6))
        )
        if dynamic_high_cap_enabled:
            teacher_valid = (
                teacher_area >= float(_safe_get_cfg(cfg, "GKD_TEACHER_AREA_MIN", 0.005))
                and teacher_area <= float(_safe_get_cfg(cfg, "GKD_TEACHER_AREA_MAX", 0.80))
            )
            if (
                teacher_valid
                and teacher_iou < float(_safe_get_cfg(cfg, "GKD_TEACHER_DESPL_HIGH_CAP_IOU", 0.55))
                and grade == 2
            ):
                grade = 1
                stats["num_dynamic_high_cap"] += 1
                reason.append("dynamic_high_cap")

        if (
            bool(_safe_get_cfg(cfg, "GKD_ENABLE_ENTROPY_GRADE", False))
            and entropy_cpu is not None
            and idx < int(entropy_cpu.numel())
        ):
            entropy_mean = float(entropy_cpu[idx].item())
            if entropy_mean > float(_safe_get_cfg(cfg, "GKD_ENTROPY_LOW_MEAN", 0.65)):
                if grade != 0:
                    stats["num_dynamic_low"] += 1
                grade = 0
                stats["num_entropy_low"] += 1
                reason.append("entropy_low")
            elif (
                entropy_mean > float(_safe_get_cfg(cfg, "GKD_ENTROPY_NORMAL_MEAN", 0.50))
                and grade == 2
            ):
                grade = 1
                stats["num_entropy_normal_cap"] += 1
                reason.append("entropy_normal_cap")

        if grade == 2:
            sample_weight = float(_safe_get_cfg(cfg, "GKD_SAMPLE_W_HIGH", 1.15))
            stats["num_high"] += 1
        elif grade == 1:
            sample_weight = float(_safe_get_cfg(cfg, "GKD_SAMPLE_W_NORMAL", 1.00))
            stats["num_normal"] += 1
        else:
            sample_weight = float(_safe_get_cfg(cfg, "GKD_SAMPLE_W_LOW", 0.60))
            stats["num_low"] += 1

        grades.append(grade)
        raw_grades.append(raw_grade)
        q_scores.append(float(q_score))
        sample_weights.append(sample_weight)
        grade_reason = ";".join(reason) if reason else "none"
        grade_reasons.append(grade_reason)
        stats["q_score_mean"] += float(q_score)
        stats["sample_weight_mean"] += sample_weight
        stats["area_mean"] += area
        stats["num_cc_mean"] += float(num_cc)
        stats["largest_cc_ratio_mean"] += largest_cc_ratio
        stats["edge_touch_ratio_mean"] += edge_touch
        if iou >= 0:
            stats["despl_fixed_iou_mean"] += iou
            stats["despl_fixed_iou_count"] += 1

        per_sample.append(
            {
                "area": area,
                "num_cc": int(num_cc),
                "largest_cc_ratio": largest_cc_ratio,
                "edge_touch_ratio": edge_touch,
                "despl_fixed_iou": iou,
                "teacher_area": teacher_area,
                "teacher_despl_iou": teacher_iou,
                "q_score": float(q_score),
                "raw_grade": int(raw_grade),
                "grade": int(grade),
                "final_grade": int(grade),
                "grade_reason": grade_reason,
                "branch_type": "high" if grade == 2 else "normal" if grade == 1 else "low",
                "sample_weight": float(sample_weight),
            }
        )

    denom = max(batch_size, 1)
    for key in (
        "q_score_mean",
        "sample_weight_mean",
        "area_mean",
        "num_cc_mean",
        "largest_cc_ratio_mean",
        "edge_touch_ratio_mean",
    ):
        stats[key] = float(stats[key]) / denom
    if stats["despl_fixed_iou_count"] > 0:
        stats["despl_fixed_iou_mean"] = (
            float(stats["despl_fixed_iou_mean"]) / stats["despl_fixed_iou_count"]
        )
    else:
        stats["despl_fixed_iou_mean"] = -1.0
    if stats["teacher_despl_iou_count"] > 0:
        stats["teacher_despl_iou_mean"] = (
            float(stats["teacher_despl_iou_mean"]) / stats["teacher_despl_iou_count"]
        )
    else:
        stats["teacher_despl_iou_mean"] = -1.0
    if stats["teacher_area_count"] > 0:
        stats["teacher_area_mean"] = float(stats["teacher_area_mean"]) / stats["teacher_area_count"]
    else:
        stats["teacher_area_mean"] = -1.0

    device = p_despl.device
    grade_tensor = torch.tensor(grades, dtype=torch.long, device=device)
    raw_grade_tensor = torch.tensor(raw_grades, dtype=torch.long, device=device)
    q_score_tensor = torch.tensor(q_scores, dtype=torch.float32, device=device)
    sample_weight_raw = torch.tensor(
        sample_weights, dtype=torch.float32, device=device
    ).view(batch_size, 1, 1, 1)

    return {
        "grade": grade_tensor,
        "raw_grade": raw_grade_tensor,
        "q_score": q_score_tensor,
        "sample_weight_raw": sample_weight_raw,
        "stats": stats,
        "per_sample": per_sample,
        "grade_reason": grade_reasons,
        "teacher_despl_iou": torch.tensor(
            [float(item["teacher_despl_iou"]) for item in per_sample],
            dtype=torch.float32,
            device=device,
        ),
        "teacher_area": torch.tensor(
            [float(item["teacher_area"]) for item in per_sample],
            dtype=torch.float32,
            device=device,
        ),
    }


def compute_pixel_weight(p_despl: torch.Tensor, cfg):
    p = p_despl.float().clamp(0.0, 1.0)
    conf = (p - 0.5).abs() * 2.0
    conf = conf.clamp(0.0, 1.0)

    w_min = float(_safe_get_cfg(cfg, "GKD_PIXEL_W_MIN", 0.60))
    pixel_weight = w_min + (1.0 - w_min) * conf
    return pixel_weight


def apply_gkd_strength(sample_weight_raw, pixel_weight_raw, epoch: int, cfg):
    if epoch <= 20:
        strength = 1.0
    else:
        strength = float(_safe_get_cfg(cfg, "GKD_LATE_STRENGTH", 0.30))

    sample_weight = 1.0 + strength * (sample_weight_raw - 1.0)
    pixel_weight = 1.0 + strength * (pixel_weight_raw - 1.0)
    return sample_weight, pixel_weight, strength


def reduce_loss_per_sample(loss_map, weight=None, eps=1e-6):
    if loss_map.ndim != 4:
        raise RuntimeError(f"loss_map must be [B,1,H,W], got {list(loss_map.shape)}")
    if weight is None:
        return loss_map.mean(dim=(1, 2, 3))
    if list(weight.shape) != list(loss_map.shape):
        raise RuntimeError(
            f"weight/loss_map shape mismatch: {list(weight.shape)} != {list(loss_map.shape)}"
        )
    weighted = loss_map * weight
    denom = weight.sum(dim=(1, 2, 3)).clamp_min(float(eps))
    return weighted.sum(dim=(1, 2, 3)) / denom


def summarize_tensor(x: torch.Tensor, prefix: str):
    flat = x.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_lt_0_7_ratio": 0.0,
            f"{prefix}_gt_0_95_ratio": 0.0,
        }
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).item()),
        f"{prefix}_min": float(flat.min().item()),
        f"{prefix}_max": float(flat.max().item()),
        f"{prefix}_p10": float(torch.quantile(flat, 0.10).item()),
        f"{prefix}_p50": float(torch.quantile(flat, 0.50).item()),
        f"{prefix}_p90": float(torch.quantile(flat, 0.90).item()),
        f"{prefix}_lt_0_7_ratio": float((flat < 0.7).float().mean().item()),
        f"{prefix}_gt_0_95_ratio": float((flat > 0.95).float().mean().item()),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if bool(mask.any().item()):
        return float(values[mask].detach().mean().item())
    return 0.0


def _quality_stats_for_info(q_info):
    stats = q_info["stats"]
    teacher_iou_values = stats.get("teacher_despl_iou_values", [])
    teacher_area_values = stats.get("teacher_area_values", [])
    if teacher_iou_values:
        teacher_iou_tensor = torch.tensor(teacher_iou_values, dtype=torch.float32)
        teacher_iou_p10 = float(torch.quantile(teacher_iou_tensor, 0.10).item())
        teacher_iou_p50 = float(torch.quantile(teacher_iou_tensor, 0.50).item())
        teacher_iou_p90 = float(torch.quantile(teacher_iou_tensor, 0.90).item())
    else:
        teacher_iou_p10 = -1.0
        teacher_iou_p50 = -1.0
        teacher_iou_p90 = -1.0
    return {
        "num_samples": int(stats.get("num_samples", 0)),
        "num_high": int(stats.get("num_high", 0)),
        "num_normal": int(stats.get("num_normal", 0)),
        "num_low": int(stats.get("num_low", 0)),
        "num_raw_high": int(stats.get("num_raw_high", 0)),
        "num_raw_normal": int(stats.get("num_raw_normal", 0)),
        "num_raw_low": int(stats.get("num_raw_low", 0)),
        "num_strict_high_block": int(stats.get("num_strict_high_block", 0)),
        "num_high_downgrade": int(stats.get("num_high_downgrade", 0)),
        "num_static_low": int(stats.get("num_static_low", 0)),
        "num_teacher_low": int(stats.get("num_teacher_low", 0)),
        "num_teacher_normal_cap": int(stats.get("num_teacher_normal_cap", 0)),
        "num_dynamic_low": int(stats.get("num_dynamic_low", 0)),
        "num_dynamic_high_cap": int(stats.get("num_dynamic_high_cap", 0)),
        "num_entropy_low": int(stats.get("num_entropy_low", 0)),
        "num_entropy_normal_cap": int(stats.get("num_entropy_normal_cap", 0)),
        "q_score_mean": float(stats.get("q_score_mean", 0.0)),
        "area_mean": float(stats.get("area_mean", 0.0)),
        "num_cc_mean": float(stats.get("num_cc_mean", 0.0)),
        "largest_cc_ratio_mean": float(stats.get("largest_cc_ratio_mean", 0.0)),
        "edge_touch_ratio_mean": float(stats.get("edge_touch_ratio_mean", 0.0)),
        "despl_fixed_iou_mean": float(stats.get("despl_fixed_iou_mean", -1.0)),
        "despl_fixed_iou_count": int(stats.get("despl_fixed_iou_count", 0)),
        "teacher_despl_iou_mean": float(stats.get("teacher_despl_iou_mean", -1.0)),
        "teacher_despl_iou_p10": teacher_iou_p10,
        "teacher_despl_iou_p50": teacher_iou_p50,
        "teacher_despl_iou_p90": teacher_iou_p90,
        "teacher_area_mean": float(stats.get("teacher_area_mean", -1.0)),
    }


def compute_plain_and_reweight_loss_for_audit(
    student_logits,
    teacher_logits,
    mixed_target,
    p_despl,
    p_fixed,
    epoch,
    cfg,
):
    bce_map = F.binary_cross_entropy_with_logits(
        student_logits,
        mixed_target,
        reduction="none",
    )
    plain_per_sample = reduce_loss_per_sample(bce_map, weight=None)
    plain_loss = plain_per_sample.mean()

    teacher_prob = torch.sigmoid(teacher_logits).detach() if teacher_logits is not None else None
    q_info = compute_gkd_quality(
        p_despl=p_despl,
        p_fixed=p_fixed,
        cfg=cfg,
        teacher_prob=teacher_prob,
        epoch=epoch,
    )
    pixel_weight_raw = compute_pixel_weight(p_despl=p_despl, cfg=cfg).to(student_logits.device)
    sample_weight_raw = q_info["sample_weight_raw"].to(student_logits.device)
    sample_weight, pixel_weight, strength = apply_gkd_strength(
        sample_weight_raw=sample_weight_raw,
        pixel_weight_raw=pixel_weight_raw,
        epoch=epoch,
        cfg=cfg,
    )

    weight = sample_weight * pixel_weight
    reweight_loss = (bce_map * weight).sum() / weight.sum().clamp_min(1e-6)
    relative_delta = (reweight_loss - plain_loss) / plain_loss.clamp_min(1e-6)

    sample_stats = summarize_tensor(sample_weight.view(-1), "sample_weight")
    pixel_stats = summarize_tensor(pixel_weight, "pixel_weight")
    info = {
        "mode": "audit",
        "plain_loss": float(plain_loss.detach().item()),
        "reweight_loss": float(reweight_loss.detach().item()),
        "relative_delta": float(relative_delta.detach().item()),
        "strength": float(strength),
        "per_sample": q_info["per_sample"],
        "sample_weight_mean": sample_stats["sample_weight_mean"],
        "sample_weight_std": sample_stats["sample_weight_std"],
        "sample_weight_min": sample_stats["sample_weight_min"],
        "sample_weight_max": sample_stats["sample_weight_max"],
    }
    info.update(_quality_stats_for_info(q_info))
    info.update(pixel_stats)
    info["reweight_loss_tensor"] = reweight_loss
    return info


def compute_entropy_pixel_weight(
    p_despl,
    teacher_prob,
    p_fixed,
    epoch,
    cfg,
):
    fallback = compute_pixel_weight(p_despl, cfg)
    start_epoch = int(_safe_get_cfg(cfg, "GKD_ENTROPY_START_EPOCH", 2))
    if epoch < start_epoch:
        stats = {
            "entropy_mean": -1.0,
            "entropy_std": 0.0,
            "entropy_p50": -1.0,
            "entropy_high_ratio": 0.0,
            "entropy_pixel_weight_mean": float(fallback.detach().mean().item()),
        }
        return fallback, stats

    candidates_cfg = _safe_get_cfg(
        cfg,
        "GKD_ENTROPY_CANDIDATES",
        ["despl", "teacher", "fixed"],
    )
    candidates_cfg = {str(value).lower() for value in candidates_cfg}
    candidates = []
    if "despl" in candidates_cfg:
        candidates.append(p_despl.float().clamp(0.0, 1.0))
    if "teacher" in candidates_cfg and teacher_prob is not None:
        candidates.append(teacher_prob.detach().float().clamp(0.0, 1.0))
    if "fixed" in candidates_cfg and p_fixed is not None:
        candidates.append(p_fixed.float().clamp(0.0, 1.0))

    if len(candidates) < 2:
        stats = {
            "entropy_mean": -1.0,
            "entropy_std": 0.0,
            "entropy_p50": -1.0,
            "entropy_high_ratio": 0.0,
            "entropy_pixel_weight_mean": float(fallback.detach().mean().item()),
        }
        return fallback, stats

    v_bar = torch.stack(candidates, dim=0).mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
    entropy = -v_bar * torch.log(v_bar) - (1.0 - v_bar) * torch.log(1.0 - v_bar)
    entropy_norm = (entropy / math.log(2.0)).clamp(0.0, 1.0)
    weight_min = float(_safe_get_cfg(cfg, "GKD_ENTROPY_WEIGHT_MIN", 0.50))
    pixel_weight = weight_min + (1.0 - weight_min) * (1.0 - entropy_norm)
    entropy_flat = entropy_norm.detach().float().reshape(-1).cpu()
    stats = {
        "entropy_mean": float(entropy_flat.mean().item()),
        "entropy_std": float(entropy_flat.std(unbiased=False).item()),
        "entropy_p50": float(torch.quantile(entropy_flat, 0.50).item()),
        "entropy_high_ratio": float((entropy_flat > 0.75).float().mean().item()),
        "entropy_pixel_weight_mean": float(pixel_weight.detach().mean().item()),
    }
    return pixel_weight, stats


def compute_gkd_branch_loss(
    student_logits,
    teacher_logits,
    mixed_target,
    p_despl,
    p_fixed,
    epoch,
    cfg,
):
    student_prob = torch.sigmoid(student_logits)
    teacher_prob = torch.sigmoid(teacher_logits).detach()
    bce_map = F.binary_cross_entropy_with_logits(
        student_logits,
        mixed_target,
        reduction="none",
    )
    plain_bce = reduce_loss_per_sample(bce_map, weight=None).mean()

    disable_after = int(_safe_get_cfg(cfg, "GKD_DISABLE_AFTER_EPOCH", -1))
    if disable_after > 0 and int(epoch) > disable_after:
        batch_size = int(student_logits.shape[0])
        loss = plain_bce
        info = {
            "mode": "branch",
            "strength": 0.0,
            "branch_strength": 0.0,
            "disabled_after_epoch": True,
            "loss_used": "plain_bce",
            "plain_bce": float(plain_bce.detach().item()),
            "branch_loss": float(loss.detach().item()),
            "branch_delta": 0.0,
            "low_loss": 0.0,
            "normal_loss": 0.0,
            "high_loss": 0.0,
            "high_bce": 0.0,
            "high_l1": 0.0,
            "high_mse": 0.0,
            "teacher_l1": 0.0,
            "teacher_bce": float(plain_bce.detach().item()),
            "high_extra_ratio": 0.0,
            "num_samples": batch_size,
            "num_high": 0,
            "num_normal": 0,
            "num_low": 0,
            "num_raw_high": 0,
            "num_raw_normal": 0,
            "num_raw_low": 0,
            "num_strict_high_block": 0,
            "num_high_downgrade": 0,
            "num_static_low": 0,
            "num_teacher_low": 0,
            "num_teacher_normal_cap": 0,
            "num_dynamic_low": 0,
            "num_dynamic_high_cap": 0,
            "teacher_despl_iou_mean": -1.0,
            "teacher_despl_iou_p10": -1.0,
            "teacher_despl_iou_p50": -1.0,
            "teacher_despl_iou_p90": -1.0,
            "teacher_area_mean": -1.0,
            "pixel_weight_mean": 1.0,
            "pixel_weight_std": 0.0,
            "entropy_mean": -1.0,
            "entropy_std": 0.0,
            "entropy_high_ratio": 0.0,
            "entropy_weight_used": False,
            "per_sample": [],
            "pseudo_box_bg_enabled": bool(_safe_get_cfg(cfg, "GKD_USE_PSEUDO_BOX_BG", False)),
        }
        return loss, info

    q_info = compute_gkd_quality(
        p_despl=p_despl,
        p_fixed=p_fixed,
        cfg=cfg,
        teacher_prob=teacher_prob,
        epoch=epoch,
    )
    grade = q_info["grade"].to(student_logits.device)

    if bool(_safe_get_cfg(cfg, "GKD_USE_ENTROPY_PIXEL_WEIGHT", True)):
        pixel_weight, entropy_stats = compute_entropy_pixel_weight(
            p_despl=p_despl,
            teacher_prob=teacher_prob,
            p_fixed=p_fixed,
            epoch=epoch,
            cfg=cfg,
        )
        entropy_weight_used = True
    else:
        pixel_weight = torch.ones_like(p_despl)
        _, entropy_stats = compute_entropy_pixel_weight(
            p_despl=p_despl,
            teacher_prob=teacher_prob,
            p_fixed=p_fixed,
            epoch=epoch,
            cfg=cfg,
        )
        entropy_stats["entropy_pixel_weight_mean"] = float(pixel_weight.detach().mean().item())
        entropy_weight_used = False
    pixel_weight = pixel_weight.to(student_logits.device)

    bce_per_sample = reduce_loss_per_sample(bce_map, weight=pixel_weight)
    teacher_binary = (teacher_prob > 0.5).float()
    teacher_bce_map = F.binary_cross_entropy_with_logits(
        student_logits,
        teacher_binary,
        reduction="none",
    )
    teacher_bce_per_sample = reduce_loss_per_sample(teacher_bce_map, weight=pixel_weight)

    teacher_l1_map = torch.abs(student_prob - teacher_prob)
    teacher_l1_per_sample = reduce_loss_per_sample(teacher_l1_map, weight=pixel_weight)

    despl_l1_map = torch.abs(student_prob - p_despl)
    despl_l1_per_sample = reduce_loss_per_sample(despl_l1_map, weight=pixel_weight)

    despl_mse_map = (student_prob - p_despl) ** 2
    despl_mse_per_sample = reduce_loss_per_sample(despl_mse_map, weight=pixel_weight)

    branch_strength = (
        1.0
        if epoch <= 20
        else float(_safe_get_cfg(cfg, "GKD_BRANCH_LATE_STRENGTH", 0.30))
    )
    l1_weight = float(_safe_get_cfg(cfg, "GKD_BRANCH_L1_WEIGHT", 1.0))
    mse_weight = float(_safe_get_cfg(cfg, "GKD_BRANCH_MSE_WEIGHT", 1.0))
    low_weight = float(_safe_get_cfg(cfg, "GKD_BRANCH_LOW_WEIGHT", 1.0))
    normal_weight = float(_safe_get_cfg(cfg, "GKD_BRANCH_NORMAL_WEIGHT", 1.0))
    high_weight = float(_safe_get_cfg(cfg, "GKD_BRANCH_HIGH_WEIGHT", 1.0))
    low_mode = str(_safe_get_cfg(cfg, "GKD_BRANCH_LOW_MODE", "teacher_l1"))
    low_teacher_bce_weight = float(_safe_get_cfg(cfg, "GKD_LOW_TEACHER_BCE_WEIGHT", 0.3))
    low_teacher_l1_weight = float(_safe_get_cfg(cfg, "GKD_LOW_TEACHER_L1_WEIGHT", 0.3))

    low_mask = grade == 0
    normal_mask = grade == 1
    high_mask = grade == 2

    if low_mode == "teacher_l1":
        low_loss = teacher_l1_per_sample
    elif low_mode == "teacher_l1_plus_teacher_bce":
        low_loss = teacher_l1_per_sample + low_teacher_bce_weight * teacher_bce_per_sample
    elif low_mode == "teacher_bce_l1":
        low_loss = (
            low_teacher_bce_weight * teacher_bce_per_sample
            + low_teacher_l1_weight * teacher_l1_per_sample
        )
    else:
        raise RuntimeError(f"Unknown GKD_BRANCH_LOW_MODE: {low_mode}")
    normal_loss = bce_per_sample
    high_extra = (
        branch_strength * l1_weight * despl_l1_per_sample
        + branch_strength * mse_weight * despl_mse_per_sample
    )
    high_loss = (
        bce_per_sample
        + high_extra
    )

    loss_per_sample = torch.zeros_like(bce_per_sample)
    loss_per_sample[low_mask] = low_weight * low_loss[low_mask]
    loss_per_sample[normal_mask] = normal_weight * normal_loss[normal_mask]
    loss_per_sample[high_mask] = high_weight * high_loss[high_mask]
    loss = loss_per_sample.mean()
    branch_delta = (loss - plain_bce) / plain_bce.clamp_min(1e-6)
    high_bce_mean_tensor = bce_per_sample[high_mask].mean() if bool(high_mask.any().item()) else student_logits.sum() * 0.0
    high_extra_mean_tensor = high_extra[high_mask].mean() if bool(high_mask.any().item()) else student_logits.sum() * 0.0
    high_extra_ratio = high_extra_mean_tensor / high_bce_mean_tensor.clamp_min(1e-6)

    pixel_stats = summarize_tensor(pixel_weight, "pixel_weight")
    info = {
        "mode": "branch",
        "strength": float(branch_strength),
        "branch_strength": float(branch_strength),
        "plain_bce": float(plain_bce.detach().item()),
        "branch_loss": float(loss.detach().item()),
        "branch_delta": float(branch_delta.detach().item()),
        "low_loss": _masked_mean(low_loss, low_mask),
        "normal_loss": _masked_mean(normal_loss, normal_mask),
        "high_loss": _masked_mean(high_loss, high_mask),
        "high_bce": _masked_mean(bce_per_sample, high_mask),
        "high_l1": _masked_mean(despl_l1_per_sample, high_mask),
        "high_mse": _masked_mean(despl_mse_per_sample, high_mask),
        "teacher_l1": float(teacher_l1_per_sample.detach().mean().item()),
        "teacher_bce": float(teacher_bce_per_sample.detach().mean().item()),
        "high_extra_ratio": float(high_extra_ratio.detach().item()),
        "disabled_after_epoch": False,
        "loss_used": "branch",
        "entropy_weight_used": bool(entropy_weight_used),
        "per_sample": q_info["per_sample"],
        "pseudo_box_bg_enabled": bool(_safe_get_cfg(cfg, "GKD_USE_PSEUDO_BOX_BG", False)),
    }
    info.update(_quality_stats_for_info(q_info))
    info.update(pixel_stats)
    info.update(entropy_stats)
    return loss, info
