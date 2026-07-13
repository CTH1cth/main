import math

import torch
import torch.nn.functional as F


class TemporalTeacherMemory:
    """CPU-backed per-sample teacher probability moments."""

    def __init__(self, num_samples, height, width, dtype=torch.float16):
        self.num_samples = int(num_samples)
        self.height = int(height)
        self.width = int(width)
        if self.num_samples <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError(
                "TemporalTeacherMemory dimensions must be positive: "
                f"num_samples={self.num_samples}, height={self.height}, width={self.width}."
            )
        if isinstance(dtype, str):
            dtype_name = dtype.lower()
            dtype_map = {"float16": torch.float16, "float32": torch.float32}
            if dtype_name not in dtype_map:
                raise ValueError(f"Unsupported temporal memory dtype: {dtype}")
            dtype = dtype_map[dtype_name]
        if dtype not in {torch.float16, torch.float32}:
            raise ValueError(f"Temporal memory dtype must be float16/float32, got {dtype}.")
        self.dtype = dtype
        shape = (self.num_samples, 1, self.height, self.width)
        self.mean = torch.zeros(shape, dtype=dtype, device="cpu")
        self.second = torch.zeros(shape, dtype=dtype, device="cpu")
        self.count = torch.zeros((self.num_samples,), dtype=torch.int32, device="cpu")

    def _validate_indices(self, indices):
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long)
        indices = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if indices.ndim != 1:
            raise RuntimeError(f"Temporal memory indices must be [B], got {list(indices.shape)}.")
        if indices.numel() == 0:
            raise RuntimeError("Temporal memory indices cannot be empty.")
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= self.num_samples:
            raise RuntimeError(
                "Temporal memory index out of range: "
                f"min={min_index}, max={max_index}, num_samples={self.num_samples}."
            )
        if int(torch.unique(indices).numel()) != int(indices.numel()):
            raise RuntimeError("Temporal memory update/fetch received duplicate sample indices.")
        return indices

    def fetch(self, indices, device):
        indices_cpu = self._validate_indices(indices)
        mean = self.mean.index_select(0, indices_cpu).float().to(device)
        second = self.second.index_select(0, indices_cpu).float().to(device)
        count = self.count.index_select(0, indices_cpu).long().to(device)
        return mean, second, count

    @torch.no_grad()
    def update(self, indices, teacher_prob, rho):
        indices_cpu = self._validate_indices(indices)
        rho = float(rho)
        if not 0.0 < rho < 1.0:
            raise ValueError(f"Temporal memory rho must be in (0,1), got {rho}.")
        expected_shape = (int(indices_cpu.numel()), 1, self.height, self.width)
        if not torch.is_tensor(teacher_prob) or tuple(teacher_prob.shape) != expected_shape:
            raise RuntimeError(
                f"Temporal teacher probability must be {list(expected_shape)}, "
                f"got {list(teacher_prob.shape) if torch.is_tensor(teacher_prob) else type(teacher_prob)}."
            )
        probability = teacher_prob.detach().float().to(device="cpu")
        if not bool(torch.isfinite(probability).all().item()):
            raise RuntimeError("Temporal teacher probability contains NaN/Inf.")
        prob_min = float(probability.min().item())
        prob_max = float(probability.max().item())
        if prob_min < -1e-6 or prob_max > 1.0 + 1e-6:
            raise RuntimeError(
                f"Temporal teacher probability is outside [0,1]: min={prob_min}, max={prob_max}."
            )
        probability = probability.clamp(0.0, 1.0)
        old_mean = self.mean.index_select(0, indices_cpu).float()
        old_second = self.second.index_select(0, indices_cpu).float()
        old_count = self.count.index_select(0, indices_cpu)
        first = (old_count == 0).view(-1, 1, 1, 1)
        next_mean = torch.where(first, probability, rho * old_mean + (1.0 - rho) * probability)
        probability_sq = probability.square()
        next_second = torch.where(
            first,
            probability_sq,
            rho * old_second + (1.0 - rho) * probability_sq,
        )
        self.mean.index_copy_(0, indices_cpu, next_mean.to(self.dtype))
        self.second.index_copy_(0, indices_cpu, next_second.to(self.dtype))
        self.count.index_copy_(0, indices_cpu, old_count + 1)

    @torch.no_grad()
    def clear(self):
        self.mean.zero_()
        self.second.zero_()
        self.count.zero_()

    def state_dict(self):
        return {
            "num_samples": self.num_samples,
            "height": self.height,
            "width": self.width,
            "dtype": str(self.dtype).replace("torch.", ""),
            "mean": self.mean.clone(),
            "second": self.second.clone(),
            "count": self.count.clone(),
        }

    @torch.no_grad()
    def load_state_dict(self, state):
        expected_meta = (self.num_samples, self.height, self.width)
        actual_meta = (
            int(state.get("num_samples", -1)),
            int(state.get("height", -1)),
            int(state.get("width", -1)),
        )
        if actual_meta != expected_meta:
            raise RuntimeError(
                f"Temporal memory metadata mismatch: {actual_meta} != {expected_meta}."
            )
        expected_shape = tuple(self.mean.shape)
        for name, target in (("mean", self.mean), ("second", self.second)):
            value = state.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
                raise RuntimeError(f"Temporal memory {name} shape mismatch.")
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"Temporal memory {name} contains NaN/Inf.")
            target.copy_(value.to(device="cpu", dtype=self.dtype))
        count = state.get("count")
        if not torch.is_tensor(count) or tuple(count.shape) != tuple(self.count.shape):
            raise RuntimeError("Temporal memory count shape mismatch.")
        if bool((count < 0).any().item()):
            raise RuntimeError("Temporal memory count contains negative values.")
        self.count.copy_(count.to(device="cpu", dtype=torch.int32))


def get_tepr_scale(cfg, epoch):
    if not bool(getattr(cfg, "USE_TEPR_LITE", False)):
        return 0.0
    start = int(getattr(cfg, "TEPR_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "TEPR_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "TEPR_STOP_EPOCH", 21))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        return float(epoch - start + 1) / float(max(1, ramp_end - start + 1))
    return 1.0


def build_teacher_region_masks(batch, device):
    fg_core = batch["pu_fg_core"].to(device, non_blocking=True).float() > 0.5
    bg_core = batch["pu_bg_core"].to(device, non_blocking=True).float() > 0.5
    extent = batch["pu_extent"].to(device, non_blocking=True).float() > 0.5
    unknown = batch["pu_unknown"].to(device, non_blocking=True).float() > 0.5
    bg_core = bg_core & (~fg_core)
    extent = extent & (~fg_core) & (~bg_core)
    unknown = unknown & (~fg_core) & (~bg_core) & (~extent)
    other = ~(fg_core | bg_core | extent | unknown)
    return {
        "fg_core": fg_core,
        "bg_core": bg_core,
        "extent": extent,
        "unknown": unknown,
        "other": other,
    }


def compute_dino_core_margin_68(cfg, batch, fg_core, bg_core, output_size, device, prefix="ESA"):
    feat = batch["feature"].to(device, non_blocking=True).float()
    if feat.ndim != 4 or feat.shape[1] != 384:
        raise RuntimeError(f"{prefix} expects cached DINO feature [B,384,H,W], got {list(feat.shape)}.")
    feature_size = int(getattr(cfg, f"{prefix}_FEATURE_SIZE", feat.shape[-1]))
    if feat.shape[-2:] != (feature_size, feature_size):
        raise RuntimeError(
            f"{prefix} expects feature spatial {feature_size}x{feature_size}, got {list(feat.shape[-2:])}."
        )
    feat_norm = F.normalize(feat, dim=1)
    fg_core_feat = F.interpolate(fg_core.float(), size=feat.shape[-2:], mode="nearest") > 0.5
    bg_core_feat = F.interpolate(bg_core.float(), size=feat.shape[-2:], mode="nearest") > 0.5
    margin_feat = torch.zeros(
        (feat.shape[0], 1, feat.shape[-2], feat.shape[-1]),
        device=device,
        dtype=feat.dtype,
    )
    skipped_no_fg = 0
    skipped_no_bg = 0
    proto_valid = torch.zeros(feat.shape[0], device=device, dtype=torch.bool)
    for idx in range(int(feat.shape[0])):
        fg_mask = fg_core_feat[idx, 0]
        bg_mask = bg_core_feat[idx, 0]
        if int(fg_mask.sum().detach().item()) <= 0:
            skipped_no_fg += 1
            continue
        if int(bg_mask.sum().detach().item()) <= 0:
            skipped_no_bg += 1
            continue
        proto_valid[idx] = True
        fg_proto = F.normalize(feat_norm[idx, :, fg_mask].mean(dim=1), dim=0)
        bg_proto = F.normalize(feat_norm[idx, :, bg_mask].mean(dim=1), dim=0)
        if bool(getattr(cfg, f"{prefix}_DETACH_PROTO", True)):
            fg_proto = fg_proto.detach()
            bg_proto = bg_proto.detach()
        sim_fg = (feat_norm[idx] * fg_proto.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        sim_bg = (feat_norm[idx] * bg_proto.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        margin_feat[idx] = sim_fg - sim_bg
    margin_68 = F.interpolate(
        margin_feat,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    if bool(getattr(cfg, f"{prefix}_DETACH_MASK", True)):
        margin_68 = margin_68.detach()
    return margin_68, {
        f"{prefix.lower()}_skipped_no_fg_proto": skipped_no_fg,
        f"{prefix.lower()}_skipped_no_bg_proto": skipped_no_bg,
        f"{prefix.lower()}_proto_valid": proto_valid.detach(),
    }


def _masked_stats(value, mask, empty_value=1.0):
    mask_f = mask.float()
    count = float(mask_f.sum().detach().item())
    if count <= 0.0:
        return float(empty_value), 0.0, 0.0
    value_sum = float((value * mask_f).sum().detach().item())
    return value_sum / count, value_sum, count


def _masked_histogram(value, mask, bins=4096, value_min=0.0, value_max=1.0):
    selected = value.detach().float()[mask]
    if selected.numel() == 0:
        return torch.zeros(int(bins), dtype=torch.float32)
    return torch.histc(
        selected.clamp(float(value_min), float(value_max)),
        bins=int(bins),
        min=float(value_min),
        max=float(value_max),
    ).cpu()


def histogram_quantile_from_hist(histogram, quantile, value_min=0.0, value_max=1.0):
    histogram = torch.as_tensor(histogram, dtype=torch.float64).reshape(-1)
    total = float(histogram.sum().item())
    if total <= 0.0:
        return 0.0
    quantile = float(quantile)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Histogram quantile must be in [0,1], got {quantile}.")
    threshold = quantile * total
    index = int(
        torch.searchsorted(
            histogram.cumsum(0),
            torch.tensor(threshold, dtype=torch.float64),
        ).item()
    )
    index = min(max(index, 0), int(histogram.numel()) - 1)
    width = float(value_max) - float(value_min)
    return float(value_min) + width * float(index + 0.5) / float(histogram.numel())


def _assert_mask_value(value, mask, expected, name, tolerance=1e-5):
    if not bool(mask.any().item()):
        return
    max_error = float((value[mask] - float(expected)).abs().max().detach().item())
    if max_error > float(tolerance):
        raise RuntimeError(
            f"TEPR-v1.1 {name} invariant failed: expected={float(expected):.6f}, "
            f"max_error={max_error:.8f}."
        )


def build_tepr_lite_v11_teacher_weight_map(
    cfg,
    batch,
    teacher_prob,
    temporal_mean,
    temporal_second,
    history_count,
    epoch,
    device,
):
    """Build state-conditional TEPR-v1.1 teacher routing weights."""
    expected_shape = tuple(teacher_prob.shape)
    if teacher_prob.ndim != 4 or int(teacher_prob.shape[1]) != 1:
        raise RuntimeError(
            f"TEPR-v1.1 teacher probability must be [B,1,H,W], got {list(teacher_prob.shape)}."
        )
    for name, tensor in (("temporal_mean", temporal_mean), ("temporal_second", temporal_second)):
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"TEPR-v1.1 {name} shape mismatch: {list(tensor.shape)} != {list(expected_shape)}."
            )
    if history_count.ndim != 1 or int(history_count.shape[0]) != int(teacher_prob.shape[0]):
        raise RuntimeError(
            f"TEPR-v1.1 history_count must be [B], got {list(history_count.shape)}."
        )
    if not all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in (teacher_prob, temporal_mean, temporal_second)
    ):
        raise RuntimeError("TEPR-v1.1 input contains NaN/Inf.")
    teacher_prob_min = float(teacher_prob.min().detach().item())
    teacher_prob_max = float(teacher_prob.max().detach().item())
    if teacher_prob_min < -1e-6 or teacher_prob_max > 1.0 + 1e-6:
        raise RuntimeError(
            "TEPR-v1.1 teacher probability is outside [0,1]: "
            f"min={teacher_prob_min}, max={teacher_prob_max}."
        )

    scale = float(get_tepr_scale(cfg, epoch))
    mean = temporal_mean.float().clamp(0.0, 1.0)
    second = temporal_second.float().clamp(0.0, 1.0)
    variance = (second - mean.square()).clamp(0.0, 0.25)
    variance_tau = float(getattr(cfg, "TEPR_VARIANCE_TAU", 0.02))
    confidence_gamma = float(getattr(cfg, "TEPR_CONF_GAMMA", 1.0))
    stability = torch.exp(-variance / variance_tau)
    bg_confidence = F.relu(2.0 * (0.5 - mean)).clamp(0.0, 1.0)
    bg_reliability_observed = bg_confidence.pow(confidence_gamma) * stability
    min_history = int(getattr(cfg, "TEPR_MIN_HISTORY", 3))
    history_valid = history_count >= min_history
    insufficient_mode = str(
        getattr(cfg, "TEPR_INSUFFICIENT_HISTORY_MODE", "dino_only")
    ).lower()
    if insufficient_mode != "dino_only":
        raise RuntimeError(
            "TEPR-v1.1 only supports TEPR_INSUFFICIENT_HISTORY_MODE='dino_only', "
            f"got {insufficient_mode!r}."
        )
    enough_history = history_valid.view(-1, 1, 1, 1)
    bg_reliability = torch.where(
        enough_history,
        bg_reliability_observed,
        torch.ones_like(bg_reliability_observed),
    )

    masks = build_teacher_region_masks(batch, device)
    fg_core = masks["fg_core"]
    bg_core = masks["bg_core"]
    extent = masks["extent"]
    unknown = masks["unknown"]
    other = masks["other"]
    teacher_fg = teacher_prob >= 0.5
    teacher_bg = ~teacher_fg
    fg_conflict = fg_core & teacher_bg
    bg_conflict = bg_core & teacher_fg
    core_conflict = fg_conflict | bg_conflict
    core_no_conflict = (fg_core | bg_core) & (~core_conflict)
    extent_teacher_fg = extent & teacher_fg
    extent_teacher_bg = extent & teacher_bg

    margin_68, margin_stats = compute_dino_core_margin_68(
        cfg,
        batch,
        fg_core,
        bg_core,
        teacher_prob.shape[-2:],
        device,
        prefix="ESA",
    )
    margin_tau = float(getattr(cfg, "TEPR_MARGIN_TAU", 0.05))
    fg_tendency = torch.sigmoid(margin_68 / margin_tau)
    extent_dino_lambda = float(
        getattr(cfg, "TEPR_EXTENT_DINO_LAMBDA", math.log(4.0))
    )
    extent_floor = float(getattr(cfg, "TEPR_EXTENT_BG_WEIGHT_FLOOR", 0.25))
    dino_ceiling = torch.exp(-extent_dino_lambda * fg_tendency).clamp(extent_floor, 1.0)
    extent_bg_weight = extent_floor + bg_reliability * (dino_ceiling - extent_floor)

    raw = torch.full_like(
        teacher_prob,
        float(getattr(cfg, "TEPR_OTHER_WEIGHT", 1.0)),
        dtype=torch.float32,
        device=device,
    )
    raw = torch.where(
        unknown,
        torch.full_like(raw, float(getattr(cfg, "TEPR_UNKNOWN_WEIGHT", 0.50))),
        raw,
    )
    extent_fg_weight = float(getattr(cfg, "TEPR_EXTENT_FG_WEIGHT", 1.0))
    raw = torch.where(
        extent_teacher_fg,
        torch.full_like(raw, extent_fg_weight),
        raw,
    )
    raw = torch.where(extent_teacher_bg, extent_bg_weight, raw)
    raw = torch.where(
        core_conflict,
        torch.full_like(raw, float(getattr(cfg, "TEPR_CORE_CONFLICT_WEIGHT", 0.20))),
        raw,
    )
    weight_min = float(getattr(cfg, "TEPR_WEIGHT_MIN", 0.20))
    weight_max = float(getattr(cfg, "TEPR_WEIGHT_MAX", 1.00))
    raw = raw.clamp(weight_min, weight_max)

    _assert_mask_value(raw, extent_teacher_fg, extent_fg_weight, "extent teacher-foreground")
    _assert_mask_value(raw, unknown, float(getattr(cfg, "TEPR_UNKNOWN_WEIGHT", 0.50)), "unknown")
    _assert_mask_value(
        raw,
        core_conflict,
        float(getattr(cfg, "TEPR_CORE_CONFLICT_WEIGHT", 0.20)),
        "core conflict",
    )
    _assert_mask_value(raw, core_no_conflict, 1.0, "core non-conflict")
    _assert_mask_value(raw, other, float(getattr(cfg, "TEPR_OTHER_WEIGHT", 1.0)), "other")
    if bool(extent_teacher_bg.any().item()):
        extent_bg_min = float(raw[extent_teacher_bg].min().detach().item())
        extent_bg_max = float(raw[extent_teacher_bg].max().detach().item())
        if extent_bg_min < extent_floor - 1e-5 or extent_bg_max > 1.0 + 1e-5:
            raise RuntimeError(
                "TEPR-v1.1 extent teacher-background weight out of range: "
                f"min={extent_bg_min:.8f}, max={extent_bg_max:.8f}, floor={extent_floor:.8f}."
            )

    effective = torch.ones_like(raw) if scale <= 0.0 else ((1.0 - scale) + scale * raw)
    effective = effective.clamp(weight_min, weight_max)
    if not bool(torch.isfinite(effective).all().item()):
        raise RuntimeError("TEPR-v1.1 teacher map contains NaN/Inf.")
    map_min = float(effective.min().detach().item())
    map_max = float(effective.max().detach().item())
    if map_min < weight_min - 1e-5 or map_max > weight_max + 1e-5:
        raise RuntimeError(
            f"TEPR-v1.1 teacher map out of range [{weight_min},{weight_max}]: "
            f"{map_min}/{map_max}."
        )

    margin_fg_like = float(getattr(cfg, "ESA_MARGIN_FG_LIKE", 0.05))
    margin_bg_like = float(getattr(cfg, "ESA_MARGIN_BG_LIKE", -0.05))
    extent_bg_fg_like = extent_teacher_bg & (margin_68 >= margin_fg_like)
    extent_bg_bg_like = extent_teacher_bg & (margin_68 <= margin_bg_like)
    extent_bg_ambiguous = extent_teacher_bg & (~extent_bg_fg_like) & (~extent_bg_bg_like)

    region_means = {}
    region_sums = {}
    region_counts = {}
    for name, mask in masks.items():
        mean_value, value_sum, count = _masked_stats(effective, mask, empty_value=1.0)
        region_means[name] = mean_value
        region_sums[name] = value_sum
        region_counts[name] = count

    conditional_values = {
        "core_conflict_map": (effective, core_conflict),
        "core_no_conflict_map": (effective, core_no_conflict),
        "extent_teacher_fg_map": (effective, extent_teacher_fg),
        "extent_teacher_bg_map": (effective, extent_teacher_bg),
        "extent_bg_temporal_reliability": (bg_reliability, extent_teacher_bg),
        "extent_bg_dino_ceiling": (dino_ceiling, extent_teacher_bg),
        "extent_bg_final_weight": (extent_bg_weight, extent_teacher_bg),
        "extent_bg_fg_like_weight": (effective, extent_bg_fg_like),
        "extent_bg_ambiguous_weight": (effective, extent_bg_ambiguous),
        "extent_bg_bg_like_weight": (effective, extent_bg_bg_like),
        "unknown_map": (effective, unknown),
        "other_map": (effective, other),
        "teacher_fg_fg_core": (teacher_fg.float(), fg_core),
        "teacher_fg_extent": (teacher_fg.float(), extent),
    }
    conditional_means = {}
    conditional_sums = {}
    conditional_counts = {}
    for name, (value, mask) in conditional_values.items():
        mean_value, value_sum, count = _masked_stats(value, mask, empty_value=0.0)
        conditional_means[name] = mean_value
        conditional_sums[name] = value_sum
        conditional_counts[name] = count

    variance_hist = torch.histc(
        variance.detach().float(), bins=4096, min=0.0, max=0.25
    ).cpu()
    reliability_hist = _masked_histogram(bg_reliability, extent_teacher_bg)
    stats = {
        "tepr_scale": scale,
        "history_count_min": int(history_count.min().detach().item()),
        "history_count_mean": float(history_count.float().mean().detach().item()),
        "history_count_max": int(history_count.max().detach().item()),
        "history_valid_ratio": float(history_valid.float().mean().detach().item()),
        "temporal_mean_min": float(mean.min().detach().item()),
        "temporal_mean_mean": float(mean.mean().detach().item()),
        "temporal_mean_max": float(mean.max().detach().item()),
        "temporal_var_min": float(variance.min().detach().item()),
        "temporal_var_mean": float(variance.mean().detach().item()),
        "temporal_var_max": float(variance.max().detach().item()),
        "temporal_conf_mean": float(bg_confidence.mean().detach().item()),
        "temporal_stability_mean": float(stability.mean().detach().item()),
        "temporal_reliability_mean": float(bg_reliability.mean().detach().item()),
        "core_conflict_mean": float(core_conflict.float().mean().detach().item()),
        "core_conflict_max": float(core_conflict.float().max().detach().item()),
        "extent_conflict_mean": float(extent_teacher_bg.float().mean().detach().item()),
        "extent_conflict_max": float(extent_teacher_bg.float().max().detach().item()),
        "dino_margin_min": float(margin_68.min().detach().item()),
        "dino_margin_mean": float(margin_68.mean().detach().item()),
        "dino_margin_max": float(margin_68.max().detach().item()),
        "fg_tendency_mean": float(fg_tendency.mean().detach().item()),
        "fg_core_conflict_count": int(fg_conflict.sum().detach().item()),
        "fg_core_count": int(fg_core.sum().detach().item()),
        "bg_core_conflict_count": int(bg_conflict.sum().detach().item()),
        "bg_core_count": int(bg_core.sum().detach().item()),
        "extent_teacher_fg_count": int(extent_teacher_fg.sum().detach().item()),
        "extent_teacher_bg_count": int(extent_teacher_bg.sum().detach().item()),
        "extent_count": int(extent.sum().detach().item()),
        "teacher_map_raw_min": float(raw.min().detach().item()),
        "teacher_map_raw_mean": float(raw.mean().detach().item()),
        "teacher_map_raw_max": float(raw.max().detach().item()),
        "teacher_map_min": map_min,
        "teacher_map_mean": float(effective.mean().detach().item()),
        "teacher_map_max": map_max,
        "teacher_map_lt_030_ratio": float((effective < 0.30).float().mean().detach().item()),
        "teacher_map_lt_050_ratio": float((effective < 0.50).float().mean().detach().item()),
        "teacher_map_gt_090_ratio": float((effective > 0.90).float().mean().detach().item()),
        "region_means": region_means,
        "region_sums": region_sums,
        "region_counts": region_counts,
        "conditional_means": conditional_means,
        "conditional_sums": conditional_sums,
        "conditional_counts": conditional_counts,
        "temporal_var_hist": variance_hist,
        "extent_bg_reliability_hist": reliability_hist,
        "map_pixel_count": int(effective.numel()),
        "map_sum": float(effective.sum().detach().item()),
        "map_lt_030_count": int((effective < 0.30).sum().detach().item()),
        "map_lt_050_count": int((effective < 0.50).sum().detach().item()),
        "map_gt_090_count": int((effective > 0.90).sum().detach().item()),
        "esa_skipped_no_fg_proto": int(margin_stats.get("esa_skipped_no_fg_proto", 0)),
        "esa_skipped_no_bg_proto": int(margin_stats.get("esa_skipped_no_bg_proto", 0)),
    }
    return effective.detach(), stats


def build_tepr_lite_teacher_weight_map(
    cfg,
    batch,
    teacher_prob,
    temporal_mean,
    temporal_second,
    history_count,
    epoch,
    device,
):
    expected_shape = tuple(teacher_prob.shape)
    for name, tensor in (("temporal_mean", temporal_mean), ("temporal_second", temporal_second)):
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(f"TEPR {name} shape mismatch: {list(tensor.shape)} != {list(expected_shape)}.")
    if history_count.ndim != 1 or int(history_count.shape[0]) != int(teacher_prob.shape[0]):
        raise RuntimeError(f"TEPR history_count must be [B], got {list(history_count.shape)}.")
    if not all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in (teacher_prob, temporal_mean, temporal_second)
    ):
        raise RuntimeError("TEPR input contains NaN/Inf.")

    scale = float(get_tepr_scale(cfg, epoch))
    min_history = int(getattr(cfg, "TEPR_MIN_HISTORY", 3))
    history_valid = history_count >= min_history
    if scale > 0.0 and not bool(history_valid.all().item()):
        raise RuntimeError(
            "TEPR is active with insufficient history: "
            f"epoch={epoch}, min_count={int(history_count.min().item())}, required={min_history}."
        )

    mean = temporal_mean.float().clamp(0.0, 1.0)
    second = temporal_second.float().clamp(0.0, 1.0)
    variance = (second - mean.square()).clamp(0.0, 0.25)
    confidence = (2.0 * (mean - 0.5).abs()).clamp(0.0, 1.0)
    variance_tau = float(getattr(cfg, "TEPR_VARIANCE_TAU", 0.02))
    confidence_gamma = float(getattr(cfg, "TEPR_CONF_GAMMA", 1.0))
    stability = torch.exp(-variance / variance_tau)
    reliability = confidence.pow(confidence_gamma) * stability
    reliability = reliability * history_valid.view(-1, 1, 1, 1).float()

    masks = build_teacher_region_masks(batch, device)
    fg_core = masks["fg_core"]
    bg_core = masks["bg_core"]
    extent = masks["extent"]
    unknown = masks["unknown"]
    core_conflict = (
        fg_core.float() * F.relu(2.0 * (0.5 - mean))
        + bg_core.float() * F.relu(2.0 * (mean - 0.5))
    )
    core_weight = torch.exp(-float(getattr(cfg, "TEPR_CORE_LAMBDA", math.log(5.0))) * core_conflict)

    margin_68, margin_stats = compute_dino_core_margin_68(
        cfg,
        batch,
        fg_core,
        bg_core,
        teacher_prob.shape[-2:],
        device,
        prefix="ESA",
    )
    margin_tau = float(getattr(cfg, "TEPR_MARGIN_TAU", 0.05))
    fg_tendency = torch.sigmoid(margin_68 / margin_tau)
    historical_bg_tendency = F.relu(2.0 * (0.5 - mean))
    extent_conflict = extent.float() * fg_tendency * historical_bg_tendency
    extent_conflict_weight = torch.exp(
        -float(getattr(cfg, "TEPR_EXTENT_LAMBDA", math.log(4.0))) * extent_conflict
    )
    extent_temporal_min = float(getattr(cfg, "TEPR_EXTENT_TEMPORAL_MIN", 0.60))
    extent_temporal_weight = extent_temporal_min + (1.0 - extent_temporal_min) * reliability
    extent_weight = extent_temporal_weight * extent_conflict_weight
    unknown_min = float(getattr(cfg, "TEPR_UNKNOWN_WEIGHT_MIN", 0.50))
    unknown_max = float(getattr(cfg, "TEPR_UNKNOWN_WEIGHT_MAX", 0.80))
    unknown_weight = unknown_min + (unknown_max - unknown_min) * reliability

    raw = torch.ones_like(teacher_prob, dtype=torch.float32, device=device)
    raw = torch.where(unknown, unknown_weight, raw)
    raw = torch.where(extent, extent_weight, raw)
    raw = torch.where(fg_core | bg_core, core_weight, raw)
    weight_min = float(getattr(cfg, "TEPR_WEIGHT_MIN", 0.20))
    weight_max = float(getattr(cfg, "TEPR_WEIGHT_MAX", 1.00))
    raw = raw.clamp(weight_min, weight_max)
    effective = torch.ones_like(raw) if scale <= 0.0 else ((1.0 - scale) + scale * raw)
    effective = effective.clamp(weight_min, weight_max)
    if not bool(torch.isfinite(effective).all().item()):
        raise RuntimeError("TEPR teacher map contains NaN/Inf.")
    map_min = float(effective.min().detach().item())
    map_max = float(effective.max().detach().item())
    if map_min < weight_min - 1e-5 or map_max > weight_max + 1e-5:
        raise RuntimeError(
            f"TEPR teacher map out of range [{weight_min},{weight_max}]: {map_min}/{map_max}."
        )

    region_means = {}
    region_sums = {}
    region_counts = {}
    for name, mask in masks.items():
        mean_value, value_sum, count = _masked_stats(effective, mask, empty_value=1.0)
        region_means[name] = mean_value
        region_sums[name] = value_sum
        region_counts[name] = count
    variance_hist = torch.histc(variance.detach().float(), bins=4096, min=0.0, max=0.25).cpu()
    stats = {
        "tepr_scale": scale,
        "history_count_min": int(history_count.min().detach().item()),
        "history_count_mean": float(history_count.float().mean().detach().item()),
        "history_count_max": int(history_count.max().detach().item()),
        "temporal_mean_min": float(mean.min().detach().item()),
        "temporal_mean_mean": float(mean.mean().detach().item()),
        "temporal_mean_max": float(mean.max().detach().item()),
        "temporal_var_min": float(variance.min().detach().item()),
        "temporal_var_mean": float(variance.mean().detach().item()),
        "temporal_var_max": float(variance.max().detach().item()),
        "temporal_conf_mean": float(confidence.mean().detach().item()),
        "temporal_stability_mean": float(stability.mean().detach().item()),
        "temporal_reliability_mean": float(reliability.mean().detach().item()),
        "core_conflict_mean": float(core_conflict.mean().detach().item()),
        "core_conflict_max": float(core_conflict.max().detach().item()),
        "extent_conflict_mean": float(extent_conflict.mean().detach().item()),
        "extent_conflict_max": float(extent_conflict.max().detach().item()),
        "dino_margin_min": float(margin_68.min().detach().item()),
        "dino_margin_mean": float(margin_68.mean().detach().item()),
        "dino_margin_max": float(margin_68.max().detach().item()),
        "fg_tendency_mean": float(fg_tendency.mean().detach().item()),
        "teacher_map_raw_min": float(raw.min().detach().item()),
        "teacher_map_raw_mean": float(raw.mean().detach().item()),
        "teacher_map_raw_max": float(raw.max().detach().item()),
        "teacher_map_min": map_min,
        "teacher_map_mean": float(effective.mean().detach().item()),
        "teacher_map_max": map_max,
        "teacher_map_lt_030_ratio": float((effective < 0.30).float().mean().detach().item()),
        "teacher_map_lt_050_ratio": float((effective < 0.50).float().mean().detach().item()),
        "teacher_map_gt_090_ratio": float((effective > 0.90).float().mean().detach().item()),
        "region_means": region_means,
        "region_sums": region_sums,
        "region_counts": region_counts,
        "temporal_var_hist": variance_hist,
        "map_pixel_count": int(effective.numel()),
        "map_sum": float(effective.sum().detach().item()),
        "map_lt_030_count": int((effective < 0.30).sum().detach().item()),
        "map_lt_050_count": int((effective < 0.50).sum().detach().item()),
        "map_gt_090_count": int((effective > 0.90).sum().detach().item()),
        "esa_skipped_no_fg_proto": int(margin_stats.get("esa_skipped_no_fg_proto", 0)),
        "esa_skipped_no_bg_proto": int(margin_stats.get("esa_skipped_no_bg_proto", 0)),
    }
    return effective.detach(), stats


def temporal_variance_p90_from_hist(histogram):
    histogram = torch.as_tensor(histogram, dtype=torch.float64).reshape(-1)
    total = float(histogram.sum().item())
    if total <= 0.0:
        return 0.0
    threshold = 0.90 * total
    index = int(
        torch.searchsorted(
            histogram.cumsum(0),
            torch.tensor(threshold, dtype=torch.float64),
        ).item()
    )
    index = min(max(index, 0), int(histogram.numel()) - 1)
    return 0.25 * float(index + 0.5) / float(histogram.numel())
