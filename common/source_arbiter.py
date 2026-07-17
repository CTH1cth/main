from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceArbiter(nn.Module):
    """Small training-only router for DABE-PU versus EMA-teacher supervision."""

    def __init__(
        self,
        input_channels=18,
        hidden_1=32,
        hidden_2=32,
        hidden_3=16,
        dilation=2,
        residual_bound=1.5,
        zero_init_head=True,
    ):
        super().__init__()
        self.residual_bound = float(residual_bound)
        self.body = nn.Sequential(
            nn.Conv2d(int(input_channels), int(hidden_1), 3, padding=1),
            nn.GroupNorm(8, int(hidden_1)),
            nn.GELU(),
            nn.Conv2d(
                int(hidden_1),
                int(hidden_2),
                3,
                padding=int(dilation),
                dilation=int(dilation),
            ),
            nn.GroupNorm(8, int(hidden_2)),
            nn.GELU(),
            nn.Conv2d(int(hidden_2), int(hidden_3), 3, padding=1),
            nn.GroupNorm(4, int(hidden_3)),
            nn.GELU(),
        )
        self.head = nn.Conv2d(int(hidden_3), 1, 1)
        if bool(zero_init_head):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, evidence):
        if evidence.ndim != 4 or int(evidence.shape[1]) != 18:
            raise RuntimeError(
                f"SourceArbiter expects [B,18,H,W], got {list(evidence.shape)}."
            )
        if not bool(torch.isfinite(evidence).all().item()):
            raise RuntimeError("SourceArbiter evidence contains NaN/Inf.")
        raw = self.head(self.body(evidence))
        residual = self.residual_bound * torch.tanh(raw)
        if not bool(torch.isfinite(residual).all().item()):
            raise RuntimeError("SourceArbiter residual contains NaN/Inf.")
        return residual


class RouteTrajectoryMemory:
    """CPU-backed previous-visit evidence for delayed source-utility labels."""

    _MAP_FIELDS = (
        "teacher_prob",
        "student_prob",
        "temporal_mean",
        "temporal_variance",
    )

    def __init__(self, num_samples, height, width, dtype="float16"):
        self.num_samples = int(num_samples)
        self.height = int(height)
        self.width = int(width)
        if self.num_samples <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("RouteTrajectoryMemory dimensions must be positive.")
        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        if isinstance(dtype, torch.dtype):
            self.dtype = dtype
        else:
            key = str(dtype).lower()
            if key not in dtype_map:
                raise ValueError(f"Unsupported route memory dtype: {dtype!r}.")
            self.dtype = dtype_map[key]
        if self.dtype not in {torch.float16, torch.float32}:
            raise ValueError("RouteTrajectoryMemory supports float16/float32 only.")
        shape = (self.num_samples, 1, self.height, self.width)
        for field in self._MAP_FIELDS:
            setattr(self, field, torch.zeros(shape, dtype=self.dtype, device="cpu"))
        self.history_count = torch.zeros(self.num_samples, dtype=torch.int32)
        self.teacher_prior = torch.zeros(self.num_samples, dtype=torch.float32)
        self.epoch = torch.full((self.num_samples,), -1, dtype=torch.int32)
        self.valid = torch.zeros(self.num_samples, dtype=torch.bool)

    def _indices(self, indices):
        value = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if value.numel() == 0:
            raise RuntimeError("Route memory indices cannot be empty.")
        if int(value.min()) < 0 or int(value.max()) >= self.num_samples:
            raise RuntimeError(
                "Route memory index out of range: "
                f"min={int(value.min())}, max={int(value.max())}, "
                f"num_samples={self.num_samples}."
            )
        if int(torch.unique(value).numel()) != int(value.numel()):
            raise RuntimeError("Route memory received duplicate sample indices.")
        return value

    def fetch(self, indices, device):
        idx = self._indices(indices)
        result = {
            field: getattr(self, field).index_select(0, idx).float().to(device)
            for field in self._MAP_FIELDS
        }
        result.update(
            {
                "history_count": self.history_count.index_select(0, idx).long().to(device),
                "teacher_prior": self.teacher_prior.index_select(0, idx).float().to(device),
                "epoch": self.epoch.index_select(0, idx).long().to(device),
                "valid": self.valid.index_select(0, idx).to(device),
            }
        )
        return result

    @torch.no_grad()
    def update(
        self,
        indices,
        teacher_prob,
        student_prob,
        temporal_mean,
        temporal_variance,
        history_count,
        teacher_prior,
        epoch,
    ):
        idx = self._indices(indices)
        batch_size = int(idx.numel())
        expected = (batch_size, 1, self.height, self.width)
        values = {
            "teacher_prob": teacher_prob,
            "student_prob": student_prob,
            "temporal_mean": temporal_mean,
            "temporal_variance": temporal_variance,
        }
        for name, value in values.items():
            if not torch.is_tensor(value) or tuple(value.shape) != expected:
                actual = list(value.shape) if torch.is_tensor(value) else type(value)
                raise RuntimeError(
                    f"Route memory {name} must be {list(expected)}, got {actual}."
                )
            value_cpu = value.detach().float().to("cpu")
            if not bool(torch.isfinite(value_cpu).all().item()):
                raise RuntimeError(f"Route memory {name} contains NaN/Inf.")
            if name != "temporal_variance":
                low = float(value_cpu.min())
                high = float(value_cpu.max())
                if low < -1e-6 or high > 1.0 + 1e-6:
                    raise RuntimeError(
                        f"Route memory {name} outside [0,1]: {low}/{high}."
                    )
            elif float(value_cpu.min()) < -1e-7:
                raise RuntimeError("Route memory temporal variance is negative.")
            getattr(self, name).index_copy_(0, idx, value_cpu.to(self.dtype))

        count = torch.as_tensor(history_count).detach().to("cpu", torch.int32).reshape(-1)
        prior = torch.as_tensor(teacher_prior).detach().to("cpu", torch.float32).reshape(-1)
        if tuple(count.shape) != (batch_size,) or tuple(prior.shape) != (batch_size,):
            raise RuntimeError("Route memory scalar state must be [B].")
        if bool((count < 0).any()) or not bool(torch.isfinite(prior).all()):
            raise RuntimeError("Route memory scalar state is invalid.")
        if float(prior.min()) < -1e-6 or float(prior.max()) > 1.0 + 1e-6:
            raise RuntimeError("Route memory teacher prior is outside [0,1].")
        epoch_value = int(epoch)
        old_valid = self.valid.index_select(0, idx)
        old_epoch = self.epoch.index_select(0, idx)
        if bool((old_valid & (old_epoch >= epoch_value)).any()):
            raise RuntimeError(
                "Route memory requires current epoch to be greater than stored epoch."
            )
        self.history_count.index_copy_(0, idx, count)
        self.teacher_prior.index_copy_(0, idx, prior)
        self.epoch.index_fill_(0, idx, epoch_value)
        self.valid.index_fill_(0, idx, True)

    @torch.no_grad()
    def clear(self):
        for field in self._MAP_FIELDS:
            getattr(self, field).zero_()
        self.history_count.zero_()
        self.teacher_prior.zero_()
        self.epoch.fill_(-1)
        self.valid.zero_()

    def state_dict(self):
        state = {
            "num_samples": self.num_samples,
            "height": self.height,
            "width": self.width,
            "dtype": str(self.dtype).replace("torch.", ""),
        }
        for field in self._MAP_FIELDS:
            state[field] = getattr(self, field).clone()
        state.update(
            {
                "history_count": self.history_count.clone(),
                "teacher_prior": self.teacher_prior.clone(),
                "epoch": self.epoch.clone(),
                "valid": self.valid.clone(),
            }
        )
        return state

    @torch.no_grad()
    def load_state_dict(self, state):
        meta = (
            int(state.get("num_samples", -1)),
            int(state.get("height", -1)),
            int(state.get("width", -1)),
        )
        expected_meta = (self.num_samples, self.height, self.width)
        if meta != expected_meta:
            raise RuntimeError(f"Route memory metadata mismatch: {meta} != {expected_meta}.")
        saved_dtype = str(state.get("dtype", "")).lower()
        expected_dtype = str(self.dtype).replace("torch.", "")
        if saved_dtype != expected_dtype:
            raise RuntimeError(
                "Route memory dtype mismatch: "
                f"{saved_dtype!r} != {expected_dtype!r}."
            )
        map_shape = (self.num_samples, 1, self.height, self.width)
        for field in self._MAP_FIELDS:
            value = state.get(field)
            if not torch.is_tensor(value) or tuple(value.shape) != map_shape:
                raise RuntimeError(f"Route memory {field} shape mismatch.")
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Route memory {field} contains NaN/Inf.")
            getattr(self, field).copy_(value.to("cpu", self.dtype))
        scalar_specs = {
            "history_count": (self.history_count, torch.int32),
            "teacher_prior": (self.teacher_prior, torch.float32),
            "epoch": (self.epoch, torch.int32),
            "valid": (self.valid, torch.bool),
        }
        for field, (target, dtype) in scalar_specs.items():
            value = state.get(field)
            if not torch.is_tensor(value) or tuple(value.shape) != (self.num_samples,):
                raise RuntimeError(f"Route memory {field} shape mismatch.")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Route memory {field} contains NaN/Inf.")
            target.copy_(value.to("cpu", dtype))
        if bool((self.history_count < 0).any()):
            raise RuntimeError("Route memory history_count contains negative values.")
        if not bool(torch.isfinite(self.teacher_prior).all()):
            raise RuntimeError("Route memory teacher_prior contains NaN/Inf.")
        if (
            float(self.teacher_prior.min()) < -1e-6
            or float(self.teacher_prior.max()) > 1.0 + 1e-6
        ):
            raise RuntimeError("Route memory teacher_prior is outside [0,1].")
        if bool((self.valid & (self.epoch < 0)).any()):
            raise RuntimeError("Route memory valid entries have invalid epochs.")


@dataclass
class UtilityTarget:
    target_teacher: torch.Tensor
    valid: torch.Tensor
    weight: torch.Tensor
    consensus: torch.Tensor
    consensus_confidence: torch.Tensor
    cross_view_difference: torch.Tensor
    source_disagreement: torch.Tensor
    utility_gap: torch.Tensor
    delta_target: torch.Tensor
    age_valid: torch.Tensor


def get_arbiter_influence_scale(epoch, cfg):
    if not bool(getattr(cfg, "USE_SOURCE_ARBITER", False)):
        return 0.0
    start = int(getattr(cfg, "SOURCE_ARBITER_START_EPOCH", 7))
    ramp_end = int(getattr(cfg, "SOURCE_ARBITER_RAMP_END_EPOCH", 15))
    stop = int(getattr(cfg, "SOURCE_ARBITER_STOP_EPOCH", 21))
    epoch = int(epoch)
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        return float(epoch - start + 1) / float(max(1, ramp_end - start + 1))
    return 1.0


def get_source_prior(static_weight, teacher_weight):
    source_sum = float(static_weight) + float(teacher_weight)
    if not source_sum > 0.0:
        raise RuntimeError(
            "Source arbitration requires static_weight + teacher_weight > 0, "
            f"got {static_weight} + {teacher_weight}."
        )
    teacher_prior = float(teacher_weight) / source_sum
    return source_sum, teacher_prior


def normalize_dabe_weight_for_arbiter(weight, mode, fixed_scale=1.0):
    if not torch.is_tensor(weight) or weight.ndim != 4:
        raise RuntimeError("DABE weight for SourceArbiter must be [B,1,H,W].")
    value = weight.detach().float()
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("SourceArbiter DABE reliability contains NaN/Inf.")
    mode = str(mode).lower()
    if mode == "raw_clamped":
        return value.clamp(0.0, 1.0)
    if mode == "per_image_max_norm":
        maximum = value.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        return (value / maximum.clamp_min(1e-6)).clamp(0.0, 1.0)
    if mode == "global_fixed_scale":
        fixed_scale = float(fixed_scale)
        if not fixed_scale > 0.0:
            raise RuntimeError(
                "SOURCE_ARBITER_DABE_WEIGHT_FIXED_SCALE must be positive."
            )
        return (value / fixed_scale).clamp(0.0, 1.0)
    raise RuntimeError(
        "Unsupported SOURCE_ARBITER_DABE_WEIGHT_INPUT_MODE="
        f"{mode!r}; expected raw_clamped/per_image_max_norm/global_fixed_scale."
    )


def build_source_arbiter_inputs(
    batch,
    teacher_prob,
    student_prob,
    temporal_mean,
    temporal_variance,
    history_count,
    margin_68,
    masks,
    teacher_prior,
    min_history,
    dino_margin_tau=0.05,
    dabe_weight_input_mode="raw_clamped",
    dabe_weight_fixed_scale=1.0,
):
    reference = teacher_prob
    expected = tuple(reference.shape)
    continuous = {
        "student_prob": student_prob,
        "temporal_mean": temporal_mean,
        "temporal_variance": temporal_variance,
        "margin_68": margin_68,
    }
    for name, value in continuous.items():
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"SourceArbiter {name} shape mismatch: {list(value.shape)} != {list(expected)}."
            )
    target = batch["pu_target_soft"].to(reference.device, non_blocking=True).float()
    reliability_raw = batch["pu_weight_map"].to(
        reference.device, non_blocking=True
    ).float()
    if tuple(target.shape) != expected or tuple(reliability_raw.shape) != expected:
        raise RuntimeError("SourceArbiter DABE target/reliability shape mismatch.")
    reliability_min = float(reliability_raw.min())
    reliability_max = float(reliability_raw.max())
    if (
        str(dabe_weight_input_mode).lower() == "raw_clamped"
        and (reliability_min < -1e-6 or reliability_max > 1.0 + 1e-6)
    ):
        raise RuntimeError(
            "DABE reliability is outside [0,1] while the explicit input mode is "
            "'raw_clamped'. Select and document a normalization mode first: "
            f"min={reliability_min}, max={reliability_max}."
        )
    reliability = normalize_dabe_weight_for_arbiter(
        reliability_raw,
        dabe_weight_input_mode,
        fixed_scale=dabe_weight_fixed_scale,
    )
    history_valid = (history_count >= int(min_history)).to(reference.dtype)
    history_valid = history_valid.view(-1, 1, 1, 1).expand_as(reference)
    prior = torch.as_tensor(
        teacher_prior,
        device=reference.device,
        dtype=reference.dtype,
    )
    if prior.ndim == 0:
        prior = prior.expand(reference.shape[0])
    prior = prior.reshape(-1, 1, 1, 1).expand_as(reference)
    teacher_confidence = (2.0 * teacher_prob - 1.0).abs().clamp(0.0, 1.0)
    signed_disagreement = (teacher_prob - target).clamp(-1.0, 1.0)
    absolute_disagreement = signed_disagreement.abs()
    student_uncertainty = (4.0 * student_prob * (1.0 - student_prob)).clamp(0.0, 1.0)
    channels = (
        target,
        reliability.clamp(0.0, 1.0),
        masks["fg_core"].float(),
        masks["bg_core"].float(),
        masks["extent"].float(),
        masks["unknown"].float(),
        masks["other"].float(),
        teacher_prob,
        teacher_confidence,
        temporal_mean.clamp(0.0, 1.0),
        (4.0 * temporal_variance).clamp(0.0, 1.0),
        history_valid,
        signed_disagreement,
        absolute_disagreement,
        torch.tanh(margin_68 / float(dino_margin_tau)),
        student_prob,
        student_uncertainty,
        prior,
    )
    evidence = torch.cat([value.detach() for value in channels], dim=1)
    if int(evidence.shape[1]) != 18 or not bool(torch.isfinite(evidence).all()):
        raise RuntimeError("SourceArbiter evidence construction failed.")
    return evidence


def compute_source_gates(
    residual_logit,
    teacher_prior,
    source_disagreement,
    influence_scale,
    residual_bound=1.5,
    eps=1e-4,
    disagreement_denom=0.5,
    use_disagreement_scale=True,
):
    del residual_bound  # residual is already bounded by SourceArbiter.forward.
    prior = float(teacher_prior)
    if not 0.0 <= prior <= 1.0:
        raise ValueError(f"Teacher prior must be in [0,1], got {prior}.")
    if prior <= float(eps):
        teacher = torch.zeros_like(residual_logit)
    elif prior >= 1.0 - float(eps):
        teacher = torch.ones_like(residual_logit)
    else:
        if bool(use_disagreement_scale):
            disagreement_denom = float(disagreement_denom)
            if disagreement_denom <= 0.0:
                raise RuntimeError(
                    "SOURCE_ARBITER_DISAGREEMENT_DENOM must be positive."
                )
            disagreement = (
                source_disagreement.abs() / disagreement_denom
            ).clamp(0.0, 1.0)
        else:
            disagreement = torch.ones_like(source_disagreement)
        prior_tensor = residual_logit.new_tensor(prior)
        prior_logit = torch.log(prior_tensor) - torch.log1p(-prior_tensor)
        teacher = torch.sigmoid(
            prior_logit
            + float(influence_scale) * disagreement * residual_logit
        )
    dabe = 1.0 - teacher
    if not bool(torch.isfinite(teacher).all()):
        raise RuntimeError("SourceArbiter gate contains NaN/Inf.")
    complement_error = float((teacher + dabe - 1.0).abs().max())
    if complement_error > 1e-6:
        raise RuntimeError(f"SourceArbiter complementary gate error={complement_error}.")
    return dabe, teacher


def normalize_source_pixel_loss(logits, target, weight, eps=1e-6):
    if tuple(logits.shape) != tuple(target.shape) or tuple(weight.shape) != tuple(logits.shape):
        raise RuntimeError("Source-normalized BCE tensors must have identical shapes.")
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denominator = weight.sum() + float(eps)
    return raw * weight * (float(raw.numel()) / denominator)


def apply_loss_space_arbitration(
    logits,
    dabe_target,
    dabe_weight,
    teacher_target,
    teacher_weight,
    gate_dabe,
    gate_teacher,
    source_sum,
    eps=1e-6,
):
    dabe_map = normalize_source_pixel_loss(logits, dabe_target, dabe_weight, eps=eps)
    teacher_map = normalize_source_pixel_loss(
        logits,
        teacher_target,
        teacher_weight,
        eps=eps,
    )
    loss = float(source_sum) * (
        gate_dabe.detach() * dabe_map + gate_teacher.detach() * teacher_map
    ).mean()
    return {
        "loss": loss,
        "loss_dabe": dabe_map.mean(),
        "loss_teacher": teacher_map.mean(),
        "dabe_loss_map": dabe_map,
        "teacher_loss_map": teacher_map,
    }


def build_delayed_utility_target(
    old_teacher_prob,
    dabe_target,
    evaluator_prob_weak,
    evaluator_prob_flip,
    old_history_count,
    old_epoch,
    current_epoch,
    old_valid,
    cfg,
):
    consensus = 0.5 * (evaluator_prob_weak + evaluator_prob_flip)
    consensus_confidence = (2.0 * consensus - 1.0).abs()
    view_difference = (evaluator_prob_weak - evaluator_prob_flip).abs()
    source_disagreement = (dabe_target - old_teacher_prob).abs()
    error_dabe = (dabe_target - consensus).square()
    error_teacher = (old_teacher_prob - consensus).square()
    gap = error_dabe - error_teacher
    bound = float(getattr(cfg, "SOURCE_ARBITER_RESIDUAL_BOUND", 1.5))
    tau = float(getattr(cfg, "SOURCE_ARBITER_UTILITY_TAU", 0.10))
    delta_target = (gap / tau).clamp(-bound, bound)
    target_teacher = torch.sigmoid(delta_target)
    age = int(current_epoch) - old_epoch
    min_age = int(getattr(cfg, "SOURCE_ARBITER_MIN_MEMORY_AGE_EPOCH", 1))
    max_age = int(getattr(cfg, "SOURCE_ARBITER_MAX_MEMORY_AGE_EPOCH", 2))
    age_valid_per_image = (age >= min_age) & (age <= max_age) & old_valid
    age_valid = age_valid_per_image.view(-1, 1, 1, 1).expand_as(consensus)
    history_valid = (
        old_history_count
        >= int(getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_HISTORY", 3))
    ).view(-1, 1, 1, 1).expand_as(consensus)
    valid = (
        age_valid
        & history_valid
        & (
            consensus_confidence
            >= float(getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_CONSENSUS_CONF", 0.30))
        )
        & (
            view_difference
            <= float(getattr(cfg, "SOURCE_ARBITER_UTILITY_MAX_VIEW_DIFF", 0.15))
        )
        & (
            source_disagreement
            >= float(getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_SOURCE_DISAGREEMENT", 0.15))
        )
    )
    min_confidence = float(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_CONSENSUS_CONF", 0.30)
    )
    confidence_weight = (
        (consensus_confidence - min_confidence)
        / max(1e-6, 1.0 - min_confidence)
    ).clamp(0.0, 1.0)
    view_weight = torch.exp(
        -view_difference
        / float(getattr(cfg, "SOURCE_ARBITER_UTILITY_VIEW_TAU", 0.05))
    )
    gap_weight = (
        gap.abs() / float(getattr(cfg, "SOURCE_ARBITER_UTILITY_GAP_SCALE", 0.10))
    ).clamp(0.0, 1.0)
    weight = valid.float() * confidence_weight * view_weight * gap_weight
    return UtilityTarget(
        target_teacher=target_teacher.detach(),
        valid=valid.detach(),
        weight=weight.detach(),
        consensus=consensus.detach(),
        consensus_confidence=consensus_confidence.detach(),
        cross_view_difference=view_difference.detach(),
        source_disagreement=source_disagreement.detach(),
        utility_gap=gap.detach(),
        delta_target=delta_target.detach(),
        age_valid=age_valid.detach(),
    )


def _weighted_mean(value, weight, eps=1e-6):
    denominator = weight.sum()
    if float(denominator.detach()) <= 0.0:
        return value.sum() * 0.0
    return (value * weight).sum() / (denominator + float(eps))


def compute_source_arbiter_loss(
    old_residual,
    current_gate_teacher,
    teacher_prior,
    utility_target,
    dino_margin,
    train_image_mask,
    cfg,
):
    train_mask = train_image_mask.to(old_residual.device).bool().view(-1, 1, 1, 1)
    train_mask = train_mask.expand_as(old_residual)
    utility_weight = utility_target.weight * train_mask.float()
    utility_bce = F.binary_cross_entropy_with_logits(
        old_residual,
        utility_target.target_teacher,
        reduction="none",
    )
    loss_utility = _weighted_mean(utility_bce, utility_weight)
    invalid = (~utility_target.valid) & train_mask
    loss_prior = _weighted_mean(old_residual.square(), invalid.float())

    current_train = train_image_mask.to(current_gate_teacher.device).bool()
    if bool(current_train.any()):
        gate_mean = current_gate_teacher[current_train].mean()
        mass_excess = (gate_mean - float(teacher_prior)).abs() - float(
            getattr(cfg, "SOURCE_ARBITER_MASS_TOLERANCE", 0.20)
        )
        loss_mass = F.relu(mass_excess).square()
        gate = current_gate_teacher[current_train]
        margin = dino_margin[current_train]
        horizontal_edge = (margin[..., :, 1:] - margin[..., :, :-1]).abs()
        vertical_edge = (margin[..., 1:, :] - margin[..., :-1, :]).abs()
        kappa = float(getattr(cfg, "SOURCE_ARBITER_SMOOTH_KAPPA", 5.0))
        horizontal = torch.exp(-kappa * horizontal_edge) * (
            gate[..., :, 1:] - gate[..., :, :-1]
        ).abs()
        vertical = torch.exp(-kappa * vertical_edge) * (
            gate[..., 1:, :] - gate[..., :-1, :]
        ).abs()
        loss_smooth = 0.5 * (horizontal.mean() + vertical.mean())
    else:
        zero = old_residual.sum() * 0.0
        gate_mean = old_residual.new_tensor(float(teacher_prior))
        loss_mass = zero
        loss_smooth = zero

    total = (
        float(getattr(cfg, "SOURCE_ARBITER_LAMBDA_UTILITY", 1.0)) * loss_utility
        + float(getattr(cfg, "SOURCE_ARBITER_LAMBDA_PRIOR", 0.10)) * loss_prior
        + float(getattr(cfg, "SOURCE_ARBITER_LAMBDA_MASS", 0.05)) * loss_mass
        + float(getattr(cfg, "SOURCE_ARBITER_LAMBDA_SMOOTH", 0.005)) * loss_smooth
    )
    if not bool(torch.isfinite(total).item()):
        raise RuntimeError("SourceArbiter loss contains NaN/Inf.")
    valid_weight = utility_target.valid.float() * train_mask.float()
    stats = {
        "loss_utility": float(loss_utility.detach()),
        "loss_prior": float(loss_prior.detach()),
        "loss_mass": float(loss_mass.detach()),
        "loss_smooth": float(loss_smooth.detach()),
        "loss_total": float(total.detach()),
        "utility_valid_ratio": float(valid_weight.mean().detach()),
        "utility_target_mean": float(
            _weighted_mean(utility_target.target_teacher, valid_weight).detach()
        ),
        "utility_gap_mean": float(
            _weighted_mean(utility_target.utility_gap.abs(), valid_weight).detach()
        ),
        "cross_view_diff_mean": float(
            _weighted_mean(
                utility_target.cross_view_difference,
                valid_weight,
            ).detach()
        ),
        "consensus_conf_mean": float(
            _weighted_mean(
                utility_target.consensus_confidence,
                valid_weight,
            ).detach()
        ),
        "gate_teacher_train_mean": float(gate_mean.detach()),
    }
    return total, stats


def source_gate_stats(
    gate_dabe,
    gate_teacher,
    residual,
    teacher_prior,
    masks,
    teacher_prob,
    dabe_target,
    bound,
):
    def masked_mean(value, mask, default=0.0):
        if not bool(mask.any()):
            return float(default)
        return float(value[mask].mean().detach())

    disagreement = (teacher_prob - dabe_target).abs() >= 0.15
    agreement = ~disagreement
    teacher_fg = teacher_prob >= 0.5
    stats = {
        "gate_teacher_mean": float(gate_teacher.mean().detach()),
        "gate_teacher_std": float(gate_teacher.std(unbiased=False).detach()),
        "gate_dabe_mean": float(gate_dabe.mean().detach()),
        "residual_mean": float(residual.mean().detach()),
        "residual_std": float(residual.std(unbiased=False).detach()),
        "residual_abs_mean": float(residual.abs().mean().detach()),
        "residual_saturation_ratio": float(
            (residual.abs() >= 0.95 * float(bound)).float().mean().detach()
        ),
        "gate_teacher_above_prior_ratio": float(
            (gate_teacher > float(teacher_prior) + 0.10).float().mean().detach()
        ),
        "gate_teacher_below_prior_ratio": float(
            (gate_teacher < float(teacher_prior) - 0.10).float().mean().detach()
        ),
        "gate_teacher_near_prior_ratio": float(
            ((gate_teacher - float(teacher_prior)).abs() < 0.02).float().mean().detach()
        ),
        "gate_teacher_agreement": masked_mean(gate_teacher, agreement),
        "gate_teacher_disagreement": masked_mean(gate_teacher, disagreement),
        "gate_teacher_fg_core_conflict": masked_mean(
            gate_teacher, masks["fg_core"] & (~teacher_fg)
        ),
        "gate_teacher_bg_core_conflict": masked_mean(
            gate_teacher, masks["bg_core"] & teacher_fg
        ),
        "gate_teacher_extent_teacher_fg": masked_mean(
            gate_teacher, masks["extent"] & teacher_fg
        ),
        "gate_teacher_extent_teacher_bg": masked_mean(
            gate_teacher, masks["extent"] & (~teacher_fg)
        ),
    }
    for name, mask in masks.items():
        stats[f"gate_teacher_{name}"] = masked_mean(gate_teacher, mask)
        stats[f"gate_dabe_{name}"] = masked_mean(gate_dabe, mask)
    return stats


def source_arbiter_parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def compute_utility_validation_stats(
    residual,
    utility_target,
    validation_image_mask,
    teacher_prior=0.5,
    fixed_teacher_score=None,
    num_bins=100,
):
    image_mask = validation_image_mask.to(residual.device).bool().view(-1, 1, 1, 1)
    pixel_mask = utility_target.valid & image_mask.expand_as(residual)
    count = int(pixel_mask.sum().detach().item())
    if count <= 0:
        return {
            "valid_pixels": 0,
            "preference_correct": 0,
            "preference_total": 0,
            "positive_correct": 0,
            "positive_total": 0,
            "negative_correct": 0,
            "negative_total": 0,
            "brier_sum": 0.0,
            "prior_brier_sum": 0.0,
            "fixed_preference_correct": 0,
            "fixed_brier_sum": 0.0,
            "score_sum": 0.0,
            "target_sum": 0.0,
            "positive_hist": [0] * int(num_bins),
            "negative_hist": [0] * int(num_bins),
            "calibration_count": [0] * int(num_bins),
            "calibration_score_sum": [0.0] * int(num_bins),
            "calibration_target_sum": [0.0] * int(num_bins),
        }
    score = torch.sigmoid(residual.detach())[pixel_mask]
    target = (utility_target.utility_gap > 0.0)[pixel_mask]
    prediction = score >= 0.5
    positive = target
    negative = ~target
    soft_target = utility_target.target_teacher[pixel_mask]
    prior = torch.as_tensor(
        teacher_prior,
        device=residual.device,
        dtype=residual.dtype,
    )
    if prior.ndim == 0:
        prior = prior.expand(residual.shape[0])
    prior_map = prior.reshape(-1, 1, 1, 1).expand_as(residual)[pixel_mask]
    if fixed_teacher_score is None:
        fixed_score = prior_map
    else:
        fixed_score = fixed_teacher_score.detach()[pixel_mask].clamp(0.0, 1.0)
    bins = int(num_bins)
    if bins < 2:
        raise RuntimeError("Utility validation requires at least two score bins.")
    bin_index = (score * bins).long().clamp(0, bins - 1)
    positive_hist = torch.bincount(
        bin_index[positive], minlength=bins
    ).to("cpu")
    negative_hist = torch.bincount(
        bin_index[negative], minlength=bins
    ).to("cpu")
    calibration_count = torch.bincount(bin_index, minlength=bins).to("cpu")
    calibration_score_sum = torch.zeros(bins, device=score.device)
    calibration_target_sum = torch.zeros(bins, device=score.device)
    calibration_score_sum.scatter_add_(0, bin_index, score)
    calibration_target_sum.scatter_add_(0, bin_index, soft_target)
    return {
        "valid_pixels": count,
        "preference_correct": int((prediction == target).sum().item()),
        "preference_total": count,
        "positive_correct": int((prediction[positive] == target[positive]).sum().item()),
        "positive_total": int(positive.sum().item()),
        "negative_correct": int((prediction[negative] == target[negative]).sum().item()),
        "negative_total": int(negative.sum().item()),
        "brier_sum": float((score - soft_target).square().sum().item()),
        "prior_brier_sum": float((prior_map - soft_target).square().sum().item()),
        "fixed_preference_correct": int(
            ((fixed_score >= 0.5) == target).sum().item()
        ),
        "fixed_brier_sum": float(
            (fixed_score - soft_target).square().sum().item()
        ),
        "score_sum": float(score.sum().item()),
        "target_sum": float(soft_target.sum().item()),
        "positive_hist": positive_hist.tolist(),
        "negative_hist": negative_hist.tolist(),
        "calibration_count": calibration_count.tolist(),
        "calibration_score_sum": calibration_score_sum.detach().cpu().tolist(),
        "calibration_target_sum": calibration_target_sum.detach().cpu().tolist(),
    }
