import math
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


class SignAwareSourceArbiter(nn.Module):
    """Two-head training-only router for teacher foreground/background signals."""

    def __init__(
        self,
        input_channels=18,
        hidden_1=32,
        hidden_2=32,
        hidden_3=16,
        dilation=2,
        positive_residual_bound=1.5,
        negative_residual_bound=4.0,
        zero_init_head=True,
    ):
        super().__init__()
        self.positive_residual_bound = float(positive_residual_bound)
        self.negative_residual_bound = float(negative_residual_bound)
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
        self.head = nn.Conv2d(int(hidden_3), 2, 1)
        if bool(zero_init_head):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, evidence):
        if evidence.ndim != 4 or int(evidence.shape[1]) != 18:
            raise RuntimeError(
                "SignAwareSourceArbiter expects [B,18,H,W], "
                f"got {list(evidence.shape)}."
            )
        if not bool(torch.isfinite(evidence).all().item()):
            raise RuntimeError("SignAwareSourceArbiter evidence contains NaN/Inf.")
        raw = self.head(self.body(evidence))
        raw_positive, raw_negative = raw.chunk(2, dim=1)
        residual_positive = self.positive_residual_bound * torch.tanh(
            raw_positive
        )
        residual_negative = self.negative_residual_bound * torch.tanh(
            raw_negative
        )
        result = {
            "raw_positive": raw_positive,
            "raw_negative": raw_negative,
            "residual_positive": residual_positive,
            "residual_negative": residual_negative,
        }
        if any(
            not bool(torch.isfinite(value).all().item())
            for value in result.values()
        ):
            raise RuntimeError("SignAwareSourceArbiter output contains NaN/Inf.")
        return result


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


@dataclass
class DirectionalUtilityTarget:
    target_teacher: torch.Tensor
    valid: torch.Tensor
    weight: torch.Tensor
    positive_valid: torch.Tensor
    negative_valid: torch.Tensor
    teacher_hard: torch.Tensor
    dabe_hard: torch.Tensor
    dabe_valid: torch.Tensor
    consensus: torch.Tensor
    cross_view_difference: torch.Tensor
    future_move: torch.Tensor
    semantic_probability: torch.Tensor
    semantic_move: torch.Tensor
    direction_agreement: torch.Tensor
    source_disagreement: torch.Tensor
    teacher_advantage: torch.Tensor
    age_valid: torch.Tensor
    history_valid: torch.Tensor
    prototype_valid: torch.Tensor


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


def _linear_arbiter_scale(epoch, start, ramp_end, stop):
    epoch = int(epoch)
    start = int(start)
    ramp_end = int(ramp_end)
    stop = int(stop)
    if start <= 0 or ramp_end < start or stop <= ramp_end:
        raise RuntimeError(
            "Invalid source-arbiter schedule: "
            f"start={start}, ramp_end={ramp_end}, stop={stop}."
        )
    if epoch < start or epoch >= stop:
        return 0.0
    if epoch <= ramp_end:
        return float(epoch - start + 1) / float(ramp_end - start + 1)
    return 1.0


def get_arbiter_train_scale(epoch, cfg):
    if not bool(getattr(cfg, "USE_SOURCE_ARBITER", False)):
        return 0.0
    if str(getattr(cfg, "SOURCE_ARBITER_MODE", "")).lower() != (
        "sign_aware_pure_loss_space"
    ):
        return get_arbiter_influence_scale(epoch, cfg)
    return _linear_arbiter_scale(
        epoch,
        getattr(cfg, "SOURCE_ARBITER_TRAIN_START_EPOCH", 7),
        getattr(cfg, "SOURCE_ARBITER_TRAIN_RAMP_END_EPOCH", 15),
        getattr(cfg, "SOURCE_ARBITER_TRAIN_STOP_EPOCH", 21),
    )


def get_arbiter_apply_scale(epoch, cfg):
    if not bool(getattr(cfg, "USE_SOURCE_ARBITER", False)):
        return 0.0
    if str(getattr(cfg, "SOURCE_ARBITER_MODE", "")).lower() != (
        "sign_aware_pure_loss_space"
    ):
        return get_arbiter_influence_scale(epoch, cfg)
    return _linear_arbiter_scale(
        epoch,
        getattr(cfg, "SOURCE_ARBITER_APPLY_START_EPOCH", 7),
        getattr(cfg, "SOURCE_ARBITER_APPLY_RAMP_END_EPOCH", 15),
        getattr(cfg, "SOURCE_ARBITER_APPLY_STOP_EPOCH", 46),
    )


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


def compute_sign_aware_source_gates(
    router_output,
    teacher_binary,
    teacher_prior,
    teacher_prob,
    dabe_soft_target,
    influence_scale,
    eps=1e-4,
    disagreement_denom=0.5,
):
    required = {
        "residual_positive",
        "residual_negative",
    }
    missing = sorted(required.difference(router_output))
    if missing:
        raise RuntimeError(f"Sign-aware router output is missing: {missing}.")
    expected = tuple(teacher_binary.shape)
    for name in (
        "teacher_prob",
        "dabe_soft_target",
        "residual_positive",
        "residual_negative",
    ):
        value = (
            router_output[name]
            if name in router_output
            else {
                "teacher_prob": teacher_prob,
                "dabe_soft_target": dabe_soft_target,
            }[name]
        )
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"Sign-aware gate {name} shape mismatch: "
                f"{list(value.shape)} != {list(expected)}."
            )
    prior = float(teacher_prior)
    if not 0.0 <= prior <= 1.0:
        raise RuntimeError(f"Teacher prior must be in [0,1], got {prior}.")
    scale = float(influence_scale)
    if not 0.0 <= scale <= 1.0:
        raise RuntimeError(f"Router apply scale must be in [0,1], got {scale}.")
    disagreement_denom = float(disagreement_denom)
    if disagreement_denom <= 0.0:
        raise RuntimeError("Sign-aware disagreement denominator must be positive.")
    disagreement = (
        (teacher_prob.detach() - dabe_soft_target.detach()).abs()
        / disagreement_denom
    ).clamp(0.0, 1.0)

    if scale <= 0.0 or prior <= float(eps) or prior >= 1.0 - float(eps):
        gate_positive = torch.full_like(teacher_prob, prior)
        gate_negative = torch.full_like(teacher_prob, prior)
    else:
        prior_tensor = teacher_prob.new_tensor(prior)
        prior_logit = torch.log(prior_tensor) - torch.log1p(-prior_tensor)
        gate_positive = torch.sigmoid(
            prior_logit
            + scale * disagreement * router_output["residual_positive"]
        )
        gate_negative = torch.sigmoid(
            prior_logit
            + scale * disagreement * router_output["residual_negative"]
        )
    teacher_sign = (teacher_binary.detach() >= 0.5).to(gate_positive.dtype)
    gate_teacher = (
        teacher_sign * gate_positive + (1.0 - teacher_sign) * gate_negative
    )
    gate_dabe = 1.0 - gate_teacher
    values = {
        "gate_teacher_positive": gate_positive,
        "gate_teacher_negative": gate_negative,
        "gate_teacher": gate_teacher,
        "gate_dabe": gate_dabe,
        "source_disagreement": disagreement,
    }
    if any(
        not bool(torch.isfinite(value).all().item())
        for value in values.values()
    ):
        raise RuntimeError("Sign-aware source gate contains NaN/Inf.")
    complement_error = float((gate_teacher + gate_dabe - 1.0).abs().max())
    if complement_error > 1e-6:
        raise RuntimeError(
            f"Sign-aware complementary gate error={complement_error}."
        )
    return values


def normalize_source_pixel_loss(logits, target, weight, eps=1e-6):
    if tuple(logits.shape) != tuple(target.shape) or tuple(weight.shape) != tuple(logits.shape):
        raise RuntimeError("Source-normalized BCE tensors must have identical shapes.")
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denominator = weight.sum() + float(eps)
    return raw * weight * (float(raw.numel()) / denominator)


def build_teacher_source_weight(
    mode,
    teacher_prob,
    ecst_weight_map=None,
):
    """Select the source-internal teacher weight without changing gate semantics."""
    if not torch.is_tensor(teacher_prob) or teacher_prob.ndim != 4:
        raise RuntimeError("Teacher probability must be a 4D tensor.")
    mode = str(mode).lower()
    if mode == "residual_over_ecst":
        if ecst_weight_map is None:
            raise RuntimeError("R1 requires an ECST teacher source weight map.")
        if tuple(ecst_weight_map.shape) != tuple(teacher_prob.shape):
            raise RuntimeError(
                "ECST teacher source weight shape mismatch: "
                f"{list(ecst_weight_map.shape)} != {list(teacher_prob.shape)}."
            )
        weight = ecst_weight_map.detach().float()
    elif mode in {"pure_loss_space", "sign_aware_pure_loss_space"}:
        if ecst_weight_map is not None:
            raise RuntimeError(
                "EGSA-R2 pure_loss_space must not receive an ECST weight map."
            )
        weight = torch.ones_like(
            teacher_prob,
            dtype=torch.float32,
            device=teacher_prob.device,
        )
    else:
        raise RuntimeError(f"Unsupported source arbiter mode: {mode!r}.")

    if weight.requires_grad or not bool(torch.isfinite(weight).all().item()):
        raise RuntimeError("Teacher source weight must be detached and finite.")
    if mode in {"pure_loss_space", "sign_aware_pure_loss_space"} and not bool(
        torch.equal(weight, torch.ones_like(weight))
    ):
        raise RuntimeError("Pure loss-space teacher source weight must be exactly one.")
    return weight


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


def apply_sign_aware_loss_space_arbitration(
    logits,
    dabe_target,
    dabe_weight,
    teacher_target,
    gate_dabe,
    gate_teacher,
    source_sum,
    eps=1e-6,
):
    teacher_weight = torch.ones_like(
        teacher_target,
        dtype=torch.float32,
        device=teacher_target.device,
    )
    return apply_loss_space_arbitration(
        logits=logits,
        dabe_target=dabe_target,
        dabe_weight=dabe_weight,
        teacher_target=teacher_target,
        teacher_weight=teacher_weight,
        gate_dabe=gate_dabe,
        gate_teacher=gate_teacher,
        source_sum=source_sum,
        eps=eps,
    )


@torch.no_grad()
def compute_r1_shadow_stats(
    main_results,
    shadow_results,
    branch_weights,
    masks,
    branch_names=None,
):
    """Compare R2 raw-teacher loss with the audit-only R1 ECST shadow."""
    if not (
        len(main_results) == len(shadow_results) == len(branch_weights)
        and len(main_results) > 0
    ):
        raise RuntimeError("R1 shadow branch lists must have the same non-zero length.")
    if branch_names is None:
        branch_names = [f"branch_{index}" for index in range(len(main_results))]
    if len(branch_names) != len(main_results):
        raise RuntimeError("R1 shadow branch names do not match branch results.")
    denominator = max(1e-12, float(sum(branch_weights)))

    def aggregate(results, key):
        return sum(
            float(weight) * result[key]
            for weight, result in zip(branch_weights, results)
        ) / denominator

    raw_teacher = aggregate(main_results, "loss_teacher")
    ecst_teacher = aggregate(shadow_results, "loss_teacher")
    raw_total = aggregate(main_results, "loss")
    ecst_total = aggregate(shadow_results, "loss")
    raw_map = sum(
        float(weight) * result["teacher_loss_map"].detach()
        for weight, result in zip(branch_weights, main_results)
    ) / denominator
    ecst_map = sum(
        float(weight) * result["teacher_loss_map"].detach()
        for weight, result in zip(branch_weights, shadow_results)
    ) / denominator
    difference = ecst_map - raw_map
    stats = {
        "shadow_raw_teacher_group": float(raw_teacher.detach()),
        "shadow_ecst_teacher_group": float(ecst_teacher.detach()),
        "shadow_teacher_group_delta": float((ecst_teacher - raw_teacher).detach()),
        "shadow_raw_total_group": float(raw_total.detach()),
        "shadow_ecst_total_group": float(ecst_total.detach()),
        "shadow_total_group_delta": float((ecst_total - raw_total).detach()),
        "shadow_teacher_map_abs_delta": float(difference.abs().mean().detach()),
    }
    for name, main_result, shadow_result in zip(
        branch_names,
        main_results,
        shadow_results,
    ):
        raw_value = main_result["loss_teacher"]
        ecst_value = shadow_result["loss_teacher"]
        stats[f"shadow_{name}_raw_teacher"] = float(raw_value.detach())
        stats[f"shadow_{name}_ecst_teacher"] = float(ecst_value.detach())
        stats[f"shadow_{name}_teacher_delta"] = float(
            (ecst_value - raw_value).detach()
        )
    for name, mask in masks.items():
        mask = mask.to(difference.device).bool()
        count = int(mask.sum().detach().item())
        stats[f"shadow_{name}_valid"] = float(count > 0)
        stats[f"shadow_{name}_teacher_delta"] = (
            float(difference[mask].mean().detach()) if count > 0 else 0.0
        )
    return stats


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


def build_confident_dabe_utility_label(
    target_soft,
    weight_map,
    masks,
    fg_threshold=0.70,
    bg_threshold=0.30,
    weight_threshold=0.70,
    use_fg_core=True,
    use_bg_core=True,
):
    expected = tuple(target_soft.shape)
    if tuple(weight_map.shape) != expected:
        raise RuntimeError("DABE utility target/weight shape mismatch.")
    fg_core = masks["fg_core"].bool() if bool(use_fg_core) else torch.zeros_like(
        target_soft, dtype=torch.bool
    )
    bg_core = masks["bg_core"].bool() if bool(use_bg_core) else torch.zeros_like(
        target_soft, dtype=torch.bool
    )
    if tuple(fg_core.shape) != expected or tuple(bg_core.shape) != expected:
        raise RuntimeError("DABE utility core mask shape mismatch.")
    unassigned = ~fg_core
    selected_bg_core = unassigned & bg_core
    unassigned = unassigned & (~selected_bg_core)
    continuous_reliable = weight_map >= float(weight_threshold)
    selected_continuous_fg = (
        unassigned
        & continuous_reliable
        & (target_soft >= float(fg_threshold))
    )
    unassigned = unassigned & (~selected_continuous_fg)
    selected_continuous_bg = (
        unassigned
        & continuous_reliable
        & (target_soft <= float(bg_threshold))
    )
    valid = (
        fg_core
        | selected_bg_core
        | selected_continuous_fg
        | selected_continuous_bg
    )
    hard = (fg_core | selected_continuous_fg).to(target_soft.dtype)
    return {
        "hard": hard.detach(),
        "valid": valid.detach(),
        "fg": (fg_core | selected_continuous_fg).detach(),
        "bg": (selected_bg_core | selected_continuous_bg).detach(),
        "fg_core": fg_core.detach(),
        "bg_core": selected_bg_core.detach(),
        "continuous_fg": selected_continuous_fg.detach(),
        "continuous_bg": selected_continuous_bg.detach(),
    }


def build_directional_utility_target(
    old_teacher_prob,
    old_student_prob,
    evaluator_prob_weak,
    evaluator_prob_flip,
    old_history_count,
    old_epoch,
    current_epoch,
    old_valid,
    dabe_target,
    dabe_weight,
    masks,
    dino_margin,
    prototype_valid,
    cfg,
):
    expected = tuple(old_student_prob.shape)
    tensors = {
        "old_teacher_prob": old_teacher_prob,
        "evaluator_prob_weak": evaluator_prob_weak,
        "evaluator_prob_flip": evaluator_prob_flip,
        "dabe_target": dabe_target,
        "dabe_weight": dabe_weight,
        "dino_margin": dino_margin,
    }
    for name, value in tensors.items():
        if tuple(value.shape) != expected:
            raise RuntimeError(
                f"Directional utility {name} shape mismatch: "
                f"{list(value.shape)} != {list(expected)}."
            )
    dabe_label = build_confident_dabe_utility_label(
        dabe_target,
        dabe_weight,
        masks,
        fg_threshold=float(
            getattr(cfg, "SOURCE_ARBITER_UTILITY_DABE_FG_THRESH", 0.70)
        ),
        bg_threshold=float(
            getattr(cfg, "SOURCE_ARBITER_UTILITY_DABE_BG_THRESH", 0.30)
        ),
        weight_threshold=float(
            getattr(cfg, "SOURCE_ARBITER_UTILITY_DABE_WEIGHT_THRESH", 0.70)
        ),
        use_fg_core=bool(
            getattr(cfg, "SOURCE_ARBITER_UTILITY_USE_FG_CORE", True)
        ),
        use_bg_core=bool(
            getattr(cfg, "SOURCE_ARBITER_UTILITY_USE_BG_CORE", True)
        ),
    )
    consensus = 0.5 * (evaluator_prob_weak + evaluator_prob_flip)
    view_difference = (evaluator_prob_weak - evaluator_prob_flip).abs()
    teacher_hard = (old_teacher_prob >= 0.5).to(old_student_prob.dtype)
    dabe_hard = dabe_label["hard"]
    source_disagreement = teacher_hard != dabe_hard
    future_move = consensus - old_student_prob
    margin_tau = float(getattr(cfg, "SOURCE_ARBITER_DINO_MARGIN_TAU", 0.05))
    semantic_probability = torch.sigmoid(dino_margin / margin_tau)
    semantic_move = semantic_probability - old_student_prob
    direction_agreement = future_move * semantic_move > 0.0

    age = int(current_epoch) - old_epoch
    age_valid_per_image = (
        (age >= int(getattr(cfg, "SOURCE_ARBITER_MIN_MEMORY_AGE_EPOCH", 1)))
        & (age <= int(getattr(cfg, "SOURCE_ARBITER_MAX_MEMORY_AGE_EPOCH", 2)))
        & old_valid
    )
    age_valid = age_valid_per_image.view(-1, 1, 1, 1).expand_as(
        old_student_prob
    )
    history_valid = (
        old_history_count
        >= int(getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_HISTORY", 3))
    ).view(-1, 1, 1, 1).expand_as(old_student_prob)
    prototype_valid = torch.as_tensor(
        prototype_valid,
        device=old_student_prob.device,
        dtype=torch.bool,
    ).reshape(-1)
    if int(prototype_valid.numel()) != int(old_student_prob.shape[0]):
        raise RuntimeError("Directional utility prototype validity must be [B].")
    prototype_valid_map = prototype_valid.view(-1, 1, 1, 1).expand_as(
        old_student_prob
    )
    view_valid = view_difference <= float(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_MAX_VIEW_DIFF", 0.15)
    )
    future_valid = future_move.abs() >= float(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_FUTURE_MOVE", 0.03)
    )
    semantic_valid = semantic_move.abs() >= float(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_SEMANTIC_MOVE", 0.05)
    )
    if bool(
        getattr(
            cfg,
            "SOURCE_ARBITER_UTILITY_REQUIRE_DIRECTION_AGREEMENT",
            True,
        )
    ):
        direction_valid = direction_agreement
    else:
        direction_valid = torch.ones_like(direction_agreement)
    valid = (
        age_valid
        & history_valid
        & view_valid
        & future_valid
        & semantic_valid
        & direction_valid
        & dabe_label["valid"]
        & source_disagreement
        & prototype_valid_map
    )
    teacher_advantage = (old_student_prob - consensus) * (
        dabe_hard - teacher_hard
    )
    utility_tau = float(getattr(cfg, "SOURCE_ARBITER_UTILITY_TAU", 0.05))
    target_teacher = torch.sigmoid(teacher_advantage / utility_tau)
    view_weight = torch.exp(
        -view_difference
        / float(getattr(cfg, "SOURCE_ARBITER_UTILITY_VIEW_TAU", 0.05))
    )
    future_weight = (
        future_move.abs()
        / float(getattr(cfg, "SOURCE_ARBITER_UTILITY_FUTURE_WEIGHT_SCALE", 0.15))
    ).clamp(0.0, 1.0)
    semantic_weight = (
        semantic_move.abs()
        / float(
            getattr(cfg, "SOURCE_ARBITER_UTILITY_SEMANTIC_WEIGHT_SCALE", 0.25)
        )
    ).clamp(0.0, 1.0)
    weight = (
        valid.float()
        * view_weight
        * future_weight
        * semantic_weight
        * dabe_weight.clamp(0.0, 1.0)
    )
    positive_valid = valid & (teacher_hard >= 0.5)
    negative_valid = valid & (teacher_hard < 0.5)
    result = DirectionalUtilityTarget(
        target_teacher=target_teacher.detach(),
        valid=valid.detach(),
        weight=weight.detach(),
        positive_valid=positive_valid.detach(),
        negative_valid=negative_valid.detach(),
        teacher_hard=teacher_hard.detach(),
        dabe_hard=dabe_hard.detach(),
        dabe_valid=dabe_label["valid"].detach(),
        consensus=consensus.detach(),
        cross_view_difference=view_difference.detach(),
        future_move=future_move.detach(),
        semantic_probability=semantic_probability.detach(),
        semantic_move=semantic_move.detach(),
        direction_agreement=direction_agreement.detach(),
        source_disagreement=source_disagreement.detach(),
        teacher_advantage=teacher_advantage.detach(),
        age_valid=age_valid.detach(),
        history_valid=history_valid.detach(),
        prototype_valid=prototype_valid_map.detach(),
    )
    if not bool(torch.isfinite(result.target_teacher).all().item()) or not bool(
        torch.isfinite(result.weight).all().item()
    ):
        raise RuntimeError("Directional utility target contains NaN/Inf.")
    return result


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


def _binary_auc(scores, labels):
    scores = scores.detach().float().reshape(-1)
    labels = labels.detach().bool().reshape(-1)
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    if positives <= 0 or negatives <= 0:
        return 0.5, False
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(
        1,
        int(scores.numel()) + 1,
        device=scores.device,
        dtype=torch.float64,
    )
    positive_rank_sum = ranks[labels].sum()
    auc = (
        positive_rank_sum
        - float(positives * (positives + 1)) / 2.0
    ) / float(positives * negatives)
    return float(auc.item()), True


def _resolve_branch_class_weights(branch_name, class_weights, cfg):
    branch_name = str(branch_name).lower()
    if branch_name not in {"positive", "negative"}:
        raise ValueError(f"Unsupported sign-aware branch: {branch_name!r}.")
    branch_values = {}
    if isinstance(class_weights, dict):
        nested = class_weights.get(branch_name, {})
        if isinstance(nested, dict):
            branch_values = nested
        else:
            branch_values = {
                "teacher": class_weights.get(
                    f"{branch_name}_teacher_class_weight", 1.0
                ),
                "dabe": class_weights.get(
                    f"{branch_name}_dabe_class_weight", 1.0
                ),
            }
    teacher_weight = float(
        branch_values.get(
            "teacher",
            getattr(
                cfg,
                f"SOURCE_ARBITER_{branch_name.upper()}_TEACHER_CLASS_WEIGHT",
                1.0,
            ),
        )
    )
    dabe_weight = float(
        branch_values.get(
            "dabe",
            getattr(
                cfg,
                f"SOURCE_ARBITER_{branch_name.upper()}_DABE_CLASS_WEIGHT",
                1.0,
            ),
        )
    )
    class_min = float(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_CLASS_WEIGHT_MIN", 0.5)
    )
    class_max = float(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_CLASS_WEIGHT_MAX", 4.0)
    )
    for name, value in (
        ("teacher", teacher_weight),
        ("dabe", dabe_weight),
    ):
        if not math.isfinite(value) or not class_min <= value <= class_max:
            raise RuntimeError(
                f"{branch_name} {name} class weight must be finite and in "
                f"[{class_min},{class_max}], got {value}."
            )
    return teacher_weight, dabe_weight


def _balanced_branch_utility_loss(
    raw,
    target,
    branch_valid,
    base_weight,
    cfg,
    branch_name,
    class_weights=None,
):
    teacher_preferred = branch_valid & (target > 0.5)
    dabe_preferred = branch_valid & (target < 0.5)
    tie = branch_valid & (~teacher_preferred) & (~dabe_preferred)
    teacher_count = int(teacher_preferred.sum().detach().item())
    dabe_count = int(dabe_preferred.sum().detach().item())
    valid_count = int(branch_valid.sum().detach().item())
    class_weight = torch.ones_like(base_weight)
    teacher_class_weight, dabe_class_weight = _resolve_branch_class_weights(
        branch_name,
        class_weights,
        cfg,
    )
    class_weight = torch.where(
        teacher_preferred,
        torch.full_like(class_weight, teacher_class_weight),
        class_weight,
    )
    class_weight = torch.where(
        dabe_preferred,
        torch.full_like(class_weight, dabe_class_weight),
        class_weight,
    )
    effective_weight = base_weight * branch_valid.float() * class_weight
    bce = F.binary_cross_entropy_with_logits(raw, target, reduction="none")
    loss = _weighted_mean(bce, effective_weight)
    score = torch.sigmoid(raw.detach())
    preference = target > 0.5
    preference_mask = branch_valid & (~tie)
    if bool(preference_mask.any().item()):
        correct = (
            (score >= 0.5) == preference
        )[preference_mask].float().mean()
        brier = (score[branch_valid] - target[branch_valid]).square().mean()
        auc, auc_valid = _binary_auc(
            score[preference_mask],
            preference[preference_mask],
        )
        teacher_recall = (
            float((score[teacher_preferred] >= 0.5).float().mean().item())
            if teacher_count > 0
            else 0.0
        )
        dabe_recall = (
            float((score[dabe_preferred] < 0.5).float().mean().item())
            if dabe_count > 0
            else 0.0
        )
    else:
        correct = raw.new_tensor(0.0)
        brier = raw.new_tensor(0.0)
        auc, auc_valid = 0.5, False
        teacher_recall = 0.0
        dabe_recall = 0.0
    return loss, {
        "valid_pixels": valid_count,
        "teacher_preferred_count": teacher_count,
        "dabe_preferred_count": dabe_count,
        "tie_count": int(tie.sum().detach().item()),
        "teacher_preferred_ratio": teacher_count / max(valid_count, 1),
        "dabe_preferred_ratio": dabe_count / max(valid_count, 1),
        "teacher_class_weight": teacher_class_weight,
        "dabe_class_weight": dabe_class_weight,
        "preference_accuracy": float(correct.detach()),
        "teacher_recall": teacher_recall,
        "dabe_recall": dabe_recall,
        "auc": auc,
        "auc_valid": float(auc_valid),
        "brier": float(brier.detach()),
    }


def _edge_aware_smoothness(gate, margin, image_mask, kappa):
    selected_gate = gate[image_mask]
    selected_margin = margin[image_mask]
    if int(selected_gate.shape[0]) <= 0:
        return gate.sum() * 0.0
    horizontal_edge = (
        selected_margin[..., :, 1:] - selected_margin[..., :, :-1]
    ).abs()
    vertical_edge = (
        selected_margin[..., 1:, :] - selected_margin[..., :-1, :]
    ).abs()
    horizontal = torch.exp(-float(kappa) * horizontal_edge) * (
        selected_gate[..., :, 1:] - selected_gate[..., :, :-1]
    ).abs()
    vertical = torch.exp(-float(kappa) * vertical_edge) * (
        selected_gate[..., 1:, :] - selected_gate[..., :-1, :]
    ).abs()
    return 0.5 * (horizontal.mean() + vertical.mean())


def compute_sign_aware_arbiter_loss(
    old_router_output,
    current_gate_positive,
    current_gate_negative,
    teacher_prior,
    utility_target,
    dino_margin,
    train_image_mask,
    cfg,
    class_weights=None,
):
    train_images = train_image_mask.to(dino_margin.device).bool().reshape(-1)
    if int(train_images.numel()) != int(dino_margin.shape[0]):
        raise RuntimeError("Sign-aware train image mask must be [B].")
    train_pixels = train_images.view(-1, 1, 1, 1).expand_as(dino_margin)
    positive_valid = utility_target.positive_valid & train_pixels
    negative_valid = utility_target.negative_valid & train_pixels
    loss_positive, positive_stats = _balanced_branch_utility_loss(
        old_router_output["raw_positive"],
        utility_target.target_teacher,
        positive_valid,
        utility_target.weight,
        cfg,
        "positive",
        class_weights=class_weights,
    )
    loss_negative, negative_stats = _balanced_branch_utility_loss(
        old_router_output["raw_negative"],
        utility_target.target_teacher,
        negative_valid,
        utility_target.weight,
        cfg,
        "negative",
        class_weights=class_weights,
    )
    active_branches = int(positive_stats["valid_pixels"] > 0) + int(
        negative_stats["valid_pixels"] > 0
    )
    if active_branches == 2:
        loss_utility = 0.5 * (loss_positive + loss_negative)
    elif positive_stats["valid_pixels"] > 0:
        loss_utility = loss_positive
    elif negative_stats["valid_pixels"] > 0:
        loss_utility = loss_negative
    else:
        loss_utility = dino_margin.sum() * 0.0

    invalid_positive = (~utility_target.positive_valid) & train_pixels
    invalid_negative = (~utility_target.negative_valid) & train_pixels
    prior_positive = _weighted_mean(
        old_router_output["raw_positive"].square(),
        invalid_positive.float(),
    )
    prior_negative = _weighted_mean(
        old_router_output["raw_negative"].square(),
        invalid_negative.float(),
    )
    loss_prior = 0.5 * (prior_positive + prior_negative)

    if bool(train_images.any().item()):
        tolerance = float(
            getattr(cfg, "SOURCE_ARBITER_MASS_TOLERANCE", 0.30)
        )
        mass_positive = F.relu(
            (current_gate_positive[train_images].mean() - float(teacher_prior)).abs()
            - tolerance
        ).square()
        mass_negative = F.relu(
            (current_gate_negative[train_images].mean() - float(teacher_prior)).abs()
            - tolerance
        ).square()
        loss_mass = 0.5 * (mass_positive + mass_negative)
        kappa = float(getattr(cfg, "SOURCE_ARBITER_SMOOTH_KAPPA", 5.0))
        smooth_positive = _edge_aware_smoothness(
            current_gate_positive,
            dino_margin,
            train_images,
            kappa,
        )
        smooth_negative = _edge_aware_smoothness(
            current_gate_negative,
            dino_margin,
            train_images,
            kappa,
        )
        loss_smooth = smooth_positive + smooth_negative
    else:
        zero = dino_margin.sum() * 0.0
        loss_mass = zero
        loss_smooth = zero

    total = (
        float(getattr(cfg, "SOURCE_ARBITER_LAMBDA_UTILITY", 1.0))
        * loss_utility
        + float(getattr(cfg, "SOURCE_ARBITER_LAMBDA_PRIOR", 0.10))
        * loss_prior
        + float(getattr(cfg, "SOURCE_ARBITER_LAMBDA_MASS", 0.01))
        * loss_mass
        + float(getattr(cfg, "SOURCE_ARBITER_LAMBDA_SMOOTH", 0.005))
        * loss_smooth
    )
    if not bool(torch.isfinite(total).item()):
        raise RuntimeError("Sign-aware source-arbiter loss contains NaN/Inf.")
    valid = utility_target.valid & train_pixels
    stats = {
        "loss_utility": float(loss_utility.detach()),
        "loss_utility_positive": float(loss_positive.detach()),
        "loss_utility_negative": float(loss_negative.detach()),
        "loss_prior": float(loss_prior.detach()),
        "loss_mass": float(loss_mass.detach()),
        "loss_smooth": float(loss_smooth.detach()),
        "loss_total": float(total.detach()),
        "utility_valid_ratio": float(valid.float().mean().detach()),
        "utility_target_mean": float(
            _weighted_mean(
                utility_target.target_teacher,
                valid.float(),
            ).detach()
        ),
        "future_move_abs_mean": float(
            _weighted_mean(
                utility_target.future_move.abs(),
                valid.float(),
            ).detach()
        ),
        "semantic_move_abs_mean": float(
            _weighted_mean(
                utility_target.semantic_move.abs(),
                valid.float(),
            ).detach()
        ),
        "direction_agreement_ratio": float(
            utility_target.direction_agreement.float().mean().detach()
        ),
        "source_disagreement_ratio": float(
            utility_target.source_disagreement.float().mean().detach()
        ),
        "dabe_confident_label_ratio": float(
            utility_target.dabe_valid.float().mean().detach()
        ),
        "active_sign_branches": float(active_branches),
    }
    for prefix, branch_stats in (
        ("positive_branch", positive_stats),
        ("negative_branch", negative_stats),
    ):
        for name, value in branch_stats.items():
            stats[f"{prefix}_{name}"] = float(value)
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
    correction = gate_teacher - float(teacher_prior)
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
        "correction_mean": float(correction.mean().detach()),
        "correction_std": float(correction.std(unbiased=False).detach()),
        "correction_abs_mean": float(correction.abs().mean().detach()),
        "correction_zero_ratio": float(
            (correction.abs() <= 1e-8).float().mean().detach()
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
    for threshold in (0.02, 0.05, 0.10):
        suffix = f"{int(round(threshold * 100)):02d}"
        stats[f"correction_pos_ge_{suffix}_ratio"] = float(
            (correction >= threshold).float().mean().detach()
        )
        stats[f"correction_neg_le_{suffix}_ratio"] = float(
            (correction <= -threshold).float().mean().detach()
        )
    for name, mask in masks.items():
        stats[f"gate_teacher_{name}"] = masked_mean(gate_teacher, mask)
        stats[f"gate_dabe_{name}"] = masked_mean(gate_dabe, mask)
        stats[f"correction_{name}"] = masked_mean(correction, mask)
    stats["correction_fg_core_conflict"] = masked_mean(
        correction, masks["fg_core"] & (~teacher_fg)
    )
    stats["correction_bg_core_conflict"] = masked_mean(
        correction, masks["bg_core"] & teacher_fg
    )
    stats["correction_extent_teacher_fg"] = masked_mean(
        correction, masks["extent"] & teacher_fg
    )
    stats["correction_extent_teacher_bg"] = masked_mean(
        correction, masks["extent"] & (~teacher_fg)
    )
    return stats


def collect_sign_aware_stats(
    gates,
    router_output,
    teacher_prior,
    masks,
    teacher_binary,
    dabe_target,
    dabe_weight,
    positive_bound=1.5,
    negative_bound=4.0,
):
    teacher_fg = teacher_binary >= 0.5
    teacher_bg = ~teacher_fg
    high_fg = (dabe_target >= 0.70) & (dabe_weight >= 0.70)
    high_bg = (dabe_target <= 0.30) & (dabe_weight >= 0.70)
    inflation = teacher_fg & high_bg
    completion = teacher_fg & high_fg
    dangerous_negative = teacher_bg & (
        masks["fg_core"] | high_fg | masks["extent"] | masks["unknown"]
    )

    def masked_mean(value, mask, default=0.0):
        if not bool(mask.any().item()):
            return float(default)
        return float(value[mask].mean().detach())

    positive = gates["gate_teacher_positive"]
    negative = gates["gate_teacher_negative"]
    applied = gates["gate_teacher"]
    residual_positive = router_output["residual_positive"]
    residual_negative = router_output["residual_negative"]
    correction_positive = positive - float(teacher_prior)
    correction_negative = negative - float(teacher_prior)
    stats = {
        "gate_teacher_positive_mean": float(positive.mean().detach()),
        "gate_teacher_positive_std": float(
            positive.std(unbiased=False).detach()
        ),
        "gate_teacher_negative_mean": float(negative.mean().detach()),
        "gate_teacher_negative_std": float(
            negative.std(unbiased=False).detach()
        ),
        "gate_teacher_mean": float(applied.mean().detach()),
        "gate_teacher_std": float(applied.std(unbiased=False).detach()),
        "gate_dabe_mean": float(gates["gate_dabe"].mean().detach()),
        "residual_positive_mean": float(residual_positive.mean().detach()),
        "residual_positive_std": float(
            residual_positive.std(unbiased=False).detach()
        ),
        "residual_positive_abs_mean": float(
            residual_positive.abs().mean().detach()
        ),
        "residual_positive_saturation_ratio": float(
            (
                residual_positive.abs()
                >= 0.95 * float(positive_bound)
            )
            .float()
            .mean()
            .detach()
        ),
        "residual_negative_mean": float(residual_negative.mean().detach()),
        "residual_negative_std": float(
            residual_negative.std(unbiased=False).detach()
        ),
        "residual_negative_abs_mean": float(
            residual_negative.abs().mean().detach()
        ),
        "residual_negative_saturation_ratio": float(
            (
                residual_negative.abs()
                >= 0.95 * float(negative_bound)
            )
            .float()
            .mean()
            .detach()
        ),
        "positive_correction_mean": float(correction_positive.mean().detach()),
        "negative_correction_mean": float(correction_negative.mean().detach()),
        "positive_above_prior_ratio": float(
            (correction_positive >= 0.01).float().mean().detach()
        ),
        "positive_below_prior_ratio": float(
            (correction_positive <= -0.01).float().mean().detach()
        ),
        "negative_above_prior_ratio": float(
            (correction_negative >= 0.01).float().mean().detach()
        ),
        "negative_below_prior_ratio": float(
            (correction_negative <= -0.01).float().mean().detach()
        ),
        "gate_negative_teacher_bg_fg_core": masked_mean(
            negative, teacher_bg & masks["fg_core"]
        ),
        "gate_negative_teacher_bg_high_fg": masked_mean(
            negative, teacher_bg & high_fg
        ),
        "gate_negative_extent_teacher_bg": masked_mean(
            negative, teacher_bg & masks["extent"]
        ),
        "gate_negative_unknown_teacher_bg": masked_mean(
            negative, teacher_bg & masks["unknown"]
        ),
        "gate_positive_teacher_fg_bg_core": masked_mean(
            positive, teacher_fg & masks["bg_core"]
        ),
        "gate_positive_teacher_fg_inflation": masked_mean(
            positive, inflation
        ),
        "gate_positive_teacher_fg_completion": masked_mean(
            positive, completion
        ),
        "dangerous_negative_ratio": float(
            dangerous_negative.float().mean().detach()
        ),
    }
    return stats


def evaluate_r2b_target_audit_admission(
    branches,
    cfg,
    full_audit,
    processed_images=None,
    expected_images=4040,
):
    """Apply the fixed R2b target-audit admission criteria."""
    minimum_valid = int(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_BRANCH_PIXELS", 10000)
    )
    minimum_class = int(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_CLASS_PIXELS", 1000)
    )
    minimum_ratio = float(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_MIN_CLASS_RATIO", 0.01)
    )
    minimum_negative_dabe_ratio = float(
        getattr(cfg, "SOURCE_ARBITER_UTILITY_NEG_DABE_MIN_RATIO", 0.02)
    )
    criteria = {
        "minimum_valid_pixels_per_branch": minimum_valid,
        "minimum_minority_count_per_branch": minimum_class,
        "minimum_minority_ratio_per_branch": minimum_ratio,
        "minimum_negative_dabe_preferred_ratio": (
            minimum_negative_dabe_ratio
        ),
        "expected_full_audit_images": int(expected_images),
    }
    failures = []
    normalized = {}
    for branch_name in ("positive", "negative"):
        values = branches.get(branch_name, {})
        if not isinstance(values, dict):
            values = {}
        valid = int(values.get("valid_pixels", 0))
        teacher = int(values.get("teacher_preferred_count", 0))
        dabe = int(values.get("dabe_preferred_count", 0))
        minority = min(teacher, dabe)
        minority_ratio = minority / max(valid, 1)
        normalized[branch_name] = {
            "valid_pixels": valid,
            "teacher_preferred_count": teacher,
            "dabe_preferred_count": dabe,
            "minority_count": minority,
            "minority_ratio": minority_ratio,
            "teacher_preferred_ratio": teacher / max(valid, 1),
            "dabe_preferred_ratio": dabe / max(valid, 1),
        }
        if valid < minimum_valid:
            failures.append(f"{branch_name}:valid_pixels")
        if minority < minimum_class:
            failures.append(f"{branch_name}:minority_count")
        if minority_ratio < minimum_ratio:
            failures.append(f"{branch_name}:minority_ratio")

    if (
        normalized["negative"]["dabe_preferred_ratio"]
        < minimum_negative_dabe_ratio
    ):
        failures.append("negative:dabe_preferred_ratio")
    if not bool(full_audit):
        failures.append("full_4040_sample_audit_required")
    if processed_images is None:
        failures.append("processed_images:missing")
    elif int(processed_images) != int(expected_images):
        failures.append(
            f"processed_images:{int(processed_images)}!={int(expected_images)}"
        )
    return {
        "passed": not failures,
        "failures": failures,
        "criteria": criteria,
        "normalized_branches": normalized,
    }


def _audit_quantile(values, quantile):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    quantile = min(1.0, max(0.0, float(quantile)))
    position = quantile * float(len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - float(lower)
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def compute_r2b_audit_class_weights(
    valid_pixels,
    teacher_preferred_pixels,
    dabe_preferred_pixels,
    weight_min=0.5,
    weight_max=4.0,
):
    valid = max(0, int(valid_pixels))
    teacher = max(0, int(teacher_preferred_pixels))
    dabe = max(0, int(dabe_preferred_pixels))
    weight_min = float(weight_min)
    weight_max = float(weight_max)
    if (
        not math.isfinite(weight_min)
        or not math.isfinite(weight_max)
        or weight_min <= 0.0
        or weight_max < weight_min
    ):
        raise ValueError(
            f"Invalid R2b audit class-weight bounds: {weight_min}, {weight_max}."
        )

    def one(count):
        if valid <= 0 or count <= 0:
            return weight_max
        value = math.sqrt(float(valid) / float(2 * count))
        return max(weight_min, min(weight_max, value))

    return {
        "teacher": float(one(teacher)),
        "dabe": float(one(dabe)),
    }


def summarize_r2b_target_audit_branch_v2(
    image_observations,
    active_batches,
    total_batches,
    weight_min=0.5,
    weight_max=4.0,
):
    observations = list(image_observations)
    valid = sum(max(0, int(row.get("valid_pixels", 0))) for row in observations)
    teacher = sum(
        max(
            0,
            int(
                row.get(
                    "teacher_preferred_pixels",
                    row.get("teacher_preferred_count", 0),
                )
            ),
        )
        for row in observations
    )
    dabe = sum(
        max(
            0,
            int(
                row.get(
                    "dabe_preferred_pixels",
                    row.get("dabe_preferred_count", 0),
                )
            ),
        )
        for row in observations
    )
    soft_teacher = sum(
        max(
            0.0,
            float(row.get("soft_teacher_mass", row.get("target_sum", 0.0))),
        )
        for row in observations
    )
    soft_teacher = min(float(valid), soft_teacher)
    soft_dabe = max(0.0, float(valid) - soft_teacher)
    hard_minority_class = "teacher" if teacher <= dabe else "dabe"
    hard_minority = min(teacher, dabe)
    minority_counts = []
    valid_images = 0
    for row in observations:
        row_valid = max(0, int(row.get("valid_pixels", 0)))
        valid_images += int(row_valid > 0)
        key = (
            "teacher_preferred_pixels"
            if hard_minority_class == "teacher"
            else "dabe_preferred_pixels"
        )
        fallback = (
            "teacher_preferred_count"
            if hard_minority_class == "teacher"
            else "dabe_preferred_count"
        )
        minority_counts.append(max(0, int(row.get(key, row.get(fallback, 0)))))
    minority_valid_images = sum(int(value > 0) for value in minority_counts)
    sorted_counts = sorted(minority_counts, reverse=True)

    def concentration(fraction):
        if hard_minority <= 0 or not sorted_counts:
            return 0.0
        count = max(1, int(math.ceil(float(len(sorted_counts)) * fraction)))
        return float(sum(sorted_counts[:count])) / float(hard_minority)

    active_batches = max(0, int(active_batches))
    total_batches = max(0, int(total_batches))
    class_weights = compute_r2b_audit_class_weights(
        valid,
        teacher,
        dabe,
        weight_min=weight_min,
        weight_max=weight_max,
    )
    result = {
        "valid_pixels": valid,
        "teacher_preferred_pixels": teacher,
        "dabe_preferred_pixels": dabe,
        "hard_minority_class": hard_minority_class,
        "hard_minority_pixels": hard_minority,
        "hard_minority_ratio": (
            float(hard_minority) / float(valid) if valid > 0 else 0.0
        ),
        "soft_teacher_mass": float(soft_teacher),
        "soft_dabe_mass": float(soft_dabe),
        "soft_minority_mass": float(min(soft_teacher, soft_dabe)),
        "valid_images": valid_images,
        "minority_valid_images": minority_valid_images,
        "minority_pixels_per_image_mean": (
            float(sum(minority_counts)) / float(len(minority_counts))
            if minority_counts
            else 0.0
        ),
        "minority_pixels_per_image_median": _audit_quantile(
            minority_counts, 0.50
        ),
        "minority_pixels_per_image_q90": _audit_quantile(
            minority_counts, 0.90
        ),
        "minority_pixels_per_image_q99": _audit_quantile(
            minority_counts, 0.99
        ),
        "minority_pixels_per_image_max": (
            max(minority_counts) if minority_counts else 0
        ),
        "top_1_percent_images_minority_share": concentration(0.01),
        "top_10_percent_images_minority_share": concentration(0.10),
        "active_batches": active_batches,
        "total_batches": total_batches,
        "active_batch_ratio": (
            float(active_batches) / float(total_batches)
            if total_batches > 0
            else 0.0
        ),
        "teacher_class_weight": class_weights["teacher"],
        "dabe_class_weight": class_weights["dabe"],
    }
    for name, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"R2b Audit v2 produced non-finite {name}.")
    return result


def evaluate_r2b_target_audit_v2_admission(
    branches,
    cfg,
    full_audit,
    processed_images=None,
    expected_images=4040,
    diagnostic_only=False,
):
    """Apply branch-specific R2b Audit v2 admission criteria."""
    branch_criteria = {
        "positive": {
            "valid_pixels": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_POS_MIN_VALID_PIXELS",
                        10000,
                    )
                ),
            ),
            "hard_minority_pixels": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_POS_MIN_HARD_MINORITY_PIXELS",
                        500,
                    )
                ),
            ),
            "hard_minority_ratio": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_POS_MIN_HARD_MINORITY_RATIO",
                        0.01,
                    )
                ),
            ),
            "soft_minority_mass": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_POS_MIN_SOFT_MINORITY_MASS",
                        2000.0,
                    )
                ),
            ),
            "minority_valid_images": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_POS_MIN_MINORITY_IMAGES",
                        128,
                    )
                ),
            ),
            "active_batch_ratio": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_POS_MIN_ACTIVE_BATCH_RATIO",
                        0.90,
                    )
                ),
            ),
        },
        "negative": {
            "valid_pixels": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_NEG_MIN_VALID_PIXELS",
                        50000,
                    )
                ),
            ),
            "hard_minority_pixels": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_NEG_MIN_HARD_MINORITY_PIXELS",
                        1000,
                    )
                ),
            ),
            "hard_minority_ratio": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_NEG_MIN_HARD_MINORITY_RATIO",
                        0.01,
                    )
                ),
            ),
            "soft_minority_mass": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_NEG_MIN_SOFT_MINORITY_MASS",
                        2000.0,
                    )
                ),
            ),
            "minority_valid_images": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_NEG_MIN_MINORITY_IMAGES",
                        256,
                    )
                ),
            ),
            "active_batch_ratio": (
                ">=",
                float(
                    getattr(
                        cfg,
                        "SOURCE_ARBITER_AUDIT_NEG_MIN_ACTIVE_BATCH_RATIO",
                        0.99,
                    )
                ),
            ),
        },
    }
    maximum_top1 = float(
        getattr(cfg, "SOURCE_ARBITER_AUDIT_MAX_TOP1P_IMAGE_SHARE", 0.25)
    )
    maximum_top10 = float(
        getattr(cfg, "SOURCE_ARBITER_AUDIT_MAX_TOP10P_IMAGE_SHARE", 0.60)
    )
    for criteria in branch_criteria.values():
        criteria["top_1_percent_images_minority_share"] = ("<=", maximum_top1)
        criteria["top_10_percent_images_minority_share"] = ("<=", maximum_top10)

    failures = []
    condition_results = {}
    for branch_name in ("positive", "negative"):
        values = branches.get(branch_name, {})
        if not isinstance(values, dict):
            values = {}
        condition_results[branch_name] = {}
        for field, (operator, threshold) in branch_criteria[branch_name].items():
            actual = float(values.get(field, 0.0))
            condition_passed = (
                actual >= threshold if operator == ">=" else actual <= threshold
            )
            condition_results[branch_name][field] = {
                "actual": actual,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(condition_passed),
            }
            if not condition_passed:
                failures.append(f"{branch_name}:{field}")

    if bool(diagnostic_only):
        failures.append("diagnostic_only")
    if not bool(full_audit):
        failures.append("full_4040_sample_audit_required")
    if processed_images is None:
        failures.append("processed_images:missing")
    elif int(processed_images) != int(expected_images):
        failures.append(
            f"processed_images:{int(processed_images)}!={int(expected_images)}"
        )
    failures = list(dict.fromkeys(failures))
    passed = not failures
    criteria = {
        branch_name: {
            field: {"operator": operator, "threshold": threshold}
            for field, (operator, threshold) in branch_criteria[
                branch_name
            ].items()
        }
        for branch_name in ("positive", "negative")
    }
    criteria.update(
        {
        "expected_full_audit_images": int(expected_images),
        }
    )
    return {
        "passed": passed,
        "mini_run_authorized": passed,
        "stage20_authorized": False,
        "diagnostic_only": bool(diagnostic_only),
        "failures": failures,
        "criteria": criteria,
        "condition_results": condition_results,
        "branches": branches,
    }


def evaluate_r2b_audit_concentration_override(
    audit_failures,
    allowed_failures,
    *,
    full_audit,
    processed_images,
    expected_images=4040,
):
    """Authorize exploration only when every failure is explicitly allowlisted."""
    failures = list(dict.fromkeys(str(item) for item in audit_failures))
    allowed = list(dict.fromkeys(str(item) for item in allowed_failures))
    allowed_set = set(allowed)
    unexpected = [item for item in failures if item not in allowed_set]
    strict_pass = len(failures) == 0
    override_passed = (
        bool(full_audit)
        and int(processed_images) == int(expected_images)
        and len(unexpected) == 0
    )
    return {
        "strict_pass": strict_pass,
        "override_passed": override_passed,
        "unexpected_failures": unexpected,
        "allowed_failures": allowed,
        "audit_failures": failures,
        "processed_images": int(processed_images),
        "expected_images": int(expected_images),
        "full_audit": bool(full_audit),
    }


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
            "utility_teacher_count": 0,
            "utility_dabe_count": 0,
            "utility_tie_count": 0,
            "utility_teacher_recall_correct": 0,
            "utility_dabe_recall_correct": 0,
            "positive_hist": [0] * int(num_bins),
            "negative_hist": [0] * int(num_bins),
            "calibration_count": [0] * int(num_bins),
            "calibration_score_sum": [0.0] * int(num_bins),
            "calibration_target_sum": [0.0] * int(num_bins),
        }
    score = torch.sigmoid(residual.detach())[pixel_mask]
    utility_gap = utility_target.utility_gap[pixel_mask]
    tie = utility_gap.abs() <= 1e-8
    target = utility_gap > 1e-8
    prediction = score >= 0.5
    positive = target & (~tie)
    negative = (utility_gap < -1e-8) & (~tie)
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
        "utility_teacher_count": int(positive.sum().item()),
        "utility_dabe_count": int(negative.sum().item()),
        "utility_tie_count": int(tie.sum().item()),
        "utility_teacher_recall_correct": int(prediction[positive].sum().item()),
        "utility_dabe_recall_correct": int((~prediction[negative]).sum().item()),
        "positive_hist": positive_hist.tolist(),
        "negative_hist": negative_hist.tolist(),
        "calibration_count": calibration_count.tolist(),
        "calibration_score_sum": calibration_score_sum.detach().cpu().tolist(),
        "calibration_target_sum": calibration_target_sum.detach().cpu().tolist(),
    }
