import torch
import torch.nn.functional as F


def area_resize(tensor, size):
    if tensor.ndim != 4:
        raise RuntimeError(f"Area resize expects BCHW, got {list(tensor.shape)}.")
    return F.interpolate(tensor.float(), size=(int(size), int(size)), mode="area")


def bilinear_resize_gain(gain_37, size):
    if gain_37.ndim != 4 or int(gain_37.shape[1]) != 1:
        raise RuntimeError(f"PSSF gain must be [B,1,H,W], got {list(gain_37.shape)}.")
    return F.interpolate(
        gain_37.float(),
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)


def bilinear_resize_retention(retention_37, size):
    """Resize horizon-level retention without changing its semantics."""
    return bilinear_resize_gain(retention_37, size).detach()


def teacher_binary_observation(teacher_prob):
    if not bool(torch.isfinite(teacher_prob).all().item()):
        raise RuntimeError("Teacher probability contains NaN/Inf.")
    return (teacher_prob.detach() > 0.5).float()


def build_pssf_state_channels(
    p0_soft_37,
    q_prev_37,
    teacher_soft_37,
    teacher_binary_37,
    student_soft_37,
    temporal_mean_37,
    temporal_var_37,
):
    tensors = (
        p0_soft_37,
        q_prev_37,
        teacher_soft_37,
        teacher_binary_37,
        student_soft_37,
        (teacher_binary_37 - q_prev_37).abs(),
        (teacher_soft_37 - student_soft_37).abs(),
        temporal_mean_37,
        temporal_var_37,
    )
    reference_shape = tuple(tensors[0].shape)
    for tensor in tensors:
        if tuple(tensor.shape) != reference_shape:
            raise RuntimeError(
                f"PSSF state tensor shape mismatch: {list(tensor.shape)} "
                f"!= {list(reference_shape)}."
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError("PSSF state input contains NaN/Inf.")
    state = torch.cat([tensor.detach().float() for tensor in tensors], dim=1)
    if int(state.shape[1]) != 9:
        raise RuntimeError(f"PSSF state must have 9 channels, got {state.shape[1]}.")
    return state


def update_supervision_state(q_prev_68, teacher_binary_68, gain_68):
    if (
        tuple(q_prev_68.shape) != tuple(teacher_binary_68.shape)
        or tuple(q_prev_68.shape) != tuple(gain_68.shape)
    ):
        raise RuntimeError(
            "PSSF state update shape mismatch: "
            f"{list(q_prev_68.shape)}, {list(teacher_binary_68.shape)}, "
            f"{list(gain_68.shape)}."
        )
    q_current = q_prev_68.float() + gain_68.detach().float() * (
        teacher_binary_68.detach().float() - q_prev_68.float()
    )
    q_current = q_current.clamp(0.0, 1.0)
    if not bool(torch.isfinite(q_current).all().item()):
        raise RuntimeError("Updated PSSF supervision state contains NaN/Inf.")
    return q_current


def update_prior_anchored_supervision_state(
    q_prev_68,
    p0_soft_68,
    teacher_binary_68,
    retention_68,
    state_step,
):
    """Apply the PPSE-v2 prior-anchored inverse-horizon state update."""
    tensors = (
        q_prev_68,
        p0_soft_68,
        teacher_binary_68,
        retention_68,
    )
    reference_shape = tuple(q_prev_68.shape)
    if len(reference_shape) != 4 or reference_shape[1] != 1:
        raise RuntimeError(
            "PPSE-v2 state tensors must be [B,1,H,W], "
            f"got {list(reference_shape)}."
        )
    for tensor in tensors:
        if tuple(tensor.shape) != reference_shape:
            raise RuntimeError(
                "PPSE-v2 state update shape mismatch: "
                f"{list(tensor.shape)} != {list(reference_shape)}."
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError("PPSE-v2 state update contains NaN/Inf input.")

    state_step = float(state_step)
    if not 0.0 < state_step <= 1.0:
        raise RuntimeError(
            f"PPSE-v2 state_step must be in (0,1], got {state_step}."
        )
    q_prev = q_prev_68.detach().float()
    p0_soft = p0_soft_68.detach().float()
    teacher_binary = teacher_binary_68.detach().float()
    retention = retention_68.detach().float().clamp(0.0, 1.0)
    for name, tensor in (
        ("q_prev", q_prev),
        ("p0_soft", p0_soft),
        ("teacher_binary", teacher_binary),
        ("retention", retention),
    ):
        if tensor.numel() and (
            float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0
        ):
            raise RuntimeError(f"PPSE-v2 {name} is outside [0,1].")

    proposal = (
        (1.0 - retention) * p0_soft
        + retention * teacher_binary
    )
    state_memory_weight = 1.0 - state_step
    p0_write_weight = state_step * (1.0 - retention)
    teacher_write_weight = state_step * retention
    coefficient_sum = (
        torch.full_like(retention, state_memory_weight)
        + p0_write_weight
        + teacher_write_weight
    )
    coefficient_sum_error = (coefficient_sum - 1.0).abs()
    coefficient_sum_error_max = float(
        coefficient_sum_error.max().item()
    )
    if coefficient_sum_error_max >= 1e-6:
        raise RuntimeError(
            "PPSE-v2 state coefficients do not sum to one: "
            f"max_error={coefficient_sum_error_max:.9g}."
        )
    teacher_write_weight_max = float(teacher_write_weight.max().item())
    teacher_write_weight_limit = (
        0.333334
        if abs(state_step - 1.0 / 3.0) < 1e-8
        else state_step + 1e-6
    )
    if teacher_write_weight_max > teacher_write_weight_limit:
        raise RuntimeError(
            "PPSE-v2 teacher write weight exceeds the inverse-horizon "
            "limit: "
            f"{teacher_write_weight_max:.9g} > "
            f"{teacher_write_weight_limit:.9g}."
        )

    q_current = (
        state_memory_weight * q_prev
        + state_step * proposal
    ).clamp(0.0, 1.0)
    prior_correction = p0_write_weight * (p0_soft - q_prev)
    teacher_correction = teacher_write_weight * (
        teacher_binary - q_prev
    )
    for tensor in (
        proposal,
        p0_write_weight,
        teacher_write_weight,
        q_current,
        prior_correction,
        teacher_correction,
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError("PPSE-v2 state update produced NaN/Inf.")
    return {
        "q_current": q_current.detach(),
        "proposal": proposal.detach(),
        "retention": retention.detach(),
        "state_memory_weight": state_memory_weight,
        "p0_write_weight": p0_write_weight.detach(),
        "teacher_write_weight": teacher_write_weight.detach(),
        "coefficient_sum_error_max": coefficient_sum_error_max,
        "prior_correction": prior_correction.detach(),
        "teacher_correction": teacher_correction.detach(),
    }


def build_retention_target(q_prev_37, teacher_binary_37, future_binary_37, eps=1e-6):
    if future_binary_37.ndim != 5:
        raise RuntimeError(
            "PSSF future observations must be [H,B,1,37,37], got "
            f"{list(future_binary_37.shape)}."
        )
    if tuple(q_prev_37.shape) != tuple(teacher_binary_37.shape):
        raise RuntimeError("PSSF retention q/teacher shape mismatch.")
    if tuple(future_binary_37.shape[1:]) != tuple(q_prev_37.shape):
        raise RuntimeError("PSSF retention future/context shape mismatch.")
    eps = float(eps)
    future_center = future_binary_37.detach().float().mean(dim=0)
    innovation = teacher_binary_37.detach().float() - q_prev_37.detach().float()
    numerator = (future_center - q_prev_37.detach().float()) * innovation
    gain_unclipped = numerator / (innovation.square() + eps)
    gain_target = gain_unclipped.clamp(0.0, 1.0)
    innovation_weight = innovation.abs()
    for tensor in (future_center, innovation, gain_target, innovation_weight):
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError("PSSF retention target contains NaN/Inf.")
    return {
        "future_center": future_center,
        "innovation": innovation,
        "gain_unclipped": gain_unclipped,
        "gain_target": gain_target,
        "retention_unclipped": gain_unclipped,
        "retention_target": gain_target,
        "innovation_weight": innovation_weight,
    }


def weighted_gain_loss(prediction, target, weight, eps=1e-6):
    if tuple(prediction.shape) != tuple(target.shape) or tuple(target.shape) != tuple(
        weight.shape
    ):
        raise RuntimeError("PSSF weighted gain loss shape mismatch.")
    error = prediction.float() - target.detach().float()
    weight = weight.detach().float()
    denominator = weight.sum()
    loss = (weight * error.square()).sum() / (denominator + float(eps))
    weighted_mae = (weight * error.abs()).sum() / (denominator + float(eps))
    return loss, weighted_mae


def weighted_retention_loss(prediction, target, weight, eps=1e-6):
    return weighted_gain_loss(prediction, target, weight, eps=eps)


def innovation_weighted_image_mean(gain, weight, eps=1e-6):
    numerator = (gain.float() * weight.float()).flatten(1).sum(dim=1)
    denominator = weight.float().flatten(1).sum(dim=1)
    return numerator / (denominator + float(eps))


def future_state_from_gain(q_prev, teacher_binary, gain):
    return q_prev.float() + gain.float() * (
        teacher_binary.float() - q_prev.float()
    )


def future_state_from_retention(q_prev, teacher_binary, retention):
    """Predict the horizon center; intentionally does not apply 1 / h."""
    return future_state_from_gain(q_prev, teacher_binary, retention)


def safe_pearson(x, y, eps=1e-12):
    x = x.detach().float().flatten(1)
    y = y.detach().float().flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).sum(dim=1)
    denominator = torch.sqrt(x.square().sum(dim=1) * y.square().sum(dim=1))
    valid = denominator > float(eps)
    result = torch.zeros_like(numerator)
    result[valid] = numerator[valid] / denominator[valid]
    return result, valid


def safe_spearman(x, y, eps=1e-12):
    x = x.detach().float().flatten(1)
    y = y.detach().float().flatten(1)
    x_rank = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
    y_rank = torch.argsort(torch.argsort(y, dim=1), dim=1).float()
    return safe_pearson(x_rank, y_rank, eps=eps)
