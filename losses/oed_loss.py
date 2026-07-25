"""OED-v1 ordinal evidence distillation.

The module is deliberately training-only.  It ranks the original detached
DABE-PU soft target independently inside each image and samples O(N) cyclic
pairs.  No teacher signal, ground truth, hard threshold, or cross-image pair
is allowed to affect the OED objective.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
import torch.nn.functional as F


OED_VERSION = "oed_v1_ordinal_evidence_distillation"
OED_MODES = {"none", "oed", "persistent_fixed_bce"}
OED_METRIC_NAMES = (
    "oed_loss",
    "oed_weighted_loss",
    "oed_valid_pair_ratio",
    "oed_rank_gap_mean",
    "oed_rank_gap_p10",
    "oed_rank_gap_p50",
    "oed_rank_gap_p90",
    "oed_pair_weight_mean",
    "oed_pair_weight_p50",
    "oed_pair_weight_p90",
    "oed_order_acc_weighted",
    "oed_order_acc_all",
    "oed_order_acc_top25_gap",
    "oed_order_acc_top50_gap",
    "oed_final_logit_mean",
    "oed_final_logit_std",
    "oed_final_prob_mean",
    "oed_final_prob_std",
    "oed_prob_saturation_low_ratio",
    "oed_prob_saturation_high_ratio",
    "oed_fixed_soft_unique_ratio",
    "oed_fixed_soft_tie_pair_ratio",
    "oed_fixed_rank_entropy",
    "oed_spearman_mean",
    "oed_spearman_median",
    "oed_teacher_conflict_weighted",
    "oed_teacher_conflict_all",
    "oed_teacher_tie_ratio",
    "oed_logit_grad_norm",
    "baseline_logit_grad_norm",
    "oed_to_baseline_grad_ratio",
)


def _default_config() -> Dict[str, Any]:
    return {
        "version": OED_VERSION,
        "enabled": False,
        "mode": "none",
        "loss_weight": 0.0,
        "num_pair_rounds": 2,
        "logit_temperature": 1.0,
        "rank_gap_power": 1.0,
        "pair_gap_eps": 1e-8,
        "rank_tie_mode": "average",
        "use_soft_fixed_only": True,
        "apply_to": "final_logits",
        "deterministic_pairing": True,
        "seed": 20260722,
        "diagnostic_interval": 100,
        "log_spearman": True,
        "log_teacher_conflict": True,
        "log_gradient_ratio": True,
        "export_debug_vis": False,
    }


def validate_oed_config(cfg: Any) -> Dict[str, Any]:
    """Validate and return the unified auxiliary-evidence configuration.

    Configs without ``OED`` are the untouched baseline and resolve to mode
    ``none``.  Full OED, unweighted OED, and persistent fixed BCE share this
    one schema so their training integration cannot be enabled together.
    """

    raw = getattr(cfg, "OED", None)
    if raw is None:
        return _default_config()
    if not isinstance(raw, dict):
        raise RuntimeError("OED must be a dictionary when configured.")

    unknown = sorted(set(raw).difference(_default_config()))
    if unknown:
        raise RuntimeError(f"Unsupported OED configuration fields: {unknown}.")
    config = {**_default_config(), **dict(raw)}
    mode = str(config["mode"]).strip().lower()
    enabled = bool(config["enabled"])
    if mode not in OED_MODES:
        raise RuntimeError(f"Unsupported OED mode: {mode!r}; expected {sorted(OED_MODES)}.")
    if enabled != (mode != "none"):
        raise RuntimeError("OED.enabled must be true exactly when OED.mode is not 'none'.")
    if str(config["version"]) != OED_VERSION:
        raise RuntimeError(
            f"OED.version={config['version']!r} does not match {OED_VERSION!r}."
        )

    loss_weight = float(config["loss_weight"])
    if not math.isfinite(loss_weight) or loss_weight < 0.0:
        raise RuntimeError("OED.loss_weight must be finite and non-negative.")
    if enabled and abs(loss_weight - 0.05) > 1e-12:
        raise RuntimeError("OED-v1 and its requested controls require loss_weight=0.05.")
    if not enabled and loss_weight != 0.0:
        raise RuntimeError("OED mode 'none' requires loss_weight=0.")

    integer_fields = ("num_pair_rounds", "seed", "diagnostic_interval")
    for field in integer_fields:
        value = config[field]
        if isinstance(value, bool) or int(value) != value:
            raise RuntimeError(f"OED.{field} must be an integer, got {value!r}.")
    if int(config["num_pair_rounds"]) <= 0:
        raise RuntimeError("OED.num_pair_rounds must be positive.")
    if int(config["diagnostic_interval"]) <= 0:
        raise RuntimeError("OED.diagnostic_interval must be positive.")

    temperature = float(config["logit_temperature"])
    gap_power = float(config["rank_gap_power"])
    gap_eps = float(config["pair_gap_eps"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError("OED.logit_temperature must be finite and positive.")
    if not math.isfinite(gap_power) or gap_power not in {0.0, 1.0}:
        raise RuntimeError("OED.rank_gap_power must be 0.0 or 1.0.")
    if not math.isfinite(gap_eps) or gap_eps < 0.0:
        raise RuntimeError("OED.pair_gap_eps must be finite and non-negative.")
    if str(config["rank_tie_mode"]).lower() != "average":
        raise RuntimeError("OED-v1 requires rank_tie_mode='average'.")
    if config["use_soft_fixed_only"] is not True:
        raise RuntimeError("OED-v1 requires use_soft_fixed_only=True.")
    if str(config["apply_to"]).lower() != "final_logits":
        raise RuntimeError("OED-v1 applies only to Student final_logits.")
    if config["deterministic_pairing"] is not True:
        raise RuntimeError("OED-v1 requires deterministic_pairing=True.")
    for field in (
        "log_spearman",
        "log_teacher_conflict",
        "log_gradient_ratio",
        "export_debug_vis",
    ):
        if not isinstance(config[field], bool):
            raise RuntimeError(f"OED.{field} must be bool.")

    forbidden = {
        "start_epoch",
        "end_epoch",
        "warmup",
        "source_conflict_gate",
        "teacher_dependent_weight",
        "hard_pair_mining_threshold",
        "adaptive_lambda",
    }
    present = sorted(forbidden.intersection(raw))
    if present:
        raise RuntimeError(f"OED-v1 forbids schedule/gating fields: {present}.")

    config["mode"] = mode
    config["loss_weight"] = loss_weight
    config["num_pair_rounds"] = int(config["num_pair_rounds"])
    config["logit_temperature"] = temperature
    config["rank_gap_power"] = gap_power
    config["pair_gap_eps"] = gap_eps
    config["seed"] = int(config["seed"])
    config["diagnostic_interval"] = int(config["diagnostic_interval"])
    return config


def average_rank_1d(values: torch.Tensor) -> torch.Tensor:
    """Return detached tie-aware average ranks in ``[0, 1]`` for one image.

    Equal values receive exactly the same rank.  The normalization follows
    rank position divided by ``max(N - 1, 1)``; consequently ``[0,0,1,1]``
    maps to ``[1/6,1/6,5/6,5/6]``.
    """

    if values.ndim != 1:
        raise ValueError(f"average_rank_1d expects [N], got {list(values.shape)}.")
    if values.numel() == 0:
        raise ValueError("average_rank_1d requires at least one value.")
    detached = values.detach().float()
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError("average_rank_1d requires finite values.")
    sorted_values, sorted_indices = torch.sort(detached, stable=True)
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    group_start = torch.cumsum(counts, dim=0) - counts
    group_end = group_start + counts - 1
    average_position = (
        group_start.to(detached.dtype) + group_end.to(detached.dtype)
    ) * 0.5
    sorted_ranks = average_position[inverse]
    ranks = torch.empty_like(sorted_ranks)
    ranks.scatter_(0, sorted_indices, sorted_ranks)
    return (ranks / float(max(detached.numel() - 1, 1))).detach()


def average_rank_map(values: torch.Tensor) -> torch.Tensor:
    """Compute average ranks independently for every image in a batch."""

    canonical, had_channel = _canonical_image_tensor(values, "values")
    flat = canonical.detach().float().flatten(1)
    ranked = torch.stack([average_rank_1d(row) for row in flat], dim=0)
    ranked = ranked.reshape_as(canonical)
    return ranked if had_channel else ranked[:, 0]


def sample_cyclic_pairs(
    num_items: int,
    num_rounds: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample O(RN) non-self pairs using permutations and cyclic shifts."""

    num_items = int(num_items)
    num_rounds = int(num_rounds)
    if num_items < 2:
        raise ValueError("Cyclic OED pairing requires at least two items.")
    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive.")
    left: List[torch.Tensor] = []
    right: List[torch.Tensor] = []
    for _ in range(num_rounds):
        permutation = torch.randperm(
            num_items,
            generator=generator,
            device=device,
        )
        shift = int(
            torch.randint(
                1,
                num_items,
                (1,),
                generator=generator,
                device=device,
            ).item()
        )
        left.append(permutation)
        right.append(permutation.roll(shifts=-shift, dims=0))
    pair_i = torch.cat(left, dim=0)
    pair_j = torch.cat(right, dim=0)
    if bool((pair_i == pair_j).any().item()):
        raise RuntimeError("OED cyclic sampler unexpectedly produced a self-pair.")
    return pair_i, pair_j


def deterministic_pair_seed(
    base_seed: int,
    epoch: int,
    global_step: int,
    batch_index: int,
    sample_index: int,
    sample_id: str,
    distributed_rank: int = 0,
) -> int:
    """Build a stable local seed without touching any global RNG state."""

    digest = hashlib.sha256(str(sample_id).encode("utf-8")).digest()
    sample_hash = int.from_bytes(digest[:8], byteorder="little", signed=False)
    # Fixed, documented integer coefficients for OED-v1 reproducibility.
    seed = (
        int(base_seed)
        + 1_000_003 * int(epoch)
        + 97_003 * int(global_step)
        + 9_973 * int(batch_index)
        + 1_009 * int(sample_index)
        + 65_537 * int(distributed_rank)
        + sample_hash
    )
    return int(seed % (2**63 - 1))


def _canonical_image_tensor(
    value: torch.Tensor,
    name: str,
) -> Tuple[torch.Tensor, bool]:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor.")
    if value.ndim == 3:
        return value.unsqueeze(1), False
    if value.ndim == 4 and int(value.shape[1]) == 1:
        return value, True
    raise ValueError(f"{name} must be [B,H,W] or [B,1,H,W], got {list(value.shape)}.")


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.detach().float(), float(q)).item())


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    den = float(denominator.detach().float().sum().item())
    if den <= 0.0:
        return 0.0
    return float(numerator.detach().float().sum().item() / den)


def _pearson(first: torch.Tensor, second: torch.Tensor, eps: float = 1e-12) -> float:
    first = first.detach().float().flatten()
    second = second.detach().float().flatten()
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    if float(denominator.item()) <= eps:
        return 0.0
    return float((first * second).sum().div(denominator).item())


def _rank_entropy(row: torch.Tensor) -> float:
    """Normalized entropy of exact tie groups in one detached rank vector."""

    _, counts = torch.unique(row.detach(), return_counts=True)
    if row.numel() <= 1:
        return 0.0
    probability = counts.float() / float(row.numel())
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float((entropy / math.log(float(row.numel()))).item())


def _diagnostic_step(config: Mapping[str, Any], global_step: int) -> bool:
    return int(global_step) % int(config["diagnostic_interval"]) == 0


def build_aux_evidence_loss(
    *,
    student_final_logits: torch.Tensor,
    fixed_soft: torch.Tensor,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    batch_index: int,
    sample_indices: Sequence[int] | torch.Tensor,
    sample_ids: Sequence[str],
    teacher_probability: torch.Tensor | None = None,
    distributed_rank: int = 0,
    force_diagnostics: bool = False,
) -> Dict[str, Any]:
    """Build the configured auxiliary evidence loss and detached diagnostics."""

    mode = str(config["mode"]).lower()
    logits, _ = _canonical_image_tensor(student_final_logits, "student_final_logits")
    soft, _ = _canonical_image_tensor(fixed_soft, "fixed_soft")
    if logits.shape != soft.shape:
        raise ValueError(
            f"OED logits/soft shape mismatch: {list(logits.shape)} != {list(soft.shape)}."
        )
    if not bool(torch.isfinite(logits).all().item()) or not bool(
        torch.isfinite(soft).all().item()
    ):
        raise RuntimeError("OED inputs must be finite.")
    batch_size = int(logits.shape[0])
    if len(sample_ids) != batch_size:
        raise ValueError("sample_ids length must equal OED batch size.")
    if torch.is_tensor(sample_indices):
        sample_index_list = [int(v) for v in sample_indices.detach().cpu().tolist()]
    else:
        sample_index_list = [int(v) for v in sample_indices]
    if len(sample_index_list) != batch_size:
        raise ValueError("sample_indices length must equal OED batch size.")

    detached_soft = soft.detach().float()
    logits_float = logits.float()
    zero = logits_float.sum() * 0.0
    if mode == "none":
        return {
            "loss": zero,
            "weighted_loss": zero,
            "rank_map": torch.empty(0, device=logits.device),
            "pair_i": torch.empty((batch_size, 0), dtype=torch.long, device=logits.device),
            "pair_j": torch.empty((batch_size, 0), dtype=torch.long, device=logits.device),
            "metrics": {name: 0.0 for name in OED_METRIC_NAMES},
            "no_valid_pair_count": 0,
            "num_pairs": 0,
            "teacher_conflict_source": "none",
            "diagnostic_active": False,
        }

    probability = logits_float.sigmoid()
    diagnostic_active = bool(
        force_diagnostics or _diagnostic_step(config, global_step)
    )
    common_metrics = {
        "oed_final_logit_mean": float(logits_float.detach().mean().item()),
        "oed_final_logit_std": float(logits_float.detach().std(unbiased=False).item()),
        "oed_final_prob_mean": float(probability.detach().mean().item()),
        "oed_final_prob_std": float(probability.detach().std(unbiased=False).item()),
        "oed_prob_saturation_low_ratio": float((probability.detach() < 0.01).float().mean().item()),
        "oed_prob_saturation_high_ratio": float((probability.detach() > 0.99).float().mean().item()),
    }

    if mode == "persistent_fixed_bce":
        aux_loss = F.binary_cross_entropy_with_logits(logits_float, detached_soft)
        weighted_loss = float(config["loss_weight"]) * aux_loss
        metrics = {name: 0.0 for name in OED_METRIC_NAMES}
        metrics.update(common_metrics)
        metrics["oed_loss"] = float(aux_loss.detach().item())
        metrics["oed_weighted_loss"] = float(weighted_loss.detach().item())
        return {
            "loss": aux_loss,
            "weighted_loss": weighted_loss,
            "rank_map": torch.empty(0, device=logits.device),
            "pair_i": torch.empty((batch_size, 0), dtype=torch.long, device=logits.device),
            "pair_j": torch.empty((batch_size, 0), dtype=torch.long, device=logits.device),
            "metrics": metrics,
            "no_valid_pair_count": 0,
            "num_pairs": 0,
            "teacher_conflict_source": "none",
            "diagnostic_active": diagnostic_active,
        }

    num_items = int(detached_soft[0].numel())
    rank_flat = torch.stack(
        [average_rank_1d(row) for row in detached_soft.flatten(1)],
        dim=0,
    )
    pair_i_rows: List[torch.Tensor] = []
    pair_j_rows: List[torch.Tensor] = []
    for sample_offset in range(batch_size):
        local_seed = deterministic_pair_seed(
            int(config["seed"]),
            int(epoch),
            int(global_step),
            int(batch_index),
            sample_index_list[sample_offset],
            str(sample_ids[sample_offset]),
            int(distributed_rank),
        )
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(local_seed)
        pair_i, pair_j = sample_cyclic_pairs(
            num_items,
            int(config["num_pair_rounds"]),
            generator,
            logits.device,
        )
        pair_i_rows.append(pair_i)
        pair_j_rows.append(pair_j)
    pair_i_batch = torch.stack(pair_i_rows, dim=0)
    pair_j_batch = torch.stack(pair_j_rows, dim=0)

    logit_flat = logits_float.flatten(1)
    rank_i = torch.gather(rank_flat, 1, pair_i_batch)
    rank_j = torch.gather(rank_flat, 1, pair_j_batch)
    logit_i = torch.gather(logit_flat, 1, pair_i_batch)
    logit_j = torch.gather(logit_flat, 1, pair_j_batch)
    signed_rank_gap = (rank_i - rank_j).detach()
    rank_gap = signed_rank_gap.abs().detach()
    valid = rank_gap > float(config["pair_gap_eps"])
    direction = signed_rank_gap.sign().detach()
    if float(config["rank_gap_power"]) == 0.0:
        pair_weight = torch.ones_like(rank_gap)
    else:
        pair_weight = rank_gap.pow(float(config["rank_gap_power"]))
    pair_weight = (pair_weight * valid.float()).detach()
    pair_loss = pair_weight * F.softplus(
        -direction * (logit_i - logit_j) / float(config["logit_temperature"])
    )
    per_sample_denominator = pair_weight.sum(dim=1)
    valid_sample = per_sample_denominator > 0.0
    per_sample_loss = pair_loss.sum(dim=1) / per_sample_denominator.clamp_min(1e-12)
    if bool(valid_sample.any().item()):
        aux_loss = per_sample_loss[valid_sample].mean()
    else:
        aux_loss = zero
    weighted_loss = float(config["loss_weight"]) * aux_loss

    valid_gap = rank_gap[valid]
    valid_weight = pair_weight[valid]
    logit_gap = logit_i - logit_j
    correct = ((signed_rank_gap * logit_gap.detach()) > 0.0) & valid
    weighted_correct = pair_weight * correct.float()
    valid_float = valid.float()
    if valid_gap.numel() > 0:
        top25_threshold = torch.quantile(valid_gap, 0.75)
        top50_threshold = torch.quantile(valid_gap, 0.50)
        top25_mask = valid & (rank_gap >= top25_threshold)
        top50_mask = valid & (rank_gap >= top50_threshold)
    else:
        top25_mask = torch.zeros_like(valid)
        top50_mask = torch.zeros_like(valid)

    rounded_soft = torch.round(detached_soft.flatten(1) * 1_000_000.0) / 1_000_000.0
    unique_ratios = [
        float(torch.unique(row).numel()) / float(max(row.numel(), 1))
        for row in rounded_soft
    ]
    rank_entropies = [_rank_entropy(row) for row in rank_flat]
    metrics: Dict[str, float] = {name: 0.0 for name in OED_METRIC_NAMES}
    metrics.update(common_metrics)
    metrics.update(
        {
            "oed_loss": float(aux_loss.detach().item()),
            "oed_weighted_loss": float(weighted_loss.detach().item()),
            "oed_valid_pair_ratio": float(valid_float.mean().item()),
            "oed_rank_gap_mean": float(valid_gap.mean().item()) if valid_gap.numel() else 0.0,
            "oed_rank_gap_p10": _quantile(valid_gap, 0.10),
            "oed_rank_gap_p50": _quantile(valid_gap, 0.50),
            "oed_rank_gap_p90": _quantile(valid_gap, 0.90),
            "oed_pair_weight_mean": float(valid_weight.mean().item()) if valid_weight.numel() else 0.0,
            "oed_pair_weight_p50": _quantile(valid_weight, 0.50),
            "oed_pair_weight_p90": _quantile(valid_weight, 0.90),
            "oed_order_acc_weighted": _ratio(weighted_correct, pair_weight),
            "oed_order_acc_all": _ratio(correct.float(), valid_float),
            "oed_order_acc_top25_gap": _ratio((correct & top25_mask).float(), top25_mask.float()),
            "oed_order_acc_top50_gap": _ratio((correct & top50_mask).float(), top50_mask.float()),
            "oed_fixed_soft_unique_ratio": float(sum(unique_ratios) / max(len(unique_ratios), 1)),
            "oed_fixed_soft_tie_pair_ratio": float((~valid).float().mean().item()),
            "oed_fixed_rank_entropy": float(sum(rank_entropies) / max(len(rank_entropies), 1)),
        }
    )

    if diagnostic_active and bool(config["log_spearman"]):
        with torch.no_grad():
            student_rank = torch.stack(
                [average_rank_1d(row) for row in probability.detach().flatten(1)],
                dim=0,
            )
            correlations = torch.tensor(
                [_pearson(rank_flat[index], student_rank[index]) for index in range(batch_size)],
                dtype=torch.float32,
            )
            metrics["oed_spearman_mean"] = float(correlations.mean().item())
            metrics["oed_spearman_median"] = float(correlations.median().item())

    teacher_conflict_source = "none"
    if (
        diagnostic_active
        and bool(config["log_teacher_conflict"])
        and teacher_probability is not None
    ):
        teacher, _ = _canonical_image_tensor(teacher_probability, "teacher_probability")
        if teacher.shape != logits.shape:
            raise ValueError("Teacher probability shape must match OED final logits.")
        teacher_conflict_source = "raw_teacher_probability"
        with torch.no_grad():
            teacher_flat = teacher.detach().float().flatten(1)
            teacher_gap = torch.gather(teacher_flat, 1, pair_i_batch) - torch.gather(
                teacher_flat, 1, pair_j_batch
            )
            teacher_tie = teacher_gap.abs() <= float(config["pair_gap_eps"])
            conflict = (direction * teacher_gap < 0.0) & valid
            metrics["oed_teacher_conflict_weighted"] = _ratio(
                pair_weight * conflict.float(), pair_weight
            )
            metrics["oed_teacher_conflict_all"] = _ratio(
                conflict.float(), valid_float
            )
            metrics["oed_teacher_tie_ratio"] = _ratio(
                (teacher_tie & valid).float(), valid_float
            )

    return {
        "loss": aux_loss,
        "weighted_loss": weighted_loss,
        "rank_map": rank_flat.reshape_as(detached_soft).detach(),
        "pair_i": pair_i_batch.detach(),
        "pair_j": pair_j_batch.detach(),
        "valid_pair_mask": valid.detach(),
        "rank_gap": rank_gap.detach(),
        "pair_weight": pair_weight.detach(),
        "pair_violation": (valid & ~correct).detach(),
        "metrics": metrics,
        "no_valid_pair_count": int((~valid_sample).sum().item()),
        "num_pairs": int(pair_i_batch.numel()),
        "teacher_conflict_source": teacher_conflict_source,
        "diagnostic_active": diagnostic_active,
    }


def compute_logit_gradient_diagnostics(
    *,
    oed_loss: torch.Tensor,
    baseline_loss: torch.Tensor,
    student_final_logits: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Measure auxiliary and baseline final-logit gradients without side effects."""

    grad_oed = torch.autograd.grad(
        oed_loss,
        student_final_logits,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    grad_baseline = torch.autograd.grad(
        baseline_loss,
        student_final_logits,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )[0]
    oed_norm = 0.0 if grad_oed is None else float(grad_oed.detach().float().norm().item())
    baseline_norm = (
        0.0
        if grad_baseline is None
        else float(grad_baseline.detach().float().norm().item())
    )
    return {
        "oed_logit_grad_norm": oed_norm,
        "baseline_logit_grad_norm": baseline_norm,
        "oed_to_baseline_grad_ratio": oed_norm / (baseline_norm + float(eps)),
        "baseline_final_logit_grad_present": grad_baseline is not None,
    }


def new_oed_epoch_accumulator() -> Dict[str, Any]:
    """Create the light-weight epoch accumulator used by the training logger."""

    return {
        "metric_sums": {name: 0.0 for name in OED_METRIC_NAMES},
        "metric_counts": {name: 0 for name in OED_METRIC_NAMES},
        "oed_no_valid_pair_count": 0,
        "batches": 0,
    }


def update_oed_epoch_accumulator(
    accumulator: MutableMapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    """Accumulate one detached OED result without retaining its graph."""

    metrics = result["metrics"]
    for name in OED_METRIC_NAMES:
        if name not in metrics:
            continue
        value = float(metrics[name])
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite OED diagnostic: {name}={value}.")
        # Sparse diagnostics are only counted when they were actually run.
        if name in {
            "oed_spearman_mean",
            "oed_spearman_median",
            "oed_teacher_conflict_weighted",
            "oed_teacher_conflict_all",
            "oed_teacher_tie_ratio",
            "oed_logit_grad_norm",
            "baseline_logit_grad_norm",
            "oed_to_baseline_grad_ratio",
        } and not bool(result.get("diagnostic_active", False)):
            continue
        accumulator["metric_sums"][name] += value
        accumulator["metric_counts"][name] += 1
    accumulator["oed_no_valid_pair_count"] += int(result["no_valid_pair_count"])
    accumulator["batches"] += 1


def finalize_oed_epoch(accumulator: Mapping[str, Any]) -> Dict[str, float | int]:
    """Return epoch means for OED training-log emission."""

    row: Dict[str, float | int] = {}
    for name in OED_METRIC_NAMES:
        count = int(accumulator["metric_counts"][name])
        row[name] = (
            float(accumulator["metric_sums"][name]) / float(count)
            if count > 0
            else 0.0
        )
    row["oed_no_valid_pair_count"] = int(accumulator["oed_no_valid_pair_count"])
    row["oed_batches"] = int(accumulator["batches"])
    return row
