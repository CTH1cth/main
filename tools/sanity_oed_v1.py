"""Bounded real-cache sanity audit for OED-v1.

This command reads at most five training samples and executes one real Student
and Teacher forward.  It may call backward for gradient isolation checks, but
never calls optimizer.step, never updates the real EMA teacher, never saves a
checkpoint, and never runs validation or evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
WORKDIR_ROOT = (PROJECT_ROOT / "workdir").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import config_to_dict, load_config  # noqa: E402
from losses.oed_loss import (  # noqa: E402
    OED_VERSION,
    average_rank_1d,
    build_aux_evidence_loss,
    compute_logit_gradient_diagnostics,
    validate_oed_config,
)
from model import build_seg_head, update_ema  # noqa: E402
from train import (  # noqa: E402
    build_optimizer_scheduler,
    complex_head_post_reset_lr,
    extract_logits,
    forward_seg_head,
    get_dabe_pu_despl_schedule,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
    validate_oed_baseline_contract,
    weighted_bce_with_logits,
)


BASELINE_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones_noecst.py"
)
FORBIDDEN_GT_FIELDS = {
    "gt",
    "gt_path",
    "mask",
    "mask_path",
    "ground_truth",
    "ground_truth_path",
}


def _resolve_inside(path_value: str, root: Path, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} must stay inside {root}, got {path}.") from error
    return path


def _prepare_output_dir(path_value: str) -> Path:
    path = _resolve_inside(path_value, WORKDIR_ROOT, "--out")
    if path == WORKDIR_ROOT:
        raise RuntimeError("--out must be a dedicated subdirectory of workdir.")
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)
        if any(path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    return path


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(value)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak_delta(device: torch.device, before: int) -> int | None:
    if device.type != "cuda":
        return None
    return max(0, int(torch.cuda.max_memory_allocated(device)) - int(before))


def _tensor_stats(value: torch.Tensor) -> Dict[str, Any]:
    tensor = value.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "requires_grad": bool(value.requires_grad),
    }


def _baseline_loss(
    cfg: Any,
    student_out: Mapping[str, torch.Tensor],
    student_logits: torch.Tensor,
    fixed_soft: torch.Tensor,
    teacher_probability: torch.Tensor,
    epoch: int,
) -> Dict[str, Any]:
    """Reproduce the protected no-ECST final/coarse/base loss group."""

    branch_logits = [student_logits]
    branch_names = ["final"]
    branch_weights = [1.0]
    if bool(cfg.USE_NDR_COARSE_AUX):
        branch_logits.append(resize_logits_for_loss(student_out["coarse_logits_68"], cfg))
        branch_names.append("coarse")
        branch_weights.append(float(cfg.LAMBDA_NDR_COARSE_AUX))
    if bool(cfg.USE_BASE_AUX_LOSS):
        branch_logits.append(resize_logits_for_loss(student_out["base_logits"], cfg))
        branch_names.append("base")
        branch_weights.append(
            float(cfg.LAMBDA_BASE_AUX)
            if int(epoch) < int(cfg.FINETUNE_RESET_EPOCH)
            else float(cfg.LAMBDA_BASE_AUX_AFTER_RESET)
        )
    eps = float(cfg.DABE_PU_WEIGHTED_BCE_EPS)
    ones = torch.ones_like(fixed_soft)
    teacher_binary = (teacher_probability.detach() >= 0.5).float()
    static_losses = [
        weighted_bce_with_logits(logits, fixed_soft.detach(), ones, eps=eps)
        for logits in branch_logits
    ]
    teacher_losses = [
        weighted_bce_with_logits(logits, teacher_binary, ones, eps=eps)
        for logits in branch_logits
    ]
    denominator = float(sum(branch_weights))
    static_group = sum(
        weight * value for weight, value in zip(branch_weights, static_losses)
    ) / denominator
    teacher_group = sum(
        weight * value for weight, value in zip(branch_weights, teacher_losses)
    ) / denominator
    static_weight, teacher_weight = get_dabe_pu_despl_schedule(epoch, cfg)
    loss = float(static_weight) * static_group + float(teacher_weight) * teacher_group
    return {
        "loss": loss,
        "static_group": static_group,
        "teacher_group": teacher_group,
        "static_weight": float(static_weight),
        "teacher_weight": float(teacher_weight),
        "branch_names": branch_names,
        "branch_weights": branch_weights,
        "teacher_binary": teacher_binary,
    }


def _config_difference_audit(cfg: Any) -> List[str]:
    baseline = config_to_dict(load_config(BASELINE_CONFIG))
    current = config_to_dict(cfg)
    keys = sorted(set(baseline).union(current))
    differences = [key for key in keys if baseline.get(key) != current.get(key)]
    if differences != ["EXP_NAME", "OED"]:
        raise RuntimeError(
            "OED config changed non-auxiliary baseline fields: "
            f"differences={differences}."
        )
    return differences


def _analytic_oed_audit(config: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """Run the required formula endpoints inside the single sanity command."""

    tie_values = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
    tie_ranks = average_rank_1d(tie_values)
    expected_ranks = torch.tensor(
        [1.0 / 6.0, 1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0],
        device=device,
    )
    rank_error = float((tie_ranks - expected_ranks).abs().max().item())
    transformed_error = float(
        (tie_ranks - average_rank_1d(tie_values.pow(3))).abs().max().item()
    )

    fixed = torch.tensor([[[[0.1, 0.3, 0.7, 0.9]]]], device=device)
    correct = torch.tensor(
        [[[[-2.0, -1.0, 1.0, 2.0]]]], device=device, requires_grad=True
    )
    reversed_logits = torch.tensor(
        [[[[2.0, 1.0, -1.0, -2.0]]]], device=device, requires_grad=True
    )

    def build(logits: torch.Tensor, *, teacher: torch.Tensor | None = None):
        return build_aux_evidence_loss(
            student_final_logits=logits,
            fixed_soft=fixed,
            config=config,
            epoch=3,
            global_step=17,
            batch_index=2,
            sample_indices=[11],
            sample_ids=["analytic/sample"],
            teacher_probability=teacher,
            force_diagnostics=True,
        )

    correct_result = build(correct, teacher=fixed)
    reversed_result = build(reversed_logits, teacher=1.0 - fixed)
    shifted_result = build(correct + 13.0, teacher=fixed)
    teacher_changed_result = build(correct, teacher=1.0 - fixed)

    all_tie_fixed = torch.full((1, 1, 2, 2), 0.5, device=device)
    all_tie_logits = torch.randn((1, 1, 2, 2), device=device, requires_grad=True)
    all_tie_result = build_aux_evidence_loss(
        student_final_logits=all_tie_logits,
        fixed_soft=all_tie_fixed,
        config=config,
        epoch=3,
        global_step=17,
        batch_index=2,
        sample_indices=[12],
        sample_ids=["analytic/all-tie"],
        force_diagnostics=True,
    )
    all_tie_grad = torch.autograd.grad(
        all_tie_result["loss"], all_tie_logits, retain_graph=False
    )[0]

    direction_fixed = torch.tensor([[[[0.0, 1.0]]]], device=device)
    direction_logits = torch.tensor(
        [[[[1.0, -1.0]]]], device=device, requires_grad=True
    )
    direction_result = build_aux_evidence_loss(
        student_final_logits=direction_logits,
        fixed_soft=direction_fixed,
        config=config,
        epoch=1,
        global_step=0,
        batch_index=0,
        sample_indices=[0],
        sample_ids=["analytic/direction"],
        force_diagnostics=True,
    )
    direction_grad = torch.autograd.grad(direction_result["loss"], direction_logits)[0]

    audit = {
        "tie_average_rank_max_abs_error": rank_error,
        "monotonic_transform_rank_max_abs_error": transformed_error,
        "correct_order_loss": float(correct_result["loss"].detach().item()),
        "reversed_order_loss": float(reversed_result["loss"].detach().item()),
        "correct_order_loss_is_lower": bool(
            correct_result["loss"].detach() < reversed_result["loss"].detach()
        ),
        "global_logit_shift_loss_max_abs_error": float(
            (correct_result["loss"] - shifted_result["loss"]).detach().abs().item()
        ),
        "teacher_change_loss_max_abs_error": float(
            (correct_result["loss"] - teacher_changed_result["loss"]).detach().abs().item()
        ),
        "teacher_change_pairs_unchanged": bool(
            torch.equal(correct_result["pair_i"], teacher_changed_result["pair_i"])
            and torch.equal(correct_result["pair_j"], teacher_changed_result["pair_j"])
        ),
        "all_tie_loss": float(all_tie_result["loss"].detach().item()),
        "all_tie_valid_pair_ratio": float(
            all_tie_result["metrics"]["oed_valid_pair_ratio"]
        ),
        "all_tie_gradient_norm": float(all_tie_grad.norm().item()),
        "violation_low_rank_gradient_positive": bool(direction_grad[0, 0, 0, 0] > 0),
        "violation_high_rank_gradient_negative": bool(direction_grad[0, 0, 0, 1] < 0),
        "pair_indices_reproducible": bool(
            torch.equal(correct_result["pair_i"], build(correct)["pair_i"])
            and torch.equal(correct_result["pair_j"], build(correct)["pair_j"])
        ),
    }
    required = (
        rank_error <= 1e-7
        and transformed_error <= 1e-7
        and audit["correct_order_loss_is_lower"]
        and audit["global_logit_shift_loss_max_abs_error"] <= 1e-6
        and audit["teacher_change_loss_max_abs_error"] == 0.0
        and audit["teacher_change_pairs_unchanged"]
        and audit["all_tie_loss"] == 0.0
        and audit["all_tie_valid_pair_ratio"] == 0.0
        and audit["all_tie_gradient_norm"] == 0.0
        and audit["violation_low_rank_gradient_positive"]
        and audit["violation_high_rank_gradient_negative"]
        and audit["pair_indices_reproducible"]
    )
    if not required:
        raise RuntimeError(f"Analytic OED endpoint audit failed: {audit}")
    return audit


def _ema_and_reset_smoke(cfg: Any) -> Dict[str, Any]:
    student = torch.nn.Linear(3, 2, bias=True)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        student.weight.add_(1.0)
    before = teacher.weight.detach().clone()
    update_ema(student, teacher, global_step=1, ema_weight=float(cfg.EMA_WEIGHT))
    ema_changed = not torch.equal(before, teacher.weight.detach())
    expected = 0.5 * before + 0.5 * student.weight.detach()
    ema_error = float((teacher.weight.detach() - expected).abs().max().item())

    optimizer, scheduler = build_optimizer_scheduler(cfg, student)
    reset_optimizer, reset_scheduler = build_optimizer_scheduler(
        cfg,
        student,
        lr=complex_head_post_reset_lr(cfg),
    )
    result = {
        "ema_utility_unchanged": True,
        "ema_changed_teacher": ema_changed,
        "ema_formula_max_abs_error": ema_error,
        "optimizer_type": type(optimizer).__name__,
        "scheduler_type": type(scheduler).__name__,
        "reset_optimizer_type": type(reset_optimizer).__name__,
        "reset_scheduler_type": type(reset_scheduler).__name__,
        "reset_epoch": int(cfg.FINETUNE_RESET_EPOCH),
        "reset_timing": str(cfg.FINETUNE_RESET_TIMING),
        "reset_lr": float(reset_optimizer.param_groups[0]["lr"]),
        "reset_rebuild_optimizer": bool(cfg.FINETUNE_RESET_REBUILD_OPTIMIZER),
        "reset_rebuild_scheduler": bool(cfg.FINETUNE_RESET_REBUILD_SCHEDULER),
    }
    if not ema_changed or ema_error > 1e-7:
        raise RuntimeError(f"EMA utility smoke failed: {result}")
    if result["optimizer_type"] != "AdamW" or result["scheduler_type"] != "StepLR":
        raise RuntimeError(f"Optimizer/scheduler smoke failed: {result}")
    return result


def _violation_map(result: Mapping[str, Any], sample_index: int, height: int, width: int) -> torch.Tensor:
    pair_i = result["pair_i"][sample_index]
    pair_j = result["pair_j"][sample_index]
    violation = result["pair_violation"][sample_index].float()
    count = torch.zeros(height * width, device=pair_i.device)
    total = torch.zeros(height * width, device=pair_i.device)
    count.scatter_add_(0, pair_i, violation)
    count.scatter_add_(0, pair_j, violation)
    total.scatter_add_(0, pair_i, torch.ones_like(violation))
    total.scatter_add_(0, pair_j, torch.ones_like(violation))
    return (count / total.clamp_min(1.0)).reshape(height, width).detach().cpu()


def _render_debug(
    out_path: Path,
    local_index: int,
    batch: Mapping[str, Any],
    student_probability: torch.Tensor,
    teacher_probability: torch.Tensor,
    oed_result: Mapping[str, Any],
) -> None:
    rgb = batch["image_68"][local_index].permute(1, 2, 0).float().clamp(0.0, 1.0)
    soft = batch["pu_target_soft"][local_index, 0].float()
    rank = oed_result["rank_map"][local_index, 0].float().cpu()
    student = student_probability[local_index, 0].detach().float().cpu()
    teacher = teacher_probability[local_index, 0].detach().float().cpu()
    violation = _violation_map(
        oed_result,
        local_index,
        int(soft.shape[-2]),
        int(soft.shape[-1]),
    )
    metrics = oed_result["metrics"]

    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    panels = (
        (rgb, "RGB", None),
        (soft, "DABE-PU soft", "viridis"),
        (rank, "DABE-PU average rank", "viridis"),
        (student, "Student final probability", "viridis"),
        (violation, "Sampled pair violation map", "magma"),
        (teacher, "Raw Teacher probability", "viridis"),
    )
    for axis, (image, title, cmap) in zip(axes.flat[:6], panels):
        axis.imshow(image.numpy(), cmap=cmap, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    axes.flat[6].axis("off")
    axes.flat[6].set_title("DABE–Teacher order conflict")
    axes.flat[6].text(
        0.02,
        0.92,
        "diagnostic only\n"
        f"weighted: {metrics['oed_teacher_conflict_weighted']:.4f}\n"
        f"all: {metrics['oed_teacher_conflict_all']:.4f}\n"
        f"teacher tie: {metrics['oed_teacher_tie_ratio']:.4f}",
        va="top",
        family="monospace",
    )
    axes.flat[7].axis("off")
    axes.flat[7].set_title("OED sample summary")
    axes.flat[7].text(
        0.02,
        0.92,
        f"{batch['dataset'][local_index]}/{batch['stem'][local_index]}\n"
        f"valid pairs: {metrics['oed_valid_pair_ratio']:.4f}\n"
        f"rank gap: {metrics['oed_rank_gap_mean']:.4f}\n"
        f"order acc: {metrics['oed_order_acc_weighted']:.4f}\n"
        f"OED loss: {metrics['oed_loss']:.6f}",
        va="top",
        family="monospace",
    )
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def _run(config_path: Path, out_dir: Path, max_samples: int, device: torch.device) -> Dict[str, Any]:
    cfg = load_config(config_path)
    oed_config = validate_oed_config(cfg)
    validate_oed_baseline_contract(cfg, oed_config)
    if str(oed_config["version"]) != OED_VERSION or str(oed_config["mode"]) != "oed":
        raise RuntimeError("sanity_oed_v1.py accepts the full OED-v1 config only.")
    config_differences = _config_difference_audit(cfg)
    analytic_audit = _analytic_oed_audit(oed_config, device)

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    dataset = CachedTrainDataset(cfg, max_samples=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=min(max_samples, len(dataset)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    leaked_gt = sorted(FORBIDDEN_GT_FIELDS.intersection(batch))
    if leaked_gt:
        raise RuntimeError(f"Sanity training batch leaked GT fields: {leaked_gt}.")
    batch_size = int(batch["feature"].shape[0])
    if not 1 <= batch_size <= max_samples <= 5:
        raise RuntimeError(f"Invalid bounded sanity batch size: {batch_size}.")

    student = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher = copy.deepcopy(student).to(device)
    student.train()
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    set_model_epoch(student, 1)
    set_model_epoch(teacher, 1)

    model_input = make_model_input(cfg, batch, device)
    image_68 = make_image_68(cfg, batch, device)
    student_out = forward_seg_head(
        student,
        model_input,
        cfg,
        image_68=image_68,
        return_aux=True,
    )
    student_logits = resize_logits_for_loss(extract_logits(student_out), cfg)
    with torch.no_grad():
        teacher_out = forward_seg_head(
            teacher,
            model_input,
            cfg,
            image_68=image_68,
            return_aux=False,
        )
        teacher_probability = resize_logits_for_loss(
            extract_logits(teacher_out), cfg
        ).sigmoid()
    fixed_soft = batch["pu_target_soft"].to(device).float()
    if tuple(fixed_soft.shape) != tuple(student_logits.shape):
        raise RuntimeError("Real fixed soft map and final logits shapes differ.")
    if not bool(((fixed_soft > 0.0) & (fixed_soft < 1.0)).any().item()):
        raise RuntimeError("DABE-PU soft source appears to be binary.")

    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory_before = int(torch.cuda.memory_allocated(device))
    else:
        baseline_memory_before = 0
    baseline_start = time.perf_counter()
    baseline = _baseline_loss(
        cfg,
        student_out,
        student_logits,
        fixed_soft,
        teacher_probability,
        epoch=1,
    )
    _synchronize(device)
    baseline_seconds = time.perf_counter() - baseline_start
    baseline_peak_delta = _cuda_peak_delta(device, baseline_memory_before)

    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if device.type == "cuda"
        else []
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        oed_memory_before = int(torch.cuda.memory_allocated(device))
    else:
        oed_memory_before = 0
    oed_start = time.perf_counter()
    oed_result = build_aux_evidence_loss(
        student_final_logits=student_logits,
        fixed_soft=fixed_soft,
        config=oed_config,
        epoch=1,
        global_step=0,
        batch_index=0,
        sample_indices=batch["sample_index"],
        sample_ids=[
            f"{dataset_name}/{stem}"
            for dataset_name, stem in zip(batch["dataset"], batch["stem"])
        ],
        teacher_probability=teacher_probability,
        force_diagnostics=True,
    )
    _synchronize(device)
    oed_seconds = time.perf_counter() - oed_start
    oed_peak_delta = _cuda_peak_delta(device, oed_memory_before)
    cpu_rng_unchanged = torch.equal(cpu_rng_before, torch.random.get_rng_state())
    cuda_rng_after = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    cuda_rng_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(cuda_rng_before, cuda_rng_after)
    )
    if not cpu_rng_unchanged or not cuda_rng_unchanged:
        raise RuntimeError("OED changed a global torch RNG state.")

    repeated = build_aux_evidence_loss(
        student_final_logits=student_logits,
        fixed_soft=fixed_soft,
        config=oed_config,
        epoch=1,
        global_step=0,
        batch_index=0,
        sample_indices=batch["sample_index"],
        sample_ids=[
            f"{dataset_name}/{stem}"
            for dataset_name, stem in zip(batch["dataset"], batch["stem"])
        ],
        teacher_probability=teacher_probability,
        force_diagnostics=False,
    )
    if not torch.equal(oed_result["pair_i"], repeated["pair_i"]) or not torch.equal(
        oed_result["pair_j"], repeated["pair_j"]
    ):
        raise RuntimeError("OED local deterministic pairing is not reproducible.")

    if bool((oed_result["pair_i"] == oed_result["pair_j"]).any().item()):
        raise RuntimeError("OED sanity found a self-pair.")
    expected_valid = oed_result["rank_gap"] > float(oed_config["pair_gap_eps"])
    if not torch.equal(expected_valid, oed_result["valid_pair_mask"]):
        raise RuntimeError("OED tie exclusion does not match pair_gap_eps.")

    gradient_metrics = compute_logit_gradient_diagnostics(
        oed_loss=oed_result["loss"],
        baseline_loss=baseline["loss"],
        student_final_logits=student_logits,
    )
    oed_result["metrics"].update(
        {key: value for key, value in gradient_metrics.items() if isinstance(value, float)}
    )
    total_loss = baseline["loss"] + oed_result["weighted_loss"]

    none_config = dict(oed_config)
    none_config.update(enabled=False, mode="none", loss_weight=0.0)
    disabled_result = build_aux_evidence_loss(
        student_final_logits=student_logits,
        fixed_soft=fixed_soft,
        config=none_config,
        epoch=1,
        global_step=0,
        batch_index=0,
        sample_indices=batch["sample_index"],
        sample_ids=[str(value) for value in batch["stem"]],
    )
    disabled_total = baseline["loss"] + disabled_result["weighted_loss"]
    baseline_exact_when_disabled = bool(
        torch.equal(baseline["loss"].detach(), disabled_total.detach())
    )
    if not baseline_exact_when_disabled:
        raise RuntimeError("Disabling OED did not exactly recover baseline loss.")
    if not bool(torch.isfinite(total_loss).item()):
        raise RuntimeError("OED sanity total loss is not finite.")

    student.zero_grad(set_to_none=True)
    total_loss.backward()
    student_has_grad = any(
        parameter.grad is not None and bool((parameter.grad != 0).any().item())
        for parameter in student.parameters()
    )
    teacher_has_oed_grad = any(parameter.grad is not None for parameter in teacher.parameters())
    if not student_has_grad or teacher_has_oed_grad:
        raise RuntimeError("OED gradient isolation sanity failed.")
    if float(gradient_metrics["oed_logit_grad_norm"]) <= 0.0:
        raise RuntimeError("OED produced zero final-logit gradient on the real batch.")

    rank_flat = oed_result["rank_map"].flatten(1)
    soft_unique_ratios = [
        float(torch.unique(torch.round(row * 1_000_000.0) / 1_000_000.0).numel())
        / float(row.numel())
        for row in fixed_soft.detach().flatten(1)
    ]
    png_paths = []
    for local_index in range(batch_size):
        path = out_dir / (
            f"{local_index:02d}_{batch['dataset'][local_index]}__"
            f"{batch['stem'][local_index]}.png"
        )
        _render_debug(
            path,
            local_index,
            batch,
            student_logits.sigmoid(),
            teacher_probability,
            oed_result,
        )
        png_paths.append(str(path))

    payload_path = out_dir / "oed_diagnostic_payload.pt"
    payload = {
        "schema_version": "oed_v1_sanity_payload_v1",
        "fixed_soft": fixed_soft.detach().cpu(),
        "rank_map": oed_result["rank_map"].detach().cpu(),
        "student_probability": student_logits.sigmoid().detach().cpu(),
        "teacher_probability": teacher_probability.detach().cpu(),
        "pair_i": oed_result["pair_i"].detach().cpu(),
        "pair_j": oed_result["pair_j"].detach().cpu(),
        "pair_violation": oed_result["pair_violation"].detach().cpu(),
        "metrics": dict(oed_result["metrics"]),
    }
    torch.save(payload, payload_path)

    ema_reset = _ema_and_reset_smoke(cfg)
    summary = {
        "status": "ok",
        "schema_version": "oed_v1_sanity_v1",
        "config": str(config_path),
        "baseline_config": str(BASELINE_CONFIG),
        "device": str(device),
        "max_samples": int(max_samples),
        "batch_size": batch_size,
        "samples": [
            {
                "sample_index": int(batch["sample_index"][index]),
                "dataset": str(batch["dataset"][index]),
                "stem": str(batch["stem"][index]),
            }
            for index in range(batch_size)
        ],
        "source_and_shape": {
            "fixed_soft_source": "batch[pu_target_soft] from DABE-PU target_soft_68",
            "fixed_soft_shape": list(fixed_soft.shape),
            "fixed_soft_min": float(fixed_soft.min().item()),
            "fixed_soft_max": float(fixed_soft.max().item()),
            "fixed_soft_mean": float(fixed_soft.mean().item()),
            "fixed_soft_unique_ratio": float(sum(soft_unique_ratios) / len(soft_unique_ratios)),
            "fixed_soft_is_nonbinary": True,
            "fixed_soft_rank_min": float(rank_flat.min().item()),
            "fixed_soft_rank_max": float(rank_flat.max().item()),
            "fixed_soft_tie_pair_ratio": float(oed_result["metrics"]["oed_fixed_soft_tie_pair_ratio"]),
            "final_logits_variable": "student_logits",
            "final_logits_shape": list(student_logits.shape),
        },
        "loss_and_pairs": {
            "oed_num_pairs": int(oed_result["num_pairs"]),
            "oed_valid_pair_ratio": float(oed_result["metrics"]["oed_valid_pair_ratio"]),
            "oed_rank_gap_mean": float(oed_result["metrics"]["oed_rank_gap_mean"]),
            "oed_loss": float(oed_result["loss"].detach().item()),
            "oed_weighted_loss": float(oed_result["weighted_loss"].detach().item()),
            "baseline_loss": float(baseline["loss"].detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "baseline_static_weight": float(baseline["static_weight"]),
            "baseline_teacher_weight": float(baseline["teacher_weight"]),
            "baseline_branches": list(baseline["branch_names"]),
            "baseline_branch_weights": list(baseline["branch_weights"]),
            "no_self_pairs": True,
            "ties_excluded": True,
            "pair_reproducible": True,
        },
        "gradient_isolation": {
            **gradient_metrics,
            "student_has_grad": student_has_grad,
            "teacher_has_oed_grad": teacher_has_oed_grad,
            "dino_has_grad": bool(getattr(model_input, "grad", None) is not None),
            "fixed_soft_has_grad": bool(fixed_soft.grad is not None),
            "rank_requires_grad": bool(oed_result["rank_map"].requires_grad),
            "pair_weight_requires_grad": bool(oed_result["pair_weight"].requires_grad),
        },
        "baseline_and_runtime_contract": {
            "config_differences_vs_baseline": config_differences,
            "baseline_loss_exact_when_oed_disabled": baseline_exact_when_disabled,
            "global_cpu_rng_unchanged_by_oed": cpu_rng_unchanged,
            "global_cuda_rng_unchanged_by_oed": cuda_rng_unchanged,
            "inference_branch_added": False,
            "model_config_changed": False,
            "training_gt_used": False,
            "optimizer_step_called": False,
            "real_teacher_ema_update_called": False,
            "checkpoint_loaded": False,
            "checkpoint_written": False,
            "validation_called": False,
            "evaluation_called": False,
            **ema_reset,
        },
        "analytic_oed_audit": analytic_audit,
        "diagnostics": dict(oed_result["metrics"]),
        "runtime": {
            "baseline_loss_seconds": baseline_seconds,
            "oed_loss_seconds": oed_seconds,
            "oed_to_baseline_loss_time_ratio": oed_seconds / max(baseline_seconds, 1e-12),
            "baseline_incremental_peak_memory_bytes": baseline_peak_delta,
            "oed_incremental_peak_memory_bytes": oed_peak_delta,
            "memory_measurement": "cuda_peak_allocated" if device.type == "cuda" else "unavailable_on_cpu",
        },
        "outputs": {
            "summary_json": str(out_dir / "sanity_oed_v1.json"),
            "diagnostic_payload": str(payload_path),
            "pngs": png_paths,
        },
        "interpretation": "Sanity validates implementation mechanics only; it does not establish OED effectiveness.",
    }
    summary_path = out_dir / "sanity_oed_v1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one bounded real-cache OED-v1 sanity batch (never full training/eval)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 1 <= int(args.max_samples) <= 5:
        parser.error("--max_samples must be in [1,5].")
    config_path = _resolve_inside(args.config, PROJECT_ROOT, "--config")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    out_dir = _prepare_output_dir(args.out)
    summary = _run(
        config_path,
        out_dir,
        int(args.max_samples),
        _device(args.device),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
