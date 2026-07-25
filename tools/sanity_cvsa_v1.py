#!/usr/bin/env python3
"""At-most-five-sample CVSA causal, numerical and gradient-isolation sanity."""

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import ensure_dir, load_config, read_jsonl, torch_load  # noqa: E402
from model import build_seg_head  # noqa: E402
from models.supervision.cvsa import (  # noqa: E402
    CVSAPatchRouter,
    align_hflip_view,
    build_cvsa_batch,
    build_cvsa_diagnostic_payload,
    build_route_target,
    compute_source_risk,
    cvsa_router_parameter_count,
    export_cvsa_visualization,
    validate_cvsa_cache_manifests,
)
from train import (  # noqa: E402
    build_cvsa_segmentation_group,
    build_optimizer_scheduler,
    extract_logits,
    forward_seg_head,
    make_hflip_image_68,
    make_hflip_model_input,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
    validate_cvsa_config,
)


DEFAULT_CONFIG = (
    "configs/dinov1_s8_dabepu_v11_cvsa_v1_hflip_router_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)


def _assert_unit(name, tensor):
    if not bool(torch.isfinite(tensor).all().item()):
        raise AssertionError(f"{name} contains NaN/Inf.")
    minimum = float(tensor.min().item())
    maximum = float(tensor.max().item())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise AssertionError(f"{name} range is [{minimum}, {maximum}].")


def _gradient_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().item())
    return total**0.5


def _manifest_provenance_checks(feature_root, fixed_root, max_samples):
    feature_rows = read_jsonl(Path(feature_root) / "manifest_train.jsonl")[
        :max_samples
    ]
    fixed_rows = read_jsonl(Path(fixed_root) / "manifest_train.jsonl")[:max_samples]
    if len(feature_rows) != max_samples or len(fixed_rows) != max_samples:
        raise AssertionError("CVSA sanity manifests do not contain the requested rows.")
    for feature_row, fixed_row in zip(feature_rows, fixed_rows):
        key = (feature_row["dataset"], feature_row["stem"])
        if key != (fixed_row["dataset"], fixed_row["stem"]):
            raise AssertionError("Feature/fixed manifest order differs.")
        if feature_row.get("generation_call_chain") != (
            "source_rgb->horizontal_flip->DINO_key"
        ):
            raise AssertionError(f"Feature call-chain mismatch for {key}.")
        if fixed_row.get("generation_call_chain") != (
            "source_rgb->horizontal_flip + "
            "independent_hflip_DINO->DABE-PU-v1.1"
        ):
            raise AssertionError(f"DABE call-chain mismatch for {key}.")
        for row in (feature_row, fixed_row):
            if row.get("independently_generated") is not True:
                raise AssertionError(f"Independent-generation marker missing for {key}.")
            if row.get("view") != "hflip":
                raise AssertionError(f"View marker mismatch for {key}.")
            if row.get("training_gt_used") is not False or "gt_path" in row:
                raise AssertionError(f"Training-GT provenance leak for {key}.")
        feature_payload = torch_load(feature_row["cache_path"], map_location="cpu")
        fixed_payload = torch_load(fixed_row["cache_path"], map_location="cpu")
        if feature_payload.get("generation_call_chain") != feature_row[
            "generation_call_chain"
        ]:
            raise AssertionError(f"Feature payload provenance mismatch for {key}.")
        if fixed_payload.get("generation_call_chain") != fixed_row[
            "generation_call_chain"
        ]:
            raise AssertionError(f"Fixed payload provenance mismatch for {key}.")
        if fixed_payload.get("source_feature_checksum") != feature_row["checksum"]:
            raise AssertionError(f"DABE feature provenance checksum mismatch for {key}.")
    return [f"{row['dataset']}/{row['stem']}" for row in feature_rows]


def _analytic_checks(device, config):
    lower = torch.full((1, 1, 37, 37), 0.2, device=device)
    higher = torch.full((1, 1, 37, 37), 0.8, device=device)
    valid = torch.ones(1, dtype=torch.bool, device=device)
    preference = build_route_target(
        fixed_risk_37=higher,
        teacher_risk_37=lower,
        fixed_valid=valid,
        teacher_valid=valid,
        route_temperature=config["route_temperature"],
    )["route_target_37"]
    if not bool((preference > 0.5).all().item()):
        raise AssertionError("Lower teacher risk must prefer teacher.")
    inverse = build_route_target(
        fixed_risk_37=lower,
        teacher_risk_37=higher,
        fixed_valid=valid,
        teacher_valid=valid,
        route_temperature=config["route_temperature"],
    )["route_target_37"]
    if not bool((inverse < 0.5).all().item()):
        raise AssertionError("Lower fixed risk must prefer fixed.")

    torch.manual_seed(20260722)
    feature_a = torch.randn(1, 384, 37, 37, device=device)
    feature_b = torch.randn_like(feature_a)
    source_valid = torch.full((1, 1, 37, 37), 0.5, device=device)
    source_zero = torch.zeros_like(source_valid)
    source_one = torch.ones_like(source_valid)
    kwargs = {
        "risk_eq_weight": config["risk_eq_weight"],
        "risk_sem_weight": config["risk_sem_weight"],
        "semantic_temperature": config["semantic_temperature"],
        "min_proto_mass_patches": config["min_proto_mass_patches"],
        "eps": config["eps"],
    }
    valid_risk = compute_source_risk(
        feature_a, feature_b, source_valid, source_valid, **kwargs
    )
    zero_risk = compute_source_risk(
        feature_a, feature_b, source_zero, source_zero, **kwargs
    )
    one_risk = compute_source_risk(
        feature_a, feature_b, source_one, source_one, **kwargs
    )
    if not bool(valid_risk["valid"].all().item()):
        raise AssertionError("Synthetic balanced source should be valid.")
    for name, risk in (("zero", zero_risk), ("one", one_risk)):
        if bool(risk["valid"].any().item()):
            raise AssertionError(f"Synthetic all-{name} source should be invalid.")
        if not torch.equal(risk["risk_total_37"], torch.ones_like(risk["risk_total_37"])):
            raise AssertionError(f"Invalid all-{name} risk must equal one.")
    fixed_only = build_route_target(
        valid_risk["risk_total_37"],
        zero_risk["risk_total_37"],
        valid_risk["valid"],
        zero_risk["valid"],
        config["route_temperature"],
    )
    teacher_only = build_route_target(
        zero_risk["risk_total_37"],
        valid_risk["risk_total_37"],
        zero_risk["valid"],
        valid_risk["valid"],
        config["route_temperature"],
    )
    both_invalid = build_route_target(
        zero_risk["risk_total_37"],
        one_risk["risk_total_37"],
        zero_risk["valid"],
        one_risk["valid"],
        config["route_temperature"],
    )
    if not torch.equal(fixed_only["route_target_37"], torch.zeros_like(lower)):
        raise AssertionError("Fixed-only valid source must route to fixed.")
    if not torch.equal(teacher_only["route_target_37"], torch.ones_like(lower)):
        raise AssertionError("Teacher-only valid source must route to teacher.")
    if not torch.equal(
        both_invalid["route_target_37"], torch.full_like(lower, 0.5)
    ) or not torch.equal(
        both_invalid["route_target_weight_37"], torch.zeros_like(lower)
    ):
        raise AssertionError("Both-invalid route target/weight contract failed.")
    return {
        "teacher_preference_mean": float(preference.mean().item()),
        "fixed_preference_mean": float(inverse.mean().item()),
        "invalid_zero_risk": float(zero_risk["risk_total_37"].mean().item()),
        "invalid_one_risk": float(one_risk["risk_total_37"].mean().item()),
        "both_invalid_weight": float(
            both_invalid["route_target_weight_37"].mean().item()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Run CVSA-v1 <=5-sample sanity.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--fixed_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= int(args.max_samples) <= 5:
        raise ValueError("--max_samples is strictly limited to 1..5.")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Sanity output directory is not empty: {output_dir}")
    ensure_dir(output_dir)

    cfg = load_config(args.config)
    cfg.CVSA = {
        **dict(cfg.CVSA),
        "feature_cache_hflip_root": str(Path(args.feature_root).resolve()),
        "fixed_cache_hflip_root": str(Path(args.fixed_root).resolve()),
    }
    validate_cvsa_config(cfg)
    validate_cvsa_cache_manifests(cfg, max_samples=args.max_samples)
    sample_keys = _manifest_provenance_checks(
        args.feature_root, args.fixed_root, args.max_samples
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = dict(cfg.CVSA)
    analytic = _analytic_checks(device, config)

    dataset = CachedTrainDataset(cfg, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.max_samples,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    feature_a = make_model_input(cfg, batch, device)
    feature_b = make_hflip_model_input(cfg, batch, device)
    if not torch.equal(align_hflip_view(align_hflip_view(feature_b)), feature_b):
        raise AssertionError("HFlip roundtrip failed for the real feature batch.")
    image_a = make_image_68(cfg, batch, device)
    image_b = make_hflip_image_68(cfg, batch, device)
    fixed_a = batch["pu_target_soft"].to(device).float()
    fixed_b = batch["cvsa_fixed_hflip_68"].to(device).float()

    student = build_seg_head(dataset.in_channels, cfg).to(device).eval()
    teacher = build_seg_head(dataset.in_channels, cfg).to(device).eval()
    teacher.load_state_dict(student.state_dict(), strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    set_model_epoch(student, 1)
    set_model_epoch(teacher, 1)
    with torch.no_grad():
        student_out = forward_seg_head(
            student, feature_a, cfg, image_68=image_a, return_aux=True
        )
        student_logits = resize_logits_for_loss(extract_logits(student_out), cfg)
        teacher_a_out = forward_seg_head(
            teacher, feature_a, cfg, image_68=image_a, return_aux=False
        )
        teacher_b_out = forward_seg_head(
            teacher, feature_b, cfg, image_68=image_b, return_aux=False
        )
        teacher_a = resize_logits_for_loss(
            extract_logits(teacher_a_out), cfg
        ).sigmoid()
        teacher_b = resize_logits_for_loss(
            extract_logits(teacher_b_out), cfg
        ).sigmoid()
        teacher_binary = (teacher_a >= 0.5).float()

    router = CVSAPatchRouter(
        feature_channels=dataset.in_channels,
        feature_dim=config["router_feature_dim"],
        hidden_dim=config["router_hidden_dim"],
        gn_groups=config["router_gn_groups"],
        init_teacher_prob=config["router_init_teacher_prob"],
        eps=config["eps"],
    ).to(device)
    result = build_cvsa_batch(
        config,
        feature_a,
        feature_b,
        fixed_a,
        fixed_b,
        teacher_a,
        teacher_b,
        teacher_binary,
        router=router,
    )
    init_gate_mean = float(result["router_gate_37"].mean().item())
    if abs(init_gate_mean - 0.05) > 1e-6:
        raise AssertionError(f"Initial router gate is {init_gate_mean}, expected 0.05.")
    for field in (
        "fixed_eq_risk_37",
        "teacher_eq_risk_37",
        "fixed_sem_risk_37",
        "teacher_sem_risk_37",
        "fixed_total_risk_37",
        "teacher_total_risk_37",
        "route_target_37",
        "route_target_weight_37",
        "router_gate_37",
        "mixed_target_68",
    ):
        _assert_unit(field, result[field])
    for field in (
        "route_target_37",
        "route_target_weight_37",
        "mixed_target_68",
    ):
        if result[field].requires_grad:
            raise AssertionError(f"{field} must be detached.")

    initial_state = copy.deepcopy(router.state_dict())
    route_target_before = result["route_target_37"].clone()
    gate_before = result["router_gate_37"].detach().clone()
    with torch.no_grad():
        router.router_out.bias.add_(1.0)
    changed = build_cvsa_batch(
        config,
        feature_a,
        feature_b,
        fixed_a,
        fixed_b,
        teacher_a,
        teacher_b,
        teacher_binary,
        router=router,
    )
    if not torch.equal(route_target_before, changed["route_target_37"]):
        raise AssertionError("Router action changed its own route target.")
    if torch.equal(gate_before, changed["router_gate_37"].detach()):
        raise AssertionError("Router perturbation did not change its gate.")
    router.load_state_dict(initial_state, strict=True)
    result = build_cvsa_batch(
        config,
        feature_a,
        feature_b,
        fixed_a,
        fixed_b,
        teacher_a,
        teacher_b,
        teacher_binary,
        router=router,
    )

    schedule_config = {
        **config,
        "alpha": 0.999,
        "teacher_weight": 0.001,
        "fixed_weight": 0.999,
        "epoch_ratio": 0.777,
        "post_reset_teacher_ratio": 0.888,
    }
    schedule_result = build_cvsa_batch(
        schedule_config,
        feature_a,
        feature_b,
        fixed_a,
        fixed_b,
        teacher_a,
        teacher_b,
        teacher_binary,
        router=router,
    )
    if not torch.equal(result["mixed_target_68"], schedule_result["mixed_target_68"]):
        raise AssertionError("Legacy schedule fields changed the CVSA mixed target.")
    forbidden_arguments = {
        "epoch",
        "history",
        "future_teacher",
        "global_alpha",
        "teacher_ratio",
    }
    signature_arguments = set(inspect.signature(build_cvsa_batch).parameters)
    if forbidden_arguments & signature_arguments:
        raise AssertionError("CVSA target builder exposes schedule/history inputs.")

    router.zero_grad(set_to_none=True)
    final_logits = student_logits.detach().clone().requires_grad_(True)
    coarse_logits = student_out["coarse_logits_68"].detach().clone().requires_grad_(True)
    base_logits = student_out["base_logits"].detach().clone().requires_grad_(True)
    seg_out = {
        "coarse_logits_68": coarse_logits,
        "base_logits": base_logits,
    }
    seg_group = build_cvsa_segmentation_group(
        cfg, 1, seg_out, final_logits, result["mixed_target_68"]
    )
    seg_group["loss"].backward()
    router_grad_from_seg = _gradient_norm(router.parameters())
    if router_grad_from_seg != 0.0:
        raise AssertionError("Segmentation loss leaked gradient into router.")
    router.zero_grad(set_to_none=True)
    result["loss_route"].backward()
    router_grad_from_route = _gradient_norm(router.parameters())
    if router_grad_from_route <= 0.0:
        raise AssertionError("Router loss produced no router gradient.")
    student_grad_from_route = _gradient_norm(student.parameters())
    teacher_grad_from_route = _gradient_norm(teacher.parameters())
    if student_grad_from_route != 0.0 or teacher_grad_from_route != 0.0:
        raise AssertionError("Router loss leaked gradient into student/teacher.")

    direct_config = {**config, "route_mode": "direct", "version": "cvsa_v1_hflip_direct"}
    direct = build_cvsa_batch(
        direct_config,
        feature_a,
        feature_b,
        fixed_a,
        fixed_b,
        teacher_a,
        teacher_b,
        teacher_binary,
        router=None,
    )
    if not torch.equal(direct["router_gate_37"], direct["route_target_37"]):
        raise AssertionError("Direct gate must equal route target.")
    if float(direct["loss_route"].item()) != 0.0:
        raise AssertionError("Direct route loss must be zero.")
    direct_optimizer, _ = build_optimizer_scheduler(cfg, student, cvsa_router=None)
    if len(direct_optimizer.param_groups) != 1:
        raise AssertionError("Direct mode must not add an optimizer parameter group.")

    payload = build_cvsa_diagnostic_payload(
        epoch=1,
        local_index=0,
        batch=batch,
        image_68=image_a,
        student_prob_68=student_logits.sigmoid(),
        result=result,
    )
    payload_path = output_dir / "cvsa_sanity_payload.pt"
    visual_path = output_dir / "cvsa_sanity_visual.png"
    torch.save(payload, payload_path)
    export_cvsa_visualization(payload, visual_path, gt=None)

    summary = {
        "status": "PASS",
        "max_samples": int(args.max_samples),
        "samples": sample_keys,
        "device": str(device),
        "training_started": False,
        "optimizer_step_called": False,
        "ema_update_called": False,
        "checkpoint_saved": False,
        "validation_run": False,
        "evaluation_run": False,
        "training_gt_used": False,
        "provenance_checked": True,
        "flip_roundtrip_exact": True,
        "risk_ranges_checked": True,
        "route_target_independent_of_router": True,
        "legacy_schedule_invariance": True,
        "future_teacher_used": False,
        "temporal_history_allocated": False,
        "router_initial_gate_mean": init_gate_mean,
        "router_parameter_count": cvsa_router_parameter_count(router),
        "router_grad_from_seg": router_grad_from_seg,
        "router_grad_from_route": router_grad_from_route,
        "student_grad_from_route": student_grad_from_route,
        "teacher_grad_from_route": teacher_grad_from_route,
        "direct_router_parameter_count": 0,
        "direct_route_loss": float(direct["loss_route"].item()),
        "direct_optimizer_param_groups": len(direct_optimizer.param_groups),
        "mixed_target_shared_by_final_coarse_base": True,
        "analytic": analytic,
        "visualization": str(visual_path.resolve()),
        "payload": str(payload_path.resolve()),
    }
    result_path = output_dir / "sanity_result.json"
    result_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
