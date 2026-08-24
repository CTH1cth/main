"""Run deterministic LCIC mechanism and protocol checks without data or training."""

import json
from pathlib import Path

import torch

from common.dabev2hard_static_only import validate_dabev2hard_static_only_config
from common.r1hard_linear_pure_student import validate_gbsp_lcic_config
from common.utils import load_config
from model import SimpleConvSegHead, build_seg_head
from models.lcic import (
    LCICHead,
    build_local_dino_affinity,
    local_affinity_propagate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "a_linear": ROOT / "configs/dinov1_s8_gbsp_lcic_a_linear.py",
    "b_consensus": ROOT / "configs/dinov1_s8_gbsp_lcic_b_consensus.py",
    "c_innovation": ROOT / "configs/dinov1_s8_gbsp_lcic_c_innovation.py",
    "d_full": ROOT / "configs/dinov1_s8_gbsp_lcic_d_full.py",
    "e_dwlite": ROOT / "configs/dinov1_s8_gbsp_lcic_e_dwlite.py",
}


def main():
    torch.manual_seed(20260818)
    uniform_feature = torch.full((2, 384, 7, 7), 0.25)
    affinity, affinity_diagnostics = build_local_dino_affinity(
        uniform_feature, return_diagnostics=True
    )
    local_consensus_feature = local_affinity_propagate(
        affinity, uniform_feature
    )
    uniform_innovation_error = float(
        (uniform_feature - local_consensus_feature).abs().max().item()
    )

    constant_logits = torch.full((2, 1, 7, 7), 2.75)
    constant_consensus_error = float(
        (
            local_affinity_propagate(affinity, constant_logits)
            - constant_logits
        )
        .abs()
        .max()
        .item()
    )

    torch.manual_seed(3407)
    linear = SimpleConvSegHead(384)
    torch.manual_seed(3407)
    full = LCICHead(384, use_consensus=True, use_innovation=True)
    test_feature = torch.linspace(
        -1.0, 1.0, steps=2 * 384 * 7 * 7
    ).reshape(2, 384, 7, 7)
    initialization_error = float(
        (linear(test_feature) - full(test_feature)).abs().max().item()
    )

    gradient_feature = torch.randn(2, 384, 7, 7)
    gradient_model = LCICHead(384, use_consensus=True, use_innovation=True)
    gradient_affinity = build_local_dino_affinity(gradient_feature)
    gradient_model(gradient_feature).square().mean().backward()
    gradient_contract = {
        "anchor_weight_grad_not_none": gradient_model.anchor.weight.grad is not None,
        "innovation_weight_grad_not_none": (
            gradient_model.innovation_head.weight.grad is not None
        ),
        "alpha_grad_not_none": gradient_model.alpha.grad is not None,
        "beta_grad_not_none": gradient_model.beta.grad is not None,
        "dino_input_grad_is_none": gradient_feature.grad is None,
        "affinity_requires_grad": gradient_affinity.requires_grad,
    }

    variants = {}
    for variant, path in CONFIGS.items():
        cfg = load_config(path)
        audit = validate_gbsp_lcic_config(cfg)
        static_audit = validate_dabev2hard_static_only_config(cfg)
        model = build_seg_head(384, cfg)
        variants[variant] = {
            "exp_name": cfg.EXP_NAME,
            "head_type": cfg.HEAD_TYPE,
            "trainable_params": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "decoder_only_audit": audit["status"],
            "static_only_audit": static_audit["status"],
            "protected_protocol_fields_match": all(
                item["matches"]
                for item in audit["protected_protocol_fields"].values()
            ),
        }

    checks = {
        "affinity_row_sum": float(
            affinity_diagnostics["row_sum_max_abs_error"].item()
        )
        < 1e-5,
        "self_similarity": (
            float(affinity_diagnostics["self_similarity_min"].item()) == 1.0
            and float(affinity_diagnostics["self_similarity_max"].item()) == 1.0
        ),
        "uniform_region_innovation": uniform_innovation_error < 1e-5,
        "constant_logit_consensus": constant_consensus_error < 1e-5,
        "initialization_equivalence": initialization_error < 1e-6,
        "gradient_contract": (
            all(
                gradient_contract[name]
                for name in (
                    "anchor_weight_grad_not_none",
                    "innovation_weight_grad_not_none",
                    "alpha_grad_not_none",
                    "beta_grad_not_none",
                    "dino_input_grad_is_none",
                )
            )
            and not gradient_contract["affinity_requires_grad"]
        ),
        "all_config_audits": all(
            item["decoder_only_audit"] == "PASS"
            and item["static_only_audit"] == "PASS"
            and item["protected_protocol_fields_match"]
            for item in variants.values()
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "measurements": {
            "affinity_row_sum_max_abs_error": float(
                affinity_diagnostics["row_sum_max_abs_error"].item()
            ),
            "self_similarity_min": float(
                affinity_diagnostics["self_similarity_min"].item()
            ),
            "self_similarity_max": float(
                affinity_diagnostics["self_similarity_max"].item()
            ),
            "uniform_innovation_max_abs": uniform_innovation_error,
            "constant_consensus_max_abs": constant_consensus_error,
            "linear_vs_full_initialization_max_abs": initialization_error,
            "dense_affinity_materialized": affinity_diagnostics[
                "dense_affinity_materialized"
            ],
        },
        "gradient_contract": gradient_contract,
        "variants": variants,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
