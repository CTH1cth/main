from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from common.bitc import (
    BITC_GRID,
    build_bitc_retrieval_cache,
    build_bitc_teacher_map,
    build_grouped_counterfactuals,
    build_query_norm_substitute,
    checkerboard_group_masks,
    validate_bitc_config,
)


def _cache_batch(batch_size=2, channels=8, topk=4):
    generator = torch.Generator().manual_seed(17)
    feature = torch.randn(
        batch_size, channels, BITC_GRID, BITC_GRID, generator=generator
    )
    nodes = BITC_GRID * BITC_GRID
    anchors = torch.arange(0, nodes, 2)
    anchor_mask = torch.zeros(batch_size, nodes, dtype=torch.bool)
    anchor_mask[:, anchors] = True
    choices = anchors[torch.randint(0, anchors.numel(), (batch_size, nodes, topk), generator=generator)]
    weights = torch.rand(batch_size, nodes, topk, generator=generator)
    weights = weights / weights.sum(dim=2, keepdim=True)
    nearest = choices[:, :, 0]
    random_index = anchors[
        torch.randint(0, anchors.numel(), (batch_size, nodes), generator=generator)
    ]
    batch = {
        "bitc_background_anchor_mask": anchor_mask,
        "bitc_topk_indices": choices,
        "bitc_topk_weights": weights,
        "bitc_nearest_bg_index": nearest,
        "bitc_fixed_random_bg_index": random_index,
    }
    return feature, batch


def test_query_norm_is_preserved_for_all_intervention_modes():
    feature, batch = _cache_batch()
    original_norm = torch.linalg.vector_norm(feature.float(), dim=1)
    for mode in (
        "weighted_bg_query_norm",
        "random_bg_query_norm",
        "nearest_bg_query_norm",
        "identity_query_norm",
    ):
        substitute, stats = build_query_norm_substitute(feature, batch, mode)
        substitute_norm = torch.linalg.vector_norm(substitute.float(), dim=1)
        relative_error = (substitute_norm - original_norm).abs() / original_norm.clamp_min(1e-12)
        assert float(relative_error.max()) < 1e-4
        assert abs(stats["bitc_norm_ratio_mean"] - 1.0) < 1e-4
        assert torch.isfinite(substitute).all()
        assert not substitute.requires_grad


def test_dabe_retrieval_indices_weights_and_random_seed_are_valid():
    generator = torch.Generator().manual_seed(3)
    feature = torch.randn(8, BITC_GRID, BITC_GRID, generator=generator)
    rgb = torch.rand(3, BITC_GRID, BITC_GRID, generator=generator)
    anchor = torch.zeros(1, BITC_GRID, BITC_GRID)
    anchor[:, ::2, ::2] = 1.0
    params = {
        "K_RECON": 5,
        "TAU_RECON": 0.07,
        "SIGMA_COLOR_RECON": 0.05,
        "LAMBDA_COLOR_RECON": 0.20,
    }
    first = build_bitc_retrieval_cache(
        feature, rgb, anchor, params, seed=23, sample_key="TR-CAMO::x"
    )
    second = build_bitc_retrieval_cache(
        feature, rgb, anchor, params, seed=23, sample_key="TR-CAMO::x"
    )
    anchor_membership = torch.zeros(BITC_GRID * BITC_GRID, dtype=torch.bool)
    anchor_membership[first["background_anchor_indices"].long()] = True
    assert anchor_membership[first["topk_indices"].long()].all()
    assert anchor_membership[first["nearest_bg_index"].long()].all()
    assert anchor_membership[first["fixed_random_bg_index"].long()].all()
    assert torch.allclose(
        first["topk_weights"].float().sum(dim=1),
        torch.ones(BITC_GRID * BITC_GRID),
        atol=2e-3,
    )
    assert torch.equal(
        first["fixed_random_bg_index"], second["fixed_random_bg_index"]
    )


def test_checkerboard_groups_cover_once_and_identity_intervention_is_exact():
    feature, batch = _cache_batch(batch_size=1)
    masks = checkerboard_group_masks()
    assert torch.equal(masks.sum(dim=0), torch.ones_like(masks[0], dtype=torch.int64))
    assert not bool((masks[0] & masks[1]).any())
    identity, _ = build_query_norm_substitute(feature, batch, "identity_query_norm")
    grouped, grouped_masks = build_grouped_counterfactuals(feature, identity)
    assert torch.equal(grouped_masks, masks)
    assert torch.equal(grouped[0], feature[0])
    assert torch.equal(grouped[1], feature[0])

    teacher = nn.Conv2d(feature.shape[1], 1, kernel_size=1).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        original_logits = teacher(feature)
        counterfactual_logits = teacher(grouped).reshape(
            2, 1, 1, BITC_GRID, BITC_GRID
        )
    assert float((counterfactual_logits - original_logits.unsqueeze(0)).abs().max()) < 1e-4
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not feature.requires_grad


def test_centering_mad_direction_range_and_detach():
    batch_size = 1
    masks = checkerboard_group_masks()
    original = torch.zeros(batch_size, 1, BITC_GRID, BITC_GRID)
    difference = torch.empty(2, batch_size, 1, BITC_GRID, BITC_GRID)
    difference[0].fill_(2.0)
    difference[1].fill_(-3.0)
    teacher_binary_37 = torch.zeros(1, 1, BITC_GRID, BITC_GRID)
    teacher_binary_37[:, :, :, : BITC_GRID // 2] = 1.0
    signed_support = torch.where(
        teacher_binary_37 > 0.5,
        torch.ones_like(teacher_binary_37),
        -torch.ones_like(teacher_binary_37),
    )
    # Add directionally correct local responses only at intervened positions.
    difference[0] += masks[0].float() * signed_support
    difference[1] += masks[1].float() * signed_support
    counterfactual = (original.unsqueeze(0) - difference).reshape(
        2 * batch_size, 1, BITC_GRID, BITC_GRID
    )
    teacher_binary = F.interpolate(
        teacher_binary_37, size=(68, 68), mode="nearest"
    )
    result = build_bitc_teacher_map(
        original_coarse_logits_37=original,
        counterfactual_coarse_logits_37=counterfactual,
        group_masks=masks,
        teacher_binary_68=teacher_binary,
    )
    centered = result["response_centered_37"]
    assert torch.isfinite(centered).all()
    assert float(result["teacher_map_68"].min()) >= 0.2 - 1e-6
    assert float(result["teacher_map_68"].max()) <= 1.0 + 1e-6
    assert not result["teacher_map_68"].requires_grad
    # Positive response supports Teacher-FG, negative response supports Teacher-BG.
    assert result["stats"]["bitc_map_teacher_fg_mean"] > 0.6
    assert result["stats"]["bitc_map_teacher_bg_mean"] > 0.6


def test_constant_weighted_bce_equals_plain_bce_and_student_gets_gradient():
    logits = torch.randn(2, 1, 7, 7, requires_grad=True)
    target = torch.randint(0, 2, logits.shape).float()
    weight = torch.full_like(logits, 0.37)
    elementwise = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = (elementwise * weight).sum() / weight.sum()
    plain = F.binary_cross_entropy_with_logits(logits, target)
    assert torch.allclose(weighted, plain, atol=1e-7, rtol=1e-6)
    weighted.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert not weight.requires_grad


def test_bitc_config_disables_ecst_and_preserves_training_contract():
    values = {
        "USE_BITC": True,
        "BITC_VERSION": "v1",
        "BITC_SUBSTITUTION_MODE": "weighted_bg_query_norm",
        "BITC_CACHE_ROOT": "/tmp/not-read-by-validation",
        "BITC_NUM_GROUPS": 2,
        "BITC_GROUP_MODE": "checkerboard_2",
        "BITC_BATCH_INTERVENTIONS": True,
        "BITC_USE_COARSE_ONLY": True,
        "BITC_CENTER_MODE": "non_intervened_median",
        "BITC_RESPONSE_NORMALIZATION": "per_image_mad",
        "BITC_RESPONSE_CLIP": 6.0,
        "BITC_WEIGHT_FLOOR": 0.20,
        "BITC_APPLY_TO_FINAL": True,
        "BITC_APPLY_TO_COARSE": True,
        "BITC_APPLY_TO_BASE": True,
        "BITC_WEIGHTED_NORMALIZE": True,
        "BITC_DETACH_RESPONSE": True,
        "BITC_DETACH_MAP": True,
        "USE_ECST": False,
        "USE_TEPR_LITE": False,
        "TEACHER_ROUTING_MODE": "bitc_v1",
        "STATIC_WEIGHT_MODE": "ones",
        "DABE_PU_STATIC_SOURCE": "target_soft_68",
        "DABE_PU_STATIC_TARGET_MODE": "soft",
        "TEACHER_TARGET_MODE": "binary",
        "DABE_PU_VERSION": "pu_v11",
        "P_INIT_MODE": "dabe_pu_v11_desplsched",
        "TEACHER_FUSION_MODE": "dabe_pu_despl_sched",
        "USE_DABE_PU": True,
        "USE_DABE_PU_DESPL_SCHEDULE": True,
        "USE_DABE_PU_STATIC_LOSS": True,
        "USE_TEACHER_BINARY_FULL_LOSS": True,
        "USE_TEACHER_SOFT_FULL_LOSS": False,
        "HEAD_TYPE": "dagp_safe",
        "USE_DAGP_SAFE_HEAD": True,
        "USE_NDR_BRANCH": True,
        "USE_NDR_COARSE_AUX": True,
        "USE_BASE_AUX_LOSS": True,
        "MAX_EPOCH": 45,
        "DABE_PU_DESPL_TEACHER_ONLY_START": 21,
        "FINETUNE_RESET_EPOCH": 20,
        "FINETUNE_RESET_TIMING": "after_epoch",
        "LOSS_SIZE": 68,
    }
    assert validate_bitc_config(SimpleNamespace(**values))
