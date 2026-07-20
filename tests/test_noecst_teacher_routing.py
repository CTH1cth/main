from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from common.static_weight import build_effective_static_weight
from common.teacher_routing import (
    LEGACY_TEACHER_ROUTING_MODE,
    build_identity_teacher_route,
    get_teacher_routing_mode,
    teacher_routing_protocol_fingerprint,
    teacher_routing_uses_ecst,
    validate_teacher_routing_config,
)
from common.utils import config_to_dict, load_config
from train import (
    get_dabe_pu_despl_schedule,
    teacher_route_bce_with_logits,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones.py"
)
NOECST_CONFIG = (
    ROOT
    / "configs"
    / "dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones_noecst.py"
)


def load_pair():
    return load_config(BASE_CONFIG), load_config(NOECST_CONFIG)


def test_resolved_configs_differ_only_in_protocol_fields():
    baseline, noecst = load_pair()
    left = config_to_dict(baseline)
    right = config_to_dict(noecst)
    differing = {
        key
        for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    }
    assert differing == {"EXP_NAME", "USE_ECST", "TEACHER_ROUTING_MODE"}
    assert teacher_routing_protocol_fingerprint(
        baseline
    ) == teacher_routing_protocol_fingerprint(noecst)


def test_explicit_modes_and_legacy_dispatch():
    baseline, noecst = load_pair()
    assert validate_teacher_routing_config(baseline) == "ecst"
    assert validate_teacher_routing_config(noecst) == "none"
    assert teacher_routing_uses_ecst(baseline)
    assert not teacher_routing_uses_ecst(noecst)
    legacy = SimpleNamespace(USE_ECST=False, USE_RAST=True)
    assert get_teacher_routing_mode(legacy) == LEGACY_TEACHER_ROUTING_MODE
    assert not teacher_routing_uses_ecst(legacy)


def test_noecst_rejects_other_teacher_router():
    _, noecst = load_pair()
    noecst.USE_RAST = True
    with pytest.raises(RuntimeError, match="other teacher routers"):
        validate_teacher_routing_config(noecst)


def test_identity_map_is_exact_detached_one():
    probability = torch.rand(3, 1, 68, 68, requires_grad=True)
    route_map, stats = build_identity_teacher_route(probability)
    assert route_map.shape == probability.shape
    assert route_map.dtype == torch.float32
    assert not route_map.requires_grad
    assert torch.equal(route_map, torch.ones_like(route_map))
    assert stats == {
        "routing_mode": "none",
        "teacher_map_min": 1.0,
        "teacher_map_mean": 1.0,
        "teacher_map_max": 1.0,
        "memory_active": False,
    }


@pytest.mark.parametrize("branch", ("final", "coarse", "base"))
def test_noecst_teacher_bce_matches_plain_bce_and_gradient(branch):
    del branch
    _, cfg = load_pair()
    generator = torch.Generator().manual_seed(3407)
    logits_plain = torch.randn(
        2, 1, 11, 13, generator=generator, dtype=torch.float64
    ).requires_grad_(True)
    logits_route = logits_plain.detach().clone().requires_grad_(True)
    target = (
        torch.rand(2, 1, 11, 13, generator=generator, dtype=torch.float64)
        >= 0.5
    ).to(torch.float64)
    route_map, _ = build_identity_teacher_route(target)

    plain = F.binary_cross_entropy_with_logits(
        logits_plain, target, reduction="mean"
    )
    routed = teacher_route_bce_with_logits(
        logits_route,
        target,
        route_map,
        cfg,
        routing_scale=0.0,
        apply_to_loss=True,
        eps=1e-6,
    )
    plain_gradient = torch.autograd.grad(plain, logits_plain)[0]
    routed_gradient = torch.autograd.grad(routed, logits_route)[0]
    assert abs(float(plain.item()) - float(routed.item())) < 1e-7
    assert torch.allclose(
        plain_gradient, routed_gradient, atol=1e-7, rtol=0.0
    )


def test_static_targets_maps_and_schedules_are_unchanged():
    baseline, noecst = load_pair()
    generator = torch.Generator().manual_seed(3407)
    raw_weight = torch.rand(2, 1, 68, 68, generator=generator)
    unknown = torch.rand(2, 1, 68, 68, generator=generator)
    target_soft = torch.rand(2, 1, 68, 68, generator=generator)
    batch = {"pu_unknown": unknown}

    baseline_weight, baseline_mode = build_effective_static_weight(
        baseline, batch, raw_weight, torch.device("cpu")
    )
    noecst_weight, noecst_mode = build_effective_static_weight(
        noecst, batch, raw_weight, torch.device("cpu")
    )
    assert baseline_mode == noecst_mode == "ones"
    assert torch.equal(baseline_weight, noecst_weight)
    assert torch.equal(baseline_weight, torch.ones_like(baseline_weight))
    assert torch.equal(target_soft, target_soft.clone())

    for epoch in (1, 7, 15, 20, 21, 45):
        assert get_dabe_pu_despl_schedule(
            epoch, baseline
        ) == get_dabe_pu_despl_schedule(epoch, noecst)
