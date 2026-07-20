import math

import torch

from common.source_arbiter import (
    SignAwareSourceArbiter,
    compute_sign_aware_source_gates,
    source_arbiter_parameter_count,
)


def test_sign_aware_router_shape_bounds_and_parameter_count():
    router = SignAwareSourceArbiter().double()
    result = router(torch.randn(2, 18, 7, 9, dtype=torch.float64))
    assert source_arbiter_parameter_count(router) == 19282
    assert set(result) == {
        "raw_positive",
        "raw_negative",
        "residual_positive",
        "residual_negative",
    }
    for value in result.values():
        assert value.shape == (2, 1, 7, 9)
        assert torch.isfinite(value).all()
    assert result["raw_positive"].abs().max().item() == 0.0
    assert result["raw_negative"].abs().max().item() == 0.0
    assert result["residual_positive"].abs().max().item() <= 1.5
    assert result["residual_negative"].abs().max().item() <= 4.0


def test_sign_selection_and_complementarity():
    teacher_binary = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    teacher_prob = torch.tensor([[[[0.9, 0.1]]]], dtype=torch.float64)
    dabe_soft = torch.tensor([[[[0.1, 0.9]]]], dtype=torch.float64)
    router_output = {
        "residual_positive": torch.ones_like(teacher_binary),
        "residual_negative": torch.full_like(teacher_binary, -2.0),
    }
    gates = compute_sign_aware_source_gates(
        router_output=router_output,
        teacher_binary=teacher_binary,
        teacher_prior=0.5,
        teacher_prob=teacher_prob,
        dabe_soft_target=dabe_soft,
        influence_scale=1.0,
    )
    expected = torch.tensor(
        [[[
            [
                torch.sigmoid(torch.tensor(1.0, dtype=torch.float64)).item(),
                torch.sigmoid(torch.tensor(-2.0, dtype=torch.float64)).item(),
            ]
        ]]],
        dtype=torch.float64,
    )
    assert torch.max(torch.abs(gates["gate_teacher"] - expected)).item() <= 1e-12
    assert torch.max(
        torch.abs(gates["gate_dabe"] + gates["gate_teacher"] - 1.0)
    ).item() <= 1e-12

    bypass = compute_sign_aware_source_gates(
        router_output=router_output,
        teacher_binary=teacher_binary,
        teacher_prior=0.4,
        teacher_prob=teacher_prob,
        dabe_soft_target=dabe_soft,
        influence_scale=0.0,
    )
    assert math.isclose(
        float(bypass["gate_teacher"].mean()), 0.4, abs_tol=1e-12
    )
