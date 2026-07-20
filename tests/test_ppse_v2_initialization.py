import math

import torch

from common.pssf_retention import (
    bilinear_resize_retention,
    update_prior_anchored_supervision_state,
)
from models.pssf import PredictiveSupervisionStateFilter


def test_ppse_v2_retention_head_initializes_to_point_zero_three():
    model = PredictiveSupervisionStateFilter(
        feature_channels=4,
        feature_proj_dim=4,
        state_channels=9,
        hidden_dim=8,
        gn_groups=4,
        init_gain=0.03,
        output_semantics="horizon_innovation_retention",
    )
    expected_bias = math.log(0.03 / 0.97)
    assert torch.equal(
        model.final_conv.weight,
        torch.zeros_like(model.final_conv.weight),
    )
    assert torch.allclose(
        model.final_conv.bias,
        torch.full_like(model.final_conv.bias, expected_bias),
        atol=1e-7,
    )

    output = model(
        torch.randn(2, 4, 3, 3),
        torch.randn(2, 9, 3, 3),
    )
    assert torch.allclose(output, torch.full_like(output, 0.03), atol=1e-7)
    assert model.protocol()["output_semantics"] == (
        "horizon_innovation_retention"
    )


def test_ppse_v2_initial_teacher_write_is_point_zero_one():
    retention_37 = torch.full((2, 1, 3, 3), 0.03)
    retention_68 = bilinear_resize_retention(retention_37, 5)
    p0 = torch.rand(2, 1, 5, 5)
    teacher = (torch.rand(2, 1, 5, 5) > 0.5).float()
    result = update_prior_anchored_supervision_state(
        q_prev_68=p0,
        p0_soft_68=p0,
        teacher_binary_68=teacher,
        retention_68=retention_68,
        state_step=1.0 / 3.0,
    )
    assert torch.allclose(
        result["teacher_write_weight"],
        torch.full_like(p0, 0.01),
        atol=1e-7,
    )
    assert torch.allclose(
        result["q_current"],
        0.99 * p0 + 0.01 * teacher,
        atol=1e-7,
    )
