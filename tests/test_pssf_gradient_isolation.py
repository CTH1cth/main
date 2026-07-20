from types import SimpleNamespace

import torch
from torch import nn

from common.pssf_retention import (
    build_retention_target,
    update_supervision_state,
    weighted_gain_loss,
)
from models.pssf import PredictiveSupervisionStateFilter
from train import build_pssf_segmentation_group


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.final = nn.Conv2d(3, 1, 1)
        self.coarse = nn.Conv2d(3, 1, 1)
        self.base = nn.Conv2d(3, 1, 1)

    def forward(self, value):
        return {
            "final_logits": self.final(value),
            "coarse_logits_68": self.coarse(value),
            "base_logits": self.base(value),
        }


def test_pssf_backward_does_not_reach_student_or_teacher():
    torch.manual_seed(2027)
    student_feature = nn.Conv2d(3, 4, 1).double()
    teacher = nn.Conv2d(3, 1, 1).double()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    pssf = PredictiveSupervisionStateFilter(
        feature_channels=4,
        feature_proj_dim=4,
        state_channels=9,
        hidden_dim=8,
        gn_groups=4,
        init_gain=0.01,
    ).double()

    image = torch.randn(2, 3, 5, 5, dtype=torch.float64)
    feature = student_feature(image)
    state = torch.rand(2, 9, 5, 5, dtype=torch.float64)
    prediction = pssf(feature, state)
    q_prev = torch.full_like(prediction, 0.2)
    teacher_binary = torch.ones_like(prediction)
    future = torch.stack(
        [
            torch.ones_like(prediction),
            torch.ones_like(prediction),
            torch.zeros_like(prediction),
        ],
        dim=0,
    )
    target = build_retention_target(q_prev, teacher_binary, future)
    loss, _ = weighted_gain_loss(
        prediction,
        target["gain_target"],
        target["innovation_weight"],
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in pssf.parameters())
    assert all(
        parameter.grad is None for parameter in student_feature.parameters()
    )
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_segmentation_backward_does_not_reach_pssf():
    torch.manual_seed(2027)
    decoder = TinyDecoder().double()
    pssf = PredictiveSupervisionStateFilter(
        feature_channels=4,
        feature_proj_dim=4,
        state_channels=9,
        hidden_dim=8,
        gn_groups=4,
        init_gain=0.01,
    ).double()
    feature = torch.randn(2, 4, 6, 6, dtype=torch.float64)
    state = torch.rand(2, 9, 6, 6, dtype=torch.float64)
    gain = pssf(feature, state)
    q_prev = torch.full((2, 1, 6, 6), 0.3, dtype=torch.float64)
    teacher_binary = torch.ones_like(q_prev)
    q_target = update_supervision_state(q_prev, teacher_binary, gain)
    pssf.zero_grad(set_to_none=True)

    output = decoder(torch.randn(2, 3, 6, 6, dtype=torch.float64))
    cfg = SimpleNamespace(
        LOSS_SIZE=6,
        FINETUNE_RESET_EPOCH=29,
        FINETUNE_RESET_TIMING="after_epoch",
    )
    group = build_pssf_segmentation_group(
        cfg,
        epoch=10,
        student_out=output,
        student_logits=output["final_logits"],
        q_target=q_target,
    )
    group["loss"].backward()
    assert any(parameter.grad is not None for parameter in decoder.parameters())
    assert all(parameter.grad is None for parameter in pssf.parameters())


def test_pre_and_post_reset_auxiliary_weights_are_exact():
    decoder = TinyDecoder()
    output = decoder(torch.randn(1, 3, 4, 4))
    cfg = SimpleNamespace(
        LOSS_SIZE=4,
        FINETUNE_RESET_EPOCH=29,
        FINETUNE_RESET_TIMING="after_epoch",
    )
    q_target = torch.rand(1, 1, 4, 4)
    before = build_pssf_segmentation_group(
        cfg, 29, output, output["final_logits"], q_target
    )
    after = build_pssf_segmentation_group(
        cfg, 30, output, output["final_logits"], q_target
    )
    assert before["coarse_weight"] == after["coarse_weight"] == 0.5
    assert before["base_weight"] == 0.5
    assert after["base_weight"] == 0.3
    assert before["weight_sum"] == 2.0
    assert after["weight_sum"] == 1.8

