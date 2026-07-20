import torch
import torch.nn.functional as F

from models.pssf import PredictiveSupervisionStateFilter
from train import sync_ppse_v2_actor


def _network():
    return PredictiveSupervisionStateFilter(
        feature_channels=4,
        feature_proj_dim=4,
        state_channels=9,
        hidden_dim=8,
        gn_groups=4,
        init_gain=0.03,
        output_semantics="horizon_innovation_retention",
    )


def test_ppse_v2_retention_and_segmentation_gradients_are_isolated():
    torch.manual_seed(3407)
    learner = _network()
    actor = _network()
    sync_ppse_v2_actor(learner, actor, epoch=4)
    student = torch.nn.Conv2d(3, 1, kernel_size=1)

    feature = torch.randn(2, 4, 3, 3)
    state = torch.randn(2, 9, 3, 3)
    retention_target = torch.rand(2, 1, 3, 3)
    retention_loss = (
        learner(feature.detach(), state.detach()) - retention_target
    ).square().mean()
    retention_loss.backward()

    learner_gradients = [
        parameter.grad
        for parameter in learner.parameters()
        if parameter.grad is not None
    ]
    assert learner_gradients
    assert all(bool(torch.isfinite(grad).all()) for grad in learner_gradients)
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert all(parameter.grad is None for parameter in student.parameters())

    for parameter in learner.parameters():
        parameter.grad = None
    with torch.no_grad():
        retention = actor(feature, state)
        q_target = F.interpolate(
            retention,
            size=(5, 5),
            mode="bilinear",
            align_corners=False,
        )
    segmentation_loss = F.binary_cross_entropy_with_logits(
        student(torch.randn(2, 3, 5, 5)),
        q_target.detach(),
    )
    segmentation_loss.backward()

    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in learner.parameters())
    assert all(parameter.grad is None for parameter in actor.parameters())
