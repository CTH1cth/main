import torch

from common.pssf_retention import (
    build_retention_target,
    future_state_from_gain,
    innovation_weighted_image_mean,
    weighted_gain_loss,
)


def _target(q, teacher, futures, eps=1e-6):
    q = torch.tensor([q], dtype=torch.float64).view(1, 1, 1, 1)
    teacher = torch.tensor([teacher], dtype=torch.float64).view_as(q)
    futures = torch.tensor(futures, dtype=torch.float64).view(-1, 1, 1, 1, 1)
    return build_retention_target(q, teacher, futures, eps=eps)


def test_retention_target_endpoints_and_partial_retention():
    full = _target(0.2, 1.0, [1.0, 1.0, 1.0])
    half = _target(0.2, 1.0, [0.6, 0.6, 0.6])
    withdrawn = _target(0.2, 1.0, [0.2, 0.2, 0.2])
    reverse_half = _target(0.8, 0.0, [0.4, 0.4, 0.4])

    assert abs(float(full["gain_target"]) - 1.0) < 2e-6
    assert abs(float(half["gain_target"]) - 0.5) < 2e-6
    assert float(withdrawn["gain_target"]) == 0.0
    assert abs(float(reverse_half["gain_target"]) - 0.5) < 2e-6


def test_zero_innovation_is_finite_and_has_zero_weight():
    result = _target(0.5, 0.5, [0.0, 1.0, 0.5])
    assert float(result["innovation_weight"]) == 0.0
    assert float(result["gain_target"]) == 0.0
    assert all(torch.isfinite(value).all() for value in result.values())

    prediction = torch.tensor([[[[0.9]]]], requires_grad=True)
    loss, mae = weighted_gain_loss(
        prediction,
        result["gain_target"].float(),
        result["innovation_weight"].float(),
    )
    assert float(loss) == 0.0
    assert float(mae) == 0.0
    loss.backward()
    assert float(prediction.grad) == 0.0


def test_innovation_weighted_loss_and_scalar_baseline():
    prediction = torch.tensor([[[[0.2, 0.8]]]], dtype=torch.float64)
    target = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
    weight = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
    loss, mae = weighted_gain_loss(
        prediction,
        target,
        weight,
        eps=1e-12,
    )
    assert abs(float(loss) - 0.04) < 1e-12
    assert abs(float(mae) - 0.2) < 1e-12
    scalar = innovation_weighted_image_mean(prediction, weight, eps=1e-12)
    assert abs(float(scalar) - 0.8) < 1e-12


def test_gain_reconstructs_future_state_formula():
    q_prev = torch.tensor([[[[0.2, 0.8]]]])
    teacher = torch.tensor([[[[1.0, 0.0]]]])
    gain = torch.tensor([[[[0.25, 0.75]]]])
    actual = future_state_from_gain(q_prev, teacher, gain)
    expected = q_prev + gain * (teacher - q_prev)
    assert torch.equal(actual, expected)

