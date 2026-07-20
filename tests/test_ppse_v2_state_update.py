import torch

from common.pssf_retention import (
    update_prior_anchored_supervision_state,
)


def _update(q_prev, p0, teacher, retention):
    return update_prior_anchored_supervision_state(
        q_prev_68=q_prev,
        p0_soft_68=p0,
        teacher_binary_68=teacher,
        retention_68=retention,
        state_step=1.0 / 3.0,
    )


def test_ppse_v2_coefficients_sum_to_one_and_stay_in_range():
    generator = torch.Generator().manual_seed(3407)
    q_prev = torch.rand(2, 1, 5, 5, generator=generator)
    p0 = torch.rand(2, 1, 5, 5, generator=generator)
    teacher = (
        torch.rand(2, 1, 5, 5, generator=generator) > 0.5
    ).float()
    retention = torch.rand(2, 1, 5, 5, generator=generator)

    result = _update(q_prev, p0, teacher, retention)
    coefficient_sum = (
        torch.full_like(retention, 2.0 / 3.0)
        + result["p0_write_weight"]
        + result["teacher_write_weight"]
    )
    assert float((coefficient_sum - 1.0).abs().max()) < 1e-6
    assert result["coefficient_sum_error_max"] < 1e-6
    assert float(result["q_current"].min()) >= 0.0
    assert float(result["q_current"].max()) <= 1.0
    assert bool(torch.isfinite(result["q_current"]).all())


def test_ppse_v2_retention_zero_uses_only_prior_proposal():
    q_prev = torch.tensor([[[[0.1, 0.9]]]])
    p0 = torch.tensor([[[[0.8, 0.2]]]])
    teacher = torch.tensor([[[[0.0, 1.0]]]])
    result = _update(q_prev, p0, teacher, torch.zeros_like(q_prev))

    expected = (2.0 / 3.0) * q_prev + (1.0 / 3.0) * p0
    assert torch.allclose(result["q_current"], expected, atol=1e-7)
    assert torch.equal(
        result["teacher_write_weight"],
        torch.zeros_like(q_prev),
    )
    assert torch.allclose(
        result["p0_write_weight"],
        torch.full_like(q_prev, 1.0 / 3.0),
        atol=1e-7,
    )


def test_ppse_v2_retention_one_uses_teacher_with_one_third_limit():
    q_prev = torch.tensor([[[[0.1, 0.9]]]])
    p0 = torch.tensor([[[[0.8, 0.2]]]])
    teacher = torch.tensor([[[[0.0, 1.0]]]])
    result = _update(q_prev, p0, teacher, torch.ones_like(q_prev))

    expected = (2.0 / 3.0) * q_prev + (1.0 / 3.0) * teacher
    assert torch.allclose(result["q_current"], expected, atol=1e-7)
    assert torch.equal(
        result["p0_write_weight"],
        torch.zeros_like(q_prev),
    )
    assert float(result["teacher_write_weight"].max()) <= 1.0 / 3.0


def test_ppse_v2_initial_update_matches_v1_teacher_write_strength():
    p0 = torch.tensor([[[[0.2, 0.8], [0.4, 0.6]]]])
    teacher = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    retention = torch.full_like(p0, 0.03)
    result = _update(p0, p0, teacher, retention)

    expected = 0.99 * p0 + 0.01 * teacher
    assert torch.allclose(result["q_current"], expected, atol=1e-7)
    assert torch.allclose(
        result["teacher_write_weight"],
        torch.full_like(p0, 0.01),
        atol=1e-7,
    )
    assert torch.allclose(
        result["p0_write_weight"],
        torch.full_like(p0, 0.97 / 3.0),
        atol=1e-7,
    )
