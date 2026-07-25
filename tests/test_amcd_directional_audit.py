import torch
import torch.nn.functional as F

from analysis.amcd_directional_audit import (
    compute_asymmetry,
    compute_bidirectional_coverage,
    extract_weighted_slots,
    permuted_mask,
    rolled_mask,
    run_analytic_assertions,
)


def test_weighted_slots_are_deterministic_and_nonempty():
    generator = torch.Generator().manual_seed(7)
    feature = torch.randn(12, 7, 7, generator=generator)
    mask = torch.rand(7, 7, generator=generator)
    first = extract_weighted_slots(feature, mask, 4, 0.10, num_iterations=5)
    second = extract_weighted_slots(feature, mask, 4, 0.10, num_iterations=5)
    assert first.slots.shape == (4, 12)
    assert first.assignment_map.shape == (7, 7)
    assert torch.equal(first.initialization_indices, second.initialization_indices)
    assert torch.equal(first.slots, second.slots)
    assert bool((first.slot_mass > 0).all())


def test_inverse_partition_flips_directional_asymmetry():
    generator = torch.Generator().manual_seed(11)
    feature = torch.randn(16, 8, 8, generator=generator)
    mask = torch.rand(8, 8, generator=generator)
    real = compute_asymmetry(feature, mask, 4, 0.10, 0.10, 5)
    inverse = compute_asymmetry(feature, 1.0 - mask, 4, 0.10, 0.10, 5)
    assert real["valid"] and inverse["valid"]
    assert abs(real["asymmetry_raw"] + inverse["asymmetry_raw"]) < 1e-5
    assert abs(
        real["asymmetry_normalized"] + inverse["asymmetry_normalized"]
    ) < 1e-5


def test_identical_slots_and_slot_order_are_symmetric():
    generator = torch.Generator().manual_seed(13)
    slots = F.normalize(torch.randn(8, 10, generator=generator), dim=-1)
    result = compute_bidirectional_coverage(slots, slots.clone(), 0.10)
    assert abs(result["asymmetry_raw"]) < 1e-6

    other = F.normalize(torch.randn(8, 10, generator=generator), dim=-1)
    before = compute_bidirectional_coverage(slots, other, 0.10)
    after = compute_bidirectional_coverage(slots.flip(0), other.roll(3, 0), 0.10)
    assert abs(before["asymmetry_raw"] - after["asymmetry_raw"]) < 1e-6


def test_controls_are_deterministic_and_preserve_histogram():
    mask = torch.linspace(0, 1, 37 * 37).reshape(37, 37)
    perm_a = permuted_mask(mask, 23)
    perm_b = permuted_mask(mask, 23)
    assert torch.equal(perm_a, perm_b)
    assert torch.equal(mask.flatten().sort().values, perm_a.flatten().sort().values)

    roll_a, dy_a, dx_a = rolled_mask(mask, 29)
    roll_b, dy_b, dx_b = rolled_mask(mask, 29)
    assert torch.equal(roll_a, roll_b)
    assert (dy_a, dx_a) == (dy_b, dx_b)
    assert min(dy_a, 37 - dy_a) >= 4
    assert min(dx_a, 37 - dx_a) >= 4


def test_formula_level_assertions_pass_without_gradients():
    result = run_analytic_assertions("cpu")
    assert result["identical_set_abs_asymmetry"] < 1e-6
    assert result["slot_order_abs_error"] < 1e-6
    assert result["inverse_sign_flip_abs_error"] < 1e-5
    assert result["all_inputs_require_grad_false"]
