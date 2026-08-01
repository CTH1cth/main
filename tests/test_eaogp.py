import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from common.eaogp import (
    build_dual_graph_weights,
    build_eaogp_target,
    build_same_source_background_evidence,
    get_eaogp_scale,
    validate_eaogp_config,
)
from common.ectp import build_ectp_projected_target
from common.teacher_routing import build_identity_teacher_route
from common.utils import load_config
from train import teacher_route_bce_with_logits


MAIN_ROOT = Path(__file__).resolve().parents[1]
EAOGP_CONFIG = MAIN_ROOT / "configs" / (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_eaogp_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)


def _map(value):
    return torch.full((1, 1, 68, 68), float(value), dtype=torch.float32)


def _graph():
    n, k = 37 * 37, 12
    base = torch.arange(n, dtype=torch.long).view(n, 1)
    offsets = torch.arange(1, k + 1, dtype=torch.long).view(1, k)
    idx = ((base + offsets) % n).unsqueeze(0)
    weight = torch.full((1, n, k), 1.0 / k, dtype=torch.float32)
    embedding = torch.zeros((1, 64, 37, 37), dtype=torch.float32)
    embedding[:, 0] = 1.0
    return idx, weight, embedding


def _target(
    foreground,
    background,
    teacher_prob,
    *,
    scale=1.0,
    idx=None,
    weight=None,
    embedding=None,
):
    if idx is None:
        idx, weight, embedding = _graph()
    foreground = foreground if torch.is_tensor(foreground) else _map(foreground)
    background = background if torch.is_tensor(background) else _map(background)
    teacher_prob = teacher_prob if torch.is_tensor(teacher_prob) else _map(teacher_prob)
    static = (foreground > 0.5).float()
    teacher_binary = (teacher_prob >= 0.5).float()
    return build_eaogp_target(
        foreground,
        background,
        static,
        teacher_prob,
        teacher_binary,
        embedding,
        idx,
        weight,
        scale,
    )


class TestEAOGP(unittest.TestCase):
    def test_01_same_source_background_endpoints(self):
        one = torch.ones((1, 1, 37, 37))
        zero = torch.zeros_like(one)
        b37, b68 = build_same_source_background_evidence(one, zero)
        self.assertTrue(torch.equal(b37, one))
        self.assertTrue(torch.equal(b68, torch.ones((1, 1, 68, 68))))
        b37, b68 = build_same_source_background_evidence(one, one)
        self.assertTrue(torch.equal(b37, zero))
        self.assertTrue(torch.equal(b68, torch.zeros((1, 1, 68, 68))))

    def test_02_static_target_is_strict_foreground_threshold(self):
        foreground = torch.tensor([0.49, 0.5, 0.5001]).view(1, 1, 1, 3)
        foreground = F.interpolate(foreground, size=(68, 68), mode="nearest")
        _, result = _target(foreground, _map(0), _map(0.2))
        expected = (foreground > 0.5).float()
        self.assertTrue(torch.equal(result["foreground_response_68"] > 0.5, expected.bool()))

    def test_03_anchor_confidence_is_bounded(self):
        generator = torch.Generator().manual_seed(3)
        foreground = torch.rand((1, 1, 68, 68), generator=generator)
        background = torch.rand((1, 1, 68, 68), generator=generator)
        _, result = _target(foreground, background, _map(0.3))
        anchor = result["anchor_confidence_68"]
        self.assertGreaterEqual(float(anchor.min()), 0.0)
        self.assertLessEqual(float(anchor.max()), 1.0)

    def test_04_strong_foreground_anchor(self):
        target, result = _target(1.0, 0.0, 0.25)
        self.assertTrue(torch.equal(result["anchor_confidence_68"], _map(1)))
        self.assertTrue(torch.equal(target, _map(1)))

    def test_05_strong_background_anchor(self):
        target, result = _target(0.0, 1.0, 0.75)
        self.assertTrue(torch.equal(result["anchor_confidence_68"], _map(1)))
        self.assertTrue(torch.equal(target, _map(0)))

    def test_06_dual_graph_row_normalization(self):
        idx, weight, embedding = _graph()
        result = build_dual_graph_weights(idx, weight, embedding)
        error = (result["dual_graph_weight"].sum(-1) - 1.0).abs().max()
        self.assertLessEqual(float(error), 1e-5)

    def test_07_identical_teacher_embedding_preserves_dino_weights(self):
        idx, weight, embedding = _graph()
        result = build_dual_graph_weights(idx, weight, embedding)
        self.assertLessEqual(
            float((result["dual_graph_weight"] - weight).abs().max()), 1e-7
        )

    def test_08_opposite_teacher_neighbor_is_rejected(self):
        idx, weight, embedding = _graph()
        flat = embedding.flatten(2)
        query = 0
        opposite_neighbor = int(idx[0, query, 0])
        flat[:, :, opposite_neighbor] = -flat[:, :, query]
        result = build_dual_graph_weights(idx, weight, embedding)
        self.assertLess(
            float(result["dual_graph_weight"][0, query, 0]),
            float(weight[0, query, 0]),
        )

    def test_09_zero_compatibility_row_falls_back_to_dino(self):
        idx, weight, embedding = _graph()
        flat = embedding.flatten(2)
        query = 0
        flat[:, :, query] = 0.0
        flat[:, 0, query] = 1e6
        for neighbor in idx[0, query].tolist():
            flat[:, :, neighbor] = 0.0
            flat[:, 0, neighbor] = -1e6
        result = build_dual_graph_weights(idx, weight, embedding)
        self.assertTrue(bool(result["fallback_row_mask"][0, query].item()))
        self.assertTrue(
            torch.equal(result["dual_graph_weight"][0, query], weight[0, query])
        )

    def test_10_constant_teacher_probability_is_propagation_invariant(self):
        _, result = _target(0.0, 0.0, 0.37)
        self.assertLessEqual(
            float((result["dual_graph_consensus_68"] - 0.37).abs().max()),
            1e-6,
        )

    def test_11_confident_teacher_keeps_binary_online_target(self):
        _, result_bg = _target(0.0, 0.0, 0.0)
        self.assertTrue(torch.equal(result_bg["online_target_68"], _map(0)))
        _, result_fg = _target(1.0, 1.0, 1.0)
        self.assertTrue(torch.equal(result_fg["online_target_68"], _map(1)))

    def test_12_maximum_teacher_uncertainty_uses_graph_consensus(self):
        _, result = _target(0.0, 0.0, 0.5)
        self.assertTrue(
            torch.equal(result["online_target_68"], result["dual_graph_consensus_68"])
        )

    def test_13_full_anchor_projects_to_foreground_response(self):
        target, _ = _target(1.0, 0.0, 0.2)
        self.assertTrue(torch.equal(target, _map(1)))

    def test_14_no_anchor_confident_teacher_stays_binary(self):
        target, _ = _target(0.0, 0.0, 0.0)
        self.assertTrue(torch.equal(target, _map(0)))

    def test_15_no_anchor_uncertain_teacher_uses_graph_consensus(self):
        target, result = _target(0.0, 0.0, 0.5)
        self.assertTrue(torch.equal(target, result["dual_graph_consensus_68"]))

    def test_16_zero_scale_is_bit_exact_binary_teacher(self):
        teacher_prob = torch.rand(
            (1, 1, 68, 68), generator=torch.Generator().manual_seed(16)
        )
        target, _ = _target(0.0, 0.0, teacher_prob, scale=0.0)
        self.assertTrue(torch.equal(target, (teacher_prob >= 0.5).float()))

    def test_17_foreground_background_symmetry(self):
        generator = torch.Generator().manual_seed(17)
        foreground = torch.rand((1, 1, 68, 68), generator=generator)
        background = foreground.clone()
        teacher_prob = torch.rand((1, 1, 68, 68), generator=generator)
        target, _ = _target(foreground, background, teacher_prob)
        mirrored, _ = _target(1.0 - background, 1.0 - foreground, 1.0 - teacher_prob)
        self.assertLessEqual(float((mirrored - (1.0 - target)).abs().max()), 1e-6)

    def test_18_eaogp_can_create_new_foreground(self):
        idx, weight, embedding = _graph()
        teacher_prob = torch.full((1, 1, 68, 68), 0.49)
        teacher_prob[..., 40:, :] = 1.0
        idx.fill_(37 * 37 - 1)
        target, _ = _target(
            0.0,
            0.0,
            teacher_prob,
            idx=idx,
            weight=weight,
            embedding=embedding,
        )
        original_bg = teacher_prob < 0.5
        self.assertTrue(bool((target[original_bg] > 0.5).any()))

    def test_19_eaogp_can_create_new_background(self):
        idx, weight, embedding = _graph()
        teacher_prob = torch.full((1, 1, 68, 68), 0.51)
        teacher_prob[..., 40:, :] = 0.0
        idx.fill_(37 * 37 - 1)
        target, _ = _target(
            1.0,
            1.0,
            teacher_prob,
            idx=idx,
            weight=weight,
            embedding=embedding,
        )
        original_fg = teacher_prob >= 0.5
        self.assertTrue(bool((target[original_fg] < 0.5).any()))

    def test_20_gradients_reach_only_student_logits(self):
        idx, weight, embedding = _graph()
        target, _ = _target(0.0, 0.0, 0.4, idx=idx, weight=weight, embedding=embedding)
        logits = torch.randn_like(target, requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        self.assertFalse(target.requires_grad)
        self.assertFalse(embedding.requires_grad)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_21_config_and_shared_dagp_schedule(self):
        config = load_config(EAOGP_CONFIG)
        self.assertEqual(validate_eaogp_config(config)["status"], "PASS")
        expected = {1: 0.0, 6: 0.0, 7: 1 / 9, 15: 1.0, 21: 1.0}
        for epoch, scale in expected.items():
            self.assertAlmostEqual(get_eaogp_scale(config, epoch), scale)

    def test_22_three_teacher_branches_use_plain_mean_bce(self):
        config = load_config(EAOGP_CONFIG)
        target, _ = _target(0.0, 0.0, 0.4)
        teacher_map, _ = build_identity_teacher_route(target)
        for seed, branch in enumerate(("final", "coarse", "base"), 22):
            logits = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed))
            actual = teacher_route_bce_with_logits(
                logits,
                target,
                teacher_map,
                config,
                routing_scale=0.0,
                apply_to_loss=True,
            )
            expected = F.binary_cross_entropy_with_logits(logits, target)
            self.assertLessEqual(float((actual - expected).abs()), 1e-7, branch)

    def test_23_historical_teacher_loss_paths_are_unchanged(self):
        teacher = (torch.rand((1, 1, 68, 68), generator=torch.Generator().manual_seed(23)) >= 0.5).float()
        logits = torch.randn(teacher.shape, generator=torch.Generator().manual_seed(24))
        route, _ = build_identity_teacher_route(teacher)
        noecst = load_config(MAIN_ROOT / "configs" / (
            "dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_"
            "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        ))
        actual = teacher_route_bce_with_logits(logits, teacher, route, noecst, 0.0)
        expected = F.binary_cross_entropy_with_logits(logits, teacher)
        self.assertLessEqual(float((actual - expected).abs()), 1e-7)

        ectp = load_config(MAIN_ROOT / "configs" / (
            "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ectp_"
            "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        ))
        foreground = _map(1)
        projected, _ = build_ectp_projected_target(
            foreground,
            _map(0),
            _map(1),
            _map(0),
            0.5,
            0.5,
        )
        actual = teacher_route_bce_with_logits(logits, projected, route, ectp, 0.0)
        expected = F.binary_cross_entropy_with_logits(logits, projected)
        self.assertLessEqual(float((actual - expected).abs()), 1e-7)

        full_ecst = load_config(MAIN_ROOT / "configs" / (
            "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_"
            "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        ))
        weighted_route = torch.rand(
            teacher.shape, generator=torch.Generator().manual_seed(25)
        ).mul(0.8).add(0.2)
        route_snapshot = weighted_route.clone()
        actual = teacher_route_bce_with_logits(
            logits,
            teacher,
            weighted_route,
            full_ecst,
            routing_scale=1.0,
            eps=1e-6,
        )
        loss_map = F.binary_cross_entropy_with_logits(
            logits, teacher, reduction="none"
        )
        expected = (loss_map * weighted_route).sum() / (
            weighted_route.sum() + 1e-6
        )
        self.assertTrue(torch.equal(weighted_route, route_snapshot))
        self.assertLessEqual(float((actual - expected).abs()), 1e-6)


if __name__ == "__main__":
    unittest.main()
