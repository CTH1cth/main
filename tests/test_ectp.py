import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from common.ectp import (
    ECTP_VERSION,
    accumulate_ectp_epoch,
    build_ectp_projected_target,
    finalize_ectp_epoch,
    new_ectp_epoch_accumulator,
    validate_ectp_config,
)
from common.teacher_routing import build_identity_teacher_route
from common.utils import load_config
from train import teacher_route_bce_with_logits


MAIN_ROOT = Path(__file__).resolve().parents[1]


def _map(value):
    return torch.full((1, 1, 68, 68), float(value), dtype=torch.float32)


def _valid_config():
    return SimpleNamespace(
        USE_ECTP=True,
        TEACHER_ROUTING_MODE="ectp",
        ECTP_VERSION=ECTP_VERSION,
        USE_ECST=False,
        USE_ECST_MINIMAL=False,
        USE_ECST_CLEAN=False,
        USE_BITC=False,
        DABE_CLEAN_USE_LEGACY_ECST_REGIONS=False,
        USE_DABE_CLEAN=True,
        USE_DABE_PU=False,
        DABE_CLEAN_STATIC_TARGET_SOURCE="dabe_v2_hard_68",
        DABE_CLEAN_DABE_V2_VERSION="v2",
        DABE_CLEAN_DABE_V2_HARD_THRESHOLD=0.5,
        STATIC_WEIGHT_MODE="ones",
        DABE_CLEAN_STATIC_WEIGHT_MODE="ones",
        TEACHER_TARGET_MODE="binary",
        USE_DABE_CLEAN_DESPL_SCHEDULE=True,
        USE_TEACHER_BINARY_FULL_LOSS=True,
        USE_DAGP_SAFE_HEAD=True,
        USE_NDR_BRANCH=True,
        FINETUNE_RESET_EPOCH=20,
        MAX_EPOCH=45,
        STOP_AFTER_EPOCH=0,
    )


def _build(y0, foreground, background, teacher, alpha=0.5, beta=0.5):
    return build_ectp_projected_target(
        foreground_evidence=_map(foreground),
        background_evidence=_map(background),
        static_target=_map(y0),
        teacher_binary=_map(teacher),
        static_weight=alpha,
        teacher_weight=beta,
    )


class TestECTP(unittest.TestCase):
    def test_config_contract_and_parameter_free_validation(self):
        report = validate_ectp_config(_valid_config())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["extra_numeric_parameters"], [])
        invalid = _valid_config()
        invalid.ECTP_STRENGTH = 1.0
        with self.assertRaisesRegex(RuntimeError, "ECTP_STRENGTH"):
            validate_ectp_config(invalid)

    def test_foreground_agreement_is_exactly_preserved(self):
        projected, stats = _build(1, 1, 0, 1)
        self.assertEqual(stats["support_mean"], 1.0)
        self.assertEqual(stats["conflict_ratio"], 0.0)
        self.assertTrue(torch.equal(projected, _map(1)))

    def test_background_agreement_is_exactly_preserved(self):
        projected, stats = _build(0, 0, 1, 0)
        self.assertEqual(stats["support_mean"], 1.0)
        self.assertEqual(stats["conflict_ratio"], 0.0)
        self.assertTrue(torch.equal(projected, _map(0)))

    def test_strong_static_foreground_conflict_projects_to_neutral(self):
        projected, stats = _build(1, 1, 0, 0)
        self.assertEqual(stats["support_mean"], 1.0)
        self.assertEqual(stats["overlap_rho"], 1.0)
        self.assertTrue(torch.equal(projected, _map(0.5)))
        self.assertEqual(stats["fg_protection_mass"], 0.5)
        self.assertEqual(stats["bg_protection_mass"], 0.0)

    def test_strong_static_background_conflict_projects_to_neutral(self):
        projected, stats = _build(0, 0, 1, 1)
        self.assertEqual(stats["support_mean"], 1.0)
        self.assertEqual(stats["overlap_rho"], 1.0)
        self.assertTrue(torch.equal(projected, _map(0.5)))
        self.assertEqual(stats["fg_protection_mass"], 0.0)
        self.assertEqual(stats["bg_protection_mass"], 0.5)

    def test_zero_support_preserves_teacher(self):
        # F remains above 0.5 so Y0=1 is the exact DABE-v2 hard target, while
        # B=1 makes foreground-direction support exactly zero.
        projected, stats = _build(1, 0.75, 1, 0)
        self.assertEqual(stats["support_mean"], 0.0)
        self.assertTrue(torch.equal(projected, _map(0)))

    def test_static_only_preserves_teacher(self):
        projected, stats = _build(1, 1, 0, 0, alpha=1.0, beta=0.0)
        self.assertEqual(stats["overlap_rho"], 0.0)
        self.assertTrue(torch.equal(projected, _map(0)))

    def test_teacher_only_preserves_teacher(self):
        projected, stats = _build(1, 1, 0, 0, alpha=0.0, beta=1.0)
        self.assertEqual(stats["overlap_rho"], 0.0)
        self.assertTrue(torch.equal(projected, _map(0)))

    def test_overlap_curve_uses_actual_global_weights(self):
        expected = (
            (1.00, 0.00, 0.00),
            (0.75, 0.25, 0.75),
            (0.50, 0.50, 1.00),
            (0.25, 0.75, 0.75),
            (0.00, 1.00, 0.00),
        )
        for alpha, beta, overlap in expected:
            with self.subTest(alpha=alpha, beta=beta):
                _, stats = _build(1, 1, 0, 0, alpha=alpha, beta=beta)
                self.assertAlmostEqual(stats["overlap_rho"], overlap)

    def test_foreground_background_symmetry(self):
        foreground_row = torch.tensor(
            [0.9, 0.2, 0.7, 0.1], dtype=torch.float32
        ).view(1, 1, 1, 4)
        background_row = torch.tensor(
            [0.1, 0.8, 0.3, 0.9], dtype=torch.float32
        ).view(1, 1, 1, 4)
        teacher_row = torch.tensor(
            [0.0, 1.0, 1.0, 0.0], dtype=torch.float32
        ).view(1, 1, 1, 4)
        foreground = foreground_row.repeat(1, 1, 68, 17)
        background = background_row.repeat(1, 1, 68, 17)
        teacher = teacher_row.repeat(1, 1, 68, 17)
        static = (foreground > 0.5).float()
        projected, _ = build_ectp_projected_target(
            foreground_evidence=foreground,
            background_evidence=background,
            static_target=static,
            teacher_binary=teacher,
            static_weight=0.7,
            teacher_weight=0.3,
        )
        mirrored, _ = build_ectp_projected_target(
            foreground_evidence=background,
            background_evidence=foreground,
            static_target=1.0 - static,
            teacher_binary=1.0 - teacher,
            static_weight=0.7,
            teacher_weight=0.3,
        )
        self.assertLessEqual(
            float((mirrored - (1.0 - projected)).abs().max()), 1e-6
        )

    def test_projection_never_crosses_neutral_boundary(self):
        foreground = torch.rand(
            (2, 1, 68, 68), generator=torch.Generator().manual_seed(4)
        )
        background = torch.rand(
            (2, 1, 68, 68), generator=torch.Generator().manual_seed(5)
        )
        static = (foreground > 0.5).float()
        teacher = torch.randint(
            0,
            2,
            (2, 1, 68, 68),
            generator=torch.Generator().manual_seed(6),
        ).float()
        projected, _ = build_ectp_projected_target(
            foreground_evidence=foreground,
            background_evidence=background,
            static_target=static,
            teacher_binary=teacher,
            static_weight=0.5,
            teacher_weight=0.5,
        )
        self.assertTrue(bool((projected[teacher > 0.5] >= 0.5).all()))
        self.assertTrue(bool((projected[teacher < 0.5] <= 0.5).all()))

    def test_nonconflicting_pixels_are_bit_exact(self):
        foreground = torch.rand(
            (1, 1, 68, 68), generator=torch.Generator().manual_seed(7)
        )
        background = torch.rand(
            (1, 1, 68, 68), generator=torch.Generator().manual_seed(8)
        )
        static = (foreground > 0.5).float()
        teacher = static.clone()
        teacher[..., ::3, ::2] = 1.0 - teacher[..., ::3, ::2]
        projected, stats = build_ectp_projected_target(
            foreground_evidence=foreground,
            background_evidence=background,
            static_target=static,
            teacher_binary=teacher,
            static_weight=0.5,
            teacher_weight=0.5,
        )
        nonconflict = ~stats["conflict_bool"]
        self.assertTrue(torch.equal(projected[nonconflict], teacher[nonconflict]))
        self.assertTrue(stats["nonconflict_exact_match"])

    def test_bce_gradients_only_reach_logits(self):
        foreground = torch.rand(
            (1, 1, 68, 68), generator=torch.Generator().manual_seed(9)
        )
        background = torch.rand(
            (1, 1, 68, 68), generator=torch.Generator().manual_seed(10)
        )
        static = (foreground > 0.5).float()
        teacher = 1.0 - static
        snapshots = tuple(
            value.clone() for value in (foreground, background, static, teacher)
        )
        projected, _ = build_ectp_projected_target(
            foreground_evidence=foreground,
            background_evidence=background,
            static_target=static,
            teacher_binary=teacher,
            static_weight=0.5,
            teacher_weight=0.5,
        )
        logits = torch.randn_like(projected, requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(logits, projected, reduction="mean")
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertFalse(projected.requires_grad)
        for current, snapshot in zip(
            (foreground, background, static, teacher), snapshots
        ):
            self.assertFalse(current.requires_grad)
            self.assertTrue(torch.equal(current, snapshot))

    def test_input_contract_rejects_mismatch_and_non_detached_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "not exactly"):
            _build(0, 1, 0, 0)
        foreground = _map(1).requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "detached"):
            build_ectp_projected_target(
                foreground_evidence=foreground,
                background_evidence=_map(0),
                static_target=_map(1),
                teacher_binary=_map(0),
                static_weight=0.5,
                teacher_weight=0.5,
            )

    def test_epoch_accumulator_keeps_projected_and_teacher_areas_separate(self):
        _, stats = _build(1, 1, 0, 0)
        accumulator = new_ectp_epoch_accumulator()
        accumulate_ectp_epoch(
            accumulator,
            stats,
            student_pred_area=0.25,
            teacher_pred_area=0.0,
        )
        row = finalize_ectp_epoch(accumulator)
        self.assertEqual(row["projected_target_mean"], 0.5)
        self.assertEqual(row["teacher_fg_ratio"], 0.0)
        self.assertEqual(row["teacher_pred_area"], 0.0)
        self.assertEqual(row["student_pred_area"], 0.25)
        self.assertEqual(row["fg_protection_mass"], 0.5)
        self.assertEqual(row["bg_protection_mass"], 0.0)

    def test_actual_ectp_config_and_all_three_teacher_branches(self):
        config = load_config(
            MAIN_ROOT
            / "configs"
            / "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ectp_"
            "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        self.assertEqual(validate_ectp_config(config)["status"], "PASS")
        projected, _ = _build(1, 1, 0, 0)
        teacher_map, _ = build_identity_teacher_route(_map(0))
        for seed, branch in enumerate(("final", "coarse", "base"), start=31):
            with self.subTest(branch=branch):
                logits = torch.randn(
                    projected.shape,
                    generator=torch.Generator().manual_seed(seed),
                )
                actual = teacher_route_bce_with_logits(
                    logits,
                    projected,
                    teacher_map,
                    config,
                    routing_scale=0.0,
                    apply_to_loss=True,
                )
                expected = F.binary_cross_entropy_with_logits(
                    logits, projected, reduction="mean"
                )
                self.assertLessEqual(float((actual - expected).abs()), 1e-7)

    def test_noecst_teacher_target_map_and_loss_regression(self):
        config = load_config(
            MAIN_ROOT
            / "configs"
            / "dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_"
            "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        teacher_binary = (
            torch.rand(
                (1, 1, 68, 68),
                generator=torch.Generator().manual_seed(41),
            )
            > 0.5
        ).float()
        logits = torch.randn(
            teacher_binary.shape,
            generator=torch.Generator().manual_seed(42),
        )
        teacher_map, _ = build_identity_teacher_route(teacher_binary)
        actual = teacher_route_bce_with_logits(
            logits,
            teacher_binary,
            teacher_map,
            config,
            routing_scale=0.0,
        )
        expected = F.binary_cross_entropy_with_logits(
            logits, teacher_binary, reduction="mean"
        )
        self.assertTrue(torch.equal(teacher_binary, teacher_binary.clone()))
        self.assertTrue(torch.equal(teacher_map, torch.ones_like(teacher_map)))
        self.assertLessEqual(float((actual - expected).abs()), 1e-6)

    def test_full_ecst_teacher_map_and_loss_regression(self):
        config = load_config(
            MAIN_ROOT
            / "configs"
            / "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_"
            "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        teacher_binary = (
            torch.rand(
                (1, 1, 68, 68),
                generator=torch.Generator().manual_seed(51),
            )
            > 0.5
        ).float()
        logits = torch.randn(
            teacher_binary.shape,
            generator=torch.Generator().manual_seed(52),
        )
        teacher_map = torch.rand(
            teacher_binary.shape,
            generator=torch.Generator().manual_seed(53),
        ).mul(0.8).add(0.2)
        map_snapshot = teacher_map.clone()
        actual = teacher_route_bce_with_logits(
            logits,
            teacher_binary,
            teacher_map,
            config,
            routing_scale=1.0,
            eps=1e-6,
        )
        loss_map = F.binary_cross_entropy_with_logits(
            logits, teacher_binary, reduction="none"
        )
        expected = (loss_map * teacher_map).sum() / (
            teacher_map.sum() + 1e-6
        )
        self.assertTrue(torch.equal(teacher_map, map_snapshot))
        self.assertLessEqual(float((actual - expected).abs()), 1e-6)


if __name__ == "__main__":
    unittest.main()
