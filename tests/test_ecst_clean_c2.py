import unittest
from unittest import mock

import torch

from common.ecst import TemporalTeacherMemory
from common.ecst_clean import (
    build_clean_history_bg_reliability,
    build_ecst_clean_teacher_weight_map,
    get_ecst_clean_continuous_strengths,
    get_ecst_clean_scale,
    validate_ecst_clean_config,
)
from common.utils import config_to_dict, load_config
from train import (
    accumulate_ecst_epoch,
    audit_c2_config_difference,
    build_clean_ecst_checkpoint_extra,
    log_ecst_clean_first_batch,
    new_ecst_epoch_accumulator,
)


C2_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_c2_nohist_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
C3_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_"
    "a1_residual_only_e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)


def _batch(target=0.5, latent=1.0):
    return {
        "dabe_clean_target_68": torch.full((1, 1, 68, 68), float(target)),
        "dabe_clean_recoverability_68": torch.full(
            (1, 1, 68, 68), float(latent)
        ),
        "dabe_clean_semantic_fg_tendency_37": torch.full(
            (1, 1, 37, 37), 0.5
        ),
        "sample_index": torch.tensor([0]),
        "dataset_name": ["synthetic"],
        "stem": ["sample"],
    }


def _legacy_v5_reference(cfg, batch, teacher, mean, second, count, epoch):
    target = batch["dabe_clean_target_68"].detach().float()
    recoverability = batch["dabe_clean_recoverability_68"].detach().float()
    temporal = build_clean_history_bg_reliability(
        mean,
        second,
        count,
        min_history=int(cfg.ECST_CLEAN_MIN_HISTORY),
        variance_tau=float(cfg.ECST_CLEAN_VARIANCE_TAU),
    )
    signed = (2.0 * target - 1.0).detach()
    positive = signed.clamp_min(0.0).detach()
    negative = (-signed).clamp_min(0.0).detach()
    teacher_fg = (teacher.detach().float() > 0.5).detach()
    scale = float(get_ecst_clean_scale(cfg, epoch))
    strengths = get_ecst_clean_continuous_strengths(cfg)
    weight_min = float(cfg.ECST_CLEAN_WEIGHT_MIN)
    suppression_max = 1.0 - weight_min
    erase_raw = ((1.0 - float(cfg.ECST_CLEAN_CONFLICT_FLOOR)) * positive).detach()
    erase = (scale * strengths["erase"] * erase_raw).clamp(
        0.0, suppression_max
    ).detach()
    recovery_raw = (
        (1.0 - float(cfg.ECST_CLEAN_NEGATIVE_WEIGHT_FLOOR))
        * recoverability
        * (1.0 - temporal["history_bg_reliability"])
    ).detach()
    recovery = (scale * strengths["recovery"] * recovery_raw).clamp(
        0.0, suppression_max
    ).detach()
    bg_suppression = (
        1.0 - (1.0 - erase) * (1.0 - recovery)
    ).clamp(0.0, suppression_max).detach()
    bg_weight = (1.0 - bg_suppression).clamp(weight_min, 1.0).detach()
    add_raw = ((1.0 - float(cfg.ECST_CLEAN_CONFLICT_FLOOR)) * negative).detach()
    add = (scale * strengths["add"] * add_raw).clamp(
        0.0, suppression_max
    ).detach()
    fg_weight = (1.0 - add).clamp(weight_min, 1.0).detach()
    return torch.where(teacher_fg, fg_weight, bg_weight).clamp(
        weight_min, 1.0
    ).detach()


class _Logger:
    def __init__(self):
        self.lines = []

    def log(self, value):
        self.lines.append(str(value))


class CleanECSTC2Test(unittest.TestCase):
    def setUp(self):
        self.c2 = load_config(C2_CONFIG)
        self.c3 = load_config(C3_CONFIG)

    def test_config_is_exact_three_field_ablation(self):
        self.assertTrue(validate_ecst_clean_config(self.c2))
        self.assertEqual(
            set(audit_c2_config_difference(self.c2)),
            {"EXP_NAME", "ECST_CLEAN_VERSION", "ECST_CLEAN_USE_HISTORY"},
        )
        current = config_to_dict(self.c2)
        parent = config_to_dict(self.c3)
        sentinel = object()
        differences = {
            key
            for key in set(current).union(parent)
            if current.get(key, sentinel) != parent.get(key, sentinel)
        }
        self.assertEqual(
            differences,
            {"EXP_NAME", "ECST_CLEAN_VERSION", "ECST_CLEAN_USE_HISTORY"},
        )

    def test_c2_skips_history_and_uses_latent_directly(self):
        teacher = torch.zeros((1, 1, 68, 68))
        with mock.patch(
            "common.ecst_clean.build_clean_history_bg_reliability",
            side_effect=AssertionError("history path was called"),
        ):
            route, stats, states = build_ecst_clean_teacher_weight_map(
                cfg=self.c2,
                batch=_batch(target=0.5, latent=1.0),
                teacher_prob=teacher,
                epoch=10,
                device="cpu",
                return_states=True,
            )
        self.assertFalse(stats["history_fields_consumed"])
        self.assertFalse(stats["memory_active"])
        self.assertEqual(stats["latent_effective_max_abs_error"], 0.0)
        self.assertTrue(torch.equal(states["latent_support"], states["latent_effective"]))
        self.assertFalse(any("history" in key or "temporal" in key for key in states))
        self.assertFalse(route.requires_grad)
        self.assertTrue(bool(torch.isfinite(route).all()))
        self.assertGreaterEqual(float(route.min()), 0.2 - 1e-6)
        self.assertLessEqual(float(route.max()), 1.0 + 1e-6)
        self.assertFalse(torch.equal(route, torch.ones_like(route)))

    def test_c2_teacher_directions_match_v5_definition(self):
        teacher_bg = torch.zeros((1, 1, 68, 68))
        static_bg, _, _ = build_ecst_clean_teacher_weight_map(
            self.c2, _batch(1.0, 0.0), teacher_bg, epoch=10, device="cpu",
            return_states=True,
        )
        latent_bg, _, _ = build_ecst_clean_teacher_weight_map(
            self.c2, _batch(0.5, 1.0), teacher_bg, epoch=10, device="cpu",
            return_states=True,
        )
        self.assertLess(float(static_bg.mean()), 1.0)
        self.assertLess(float(latent_bg.mean()), 1.0)

        teacher_fg = torch.ones((1, 1, 68, 68))
        fg_without_latent, _, _ = build_ecst_clean_teacher_weight_map(
            self.c2, _batch(0.0, 0.0), teacher_fg, epoch=10, device="cpu",
            return_states=True,
        )
        fg_with_latent, _, _ = build_ecst_clean_teacher_weight_map(
            self.c2, _batch(0.0, 1.0), teacher_fg, epoch=10, device="cpu",
            return_states=True,
        )
        self.assertTrue(torch.equal(fg_without_latent, fg_with_latent))
        self.assertLess(float(fg_without_latent.mean()), 1.0)

    def test_c3_map_matches_frozen_v5_formula(self):
        generator = torch.Generator().manual_seed(2027)
        batch = _batch()
        batch["dabe_clean_target_68"] = torch.rand(
            (2, 1, 68, 68), generator=generator
        )
        batch["dabe_clean_recoverability_68"] = torch.rand(
            (2, 1, 68, 68), generator=generator
        )
        batch["dabe_clean_semantic_fg_tendency_37"] = torch.rand(
            (2, 1, 37, 37), generator=generator
        )
        teacher = torch.rand((2, 1, 68, 68), generator=generator)
        mean = torch.rand((2, 1, 68, 68), generator=generator)
        variance = 0.01 * torch.rand((2, 1, 68, 68), generator=generator)
        second = mean.square() + variance
        count = torch.tensor([3, 7])
        actual, _, _ = build_ecst_clean_teacher_weight_map(
            self.c3,
            batch,
            teacher,
            mean,
            second,
            count,
            10,
            "cpu",
            True,
        )
        expected = _legacy_v5_reference(
            self.c3, batch, teacher, mean, second, count, 10
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertLessEqual(float((actual - expected).abs().max()), 1e-6)

        explicit = load_config(C3_CONFIG)
        explicit.ECST_CLEAN_USE_HISTORY = True
        explicit_map, _, _ = build_ecst_clean_teacher_weight_map(
            explicit,
            batch,
            teacher,
            mean,
            second,
            count,
            10,
            "cpu",
            True,
        )
        self.assertTrue(torch.equal(actual, explicit_map))

    def test_c2_logging_and_accumulator_do_not_fabricate_history(self):
        route, stats, _ = build_ecst_clean_teacher_weight_map(
            self.c2,
            _batch(0.75, 1.0),
            torch.zeros((1, 1, 68, 68)),
            epoch=10,
            device="cpu",
            return_states=True,
        )
        logger = _Logger()
        log_ecst_clean_first_batch(
            logger,
            self.c2,
            10,
            torch.tensor([0]),
            stats,
            torch.zeros_like(route),
            torch.zeros_like(route),
        )
        text = "\n".join(logger.lines)
        self.assertIn("history=disabled", text)
        self.assertIn("history_fields_consumed=False", text)
        for forbidden in (
            "history_count_mean",
            "history_valid_ratio",
            "history_bg_reliability",
            "temporal_mean",
            "temporal_variance",
        ):
            self.assertNotIn(forbidden, text)
        accumulator = new_ecst_epoch_accumulator()
        accumulate_ecst_epoch(
            accumulator,
            stats,
            torch.zeros_like(route),
            torch.zeros_like(route),
        )
        self.assertEqual(accumulator["memory_active_batches"], 0)
        self.assertEqual(accumulator["history_count_mean_sum"], 0.0)
        self.assertEqual(accumulator["clean_latent_support_sum"], 1.0)
        self.assertEqual(accumulator["clean_latent_effective_sum"], 1.0)

    def test_checkpoint_lifecycle_excludes_c2_memory(self):
        c2_extra = build_clean_ecst_checkpoint_extra(self.c2, None)
        self.assertFalse(c2_extra["temporal_memory_initialized"])
        self.assertFalse(c2_extra["temporal_memory_fetched"])
        self.assertFalse(c2_extra["temporal_memory_updated"])
        self.assertFalse(c2_extra["temporal_memory_saved"])
        self.assertFalse(c2_extra["history_fields_consumed"])
        self.assertNotIn("clean_ecst_temporal_memory", c2_extra)

        memory = TemporalTeacherMemory(1, 68, 68, dtype="float16")
        memory.update(
            torch.tensor([0]),
            torch.full((1, 1, 68, 68), 0.25),
            rho=0.9,
        )
        c3_extra = build_clean_ecst_checkpoint_extra(self.c3, memory)
        self.assertTrue(c3_extra["clean_ecst_history_enabled"])
        self.assertTrue(c3_extra["temporal_memory_saved"])
        self.assertIn("clean_ecst_temporal_memory", c3_extra)
        restored = TemporalTeacherMemory(1, 68, 68, dtype="float16")
        restored.load_state_dict(c3_extra["clean_ecst_temporal_memory"])
        self.assertTrue(torch.equal(restored.mean, memory.mean))
        self.assertTrue(torch.equal(restored.second, memory.second))
        self.assertTrue(torch.equal(restored.count, memory.count))


if __name__ == "__main__":
    unittest.main()
