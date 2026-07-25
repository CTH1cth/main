import copy
import unittest
from types import SimpleNamespace

import torch

from common.ecst import TemporalTeacherMemory
from common.ecst_clean import (
    ECST_CLEAN_VERSION,
    ECST_CLEAN_V2_VERSION,
    ECST_CLEAN_V3_VERSION,
    ECST_CLEAN_V4_VERSION,
    ECST_CLEAN_V5_VERSION,
    apply_directional_strength,
    apply_ecst_clean_strength,
    build_clean_history_bg_reliability,
    build_clean_negative_weight,
    build_clean_semantic_fg_tendency,
    build_clean_support_ring,
    build_ecst_clean_teacher_weight_map,
    compose_ecst_clean_directional_map,
    get_ecst_clean_scale,
    validate_ecst_clean_config,
)
from common.teacher_routing import validate_teacher_routing_config
from common.utils import load_config
from train import teacher_route_bce_with_logits, teacher_routing_apply_flag


def _cfg():
    return SimpleNamespace(
        TEACHER_ROUTING_MODE="clean_ecst",
        USE_ECST=False,
        USE_ECST_MINIMAL=False,
        USE_ECST_CLEAN=True,
        USE_DABE_CLEAN=True,
        USE_DABE_PU=False,
        DABE_PU_ROOT="",
        DABE_CLEAN_TARGET_MODE="dp",
        DABE_CLEAN_USE_LEGACY_ECST_REGIONS=False,
        DABE_CLEAN_LEGACY_ROUTING_ONLY=False,
        DABE_CLEAN_LEGACY_REGION_ROOT="",
        DABE_CLEAN_STATIC_WEIGHT_MODE="ones",
        ECST_CLEAN_VERSION=ECST_CLEAN_VERSION,
        ECST_CLEAN_START_EPOCH=7,
        ECST_CLEAN_RAMP_END_EPOCH=15,
        ECST_CLEAN_STOP_EPOCH=21,
        ECST_CLEAN_MEMORY_UPDATE_START_EPOCH=1,
        ECST_CLEAN_MEMORY_UPDATE_END_EPOCH=20,
        ECST_CLEAN_TEMPORAL_RHO=0.90,
        ECST_CLEAN_MIN_HISTORY=3,
        ECST_CLEAN_VARIANCE_TAU=0.02,
        ECST_CLEAN_USE_PREUPDATE_STATS=True,
        ECST_CLEAN_MEMORY_DTYPE="float16",
        ECST_CLEAN_RESET_MEMORY_AT_FINETUNE_RESET=True,
        ECST_CLEAN_RING_RADIUS=1,
        ECST_CLEAN_CONFLICT_FLOOR=0.20,
        ECST_CLEAN_NEGATIVE_WEIGHT_FLOOR=0.25,
        ECST_CLEAN_MARGIN_TAU=0.05,
        ECST_CLEAN_WEIGHT_MIN=0.20,
        ECST_CLEAN_WEIGHT_MAX=1.00,
        ECST_CLEAN_APPLY_TO_FINAL=True,
        ECST_CLEAN_APPLY_TO_COARSE_AUX=True,
        ECST_CLEAN_APPLY_TO_BASE_AUX=True,
        MAX_EPOCH=45,
        STOP_AFTER_EPOCH=0,
        SAVE_EVERY_EPOCH=True,
        SAVE_INTERVAL=1,
    )


def _v2_cfg():
    cfg = copy.copy(_cfg())
    cfg.ECST_CLEAN_VERSION = ECST_CLEAN_V2_VERSION
    cfg.ECST_CLEAN_START_EPOCH = 5
    cfg.ECST_CLEAN_RAMP_END_EPOCH = 10
    cfg.ECST_CLEAN_RING_RADIUS = 2
    cfg.ECST_CLEAN_STRENGTH_MAX = 1.5
    return cfg


def _v3_cfg():
    cfg = copy.copy(_v2_cfg())
    cfg.ECST_CLEAN_VERSION = ECST_CLEAN_V3_VERSION
    cfg.ECST_CLEAN_START_EPOCH = 2
    cfg.ECST_CLEAN_RAMP_END_EPOCH = 2
    return cfg


def _v4_cfg():
    cfg = copy.copy(_v2_cfg())
    cfg.ECST_CLEAN_VERSION = ECST_CLEAN_V4_VERSION
    cfg.ECST_CLEAN_STRENGTH_MODE = "directional"
    cfg.ECST_CLEAN_ERASE_STRENGTH = 2.5
    cfg.ECST_CLEAN_ADD_STRENGTH = 1.0
    cfg.ECST_CLEAN_RING_BG_STRENGTH = 2.5
    delattr(cfg, "ECST_CLEAN_STRENGTH_MAX")
    return cfg


def _v5_cfg():
    cfg = copy.copy(_v4_cfg())
    cfg.ECST_CLEAN_VERSION = ECST_CLEAN_V5_VERSION
    cfg.ECST_CLEAN_STRENGTH_MODE = "directional_continuous"
    cfg.ECST_CLEAN_RECOVERY_STRENGTH = 2.5
    cfg.ECST_CLEAN_USE_HARD_RING = False
    cfg.ECST_CLEAN_RECOVERY_MODE = "latent_rw_geometric_mean"
    delattr(cfg, "ECST_CLEAN_RING_RADIUS")
    delattr(cfg, "ECST_CLEAN_RING_BG_STRENGTH")
    return cfg


def _batch():
    target = torch.full((1, 1, 68, 68), 0.5)
    foreground_68 = torch.zeros_like(target)
    foreground_68[:, :, 30:38, 30:38] = 1.0
    background_68 = 1.0 - foreground_68

    foreground_37 = torch.zeros(1, 1, 37, 37)
    background_37 = torch.zeros_like(foreground_37)
    foreground_37[:, :, :, 24:] = 1.0
    background_37[:, :, :, :13] = 1.0
    x = torch.linspace(-1.0, 1.0, 37).view(1, 1, 1, 37)
    feature = torch.zeros(1, 384, 37, 37)
    feature[:, 0:1] = x
    feature[:, 1:2] = 1.0
    return {
        "dabe_clean_target_68": target,
        "dabe_clean_fg_evidence_37": foreground_37,
        "dabe_clean_fg_evidence_68": foreground_68,
        "dabe_clean_bg_evidence_37": background_37,
        "dabe_clean_bg_evidence_68": background_68,
        "feature": feature,
        "sample_index": torch.tensor([0]),
        "dataset_name": ["synthetic"],
        "stem": ["sample"],
    }


def _v5_batch(target=0.5, recoverability=0.0):
    batch = _batch()
    batch["dabe_clean_target_68"] = torch.full((1, 1, 68, 68), float(target))
    batch["dabe_clean_recoverability_68"] = torch.full(
        (1, 1, 68, 68), float(recoverability)
    )
    batch["dabe_clean_semantic_fg_tendency_37"] = torch.full(
        (1, 1, 37, 37), 0.5
    )
    return batch


def _build(cfg=None, batch=None, teacher=None, history_count=None, epoch=15):
    cfg = cfg or _cfg()
    batch = batch or _batch()
    teacher = teacher if teacher is not None else torch.full((1, 1, 68, 68), 0.1)
    count = (
        history_count
        if history_count is not None
        else torch.zeros((teacher.shape[0],), dtype=torch.long)
    )
    zeros = torch.zeros_like(teacher)
    return build_ecst_clean_teacher_weight_map(
        cfg=cfg,
        batch=batch,
        teacher_prob=teacher,
        temporal_mean=zeros,
        temporal_second=zeros,
        history_count=count,
        epoch=epoch,
        device="cpu",
        return_states=True,
    )


class ECSTCleanTest(unittest.TestCase):
    def test_config_contract_and_actual_config(self):
        self.assertTrue(validate_ecst_clean_config(_cfg()))
        self.assertEqual(validate_teacher_routing_config(_cfg()), "clean_ecst")
        actual = load_config(
            "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_v1_r1_"
            "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        self.assertEqual(validate_teacher_routing_config(actual), "clean_ecst")
        self.assertFalse(actual.USE_DABE_PU)
        self.assertEqual(actual.DABE_PU_ROOT, "")
        self.assertFalse(actual.DABE_CLEAN_USE_LEGACY_ECST_REGIONS)
        self.assertEqual(actual.DABE_CLEAN_LEGACY_REGION_ROOT, "")
        actual_v2 = load_config(
            "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_v2_r2_s15_"
            "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        self.assertEqual(validate_teacher_routing_config(actual_v2), "clean_ecst")
        self.assertEqual(actual_v2.ECST_CLEAN_START_EPOCH, 5)
        self.assertEqual(actual_v2.ECST_CLEAN_RAMP_END_EPOCH, 10)
        self.assertEqual(actual_v2.ECST_CLEAN_RING_RADIUS, 2)
        self.assertEqual(actual_v2.ECST_CLEAN_STRENGTH_MAX, 1.5)
        actual_v3 = load_config(
            "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_v3_"
            "instantfull_r2_s15_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        self.assertEqual(validate_teacher_routing_config(actual_v3), "clean_ecst")
        self.assertEqual(actual_v3.ECST_CLEAN_START_EPOCH, 2)
        self.assertEqual(actual_v3.ECST_CLEAN_RAMP_END_EPOCH, 2)
        self.assertEqual(actual_v3.ECST_CLEAN_RING_RADIUS, 2)
        self.assertEqual(actual_v3.ECST_CLEAN_STRENGTH_MAX, 1.5)
        actual_v4 = load_config(
            "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_v4_asym_"
            "e25_a10_r25_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        self.assertEqual(validate_teacher_routing_config(actual_v4), "clean_ecst")
        self.assertEqual(actual_v4.ECST_CLEAN_STRENGTH_MODE, "directional")
        self.assertEqual(actual_v4.ECST_CLEAN_ERASE_STRENGTH, 2.5)
        self.assertEqual(actual_v4.ECST_CLEAN_ADD_STRENGTH, 1.0)
        self.assertEqual(actual_v4.ECST_CLEAN_RING_BG_STRENGTH, 2.5)
        self.assertFalse(hasattr(actual_v4, "ECST_CLEAN_STRENGTH_MAX"))
        actual_v5 = load_config(
            "configs/dinov1_s8_dabe_clean_v1_dp_clean_ecst_v5_ab_contrec_"
            "e25_a10_c25_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        self.assertEqual(validate_teacher_routing_config(actual_v5), "clean_ecst")
        self.assertEqual(
            actual_v5.ECST_CLEAN_STRENGTH_MODE, "directional_continuous"
        )
        self.assertFalse(actual_v5.ECST_CLEAN_USE_HARD_RING)
        self.assertFalse(hasattr(actual_v5, "ECST_CLEAN_RING_RADIUS"))
        self.assertEqual(actual_v5.ECST_CLEAN_ERASE_STRENGTH, 2.5)
        self.assertEqual(actual_v5.ECST_CLEAN_ADD_STRENGTH, 1.0)
        self.assertEqual(actual_v5.ECST_CLEAN_RECOVERY_STRENGTH, 2.5)

    def test_config_rejects_every_legacy_region_entry(self):
        for field, value in (
            ("DABE_CLEAN_USE_LEGACY_ECST_REGIONS", True),
            ("DABE_CLEAN_LEGACY_ROUTING_ONLY", True),
            ("DABE_CLEAN_LEGACY_REGION_ROOT", "/forbidden"),
            ("DABE_PU_ROOT", "/forbidden"),
        ):
            cfg = copy.copy(_cfg())
            setattr(cfg, field, value)
            with self.assertRaises(RuntimeError, msg=field):
                validate_ecst_clean_config(cfg)

    def test_clean_batch_rejects_legacy_and_static_weight_maps(self):
        for forbidden_key in (
            "legacy_ecst_fg_core",
            "fg_core_pu_68",
            "weight_map",
            "pu_weight_map",
            "dabe_clean_static_weight_map",
        ):
            batch = _batch()
            batch[forbidden_key] = torch.ones(1, 1, 68, 68)
            with self.assertRaises(RuntimeError, msg=forbidden_key):
                _build(batch=batch)
        clean_batch = _batch()
        self.assertFalse(
            any("weight_map" in key or key.startswith("legacy_ecst_") for key in clean_batch)
        )

    def test_support_ring_radius_one_is_exact_and_disjoint(self):
        evidence = torch.zeros(1, 1, 7, 7)
        evidence[:, :, 3, 3] = 1.0
        support, ring = build_clean_support_ring(evidence, radius=1)
        self.assertEqual(int(support.sum()), 1)
        self.assertEqual(int(ring.sum()), 8)
        self.assertFalse(bool((support & ring).any()))
        self.assertTrue(bool(ring[0, 0, 2:5, 2:5].sum() == 8))

    def test_support_ring_radius_two_is_exact_and_disjoint(self):
        evidence = torch.zeros(1, 1, 9, 9)
        evidence[:, :, 4, 4] = 1.0
        support, ring = build_clean_support_ring(evidence, radius=2)
        self.assertEqual(int(support.sum()), 1)
        self.assertEqual(int(ring.sum()), 24)
        self.assertFalse(bool((support & ring).any()))
        self.assertEqual(int(ring[0, 0, 2:7, 2:7].sum()), 24)

    def test_schedule_preserves_the_existing_ecst_ramp(self):
        cfg = _cfg()
        actual = [get_ecst_clean_scale(cfg, epoch) for epoch in (6, 7, 14, 15, 20, 21)]
        self.assertEqual(actual, [0.0, 1.0 / 9.0, 8.0 / 9.0, 1.0, 1.0, 0.0])

    def test_v2_schedule_starts_at_five_full_at_ten_and_stops_at_twenty_one(self):
        cfg = _v2_cfg()
        actual = [
            get_ecst_clean_scale(cfg, epoch)
            for epoch in (4, 5, 7, 10, 15, 20, 21)
        ]
        self.assertEqual(actual, [0.0, 1.0 / 6.0, 0.5, 1.0, 1.0, 1.0, 0.0])

    def test_v3_schedule_is_instant_full_from_two_through_twenty(self):
        cfg = _v3_cfg()
        actual = [
            get_ecst_clean_scale(cfg, epoch)
            for epoch in (1, 2, 3, 10, 20, 21)
        ]
        self.assertEqual(actual, [0.0, 1.0, 1.0, 1.0, 1.0, 0.0])

    def test_v4_schedule_matches_early_ramp_protocol(self):
        cfg = _v4_cfg()
        actual = [
            get_ecst_clean_scale(cfg, epoch)
            for epoch in (4, 5, 7, 10, 15, 20, 21)
        ]
        self.assertEqual(actual, [0.0, 1.0 / 6.0, 0.5, 1.0, 1.0, 1.0, 0.0])

    def test_v4_rejects_the_legacy_unified_strength_field(self):
        cfg = _v4_cfg()
        cfg.ECST_CLEAN_STRENGTH_MAX = 1.5
        with self.assertRaises(RuntimeError):
            validate_ecst_clean_config(cfg)

    def test_strength_scaling_matches_v1_amplifies_v2_and_clamps(self):
        raw = torch.tensor([[[[0.8, 0.4, 0.0, 1.0]]]])
        old_formula = (1.0 + (raw - 1.0)).clamp(0.20, 1.00)
        strength_one = apply_ecst_clean_strength(
            raw, schedule_scale=1.0, strength_max=1.0
        )
        self.assertTrue(torch.equal(strength_one, old_formula))

        strength_15 = apply_ecst_clean_strength(
            raw, schedule_scale=1.0, strength_max=1.5
        )
        self.assertAlmostEqual(float(strength_15[0, 0, 0, 0]), 0.7, places=7)
        self.assertAlmostEqual(
            float(1.0 - strength_15[0, 0, 0, 0]),
            1.5 * float(1.0 - raw[0, 0, 0, 0]),
            places=7,
        )
        self.assertGreaterEqual(float(strength_15.min()), 0.20)
        self.assertLessEqual(float(strength_15.max()), 1.00)
        scale_zero = apply_ecst_clean_strength(
            raw, schedule_scale=0.0, strength_max=1.5
        )
        self.assertTrue(torch.equal(scale_zero, torch.ones_like(scale_zero)))

    def test_directional_strengths_apply_independently(self):
        raw = torch.tensor([[[[0.8]]]])
        erase = apply_directional_strength(raw, 1.0, 2.5, 0.20)
        add = apply_directional_strength(raw, 1.0, 1.0, 0.20)
        ring_bg = apply_directional_strength(raw, 1.0, 2.5, 0.20)
        self.assertAlmostEqual(float(erase), 0.5, places=7)
        self.assertAlmostEqual(float(add), 0.8, places=7)
        self.assertTrue(torch.equal(erase, ring_bg))
        self.assertLess(float(erase), float(add))

    def test_equal_directional_strengths_match_uniform_v2(self):
        conflict_weight = torch.tensor([[[[0.8, 0.6, 1.0, 1.0, 1.0]]]])
        negative_weight = torch.tensor([[[[1.0, 1.0, 0.7, 1.0, 1.0]]]])
        erase = torch.tensor([[[[1, 0, 0, 0, 0]]]], dtype=torch.bool)
        add = torch.tensor([[[[0, 1, 0, 0, 0]]]], dtype=torch.bool)
        ring_bg = torch.tensor([[[[0, 0, 1, 0, 0]]]], dtype=torch.bool)
        ring_fg = torch.tensor([[[[0, 0, 0, 1, 0]]]], dtype=torch.bool)
        raw_map = torch.ones_like(conflict_weight)
        raw_map = torch.where(erase | add, conflict_weight, raw_map)
        raw_map = torch.where(ring_bg, negative_weight, raw_map)
        raw_map = torch.where(ring_fg, torch.ones_like(raw_map), raw_map)
        uniform = apply_ecst_clean_strength(raw_map, 1.0, 1.5)
        directional, _ = compose_ecst_clean_directional_map(
            conflict_weight,
            negative_weight,
            erase,
            add,
            ring_bg,
            ring_fg,
            schedule_scale=1.0,
            erase_strength=1.5,
            add_strength=1.5,
            ring_bg_strength=1.5,
            weight_min=0.20,
        )
        self.assertTrue(torch.equal(directional, uniform))

    def test_v4_regions_do_not_overlap_and_directional_priorities_hold(self):
        cfg = _v4_cfg()
        batch = _batch()
        teacher = torch.full((1, 1, 68, 68), 0.1)
        _, ring = build_clean_support_ring(
            batch["dabe_clean_fg_evidence_68"], radius=2
        )
        ring_points = ring.nonzero(as_tuple=False)
        ring_fg_point = tuple(int(value) for value in ring_points[0])
        teacher[ring_fg_point] = 0.9
        route, stats, states = _build(
            cfg=cfg, batch=batch, teacher=teacher, epoch=10
        )
        directional_masks = (
            states["erase_conflict_effective"],
            states["add_conflict_effective"],
            states["ring_teacher_bg"],
            states["ring_teacher_fg"],
        )
        for left_index, left in enumerate(directional_masks):
            for right in directional_masks[left_index + 1 :]:
                self.assertFalse(bool((left & right).any()))
        self.assertTrue(
            torch.equal(
                route[states["ring_teacher_fg"]],
                torch.ones_like(route[states["ring_teacher_fg"]]),
            )
        )
        self.assertTrue(
            torch.equal(
                route[states["ring_teacher_bg"]],
                states["ring_bg_map"][states["ring_teacher_bg"]],
            )
        )
        self.assertEqual(stats["erase_strength"], 2.5)
        self.assertEqual(stats["add_strength"], 1.0)
        self.assertEqual(stats["ring_bg_strength"], 2.5)
        self.assertGreaterEqual(float(route.min()), 0.20)
        self.assertLessEqual(float(route.max()), 1.00)

    def test_v4_scale_zero_is_identity(self):
        route, stats, _ = _build(cfg=_v4_cfg(), epoch=4)
        self.assertEqual(stats["schedule_scale"], 0.0)
        self.assertTrue(torch.equal(route, torch.ones_like(route)))

    def test_v5_scale_schedule_and_hard_ring_absence(self):
        cfg = _v5_cfg()
        self.assertEqual(
            [get_ecst_clean_scale(cfg, epoch) for epoch in (4, 5, 10, 20, 21)],
            [0.0, 1.0 / 6.0, 1.0, 1.0, 0.0],
        )
        route4, _, states4 = _build(
            cfg=cfg, batch=_v5_batch(0.8, 1.0), epoch=4
        )
        route21, _, states21 = _build(
            cfg=cfg, batch=_v5_batch(0.8, 1.0), epoch=21
        )
        self.assertTrue(torch.equal(route4, torch.ones_like(route4)))
        self.assertTrue(torch.equal(route21, torch.ones_like(route21)))
        self.assertFalse(any("ring" in key for key in states4))
        self.assertFalse(any("ring" in key for key in states21))

    def test_v5_teacher_bg_static_and_recovery_protect_independently(self):
        cfg = _v5_cfg()
        teacher_bg = torch.zeros((1, 1, 68, 68))
        static_route, _, static_states = _build(
            cfg=cfg,
            batch=_v5_batch(target=1.0, recoverability=0.0),
            teacher=teacher_bg,
            epoch=10,
        )
        recovery_route, _, recovery_states = _build(
            cfg=cfg,
            batch=_v5_batch(target=0.5, recoverability=1.0),
            teacher=teacher_bg,
            epoch=10,
        )
        self.assertLess(float(static_route.mean()), 1.0)
        self.assertLess(float(recovery_route.mean()), 1.0)
        self.assertEqual(float(static_states["recovery_suppression"].max()), 0.0)
        self.assertEqual(float(recovery_states["erase_suppression"].max()), 0.0)

    def test_v5_history_bg_reliability_weakens_recovery_protection(self):
        cfg = _v5_cfg()
        batch = _v5_batch(target=0.5, recoverability=1.0)
        teacher_bg = torch.zeros((1, 1, 68, 68))
        route_low_history, _, _ = _build(
            cfg=cfg,
            batch=batch,
            teacher=teacher_bg,
            history_count=torch.zeros(1, dtype=torch.long),
            epoch=10,
        )
        route_reliable_bg, _, _ = _build(
            cfg=cfg,
            batch=batch,
            teacher=teacher_bg,
            history_count=torch.full((1,), 3, dtype=torch.long),
            epoch=10,
        )
        self.assertGreater(
            float(route_reliable_bg.mean()), float(route_low_history.mean())
        )

    def test_v5_teacher_fg_uses_only_negative_static_evidence(self):
        cfg = _v5_cfg()
        teacher_fg = torch.ones((1, 1, 68, 68))
        route_zero, _, _ = _build(
            cfg=cfg,
            batch=_v5_batch(target=0.0, recoverability=0.0),
            teacher=teacher_fg,
            epoch=10,
        )
        route_one, _, _ = _build(
            cfg=cfg,
            batch=_v5_batch(target=0.0, recoverability=1.0),
            teacher=teacher_fg,
            epoch=10,
        )
        self.assertTrue(torch.equal(route_zero, route_one))
        self.assertLess(float(route_zero.mean()), 1.0)

    def test_v5_noisy_or_and_floor_are_exact(self):
        route, _, states = _build(
            cfg=_v5_cfg(),
            batch=_v5_batch(target=0.75, recoverability=1.0),
            teacher=torch.zeros((1, 1, 68, 68)),
            epoch=10,
        )
        erase = states["erase_suppression"]
        recovery = states["recovery_suppression"]
        expected = (1.0 - (1.0 - erase) * (1.0 - recovery)).clamp(0.0, 0.8)
        self.assertTrue(torch.equal(states["teacher_bg_suppression"], expected))
        self.assertTrue(bool((expected >= erase).all()))
        self.assertTrue(bool((expected >= recovery).all()))
        self.assertGreaterEqual(float(route.min()), 0.20)
        self.assertLessEqual(float(route.max()), 1.00)
        self.assertTrue(
            all(
                not value.requires_grad
                for value in states.values()
                if torch.is_tensor(value)
            )
        )

    def test_v2_builder_uses_radius_two_and_strength_15(self):
        cfg = _v2_cfg()
        batch = _batch()
        support1, ring1 = build_clean_support_ring(
            batch["dabe_clean_fg_evidence_68"], radius=1
        )
        route, stats, states = _build(cfg=cfg, batch=batch, epoch=10)
        self.assertTrue(torch.equal(states["support"], support1))
        self.assertGreater(int(states["ring"].sum()), int(ring1.sum()))
        self.assertFalse(bool((states["support"] & states["ring"]).any()))
        self.assertEqual(stats["schedule_scale"], 1.0)
        self.assertEqual(stats["strength_max"], 1.5)
        self.assertEqual(stats["effective_strength"], 1.5)
        self.assertGreaterEqual(float(route.min()), 0.20)
        self.assertLessEqual(float(route.max()), 1.00)

    def test_conflict_endpoints_and_neutral_evidence(self):
        batch = _batch()
        target = batch["dabe_clean_target_68"]
        target[0, 0, 0, 0] = 1.0
        target[0, 0, 0, 1] = 0.0
        target[0, 0, 0, 2] = 0.5
        teacher = torch.full_like(target, 0.1)
        teacher[0, 0, 0, 1] = 0.9
        teacher[0, 0, 0, 2] = 0.9
        route, _, states = _build(batch=batch, teacher=teacher)
        self.assertAlmostEqual(float(route[0, 0, 0, 0]), 0.20, places=6)
        self.assertAlmostEqual(float(route[0, 0, 0, 1]), 0.20, places=6)
        self.assertAlmostEqual(float(route[0, 0, 0, 2]), 1.00, places=6)
        self.assertAlmostEqual(float(states["conflict_weight"][0, 0, 0, 2]), 1.00, places=6)

    def test_ring_priority_teacher_fg_and_teacher_bg(self):
        batch = _batch()
        _, ring = build_clean_support_ring(batch["dabe_clean_fg_evidence_68"], 1)
        points = ring.nonzero(as_tuple=False)
        fg_point = tuple(int(value) for value in points[0])
        bg_point = tuple(int(value) for value in points[1])
        batch["dabe_clean_target_68"][fg_point] = 0.0
        batch["dabe_clean_target_68"][bg_point] = 1.0
        teacher = torch.full((1, 1, 68, 68), 0.1)
        teacher[fg_point] = 0.9
        route, _, states = _build(batch=batch, teacher=teacher)
        self.assertTrue(bool(states["conflict"][fg_point]))
        self.assertTrue(bool(states["conflict"][bg_point]))
        self.assertEqual(float(route[fg_point]), 1.0)
        self.assertAlmostEqual(
            float(route[bg_point]), float(states["negative_weight"][bg_point]), places=7
        )

    def test_semantic_tendency_is_continuous_and_detached(self):
        batch = _batch()
        tendency_37, tendency_68, stats = build_clean_semantic_fg_tendency(
            batch["feature"].requires_grad_(True),
            batch["dabe_clean_fg_evidence_37"].requires_grad_(True),
            batch["dabe_clean_bg_evidence_37"].requires_grad_(True),
            margin_tau=0.05,
        )
        self.assertFalse(tendency_37.requires_grad)
        self.assertFalse(tendency_68.requires_grad)
        self.assertEqual(tuple(tendency_68.shape), (1, 1, 68, 68))
        self.assertGreater(float(tendency_37[:, :, :, 25:].mean()), float(tendency_37[:, :, :, :12].mean()))
        self.assertEqual(stats["fg_prototype_fallback_count"], 0)
        self.assertEqual(stats["bg_prototype_fallback_count"], 0)

    def test_negative_weight_monotonicity_and_stable_background_history(self):
        semantic = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        unreliable = torch.zeros_like(semantic)
        weights = build_clean_negative_weight(semantic, unreliable, 0.25)
        self.assertTrue(bool(weights[0, 0, 0, 0] > weights[0, 0, 0, 1]))
        self.assertTrue(bool(weights[0, 0, 0, 1] > weights[0, 0, 0, 2]))
        self.assertAlmostEqual(float(weights[0, 0, 0, 2]), 0.25, places=7)
        reliable = torch.ones_like(semantic)
        stable_weights = build_clean_negative_weight(semantic, reliable, 0.25)
        self.assertTrue(torch.equal(stable_weights, torch.ones_like(stable_weights)))

        temporal = build_clean_history_bg_reliability(
            history_mean=torch.zeros(2, 1, 1, 1),
            history_second=torch.zeros(2, 1, 1, 1),
            history_count=torch.tensor([2, 3]),
            min_history=3,
            variance_tau=0.02,
        )
        self.assertEqual(float(temporal["history_bg_reliability"][0]), 0.0)
        self.assertEqual(float(temporal["history_bg_reliability"][1]), 1.0)

    def test_all_routing_states_are_detached_and_bounded(self):
        batch = _batch()
        for key in (
            "dabe_clean_target_68",
            "dabe_clean_fg_evidence_37",
            "dabe_clean_fg_evidence_68",
            "dabe_clean_bg_evidence_37",
            "dabe_clean_bg_evidence_68",
            "feature",
        ):
            batch[key].requires_grad_(True)
        teacher = torch.full((1, 1, 68, 68), 0.1, requires_grad=True)
        route, _, states = _build(batch=batch, teacher=teacher)
        self.assertFalse(route.requires_grad)
        self.assertGreaterEqual(float(route.min()), 0.20)
        self.assertLessEqual(float(route.max()), 1.00)
        self.assertTrue(
            all(not value.requires_grad for value in states.values() if torch.is_tensor(value))
        )

    def test_temporal_memory_is_past_only(self):
        cfg = _cfg()
        batch = _batch()
        teacher = torch.full((1, 1, 68, 68), 0.1)
        memory = TemporalTeacherMemory(1, 68, 68, dtype="float16")
        indices = torch.tensor([0])
        mean0, second0, count0 = memory.fetch(indices, "cpu")
        route0, _, _ = build_ecst_clean_teacher_weight_map(
            cfg, batch, teacher, mean0, second0, count0, 15, "cpu", True
        )
        frozen_route = route0.clone()
        for _ in range(3):
            memory.update(indices, teacher, rho=0.9)
        self.assertTrue(torch.equal(route0, frozen_route))
        mean3, second3, count3 = memory.fetch(indices, "cpu")
        route3, stats3, _ = build_ecst_clean_teacher_weight_map(
            cfg, batch, teacher, mean3, second3, count3, 15, "cpu", True
        )
        self.assertEqual(stats3["history_valid_ratio"], 1.0)
        self.assertFalse(torch.equal(route0, route3))

    def test_three_teacher_losses_share_one_route_map(self):
        cfg = _cfg()
        route, stats, _ = _build(cfg=cfg)
        target = torch.zeros_like(route)
        branches = {
            "final": torch.zeros_like(route, requires_grad=True),
            "coarse": torch.ones_like(route, requires_grad=True),
            "base": -torch.ones_like(route, requires_grad=True),
        }
        route_ids = []
        losses = []
        for name, logits in branches.items():
            self.assertTrue(teacher_routing_apply_flag(cfg, name))
            route_ids.append(id(route))
            losses.append(
                teacher_route_bce_with_logits(
                    logits,
                    target,
                    route,
                    cfg,
                    routing_scale=stats["ecst_scale"],
                    apply_to_loss=True,
                )
            )
        self.assertEqual(len(set(route_ids)), 1)
        self.assertTrue(all(bool(torch.isfinite(loss).item()) for loss in losses))


if __name__ == "__main__":
    unittest.main()
