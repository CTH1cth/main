import math
import unittest
from types import SimpleNamespace

import torch

from common.ecst import TemporalTeacherMemory
from common.ecst_minimal import (
    build_ecst_minimal_teacher_weight_map,
    build_minimal_support_ring,
    get_ecst_minimal_scale,
)
from common.teacher_routing import validate_teacher_routing_config
from common.utils import load_config


def _cfg(radius=1):
    return SimpleNamespace(
        USE_DABE_CLEAN=True,
        USE_ECST=False,
        USE_ECST_MINIMAL=True,
        DABE_CLEAN_USE_LEGACY_ECST_REGIONS=False,
        TEACHER_ROUTING_MODE="minimal_ecst",
        ECST_MINIMAL_VERSION="evidence_protect_ring_asym_v1",
        ECST_MINIMAL_RING_RADIUS=radius,
        ECST_MINIMAL_CONFLICT_FLOOR=0.20,
        ECST_START_EPOCH=7,
        ECST_RAMP_END_EPOCH=15,
        ECST_STOP_EPOCH=21,
        ECST_WEIGHT_MIN=0.20,
        ECST_MARGIN_TAU=0.05,
        ECST_EXTENT_DINO_LAMBDA=math.log(4.0),
        ECST_EXTENT_BG_WEIGHT_FLOOR=0.25,
        ECST_MIN_HISTORY=3,
        ECST_INSUFFICIENT_HISTORY_MODE="dino_only",
        ECST_VARIANCE_TAU=0.02,
        ECST_CONF_GAMMA=1.0,
    )


def _batch():
    foreground = torch.zeros(1, 1, 37, 37)
    foreground[:, :, 15:22, 15:22] = 1.0
    background = 1.0 - foreground
    target = torch.zeros_like(foreground)
    x = torch.linspace(-1.0, 1.0, 37).view(1, 1, 1, 37)
    feature = x.expand(1, 384, 37, 37).clone()
    return {
        "dabe_clean_target_37": target,
        "dabe_clean_fg_evidence_37": foreground,
        "dabe_clean_bg_evidence_37": background,
        "feature": feature,
    }


class ECSTMinimalTest(unittest.TestCase):
    def test_ring_radius_and_schedule_contract(self):
        foreground = _batch()["dabe_clean_fg_evidence_37"]
        support1, ring1 = build_minimal_support_ring(foreground, 1)
        support2, ring2 = build_minimal_support_ring(foreground, 2)
        self.assertTrue(torch.equal(support1, support2))
        self.assertFalse(bool((support1 & ring1).any()))
        self.assertGreater(int(ring2.sum()), int(ring1.sum()))
        cfg = _cfg()
        self.assertEqual(
            [
                get_ecst_minimal_scale(cfg, epoch)
                for epoch in (6, 7, 15, 16, 20, 21)
            ],
            [0.0, 1.0 / 9.0, 1.0, 1.0, 1.0, 0.0],
        )

    def test_minimal_priority_detach_and_past_only_memory(self):
        cfg = _cfg()
        batch = _batch()
        teacher = torch.full((1, 1, 68, 68), 0.1)
        teacher[:, :, 25:43, 25:34] = 0.9
        memory = TemporalTeacherMemory(1, 68, 68, dtype="float32")
        indices = torch.tensor([0])
        mean0, second0, count0 = memory.fetch(indices, "cpu")
        route0, stats0, states0 = build_ecst_minimal_teacher_weight_map(
            cfg,
            batch,
            teacher,
            mean0,
            second0,
            count0,
            15,
            "cpu",
            return_states=True,
        )
        ring_fg = states0["ring"] & states0["teacher_fg_37"]
        ring_bg = states0["ring"] & states0["teacher_bg_37"]
        self.assertTrue(bool(ring_fg.any()) and bool(ring_bg.any()))
        self.assertTrue(
            torch.equal(
                states0["raw_37"][ring_fg],
                torch.ones_like(states0["raw_37"][ring_fg]),
            )
        )
        self.assertTrue(
            torch.equal(
                states0["raw_37"][ring_bg],
                states0["negative_verified_weight"][ring_bg],
            )
        )
        self.assertFalse(route0.requires_grad)
        self.assertTrue(
            all(
                not value.requires_grad
                for value in states0.values()
                if torch.is_tensor(value)
            )
        )
        self.assertEqual(stats0["history_valid_ratio"], 0.0)

        route_before_update = route0.clone()
        for _ in range(3):
            memory.update(indices, teacher, rho=0.9)
        self.assertTrue(torch.equal(route0, route_before_update))
        mean3, second3, count3 = memory.fetch(indices, "cpu")
        route3, stats3 = build_ecst_minimal_teacher_weight_map(
            cfg, batch, teacher, mean3, second3, count3, 15, "cpu"
        )
        self.assertEqual(stats3["history_valid_ratio"], 1.0)
        self.assertFalse(torch.equal(route0, route3))

    def test_minimal_config_r2_changes_only_radius(self):
        r1 = load_config(
            "configs/dinov1_s8_dabe_clean_v1_dp_minimal_ecst_r1_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        r2 = load_config(
            "configs/dinov1_s8_dabe_clean_v1_dp_minimal_ecst_r2_dagp_uncgate_ndr_long45_lrfloor_2e5.py"
        )
        self.assertEqual(validate_teacher_routing_config(r1), "minimal_ecst")
        self.assertEqual(validate_teacher_routing_config(r2), "minimal_ecst")
        values1 = {
            key: value for key, value in vars(r1).items() if not key.startswith("__")
        }
        values2 = {
            key: value for key, value in vars(r2).items() if not key.startswith("__")
        }
        differences = {key for key in values1 if values1[key] != values2[key]}
        self.assertEqual(differences, {"EXP_NAME", "ECST_MINIMAL_RING_RADIUS"})


if __name__ == "__main__":
    unittest.main()
