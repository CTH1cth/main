import math
import unittest
from types import SimpleNamespace

import torch

from common.ecst import (
    build_asymmetric_negative_verified_weight,
    build_ecst_teacher_weight_map,
    build_past_only_temporal_reliability,
)


def _cfg():
    return SimpleNamespace(
        USE_ECST=True,
        ECST_START_EPOCH=7,
        ECST_RAMP_END_EPOCH=15,
        ECST_STOP_EPOCH=21,
        ECST_MIN_HISTORY=3,
        ECST_INSUFFICIENT_HISTORY_MODE="dino_only",
        ECST_VARIANCE_TAU=0.02,
        ECST_CONF_GAMMA=1.0,
        ECST_MARGIN_TAU=0.05,
        ECST_EXTENT_DINO_LAMBDA=math.log(4.0),
        ECST_EXTENT_BG_WEIGHT_FLOOR=0.25,
        ECST_UNKNOWN_WEIGHT=0.50,
        ECST_EXTENT_FG_WEIGHT=1.0,
        ECST_CORE_CONFLICT_WEIGHT=0.20,
        ECST_WEIGHT_MIN=0.20,
        ECST_WEIGHT_MAX=1.0,
        ECST_FEATURE_SIZE=37,
        ECST_DETACH_PROTO=True,
        ECST_DETACH_MASK=True,
        ECST_MARGIN_FG_LIKE=0.05,
        ECST_MARGIN_BG_LIKE=-0.05,
    )


def _batch():
    fg = torch.zeros(1, 1, 68, 68)
    bg = torch.zeros_like(fg)
    extent = torch.zeros_like(fg)
    unknown = torch.zeros_like(fg)
    fg[:, :, 10:25, 10:25] = 1
    bg[:, :, 40:58, 40:58] = 1
    extent[:, :, 20:48, 20:48] = 1
    unknown[:, :, :7, :] = 1
    x = torch.linspace(-1.0, 1.0, 37).view(1, 1, 1, 37)
    feature = x.expand(1, 384, 37, 37).clone()
    return {
        "feature": feature,
        "pu_fg_core": fg,
        "pu_bg_core": bg,
        "pu_extent": extent,
        "pu_unknown": unknown,
        "legacy_ecst_fg_core": fg.clone(),
        "legacy_ecst_bg_core": bg.clone(),
        "legacy_ecst_extent": extent.clone(),
        "legacy_ecst_unknown": unknown.clone(),
    }


class ECSTSharedRegressionTest(unittest.TestCase):
    def test_shared_helpers_match_original_equations(self):
        cfg = _cfg()
        mean = torch.tensor([[[[0.1, 0.4]]]], dtype=torch.float32)
        second = torch.tensor([[[[0.02, 0.17]]]], dtype=torch.float32)
        temporal = build_past_only_temporal_reliability(
            cfg, mean, second, torch.tensor([3])
        )
        variance = (second - mean.square()).clamp(0.0, 0.25)
        expected = torch.relu(2.0 * (0.5 - mean)) * torch.exp(-variance / 0.02)
        self.assertTrue(torch.allclose(temporal["bg_reliability"], expected))

        margin = torch.tensor([[[[-0.1, 0.2]]]], dtype=torch.float32)
        negative = build_asymmetric_negative_verified_weight(
            cfg, margin, temporal["bg_reliability"]
        )
        tendency = torch.sigmoid(margin / 0.05)
        ceiling = torch.exp(-math.log(4.0) * tendency).clamp(0.25, 1.0)
        expected_negative = 0.25 + expected * (ceiling - 0.25)
        self.assertTrue(
            torch.allclose(
                negative["negative_verified_weight"], expected_negative
            )
        )
        self.assertFalse(negative["negative_verified_weight"].requires_grad)

    def test_default_and_namespaced_regions_are_bit_exact(self):
        cfg = _cfg()
        batch = _batch()
        teacher = torch.full((1, 1, 68, 68), 0.2)
        teacher[:, :, 12:23, 12:23] = 0.8
        mean = torch.full_like(teacher, 0.2)
        second = mean.square()
        count = torch.tensor([4])
        default_map, default_stats = build_ecst_teacher_weight_map(
            cfg, batch, teacher, mean, second, count, 15, "cpu"
        )
        explicit_map, explicit_stats = build_ecst_teacher_weight_map(
            cfg,
            batch,
            teacher,
            mean,
            second,
            count,
            15,
            "cpu",
            region_key_prefix="pu",
        )
        legacy_map, legacy_stats = build_ecst_teacher_weight_map(
            cfg,
            batch,
            teacher,
            mean,
            second,
            count,
            15,
            "cpu",
            region_key_prefix="legacy_ecst",
        )
        self.assertTrue(torch.equal(default_map, explicit_map))
        self.assertTrue(torch.equal(default_map, legacy_map))
        self.assertEqual(
            default_stats["state_means"], explicit_stats["state_means"]
        )
        self.assertEqual(
            default_stats["state_means"], legacy_stats["state_means"]
        )


if __name__ == "__main__":
    unittest.main()
