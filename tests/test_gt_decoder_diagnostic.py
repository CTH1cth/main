import unittest

import torch

from common.dabev2hard_static_only import validate_dabev2hard_static_only_config
from common.dataset import build_gt_hard_37, build_gt_hard_68
from common.r1hard_linear_pure_student import is_r1hard_linear_pure_student_config
from common.teacher_routing import validate_teacher_routing_config
from common.utils import load_config
from train import (
    decoder_supervision_target_37,
    gt68_cached_linear_target,
    use_gt68_cached_linear_supervision,
    use_gt_decoder_supervision,
)


GT_CONFIGS = (
    "configs/dinov1_s8_gt37_last4_linear_b16.py",
    "configs/dinov1_s8_gt37_hsd_v1_sem_triple_b16.py",
)
GT68_CACHED_LINEAR_CONFIG = "configs/dinov1_s8_gt68_linear_cached_b16.py"


class GTDecoderDiagnosticTest(unittest.TestCase):
    def test_gt_is_thresholded_then_nearest_resized_to_37(self):
        gt = torch.tensor([[[0.1, 0.9], [0.6, 0.4]]], dtype=torch.float32)
        target = build_gt_hard_37(gt)
        self.assertEqual(list(target.shape), [1, 37, 37])
        self.assertTrue(((target == 0.0) | (target == 1.0)).all().item())
        self.assertEqual(float(target[0, 0, 0]), 0.0)
        self.assertEqual(float(target[0, 0, -1]), 1.0)
        self.assertEqual(float(target[0, -1, 0]), 1.0)
        self.assertEqual(float(target[0, -1, -1]), 0.0)

    def test_gt_is_thresholded_then_nearest_resized_to_68(self):
        gt = torch.tensor([[[0.1, 0.9], [0.6, 0.4]]], dtype=torch.float32)
        target = build_gt_hard_68(gt)
        self.assertEqual(list(target.shape), [1, 68, 68])
        self.assertTrue(((target == 0.0) | (target == 1.0)).all().item())
        self.assertEqual(float(target[0, 0, 0]), 0.0)
        self.assertEqual(float(target[0, 0, -1]), 1.0)
        self.assertEqual(float(target[0, -1, 0]), 1.0)
        self.assertEqual(float(target[0, -1, -1]), 0.0)

    def test_gt_source_replaces_r1_even_when_both_are_present(self):
        cfg = load_config(GT_CONFIGS[0])
        gt = torch.zeros(2, 1, 37, 37)
        r1 = torch.ones(2, 1, 37, 37)
        target, field = decoder_supervision_target_37(
            cfg, {"gt_hard_37": gt, "r1_hard_37": r1}, torch.device("cpu")
        )
        self.assertEqual(field, "gt_hard_37")
        self.assertTrue(torch.equal(target, gt))

    def test_gt_configs_keep_pure_student_protocol(self):
        for path in GT_CONFIGS:
            with self.subTest(path=path):
                cfg = load_config(path)
                self.assertTrue(use_gt_decoder_supervision(cfg))
                self.assertTrue(is_r1hard_linear_pure_student_config(cfg))
                self.assertEqual(validate_teacher_routing_config(cfg), "none")
                audit = validate_dabev2hard_static_only_config(cfg)
                self.assertEqual(audit["status"], "PASS")
                self.assertTrue(audit["contract"]["gt_diagnostic"])
                self.assertFalse(audit["contract"]["teacher_instantiated"])
                self.assertEqual(
                    {record["field"] for record in audit["gt_pair_differences"]},
                    {
                        "EXP_NAME",
                        "GT_DIAGNOSTIC_SUPERVISION",
                        "GT_DIAGNOSTIC_REFERENCE_CONFIG",
                        "DECODER_SUPERVISION_SOURCE",
                        "GT_DIAGNOSTIC_RESIZE_MODE",
                        "GT_DIAGNOSTIC_STRICT_BINARY",
                    },
                )
                self.assertEqual(cfg.BATCH_SIZE, 16)
                self.assertEqual(cfg.SEED, 42)

    def test_gt68_cached_linear_uses_only_gt_on_the_legacy_loss_grid(self):
        cfg = load_config(GT68_CACHED_LINEAR_CONFIG)
        gt = torch.zeros(2, 1, 68, 68)
        gt[:, :, 10:20, 12:24] = 1.0
        target = gt68_cached_linear_target(
            cfg,
            {
                "gt_hard_68": gt,
                "dabe_clean_static_target_68": torch.ones_like(gt),
            },
            torch.device("cpu"),
        )
        self.assertTrue(torch.equal(target, gt))
        self.assertTrue(use_gt68_cached_linear_supervision(cfg))
        self.assertFalse(use_gt_decoder_supervision(cfg))
        self.assertTrue(is_r1hard_linear_pure_student_config(cfg))
        self.assertEqual(validate_teacher_routing_config(cfg), "none")
        audit = validate_dabev2hard_static_only_config(cfg)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(
            audit["contract"]["feature_source"],
            "legacy_single_layer_feature_cache",
        )
        self.assertEqual(audit["contract"]["student"], "single_1x1_conv")
        self.assertFalse(audit["contract"]["online_dino_forward"])
        self.assertFalse(hasattr(cfg, "USE_MULTI_LEVEL_FEATURE"))
        self.assertFalse(hasattr(cfg, "ONLINE_DINO_LAST4"))
        self.assertEqual(cfg.LOSS_SIZE, 68)
        self.assertEqual(cfg.BATCH_SIZE, 16)
        self.assertEqual(cfg.SEED, 42)


if __name__ == "__main__":
    unittest.main()
