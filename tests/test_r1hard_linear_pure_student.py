import unittest

import torch
import torch.nn.functional as F

from common.dataset import CachedTrainDataset
from common.dabev2hard_static_only import (
    validate_dabev2hard_static_only_config,
)
from common.r1hard_linear_pure_student import (
    CONFIG_PATH,
    is_r1hard_linear_pure_student_config,
)
from common.utils import load_config, torch_load
from model import SimpleConvSegHead, build_seg_head


class R1HardLinearPureStudentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(CONFIG_PATH)

    def test_config_contract_and_linear_head(self):
        report = validate_dabev2hard_static_only_config(self.cfg)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(is_r1hard_linear_pure_student_config(self.cfg))
        self.assertEqual(self.cfg.HEAD_TYPE, "simple")
        self.assertFalse(self.cfg.FINETUNE_RESET_TEACHER)

        head = build_seg_head(384, self.cfg)
        self.assertIsInstance(head, SimpleConvSegHead)
        convolutions = [
            module for module in head.modules() if isinstance(module, torch.nn.Conv2d)
        ]
        self.assertEqual(len(convolutions), 1)
        self.assertEqual(convolutions[0].kernel_size, (1, 1))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 385)

    def test_cached_target_is_exact_resized_hard_r1(self):
        dataset = CachedTrainDataset(self.cfg, max_samples=1)
        sample = dataset[0]
        key = dataset.keys[0]
        row = dataset.dabe_clean_dabe_v2_map[key]
        payload = torch_load(row["cache_path"], map_location="cpu")
        r1 = payload["residual_pass1_37"].detach().cpu().float()
        expected_soft = F.interpolate(
            r1.unsqueeze(0),
            size=(68, 68),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        expected_hard = (expected_soft > 0.5).float()

        self.assertTrue(
            torch.equal(sample["dabe_clean_dabe_v2_soft_68"], expected_soft)
        )
        self.assertTrue(
            torch.equal(sample["dabe_clean_static_target_68"], expected_hard)
        )
        self.assertTrue(torch.equal(sample["pseudo"], expected_hard))
        self.assertEqual(
            sample["dabe_clean_static_target_source"],
            "independent_dabe_v2_residual_pass1_37_gt_0.5",
        )


if __name__ == "__main__":
    unittest.main()
