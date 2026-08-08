import unittest

import torch

from tools.audit_decoder_stages import transition_category_counts


class DecoderStageAuditTest(unittest.TestCase):
    def test_gt_conditioned_transition_categories(self):
        source = torch.tensor([[[[0, 0, 1, 1, 1]]]], dtype=torch.bool)
        target = torch.tensor([[[[1, 1, 0, 0, 1]]]], dtype=torch.bool)
        gt = torch.tensor([[[[1, 0, 0, 1, 1]]]], dtype=torch.bool)
        self.assertEqual(
            transition_category_counts(source, target, gt),
            {
                "0_to_1_gt1": 1,
                "0_to_1_gt0": 1,
                "1_to_0_gt0": 1,
                "1_to_0_gt1": 1,
            },
        )

    def test_transition_shapes_must_match(self):
        with self.assertRaises(RuntimeError):
            transition_category_counts(
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1, 1, 3, 3),
                torch.zeros(1, 1, 2, 2),
            )


if __name__ == "__main__":
    unittest.main()

