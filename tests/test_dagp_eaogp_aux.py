import unittest

import torch

from model import DAGPSafeHead


class TestDAGPEAOGPAux(unittest.TestCase):
    def test_auxiliary_output_does_not_change_predictions(self):
        torch.manual_seed(3107)
        head = DAGPSafeHead(
            in_channels=384,
            hidden=64,
            topk=12,
            tau=0.07,
            warmup_epoch=6,
            ramp_start_epoch=7,
            ramp_end_epoch=15,
            use_ndr_branch=True,
            ndr_loss_size=68,
        ).eval()
        head.set_epoch(15)
        feature = torch.randn((1, 384, 37, 37))
        image = torch.randn((1, 3, 68, 68))
        with torch.no_grad():
            baseline = head(
                feature,
                image_68=image,
                return_aux=True,
                return_eaogp_aux=False,
            )
            candidate = head(
                feature,
                image_68=image,
                return_aux=True,
                return_eaogp_aux=True,
            )
        for key in ("logits", "coarse_logits_68", "base_logits"):
            self.assertLessEqual(
                float((baseline[key] - candidate[key]).abs().max()),
                1e-6,
                key,
            )
        idx = candidate["eaogp_dino_topk_idx"]
        weight = candidate["eaogp_dino_topk_weight"]
        embedding = candidate["eaogp_teacher_embedding_37"]
        self.assertEqual(tuple(idx.shape), (1, 1369, 12))
        self.assertEqual(tuple(weight.shape), (1, 1369, 12))
        self.assertEqual(tuple(embedding.shape), (1, 64, 37, 37))
        self.assertFalse(weight.requires_grad)
        self.assertFalse(embedding.requires_grad)
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertLess(int(idx.max()), 1369)
        self.assertLessEqual(float((weight.sum(-1) - 1.0).abs().max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
