import unittest

import torch
from torch import nn

from common.utils import load_config
from models.bcrd import BCRDSemV1Head
from train import build_r1_decoder_isolation_loss


def features(device="cpu", channels=8):
    return {
        key: torch.randn(1, channels, 37, 37, device=device)
        for key in ("f9", "f10", "f11", "f12")
    }


class BCRDSemTest(unittest.TestCase):
    def test_zero_init_is_bit_exact_base(self):
        model = BCRDSemV1Head(in_channels=8, dim=8, gn_groups=4)
        output = model(features())
        self.assertTrue(torch.equal(output["final_logits_37"], output["base_logits_37"]))
        for head in model.proposal_heads.values():
            self.assertTrue(torch.count_nonzero(head[-1].weight) == 0)
            self.assertTrue(torch.count_nonzero(head[-1].bias) == 0)

    def test_base_uncertainty_gate_extremes(self):
        model = BCRDSemV1Head(in_channels=8, dim=8, gn_groups=4)
        with torch.no_grad():
            model.base_head.weight.zero_()
            model.base_head.bias.fill_(12.0)
        confident = model(features())["base_uncertainty"].max().item()
        with torch.no_grad():
            model.base_head.bias.zero_()
        uncertain = model(features())["base_uncertainty"].min().item()
        self.assertLess(confident, 1e-4)
        self.assertGreater(uncertain, 0.9999)

    def test_consistency_gate_drops_on_disagreement(self):
        model = BCRDSemV1Head(in_channels=8, dim=8, gn_groups=4)
        with torch.no_grad():
            for head in model.proposal_heads.values():
                head[-1].weight.zero_()
                head[-1].bias.fill_(1.0)
        consistent = model(features())["consistency_gate"].mean().item()
        with torch.no_grad():
            for bias, key in zip((-1.0, 0.0, 1.0), ("f9", "f10", "f11")):
                model.proposal_heads[key][-1].bias.fill_(bias)
        disagreeing = model(features())["consistency_gate"].mean().item()
        self.assertGreater(consistent, 0.999)
        self.assertLess(disagreeing, consistent)
        self.assertLess(disagreeing, 0.01)

    def test_residual_is_bidirectional(self):
        model = BCRDSemV1Head(in_channels=8, dim=8, gn_groups=4)
        with torch.no_grad():
            model.base_head.weight.zero_()
            model.base_head.bias.zero_()
            for head in model.proposal_heads.values():
                head[-1].weight.zero_()
                head[-1].bias.fill_(1.0)
        positive = model(features())["applied_residual"]
        self.assertGreater(positive.min().item(), 0.0)
        with torch.no_grad():
            for head in model.proposal_heads.values():
                head[-1].bias.fill_(-1.0)
        negative = model(features())["applied_residual"]
        self.assertLess(negative.max().item(), 0.0)

    def test_all_proposal_heads_receive_gradient(self):
        model = BCRDSemV1Head(in_channels=8, dim=8, gn_groups=4)
        target = (torch.rand(1, 1, 37, 37) > 0.5).float()
        output = model(features())
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                output["final_logits_37"], target
            )
        )
        loss.backward()
        for key, head in model.proposal_heads.items():
            self.assertIsNotNone(head[-1].weight.grad, key)
            self.assertGreater(head[-1].weight.grad.abs().sum().item(), 0.0, key)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_single_batch_backward_is_finite(self):
        cfg = load_config("configs/dinov1_s8_r1hard_bcrd_sem_v1_b16.py")
        model = BCRDSemV1Head(in_channels=384, dim=32, gn_groups=4).cuda()
        output = model(features(device="cuda", channels=384))
        target = (torch.rand(1, 1, 37, 37, device="cuda") > 0.5).float()
        loss = build_r1_decoder_isolation_loss(cfg, output, target)["loss"]
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()

