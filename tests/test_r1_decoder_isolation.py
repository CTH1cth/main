import unittest

import torch
import torch.nn.functional as F
from torch import nn

from common.r1_decoder_isolation import validate_r1_decoder_isolation_config
from common.r1hard_linear_pure_student import (
    is_r1hard_linear_pure_student_config,
    validate_r1hard_linear_pure_student_config,
)
from common.utils import load_config
from eval import infer_in_channels
from model import build_seg_head
from models.hsd import F12ScaleLiftHead, HSDV1Head, Last4LinearProbe
from train import (
    build_hsd_r1_group_loss,
    build_r1_decoder_isolation_loss,
    compute_linear_floor_two_stage_lr,
)


CONFIGS = (
    "configs/dinov1_s8_r1hard_last4_linear_b16.py",
    "configs/dinov1_s8_r1hard_f12_scalelift_triple_b16.py",
    "configs/dinov1_s8_r1hard_f12_scalelift_finalbase_b16.py",
    "configs/dinov1_s8_r1hard_hsd_v1_sem_finalbase_b16.py",
    "configs/dinov1_s8_r1hard_bcrd_sem_v1_b16.py",
)


def binary_target(batch=1):
    return (torch.rand(batch, 1, 37, 37) > 0.5).float()


class R1DecoderIsolationTest(unittest.TestCase):
    def test_static_hsd_low_lr_control_decays_over_full_run(self):
        cfg = load_config(
            "configs/dinov1_s8_r1hard_hsd_v1_sem_ms148_online_lr1e4_linear1e5.py"
        )
        report = validate_r1hard_linear_pure_student_config(cfg)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["contract"]["lr_ablation"])
        num_iters = 253
        first = compute_linear_floor_two_stage_lr(1, 0, num_iters, cfg)
        middle = compute_linear_floor_two_stage_lr(23, 126, num_iters, cfg)
        last = compute_linear_floor_two_stage_lr(45, num_iters - 1, num_iters, cfg)
        self.assertAlmostEqual(first, 1e-4)
        self.assertGreater(first, middle)
        self.assertGreater(middle, last)
        self.assertAlmostEqual(last, 1e-5)

    def test_last4_probe_is_exactly_one_1x1_conv(self):
        model = Last4LinearProbe(384)
        convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        self.assertEqual(len(convolutions), 1)
        self.assertEqual(convolutions[0].kernel_size, (1, 1))
        self.assertFalse(any(isinstance(module, nn.GroupNorm) for module in model.modules()))

    def test_f12_scalelift_only_reads_f12(self):
        model = F12ScaleLiftHead(in_channels=8, channels=8, gn_groups=4)
        output = model(
            {
                "f12": torch.randn(1, 8, 37, 37),
                "f9": object(),
                "f10": object(),
                "f11": object(),
            }
        )
        self.assertEqual(list(output["base_logits_37"].shape), [1, 1, 37, 37])
        self.assertEqual(list(output["coarse_logits_74"].shape), [1, 1, 74, 74])
        self.assertEqual(list(output["final_logits"].shape), [1, 1, 148, 148])

    def test_hsd_final_base_does_not_use_coarse_bce(self):
        cfg = load_config("configs/dinov1_s8_r1hard_hsd_v1_sem_finalbase_b16.py")
        target = binary_target()
        output = {
            "final_logits": torch.randn(1, 1, 148, 148, requires_grad=True),
            "coarse_logits_74": torch.randn(1, 1, 74, 74, requires_grad=True),
            "base_logits_37": torch.randn(1, 1, 37, 37, requires_grad=True),
        }
        result = build_hsd_r1_group_loss(cfg, 1, output, target)
        expected = (
            F.binary_cross_entropy_with_logits(
                output["final_logits"], F.interpolate(target, (148, 148), mode="nearest")
            )
            + 0.25 * F.binary_cross_entropy_with_logits(output["base_logits_37"], target)
        ) / 1.25
        self.assertFalse(result["coarse_supervised"])
        self.assertEqual(result["coarse_weight"], 0.0)
        torch.testing.assert_close(result["loss"], expected)
        changed = dict(output)
        changed["coarse_logits_74"] = output["coarse_logits_74"] + 1000.0
        torch.testing.assert_close(
            build_hsd_r1_group_loss(cfg, 1, changed, target)["loss"], expected
        )

    def test_original_hsd_triple_loss_is_unchanged(self):
        cfg = load_config("configs/dinov1_s8_r1hard_hsd_v1_sem_ms148_online.py")
        target = binary_target()
        output = {
            "final_logits": torch.randn(1, 1, 148, 148),
            "coarse_logits_74": torch.randn(1, 1, 74, 74),
            "base_logits_37": torch.randn(1, 1, 37, 37),
        }
        result = build_hsd_r1_group_loss(cfg, 1, output, target)
        expected = (
            F.binary_cross_entropy_with_logits(
                output["final_logits"], F.interpolate(target, (148, 148), mode="nearest")
            )
            + 0.5
            * F.binary_cross_entropy_with_logits(
                output["coarse_logits_74"], F.interpolate(target, (74, 74), mode="nearest")
            )
            + 0.5 * F.binary_cross_entropy_with_logits(output["base_logits_37"], target)
        ) / 2.0
        self.assertTrue(result["coarse_supervised"])
        torch.testing.assert_close(result["loss"], expected)

    def test_all_configs_are_direct_hard_r1_and_pure_student(self):
        for path in CONFIGS:
            with self.subTest(path=path):
                cfg = load_config(path)
                self.assertEqual(validate_r1_decoder_isolation_config(cfg)["status"], "PASS")
                self.assertTrue(is_r1hard_linear_pure_student_config(cfg))
                self.assertTrue(cfg.R1_DECODER_DIRECT_R1_37)
                self.assertEqual(cfg.R1_HARD_SOURCE_KEY, "residual_pass1_37")
                self.assertEqual(cfg.R1_HARD_THRESHOLD, 0.5)
                self.assertFalse(cfg.USE_ECST)
                self.assertFalse(cfg.FINETUNE_RESET_TEACHER)
                self.assertEqual(cfg.FINETUNE_RESET_EPOCH, 0)
                self.assertFalse(cfg.USE_TEACHER_BINARY_FULL_LOSS)
                self.assertFalse(cfg.USE_TEACHER_SOFT_FULL_LOSS)

    def test_eval_infers_384_input_channels_for_every_decoder(self):
        for path in CONFIGS:
            with self.subTest(path=path):
                cfg = load_config(path)
                state = build_seg_head(384, cfg).state_dict()
                self.assertEqual(infer_in_channels(state), 384)

    def test_direct_target_rejects_soft_values(self):
        cfg = load_config("configs/dinov1_s8_r1hard_last4_linear_b16.py")
        with self.assertRaises(RuntimeError):
            build_r1_decoder_isolation_loss(
                cfg, torch.randn(1, 1, 37, 37), torch.full((1, 1, 37, 37), 0.5)
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_all_isolation_decoders_single_batch_backward(self):
        feature = {
            key: torch.randn(1, 384, 37, 37, device="cuda")
            for key in ("f9", "f10", "f11", "f12")
        }
        target = (torch.rand(1, 1, 37, 37, device="cuda") > 0.5).float()

        last4_cfg = load_config("configs/dinov1_s8_r1hard_last4_linear_b16.py")
        last4 = Last4LinearProbe(384).cuda()
        last4_output = last4(feature)
        self.assertTrue(torch.isfinite(last4_output).all().item())
        build_r1_decoder_isolation_loss(
            last4_cfg, last4_output, target
        )["loss"].backward()

        scale_cfg = load_config(
            "configs/dinov1_s8_r1hard_f12_scalelift_triple_b16.py"
        )
        scale = F12ScaleLiftHead(384, 64, 8).cuda()
        scale_output = scale(feature)
        self.assertTrue(
            all(
                torch.isfinite(value).all().item()
                for value in scale_output.values()
                if torch.is_tensor(value)
            )
        )
        build_r1_decoder_isolation_loss(
            scale_cfg, scale_output, target
        )["loss"].backward()

        hsd_cfg = load_config(
            "configs/dinov1_s8_r1hard_hsd_v1_sem_finalbase_b16.py"
        )
        hsd = HSDV1Head(
            384, semantic_channels=64, detail_channels=32,
            gn_groups=8, use_detail=False, output_size=148, coarse_size=74
        ).cuda()
        hsd_output = hsd(feature)
        self.assertTrue(
            all(
                torch.isfinite(value).all().item()
                for value in hsd_output.values()
                if torch.is_tensor(value)
            )
        )
        build_hsd_r1_group_loss(
            hsd_cfg, 1, hsd_output, target
        )["loss"].backward()


if __name__ == "__main__":
    unittest.main()
