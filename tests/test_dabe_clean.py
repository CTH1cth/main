import unittest

import torch

from common.dabe_clean import build_continuous_recoverability


class DABECleanContinuousRecoverabilityTest(unittest.TestCase):
    def _build(self, p_rw, gate, background, semantic):
        foreground = (p_rw * gate).clamp(0.0, 1.0)
        return build_continuous_recoverability(
            p_rw,
            gate,
            foreground,
            background,
            semantic,
            output_size=(68, 68),
        )

    def test_probability_ranges_regression_and_determinism(self):
        generator = torch.Generator().manual_seed(17)
        p_rw = torch.rand((1, 37, 37), generator=generator)
        gate = torch.rand((1, 37, 37), generator=generator)
        background = torch.rand((1, 37, 37), generator=generator)
        semantic = torch.rand((1, 37, 37), generator=generator)
        first = self._build(p_rw, gate, background, semantic)
        second = self._build(p_rw, gate, background, semantic)
        self.assertLess(first["foreground_reconstruction_error"], 1e-5)
        for key in ("latent_rw_37", "recoverability_37", "recoverability_68"):
            self.assertTrue(torch.equal(first[key], second[key]))
            self.assertGreaterEqual(float(first[key].min()), 0.0)
            self.assertLessEqual(float(first[key].max()), 1.0)
            self.assertFalse(first[key].requires_grad)

    def test_each_zero_evidence_annihilates_recoverability(self):
        ones = torch.ones((1, 37, 37), dtype=torch.float32)
        zeros = torch.zeros_like(ones)
        # latent_rw=0 because gate=1 makes foreground equal p_rw.
        self.assertEqual(float(self._build(ones, ones, zeros, ones)["recoverability_37"].max()), 0.0)
        # background_evidence=1 annihilates (1-B).
        self.assertEqual(float(self._build(ones, zeros, ones, ones)["recoverability_37"].max()), 0.0)
        # semantic tendency=0 annihilates H.
        self.assertEqual(float(self._build(ones, zeros, zeros, zeros)["recoverability_37"].max()), 0.0)

    def test_all_three_recovery_evidences_one_give_one(self):
        ones = torch.ones((1, 37, 37), dtype=torch.float32)
        zeros = torch.zeros_like(ones)
        output = self._build(ones, zeros, zeros, ones)
        self.assertTrue(torch.equal(output["recoverability_37"], ones))
        self.assertTrue(torch.equal(output["recoverability_68"], torch.ones((1, 68, 68))))

    def test_foreground_regression_is_strict(self):
        ones = torch.ones((1, 37, 37), dtype=torch.float32)
        zeros = torch.zeros_like(ones)
        with self.assertRaises(RuntimeError):
            build_continuous_recoverability(
                ones, ones, zeros, zeros, ones, output_size=(68, 68)
            )


if __name__ == "__main__":
    unittest.main()
