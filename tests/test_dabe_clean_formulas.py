import unittest

import torch
import torch.nn.functional as F

from common.dabe_clean import (
    build_background_evidence,
    build_bridge_target,
    build_clean_targets,
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


class DABECleanFormulaTest(unittest.TestCase):
    def test_bridge_and_clean_formula_endpoints_are_exact_and_detached(self):
        target = _tensor([0.0, 1.0, 0.5, 1.0])
        weight = _tensor([1.0, 1.0, 0.25, 0.0])
        target_before = target.clone()
        weight_before = weight.clone()
        bridge = build_bridge_target(target, weight)
        self.assertTrue(torch.equal(bridge, _tensor([0.0, 1.0, 0.5, 0.5])))
        self.assertFalse(bridge.requires_grad)
        self.assertTrue(torch.equal(target, target_before))
        self.assertTrue(torch.equal(weight, weight_before))

        background = build_background_evidence(
            _tensor([0.0, 1.0, 1.0]), _tensor([0.0, 0.0, 1.0])
        )
        self.assertTrue(torch.equal(background, _tensor([0.0, 1.0, 0.0])))

        result = build_clean_targets(
            _tensor([1.0, 0.0, 0.5, 0.25]),
            _tensor([0.0, 1.0, 1.0, 0.75]),
        )
        self.assertTrue(
            torch.allclose(result["target_dp"], _tensor([1.0, 0.0, 0.5, 0.3125]))
        )
        self.assertTrue(
            torch.allclose(result["target_diff"], _tensor([1.0, 0.0, 0.25, 0.25]))
        )
        self.assertTrue(all(not value.requires_grad for value in result.values()))

    def test_formula_contract_rejects_wrong_dtype_and_shape(self):
        with self.assertRaisesRegex(RuntimeError, "float32"):
            build_bridge_target(
                torch.zeros(1, 1, 2, 2, dtype=torch.float64),
                torch.ones(1, 1, 2, 2, dtype=torch.float64),
            )
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            build_clean_targets(
                torch.zeros(1, 1, 2, 2),
                torch.zeros(1, 1, 3, 3),
            )

    def test_bridge_preserves_gradient_map_and_reports_reduction_scale(self):
        target = _tensor([0.1, 0.4, 0.7, 0.9])
        weight = _tensor([0.2, 0.5, 0.8, 1.0])
        bridge = build_bridge_target(target, weight)
        self.assertTrue(
            torch.allclose(bridge - 0.5, weight * (target - 0.5), atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(0.5 - bridge, weight * (0.5 - target), atol=1e-7)
        )

        old_logits = torch.zeros_like(target, requires_grad=True)
        old_loss = (
            F.binary_cross_entropy_with_logits(old_logits, target, reduction="none")
            * weight
        ).sum() / weight.sum()
        old_grad = torch.autograd.grad(old_loss, old_logits)[0]

        bridge_logits = torch.zeros_like(target, requires_grad=True)
        bridge_loss = F.binary_cross_entropy_with_logits(
            bridge_logits, bridge, reduction="mean"
        )
        bridge_grad = torch.autograd.grad(bridge_loss, bridge_logits)[0]
        reduction_scale = target.numel() / float(weight.sum())
        self.assertTrue(
            torch.allclose(old_grad, bridge_grad * reduction_scale, atol=1e-7)
        )


if __name__ == "__main__":
    unittest.main()
