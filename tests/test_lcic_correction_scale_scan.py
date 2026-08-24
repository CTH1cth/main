from pathlib import Path

import numpy as np
import torch

from common.metrics import CODMetrics
from common.utils import load_config
from model import build_seg_head
from tools.scan_lcic_correction_scales import (
    CoreMetricAccumulator,
    build_scale_pairs,
    lcic_scaled_logits,
    parse_scales,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/dinov1_s8_gbsp_lcic_d_full.py"


def test_scale_parser_and_cartesian_grid():
    scales = parse_scales("0,0.5,1,1,2")
    assert scales == (0.0, 0.5, 1.0, 2.0)
    assert build_scale_pairs(scales[:2], scales[2:]) == (
        (0.0, 1.0),
        (0.0, 2.0),
        (0.5, 1.0),
        (0.5, 2.0),
    )


def test_unit_scales_equal_normal_lcic_and_zero_scales_equal_anchor():
    torch.manual_seed(17)
    cfg = load_config(CONFIG)
    model = build_seg_head(384, cfg).eval()
    model.alpha.data.fill_(0.3)
    model.beta.data.fill_(0.4)
    feature = torch.randn(1, 384, 7, 7)
    scaled = lcic_scaled_logits(model, feature, ((1.0, 1.0), (0.0, 0.0)))
    assert torch.allclose(scaled[0], model(feature), atol=1e-6)
    assert torch.allclose(scaled[1], model.anchor(feature), atol=1e-6)


def test_core_metrics_match_cod_metrics_for_binary_predictions():
    ground_truth = torch.tensor(
        [[[[0, 0, 1], [0, 1, 1], [0, 0, 0]]]], dtype=torch.float32
    )
    prediction = torch.tensor(
        [[[[0, 1, 1], [0, 1, 1], [0, 0, 0]]]], dtype=torch.float32
    )
    core = CoreMetricAccumulator()
    core.step(ground_truth, prediction)
    core_result = core.result()
    full = CODMetrics()
    full.step(ground_truth, prediction)
    full_result = full.get_result()
    assert np.isclose(core_result["SMeasure"], full_result["SMeasure"])
    assert np.isclose(core_result["E_ADP"], full_result["E_ADP"])
    assert np.isclose(core_result["MAE"], full_result["MAE"])
    assert np.isclose(core_result["prediction_area"], prediction.mean().item())
