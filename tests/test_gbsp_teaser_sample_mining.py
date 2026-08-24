from __future__ import annotations

import inspect

import numpy as np

from tools.gbsp_teaser_sample_mining.common import binary_metrics, patch_box
from tools.gbsp_teaser_sample_mining.pipeline import generate_method_scores


def test_patch_coordinate_mapping_exact() -> None:
    for index in np.linspace(0, 37 * 37 - 1, 100, dtype=int):
        x0, y0, x1, y1 = patch_box(int(index))
        assert (x1 - x0, y1 - y0) == (8, 8)
        assert 0 <= x0 < x1 <= 296
        assert 0 <= y0 < y1 <= 296


def test_score_generator_cannot_receive_gt() -> None:
    parameters = set(inspect.signature(generate_method_scores).parameters)
    assert "gt" not in parameters
    assert "gt_path" not in parameters
    source = inspect.getsource(generate_method_scores)
    assert "gt_occupancy" not in source
    assert "Image.open" not in source


def test_binary_metrics_known_case() -> None:
    target = np.asarray([1, 1, 0, 0], dtype=bool)
    prediction = np.asarray([1, 0, 1, 0], dtype=bool)
    result = binary_metrics(prediction, target)
    assert result["tp"] == 1 and result["fp"] == 1 and result["fn"] == 1
    assert abs(result["iou"] - 1 / 3) < 1e-12
    assert abs(result["f1"] - 0.5) < 1e-12
    assert abs(result["mae"] - 0.5) < 1e-12

