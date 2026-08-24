import torch

from common.metrics import CODMetrics
from common.utils import format_metric_table, metric_value


def test_metric_result_and_main_table_use_adaptive_e():
    gt = torch.tensor(
        [[[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]]
    )
    pred = torch.tensor(
        [[[[0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]]
    )
    metrics = CODMetrics()
    metrics.step(gt, pred)
    result = metrics.get_result()

    assert result["E_ADP"] == metrics.em.get_results()["em"]["adp"]
    assert metric_value(result, "E_phi^adp") == result["E_ADP"]

    table = format_metric_table(result)
    assert "adp E↑" in table
    assert "E_phi^adp↑" not in table
    assert "E_phi^m↑" not in table
    assert f"{float(result['E_ADP']):.4f}" in table
