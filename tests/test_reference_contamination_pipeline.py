import copy
from pathlib import Path

import torch

from tools.evaluate_reference_contamination_native import (
    _metric_error,
    _raw_score_reproduction,
    _score_pair_metrics,
)
from tools.run_reference_contamination_mechanism import (
    _parse_path_maps,
    _resolve_existing_path,
)


def _condition(offset: float = 0.0) -> dict:
    base = torch.linspace(0.0, 1.0, 37 * 37) + offset
    return {"knn8_score": base.clone(), "gbsp_score": base.square()}


def _payload() -> dict:
    condition = _condition()
    return {
        "natural": copy.deepcopy(condition),
        "oracle_clean": copy.deepcopy(condition),
        "contamination": [
            {"seed": 0, "p": 0.0, "valid": True, **copy.deepcopy(condition)},
            {"seed": 0, "p": 0.05, "valid": False},
        ],
    }


def test_native_pair_metrics_are_exact_and_finite():
    gt = torch.zeros(1, 74, 91)
    gt[:, 20:60, 30:70] = 1
    metrics = _score_pair_metrics(_condition(), gt)
    assert set(metrics) == {"knn8", "gbsp"}
    for method in metrics:
        assert set(metrics[method]) == {"AP", "AUROC"}
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics[method].values())


def test_raw_score_reproduction_detects_any_change():
    reference = _payload()
    current = copy.deepcopy(reference)
    assert _raw_score_reproduction(current, reference) == 0.0
    current["contamination"][0]["knn8_score"][12] += 1e-3
    assert _raw_score_reproduction(current, reference) > 0.0


def test_metric_error_is_zero_for_identical_results():
    value = {
        "knn8": {"AP": .7, "AUROC": .8},
        "gbsp": {"AP": .6, "AUROC": .9},
    }
    assert _metric_error(value, copy.deepcopy(value)) == 0.0


def test_path_map_relocates_a_missing_absolute_source():
    existing = Path(__file__).resolve()
    fake_source = Path("/machine_a/project") / existing.name
    mappings = _parse_path_maps([f"/machine_a/project={existing.parent}"])
    resolved = _resolve_existing_path(fake_source, mappings, role="test file")
    assert resolved == existing


def test_path_maps_use_the_longest_matching_prefix_first():
    mappings = _parse_path_maps([
        "/home/dell01/CTH=/remote/CTH",
        "/home/dell01/CTH/MY-baseline=/remote/MY-BASELINE",
    ])
    assert mappings[0][0] == "/home/dell01/CTH/MY-baseline"


def test_path_map_rejects_malformed_or_conflicting_values():
    for values in (["missing_separator"], ["/old="], ["=/new"]):
        try:
            _parse_path_maps(values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid path map: {values}")
    try:
        _parse_path_maps(["/old=/one", "/old=/two"])
    except ValueError as error:
        assert "conflicting" in str(error)
    else:
        raise AssertionError("expected conflicting path-map failure")
