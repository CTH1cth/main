from __future__ import annotations

import numpy as np
import torch

from tools.validate_reconstruction_solver import validate_cases


def test_solver_validation_accepts_mixed_feature_dimensions() -> None:
    rng = np.random.default_rng(20260807)
    queries, atoms = [], []
    for dimension in (8, 12):
        local = rng.normal(size=(4, dimension)).astype(np.float32)
        alpha = np.asarray([.1, .2, .3, .4], dtype=np.float32)
        queries.append(alpha @ local)
        atoms.append(local)
    rows = validate_cases(queries, atoms, ["synthetic", "real"], torch.device("cpu"))
    assert len(rows) == 2
    assert all(row["slsqp_success"] for row in rows)
    assert max(row["objective_gap"] for row in rows) < 1e-5
    assert max(row["simplex_sum_error"] for row in rows) < 1e-6
