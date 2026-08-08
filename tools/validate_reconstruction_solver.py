#!/usr/bin/env python3
"""Validate simplex PGD against SciPy SLSQP on synthetic and real queries."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.reconstruction import flatten_feature, retrieve_fullbc_neighbors, simplex_projected_gradient  # noqa: E402
from tools.gbsp_knn_lsr_common import write_csv, write_json  # noqa: E402
from tools.reconstruction_rescue_common import (  # noqa: E402
    load_core_inputs, load_core_rows, require_output_outside_main,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_synthetic", type=int, default=100)
    parser.add_argument("--num_real", type=int, default=100)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--geometry", choices=("l2", "raw"), default="l2")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def slsqp(query: np.ndarray, atoms: np.ndarray) -> tuple[np.ndarray, float, bool]:
    k = atoms.shape[0]
    objective = lambda alpha: float(np.square(query - alpha @ atoms).sum())
    result = minimize(
        objective, np.full(k, 1.0 / k), method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda alpha: float(alpha.sum() - 1.0)},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return np.asarray(result.x), float(result.fun), bool(result.success)


def validate_cases(
    query_batches: list[np.ndarray],
    atom_batches: list[np.ndarray],
    kinds: list[str],
    device: torch.device,
) -> list[dict]:
    """Validate heterogeneous feature dimensions without mixing stack shapes."""
    if not (len(query_batches) == len(atom_batches) == len(kinds)):
        raise ValueError("query/atom/kind case counts differ")
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = defaultdict(list)
    for index, (query, atoms) in enumerate(zip(query_batches, atom_batches)):
        if query.ndim != 1 or atoms.ndim != 2 or atoms.shape[1] != query.shape[0]:
            raise ValueError(f"invalid case shape at {index}: query={query.shape}, atoms={atoms.shape}")
        groups[(tuple(query.shape), tuple(atoms.shape))].append(index)

    results: dict[int, dict] = {}
    for indices in groups.values():
        query = torch.from_numpy(np.stack([query_batches[index] for index in indices])).to(device)
        atoms = torch.from_numpy(np.stack([atom_batches[index] for index in indices])).to(device)
        pgd = simplex_projected_gradient(query, atoms, max_iterations=64, tolerance=1e-6)
        for local_index, case_index in enumerate(indices):
            q, a = query_batches[case_index], atom_batches[case_index]
            alpha_ref, objective_ref, success = slsqp(q.astype(np.float64), a.astype(np.float64))
            alpha = pgd.coefficients[local_index].detach().cpu().double().numpy()
            objective = float(pgd.residual[local_index])
            results[case_index] = {
                "index": case_index, "kind": kinds[case_index], "slsqp_success": success,
                "pgd_converged": bool(pgd.converged[local_index]),
                "objective_pgd": objective, "objective_slsqp": objective_ref,
                "objective_gap": objective - objective_ref,
                "residual_relative_error": abs(objective - objective_ref) / max(abs(objective_ref), 1e-12),
                "alpha_linf_error": float(np.max(np.abs(alpha - alpha_ref))),
                "simplex_sum_error": float(pgd.simplex_sum_error[local_index]),
                "minimum_alpha": float(pgd.minimum_alpha[local_index]),
            }
    return [results[index] for index in range(len(query_batches))]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    query_batches, atom_batches, kinds = [], [], []
    for _ in range(args.num_synthetic):
        atoms = rng.normal(size=(args.k, 32)).astype(np.float32)
        alpha = rng.dirichlet(np.ones(args.k)).astype(np.float32)
        query_batches.append(alpha @ atoms + .01 * rng.normal(size=32).astype(np.float32))
        atom_batches.append(atoms); kinds.append("synthetic")
    real_needed = args.num_real
    rows = load_core_rows(args.core_root, split=args.split, max_samples=max(1, (real_needed + 1368) // 1369))
    for row in rows:
        core, raw, background = load_core_inputs(row, device)
        retrieval = retrieve_fullbc_neighbors(raw, background, max_k=args.k)
        feature = retrieval.normalized_features if args.geometry == "l2" else flatten_feature(raw)
        for patch in range(min(1369, real_needed)):
            index = retrieval.neighbor_indices[patch, :args.k]
            query_batches.append(feature[patch].detach().cpu().numpy())
            atom_batches.append(feature[index].detach().cpu().numpy())
            kinds.append("real")
        real_needed -= min(1369, real_needed)
        if real_needed <= 0:
            break
    output = validate_cases(query_batches, atom_batches, kinds, device)
    out = require_output_outside_main(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "solver_validation.csv", output)
    relative_errors = [row["residual_relative_error"] for row in output]
    summary = {
        "cases": len(output), "synthetic": kinds.count("synthetic"), "real": kinds.count("real"),
        "slsqp_failures": sum(not row["slsqp_success"] for row in output),
        "pgd_nonconverged": sum(not row["pgd_converged"] for row in output),
        "objective_gap_max": max(row["objective_gap"] for row in output),
        "objective_gap_p95": float(np.quantile([row["objective_gap"] for row in output], .95)),
        "residual_relative_error_median": float(np.median(relative_errors)),
        "residual_relative_error_p95": float(np.quantile(relative_errors, .95)),
        "alpha_linf_error_max": max(row["alpha_linf_error"] for row in output),
        "simplex_sum_error_max": max(row["simplex_sum_error"] for row in output),
        "minimum_alpha": min(row["minimum_alpha"] for row in output),
        "pass": all(row["slsqp_success"] for row in output)
        and float(np.median(relative_errors)) < 1e-5
        and float(np.quantile(relative_errors, .95)) < 1e-4
        and max(row["simplex_sum_error"] for row in output) < 1e-5,
    }
    write_json(out / "solver_validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
