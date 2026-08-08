#!/usr/bin/env python3
"""Original-size COD evaluation for frozen GBSP PCA-SPE calibration caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.eval_dabe_rank_calibration import FastCODContext  # noqa: E402
from common.utils import load_config, read_jsonl, torch_load  # noqa: E402
from models.gbsp_threshold_v5 import SeededMultiOtsuHysteresis  # noqa: E402
from tools.eval_gbsp_threshold import (  # noqa: E402
    _binary_metrics,
    _load_gt,
    _resize,
    _sample_keys,
    _table,
    _write_csv,
    _write_json,
)
from tools.eval_gbsp_threshold_v3 import _rank_metrics  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / "configs/dinov1_s8_gbsp_spe95.py"
DATASETS = ("CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K")
EXPECTED = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
SPE_METHODS = ("jm_spe_090", "jm_spe_095", "jm_spe_099", "gamma_spe_095")
REFERENCES = ("r1_fixed_050", "gbsp_fixed_050", "gbsp_fixed_058", "smoh")
PRIMARY = ("S_m", "F_beta_w", "E_mean", "MAE")
SECONDARY = ("F_beta_mean", "Precision", "Recall", "Area", "IoU", "Dice")
METRICS = (*PRIMARY, *SECONDARY)
DELTA_FIELDS = {
    "S_m": "delta_S_vs_fixed_058",
    "F_beta_w": "delta_F_beta_w_vs_fixed_058",
    "F_beta_mean": "delta_Fmean_vs_fixed_058",
    "E_mean": "delta_E_vs_fixed_058",
    "MAE": "delta_MAE_vs_fixed_058",
    "Precision": "delta_Precision_vs_fixed_058",
    "Recall": "delta_Recall_vs_fixed_058",
    "Area": "delta_Area_vs_fixed_058",
    "IoU": "delta_IoU_vs_fixed_058",
    "Dice": "delta_Dice_vs_fixed_058",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def _manifest(root: Path) -> list[dict]:
    path = root / "manifest_test.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    seen = set()
    for line, row in enumerate(rows, 1):
        key = (str(row.get("dataset", "")), str(row.get("stem", "")))
        cache = Path(str(row.get("cache_path", "")))
        if not all(key) or key in seen or not cache.is_file():
            raise RuntimeError(f"invalid calibration manifest {path}:{line}: {key}")
        seen.add(key)
    return rows


def _select(rows: list[dict], sample_list: str | None, limit: int) -> list[dict]:
    if sample_list:
        keys = _sample_keys(_resolve(sample_list))
        mapping = {(str(row["dataset"]), str(row["stem"])): row for row in rows}
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise KeyError(f"samples missing from calibration manifest: {missing[:5]}")
        rows = [mapping[key] for key in keys]
    if limit >= 0:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError("no samples selected")
    return rows


def _map(payload: dict, field: str, path: Path, unit: bool = False) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be Tensor[1,37,37]: {path}")
    value = value.detach().cpu().float().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field} contains NaN/Inf: {path}")
    if unit and (float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6):
        raise ValueError(f"{field} escaped [0,1]: {path}")
    return value.clamp(0.0, 1.0) if unit else value


def _method_payload(payload: dict, method: str, path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if method.startswith("jm_spe_"):
        tag = method.rsplit("_", 1)[1]
        valid = bool(payload.get("numerical_validity", {}).get(f"jm_{tag}", False))
        score = payload.get(f"jm_calibrated_map_{tag}")
        mask = payload.get(f"jm_mask_{tag}")
        tau = payload.get(f"jm_tau_{tag}")
        equivalent = payload.get(f"equivalent_minmax_threshold_jm{tag}")
        area = payload.get(f"foreground_area_jm{tag}")
        reason = payload.get("failure_reason", {}).get(f"jm_{tag}")
        bc_rate = payload.get("background_candidate_exceedance_rate") if tag == "095" else None
    elif method == "gamma_spe_095":
        valid = bool(payload.get("numerical_validity", {}).get("gamma_095", False))
        score = payload.get("gamma_calibrated_map_095")
        mask = payload.get("gamma_mask_095")
        tau = payload.get("gamma_tau_095")
        equivalent = payload.get("equivalent_minmax_threshold_gamma095")
        area = payload.get("foreground_area_gamma095")
        reason = payload.get("failure_reason", {}).get("gamma_095")
        bc_rate = payload.get("background_candidate_exceedance_rate_gamma095")
    else:
        raise ValueError(method)
    if valid:
        if not torch.is_tensor(score) or not torch.is_tensor(mask):
            raise ValueError(f"valid {method} cache lacks score/mask: {path}")
        score = _map({"value": score}, "value", path, unit=True)
        mask = _map({"value": mask}, "value", path, unit=True)
        raw = _map(payload, "raw_residual_map", path)
        raw_mask = raw > float(tau)
        if not torch.equal(mask.bool(), raw_mask):
            raise RuntimeError(f"{method} cached mask is not Q>tau: {path}")
        # Score and raw comparisons may differ only at machine-epsilon-near
        # points.  Cached production data should contain no such disagreement.
        if not torch.equal(score > 0.5, raw_mask):
            raise RuntimeError(f"{method} score threshold is not equivalent to Q>tau: {path}")
    else:
        score = torch.zeros(1, 37, 37)
        mask = torch.zeros(1, 37, 37)
    return score, mask, {
        "numerical_validity": int(valid),
        "numerical_failure": int(not valid),
        "failure_reason": reason,
        "control_limit": tau,
        "equivalent_minmax_threshold": equivalent,
        "foreground_area": area,
        "background_candidate_exceedance_rate": bc_rate,
    }


def _source(payload: dict, path: Path, identity: tuple[str, str]) -> tuple[dict, Path]:
    source_path = Path(str(payload.get("source_gbsp_path", "")))
    if not source_path.is_file():
        raise FileNotFoundError(f"source GBSP missing: {path}: {source_path}")
    source = torch_load(source_path, map_location="cpu")
    if not isinstance(source, dict) or (str(source.get("dataset")), str(source.get("stem"))) != identity:
        raise RuntimeError(f"source GBSP identity mismatch: {source_path}")
    return source, source_path


def _process(task: dict) -> dict:
    try:
        path = Path(task["cache_path"])
        payload = torch_load(path, map_location="cpu")
        identity = (task["dataset"], task["stem"])
        if not isinstance(payload, dict) or (str(payload.get("dataset")), str(payload.get("stem"))) != identity:
            raise RuntimeError(f"calibration identity mismatch: {path}")
        forbidden = (
            "gt_used_for_generation", "r1_used_for_generation", "fixed_058_used_for_generation",
            "target_area_used_for_generation", "dino_forward_used_for_generation",
            "pca_refit_for_generation", "pca_rank_changed_for_generation",
            "raw_residual_changed_for_generation", "morphology_used_for_generation",
            "graph_propagation_used_for_generation", "background_candidates_forced_for_generation",
        )
        if any(bool(payload.get(key, True)) for key in forbidden):
            raise RuntimeError(f"SPE independence violation: {path}")
        source, source_path = _source(payload, path, identity)
        minmax = _map(source, "absolute_minmax", source_path, unit=True)
        raw = _map(source, "absolute_raw", source_path)
        if not torch.equal(raw, _map(payload, "raw_residual_map", path)):
            raise RuntimeError(f"SPE changed the formal GBSP raw residual: {path}")
        bc = source.get("background_indices")
        if not torch.is_tensor(bc) or bc.ndim != 1 or bc.numel() == 0:
            raise ValueError(f"background_indices missing: {source_path}")
        bc = bc.detach().cpu().long().contiguous()

        methods = {}
        for name in task["methods"]:
            score, mask, metadata = _method_payload(payload, name, path)
            methods[name] = {"score": score, "mask": mask, "rank_score": raw, "metadata": metadata}
        for name in task["references"]:
            if name == "gbsp_fixed_050":
                methods[name] = {"score": minmax, "mask": minmax > 0.50, "rank_score": raw,
                                 "metadata": {"threshold": 0.50, "numerical_validity": 1, "numerical_failure": 0}}
            elif name == "gbsp_fixed_058":
                methods[name] = {"score": minmax, "mask": minmax > 0.58, "rank_score": raw,
                                 "metadata": {"threshold": 0.58, "numerical_validity": 1, "numerical_failure": 0}}
            elif name == "r1_fixed_050":
                dabe_path = Path(str(source.get("source_dabe_path", "")))
                dabe = torch_load(dabe_path, map_location="cpu")
                if (str(dabe.get("dataset")), str(dabe.get("stem"))) != identity:
                    raise RuntimeError(f"R1 identity mismatch: {dabe_path}")
                r1 = _map(dabe, "residual_pass1_37", dabe_path, unit=True)
                methods[name] = {"score": r1, "mask": r1 > 0.50, "rank_score": r1,
                                 "metadata": {"threshold": 0.50, "numerical_validity": 1, "numerical_failure": 0}}
            elif name == "smoh":
                result = SeededMultiOtsuHysteresis().apply(minmax, bc)
                methods[name] = {
                    "score": minmax, "mask": result.mask.float(), "rank_score": raw,
                    "metadata": {
                        "threshold_high": result.threshold_high, "threshold_low": result.threshold_low,
                        "numerical_validity": int(not result.numerical_failure),
                        "numerical_failure": int(result.numerical_failure),
                        "failure_reason": result.diagnostics.get("failure_reason"),
                        "foreground_area": float(result.mask.float().mean()),
                    },
                }
            else:
                raise ValueError(f"unknown reference: {name}")

        gt_path = Path(task["gt_path"] or payload.get("gt_path", ""))
        gt = _load_gt(gt_path)
        shape = tuple(gt.shape[-2:])
        context = FastCODContext(gt)
        rows = []
        for name in (*task["references"], *task["methods"]):
            item = methods[name]
            if name == "smoh":
                native_mask = (_resize(item["mask"], shape) > 0.5).float()
            else:
                threshold = 0.58 if name == "gbsp_fixed_058" else 0.5
                native_mask = (_resize(item["score"], shape) > threshold).float()
            native_rank = _resize(item["rank_score"], shape)
            method_valid = bool(item["metadata"].get("numerical_validity", 1))
            binary = (
                _binary_metrics(context, native_mask)
                if method_valid
                else {metric: float("nan") for metric in METRICS}
            )
            rows.append({
                "dataset": identity[0], "stem": identity[1], "cache_path": str(path),
                "source_gbsp_path": str(source_path),
                "image_path": task["image_path"] or payload.get("image_path", ""),
                "gt_path": str(gt_path), "method": name,
                **binary,
                **_rank_metrics(native_rank, gt),
                **item["metadata"],
                "foreground_area_patch": float(item["mask"].float().mean()) if method_valid else float("nan"),
                "empty_mask": int(float(item["mask"].float().mean()) == 0.0) if method_valid else float("nan"),
                "area_over_50": int(float(item["mask"].float().mean()) > 0.5) if method_valid else float("nan"),
                "pca_rank": int(payload.get("pca_rank", -1)),
                "background_candidate_count": int(payload.get("background_candidate_count", -1)),
                "theta1": payload.get("theta1"), "theta2": payload.get("theta2"), "theta3": payload.get("theta3"),
                "h0": payload.get("h0"),
            })
        fixed = next((row for row in rows if row["method"] == "gbsp_fixed_058"), None)
        if fixed is not None:
            for row in rows:
                for field, delta in DELTA_FIELDS.items():
                    row[delta] = row[field] - fixed[field]
        return {"rows": rows}
    except Exception as error:
        return {
            "dataset": task.get("dataset", ""), "stem": task.get("stem", ""),
            "error": repr(error), "traceback": traceback.format_exc(),
        }


def _init(torch_threads: int) -> None:
    torch.set_num_threads(int(torch_threads))


def _mean(values) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _fields(rows: list[dict]) -> list[str]:
    output, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                output.append(key)
                seen.add(key)
    return output


def _aggregate(rows: list[dict]) -> list[dict]:
    methods = tuple(dict.fromkeys(row["method"] for row in rows))
    extras = (
        "pixel_AP", "pixel_AUROC", "numerical_validity", "numerical_failure",
        "foreground_area_patch", "empty_mask", "area_over_50", "control_limit",
        "equivalent_minmax_threshold", "background_candidate_exceedance_rate",
        *DELTA_FIELDS.values(),
    )
    output = []
    for dataset in DATASETS:
        for method in methods:
            subset = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if subset:
                output.append({
                    "scope": "dataset", "dataset": dataset, "method": method, "num_samples": len(subset),
                    **{field: _mean(row.get(field) for row in subset) for field in (*METRICS, *extras)},
                })
    for method in methods:
        subset = [row for row in output if row["method"] == method]
        output.append({
            "scope": "dataset_macro", "dataset": "ALL", "method": method,
            "num_samples": sum(int(row["num_samples"]) for row in subset),
            **{field: _mean(row.get(field) for row in subset) for field in (*METRICS, *extras)},
        })
    return output


def _bootstrap(rows: list[dict], methods: tuple[str, ...], repetitions: int, seed: int) -> list[dict]:
    if repetitions <= 0 or "gbsp_fixed_058" not in {row["method"] for row in rows}:
        return []
    lookup = {(row["dataset"], row["stem"], row["method"]): row for row in rows}
    rng = np.random.default_rng(seed)
    output = []
    for method in methods:
        for metric in PRIMARY:
            dataset_deltas = {}
            for dataset in DATASETS:
                keys = [(row["dataset"], row["stem"]) for row in rows if row["dataset"] == dataset and row["method"] == method]
                values = [
                    float(lookup[(d, s, method)][metric]) - float(lookup[(d, s, "gbsp_fixed_058")][metric])
                    for d, s in keys if (d, s, "gbsp_fixed_058") in lookup
                ]
                deltas = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
                if deltas.size:
                    dataset_deltas[dataset] = deltas
                    samples = np.empty(repetitions, dtype=np.float64)
                    for index in range(repetitions):
                        samples[index] = float(deltas[rng.integers(0, deltas.size, deltas.size)].mean())
                    output.append({
                        "scope": "dataset", "dataset": dataset, "method": method, "metric": metric,
                        "paired_delta": float(deltas.mean()), "ci95_low": float(np.quantile(samples, .025)),
                        "ci95_high": float(np.quantile(samples, .975)), "repetitions": repetitions, "seed": seed,
                    })
            if dataset_deltas:
                samples = np.empty(repetitions, dtype=np.float64)
                for index in range(repetitions):
                    samples[index] = float(np.mean([
                        values[rng.integers(0, values.size, values.size)].mean()
                        for values in dataset_deltas.values()
                    ]))
                paired = float(np.mean([values.mean() for values in dataset_deltas.values()]))
                output.append({
                    "scope": "dataset_macro", "dataset": "ALL", "method": method, "metric": metric,
                    "paired_delta": paired, "ci95_low": float(np.quantile(samples, .025)),
                    "ci95_high": float(np.quantile(samples, .975)), "repetitions": repetitions, "seed": seed,
                })
    return output


def evaluate(args) -> None:
    cfg = load_config(_resolve(args.config))
    root = _resolve(args.calibration_root)
    rows = _select(_manifest(root), args.sample_list, args.max_samples)
    methods = tuple(args.methods or SPE_METHODS)
    references = tuple(args.compare if args.compare is not None else ("gbsp_fixed_058",))
    if not methods or set(methods) - set(SPE_METHODS):
        raise ValueError(f"invalid SPE methods: {methods}")
    if set(references) - set(REFERENCES):
        raise ValueError(f"invalid references: {references}")
    if "gbsp_fixed_058" not in references:
        references = (*references, "gbsp_fixed_058")
    out = _resolve(args.out_dir)
    if out == MAIN_ROOT or MAIN_ROOT in out.parents:
        raise ValueError("output must be outside the source repository")
    out.mkdir(parents=True, exist_ok=True)
    tasks = [{
        "dataset": str(row["dataset"]), "stem": str(row["stem"]), "cache_path": row["cache_path"],
        "image_path": row.get("image_path", ""), "gt_path": row.get("gt_path", ""),
        "methods": methods, "references": references,
    } for row in rows]
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.torch_threads,)) as pool:
        for index, result in enumerate(pool.map(_process, tasks, chunksize=1), 1):
            results.append(result)
            if index % 20 == 0 or index == len(tasks):
                print(f"GBSP-SPE eval {index}/{len(tasks)}", flush=True)
    failures = [result for result in results if "error" in result]
    valid = [result for result in results if "error" not in result]
    _write_json(out / "evaluation_failures.json", failures)
    if not valid:
        raise RuntimeError("no valid evaluations")
    per_image = [row for result in valid for row in result["rows"]]
    summary = _aggregate(per_image)
    counts = dict(Counter(task["dataset"] for task in tasks))
    stage = "test20" if len(tasks) == 20 else "diagnostic200" if len(tasks) == 200 else (
        "full6473" if len(tasks) == 6473 and counts == EXPECTED else "custom"
    )
    bootstrap = _bootstrap(per_image, methods, args.bootstrap_repetitions, args.bootstrap_seed)
    _write_csv(out / "per_image_metrics.csv", per_image, _fields(per_image))
    _write_csv(out / "per_dataset_metrics.csv", summary, _fields(summary))
    macro = [row for row in summary if row["scope"] == "dataset_macro"]
    _write_csv(out / f"{stage}_summary.csv", macro, _fields(macro))
    _write_csv(out / "primary_metric_deltas.csv", [
        {key: row.get(key) for key in ("dataset", "stem", "method", "delta_S_vs_fixed_058", "delta_F_beta_w_vs_fixed_058", "delta_E_vs_fixed_058", "delta_MAE_vs_fixed_058")}
        for row in per_image
    ])
    _write_csv(out / "secondary_metric_deltas.csv", [
        {key: row.get(key) for key in ("dataset", "stem", "method", "delta_Fmean_vs_fixed_058", "delta_Precision_vs_fixed_058", "delta_Recall_vs_fixed_058", "delta_Area_vs_fixed_058", "delta_IoU_vs_fixed_058", "delta_Dice_vs_fixed_058")}
        for row in per_image
    ])
    _write_csv(out / "bootstrap_ci95.csv", bootstrap, _fields(bootstrap))
    win_rows = []
    for method in methods:
        subset = [row for row in per_image if row["method"] == method]
        for metric in PRIMARY:
            delta = DELTA_FIELDS[metric]
            values = np.asarray([float(row[delta]) for row in subset if math.isfinite(float(row[delta]))], dtype=np.float64)
            wins = values < 0 if metric == "MAE" else values > 0
            win_rows.append({"method": method, "metric": metric, "win_rate": float(wins.mean()) if values.size else float("nan"), "num_samples": len(values)})
    _write_csv(out / "per_image_win_rates.csv", win_rows)
    invalid = {
        method: sum(int(row.get("numerical_failure", 0)) for row in per_image if row["method"] == method)
        for method in methods
    }
    rank_invariance = {}
    for method in ("gbsp_fixed_050", "gbsp_fixed_058", "smoh", *methods):
        values = [row for row in per_image if row["method"] == method]
        if values:
            rank_invariance[method] = {"pixel_AP": _mean(row["pixel_AP"] for row in values), "pixel_AUROC": _mean(row["pixel_AUROC"] for row in values)}
    numerical = {
        "schema": "gbsp_spe_calibration_eval_v1", "stage": stage,
        "num_requested": len(tasks), "num_valid": len(valid), "evaluation_failed": len(failures),
        "dataset_counts": counts, "method_invalid_counts": invalid,
        "gt_used_for_calibration": False, "fixed_058_used_in_spe_formula": False,
        "rank_metric_audit": rank_invariance, "wall_seconds": time.perf_counter() - started,
    }
    _write_json(out / "numerical_validity_summary.json", numerical)
    _write_csv(out / "downstream_1x1_results.csv", [{"method": "jm_spe_095", "status": "not_run_full_gate_required"}])
    report = [
        "# GBSP-SPE Calibration Evaluation", "", f"- Stage: {stage}",
        f"- Evaluated: {len(valid)}/{len(tasks)}", f"- Evaluation failures: {len(failures)}",
        f"- JM-SPE95 invalid: {invalid.get('jm_spe_095', 0)}", "",
        _table(macro, ("method", *PRIMARY, *SECONDARY)),
    ]
    (out / "GBSP_SPE_CALIBRATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(numerical, ensure_ascii=False, indent=2))
    if failures and args.failure_policy == "strict":
        raise RuntimeError(f"{len(failures)} GBSP-SPE evaluation failures")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", default=str(DEFAULT_CONFIG))
    value.add_argument("--calibration_root", required=True)
    value.add_argument("--sample_list")
    value.add_argument("--split", default="test", choices=("test",))
    value.add_argument("--max_samples", type=int, default=-1)
    value.add_argument("--methods", nargs="+")
    value.add_argument("--compare", nargs="*")
    value.add_argument("--bootstrap_repetitions", type=int, default=0)
    value.add_argument("--bootstrap_seed", type=int, default=20260806)
    value.add_argument("--out_dir", required=True)
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--torch_threads", type=int, default=1)
    value.add_argument("--failure-policy", choices=("record", "strict"), default="record")
    return value


if __name__ == "__main__":
    evaluate(parser().parse_args())
