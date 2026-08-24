#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

MAIN_ROOT = Path(__file__).resolve().parents[2]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.gbsp_teaser_analysis.common import (  # noqa: E402
    load_settings, read_jsonl, validate_output, write_csv, write_json,
)

PROTOCOL_SUFFIX = {"allpatch_0.5": "main", "core_0.2_0.8": "core"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _scalar(payload, key: str) -> float:
    return float(np.asarray(payload[key]).reshape(()))


def _hist_diagnostics(first: np.ndarray, second: np.ndarray, bins: np.ndarray) -> dict:
    centers = (bins[:-1] + bins[1:]) / 2
    return {
        "OVL": float(np.minimum(first, second).sum()),
        "Wasserstein": float(wasserstein_distance(centers, centers, u_weights=first, v_weights=second)),
    }


def analyze(config: str | Path, out_dir: str | Path) -> dict:
    settings = load_settings(config)
    out = validate_output(out_dir)
    rows = read_jsonl(out / "CAMO" / "manifest.jsonl")
    per_image_counts, similarity_rows, residual_rows = [], [], []
    histograms: dict[str, dict[str, np.ndarray]] = {}
    for protocol, suffix in PROTOCOL_SUFFIX.items():
        accumulator = {key: [] for key in ("bb", "fb", "bg", "fg")}
        valid_similarity = 0; valid_residual = 0
        for row in rows:
            with np.load(row["cache_path"], allow_pickle=False) as payload:
                required = (f"bb_hist_{suffix}", f"fb_hist_{suffix}", f"gbsp_bg_hist_{suffix}", f"gbsp_fg_hist_{suffix}")
                if not all(key in payload for key in required):
                    continue
                accumulator["bb"].append(payload[f"bb_hist_{suffix}"].astype(np.float64))
                accumulator["fb"].append(payload[f"fb_hist_{suffix}"].astype(np.float64))
                accumulator["bg"].append(payload[f"gbsp_bg_hist_{suffix}"].astype(np.float64))
                accumulator["fg"].append(payload[f"gbsp_fg_hist_{suffix}"].astype(np.float64))
                valid_similarity += 1; valid_residual += 1
                per_image_counts.append({
                    "dataset": "CAMO", "stem": row["stem"], "protocol": protocol,
                    "num_bg_patches": int(_scalar(payload, f"gbsp_num_bg_patches_{suffix}")),
                    "num_fg_patches": int(_scalar(payload, f"gbsp_num_fg_patches_{suffix}")),
                    "num_bb_pairs": int(_scalar(payload, f"num_bb_pairs_{suffix}")),
                    "num_fb_pairs": int(_scalar(payload, f"num_fb_pairs_{suffix}")),
                    "gbsp_candidate_count": int(_scalar(payload, "gbsp_candidate_count")),
                })
                for group in ("bb", "fb"):
                    similarity_rows.append({
                        "dataset": "CAMO", "stem": row["stem"], "protocol": protocol,
                        "group": "BG--BG" if group == "bb" else "FG--BG",
                        **{stat: _scalar(payload, f"{group}_{stat}_{suffix}") for stat in ("mean", "median", "q75", "q90")},
                        **{f"P(sim>{threshold:.1f})": _scalar(payload, f"{group}_p_gt_{threshold:.1f}_{suffix}") for threshold in settings.descriptive_thresholds},
                    })
                for group in ("bg", "fg"):
                    residual_rows.append({
                        "dataset": "CAMO", "stem": row["stem"], "protocol": protocol,
                        "group": "Background" if group == "bg" else "Foreground",
                        **{stat: _scalar(payload, f"gbsp_{group}_{stat}_{suffix}") for stat in ("mean", "median", "q75", "q90")},
                    })
        if not accumulator["bb"]:
            continue
        histograms[protocol] = {key: np.mean(np.stack(value), axis=0) for key, value in accumulator.items()}
        protocol_dir = out / "CAMO" / ("similarity" if suffix == "main" else "robustness_core")
        protocol_dir.mkdir(parents=True, exist_ok=True)
        if suffix == "main":
            np.save(out / "CAMO" / "similarity" / "bb_hist.npy", histograms[protocol]["bb"])
            np.save(out / "CAMO" / "similarity" / "fb_hist.npy", histograms[protocol]["fb"])
            residual_dir = out / "CAMO" / "residual"; residual_dir.mkdir(parents=True, exist_ok=True)
            np.save(residual_dir / "bg_hist.npy", histograms[protocol]["bg"])
            np.save(residual_dir / "fg_hist.npy", histograms[protocol]["fg"])
        else:
            for key, value in histograms[protocol].items():
                np.save(protocol_dir / f"{key}_hist.npy", value)

    def aggregate(rows_: list[dict], fields: tuple[str, ...]) -> list[dict]:
        output = []
        keys = sorted({(row["protocol"], row["group"]) for row in rows_})
        for protocol, group in keys:
            subset = [row for row in rows_ if row["protocol"] == protocol and row["group"] == group]
            output.append({
                "dataset": "CAMO", "protocol": protocol, "group": group, "valid_images": len(subset),
                **{field: float(np.mean([row[field] for row in subset])) for field in fields},
            })
        return output

    similarity_fields = ("mean", "median", "q75", "q90", *tuple(f"P(sim>{x:.1f})" for x in settings.descriptive_thresholds))
    residual_fields = ("mean", "median", "q75", "q90")
    similarity_summary = aggregate(similarity_rows, similarity_fields)
    residual_summary = aggregate(residual_rows, residual_fields)
    internal = []
    for protocol, values in histograms.items():
        internal.append({"dataset": "CAMO", "protocol": protocol, "family": "pairwise_similarity",
                         **_hist_diagnostics(values["bb"], values["fb"], settings.similarity_bins)})
        internal.append({"dataset": "CAMO", "protocol": protocol, "family": "gbsp_residual",
                         **_hist_diagnostics(values["bg"], values["fg"], settings.residual_bins)})
    diagnostics = out / "diagnostics"
    write_csv(diagnostics / "per_image_counts.csv", per_image_counts)
    write_csv(diagnostics / "per_image_similarity_statistics.csv", similarity_rows)
    write_csv(diagnostics / "per_image_residual_statistics.csv", residual_rows)
    write_csv(diagnostics / "similarity_statistics.csv", similarity_summary)
    write_csv(diagnostics / "residual_statistics.csv", residual_summary)
    write_csv(diagnostics / "internal_distribution_diagnostics.csv", internal)
    summary = {
        "images": len(rows), "formal_camo_complete": len(rows) == 250,
        "protocols": list(histograms), "image_balanced_histogram": True,
        "similarity_summary": similarity_summary, "residual_summary": residual_summary,
        "internal_diagnostics": internal,
        "cross_family_overlap_comparison_allowed": False,
    }
    write_json(diagnostics / "analysis_summary.json", summary)
    return summary


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(analyze(args.config, args.out_dir), ensure_ascii=False, indent=2))
