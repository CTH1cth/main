#!/usr/bin/env python3
"""Materialize training-ready identity+hflip soft-mean R32 GBSP targets.

The source diagnostic cache already contains independently generated,
inverse-aligned true-RGB views.  This tool only averages the identity and
hflip continuous maps.  It never reads GT and never starts training.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import read_jsonl, torch_load, write_json, write_jsonl  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
CACHE_ROOT = (PROJECT_ROOT / "datasets/cache").resolve()
SOURCE_VERSION = "gbsp_r32_true_rgb_d4_multiview_v1"
OUTPUT_VERSION = "gbsp_r32_rgbmv_id_hflip_softmean_v1"
VIEW_ORDER = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")
FUSED_VIEWS = ("identity", "hflip")
EXPECTED_COUNTS = {"TR-CAMO": 1000, "TR-COD10K": 3040}
DEFAULT_SOURCE = CACHE_ROOT / "gbsp_r32_rgb_multiview_v1/dinov1-s8/manifest_train.jsonl"
DEFAULT_OUTPUT = CACHE_ROOT / "gbsp_r32_rgbmv_id_hflip_softmean_v1/dinov1-s8"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (MAIN_ROOT / value).resolve()


def validate_output_root(path: str | Path) -> Path:
    output = _resolve(path)
    if output == CACHE_ROOT or CACHE_ROOT not in output.parents:
        raise ValueError(f"output_root must be a child of {CACHE_ROOT}: {output}")
    return output


def fuse_identity_hflip(payload: dict, source_path: str | Path) -> torch.Tensor:
    """Validate source provenance and return the equal continuous mean."""

    path = Path(source_path)
    if str(payload.get("version")) != SOURCE_VERSION:
        raise RuntimeError(f"unexpected source version: {path}")
    if tuple(payload.get("view_order", ())) != VIEW_ORDER:
        raise RuntimeError(f"unexpected source view order: {path}")
    if str(payload.get("backbone_key")) != "dinov1-s8":
        raise RuntimeError(f"unexpected source backbone: {path}")
    if not bool(payload.get("transformed_rgb_dino_forward_used")):
        raise RuntimeError(f"hflip was not produced from transformed RGB: {path}")
    if bool(payload.get("feature_only_transform_used", True)):
        raise RuntimeError(f"feature-only transform is forbidden: {path}")
    if not bool(payload.get("graph_path_used")):
        raise RuntimeError(f"GBSP graph path was not used: {path}")
    if bool(payload.get("gt_used_for_generation", True)):
        raise RuntimeError(f"source reports GT use during generation: {path}")
    if str(payload.get("pca_rank_mode")) != "fixed" or int(
        payload.get("fixed_pca_rank", -1)
    ) != 32:
        raise RuntimeError(f"source is not fixed R32: {path}")
    if int(payload.get("candidate_border_width", -1)) != 2 or abs(
        float(payload.get("candidate_top_percent", -1.0)) - 30.0
    ) > 1e-12:
        raise RuntimeError(f"source GBSP candidate protocol mismatch: {path}")

    scores = payload.get("aligned_scores_37")
    if not torch.is_tensor(scores) or tuple(scores.shape) != (6, 1, 37, 37):
        raise RuntimeError(f"invalid aligned score tensor: {path}")
    scores = scores.detach().cpu().float()
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError(f"non-finite aligned score tensor: {path}")
    if float(scores.min()) < -1e-6 or float(scores.max()) > 1.0 + 1e-6:
        raise RuntimeError(f"aligned score tensor is outside [0,1]: {path}")

    # Deliberately no post-fusion min-max: preserve the equal-view confidence.
    return (0.5 * (scores[0] + scores[1])).clamp(0.0, 1.0).contiguous()


def build_training_payload(source: dict, source_path: Path) -> dict:
    fused = fuse_identity_hflip(source, source_path)
    soft68 = F.interpolate(
        fused.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    ).squeeze(0)
    hard_area = float((soft68 > 0.50).float().mean().item())
    return {
        "dataset": str(source["dataset"]),
        "stem": str(source["stem"]),
        "image_path": str(source["image_path"]),
        "backbone_key": "dinov1-s8",
        "dabe_version": "v2",
        "source_dabe_version": "v2",
        "gbsp_version": OUTPUT_VERSION,
        "source_augs": list(FUSED_VIEWS),
        "source_num_views": len(FUSED_VIEWS),
        "generation_stage": "true_rgb_fixed_r32_identity_hflip_softmean",
        "dino_forward_used": True,
        "transformed_rgb_dino_forward_used": True,
        "feature_only_transform_used": False,
        "inverse_aligned": True,
        "graph_path_used": True,
        "graph_path_method": str(source.get("graph_path_method", "")),
        "candidate_border_width": 2,
        "candidate_top_percent": 30.0,
        "gt_used_for_generation": False,
        "foreground_seed_used": False,
        "random_walk_used": False,
        "evidence_gate_used": False,
        "fallback_used": False,
        "fallback_reason": "",
        "num_subspaces": 1,
        "pca_energy": 0.90,
        "pca_min_rank": 1,
        "pca_max_rank": 32,
        "pca_rank_mode": "fixed",
        "fixed_pca_rank": 32,
        "selected_ranks": torch.tensor([32], dtype=torch.int64),
        "view_selected_ranks": {"identity": 32, "hflip": 32},
        "gbsp_abs_minmax_37": fused,
        "static_threshold": 0.50,
        "hard_area_at_static_threshold": hard_area,
        "fusion": "identity_hflip_continuous_equal_mean",
        "fusion_weights": [0.5, 0.5],
        "post_fusion_minmax_used": False,
        "per_view_minmax_used": bool(source.get("per_view_minmax_used", False)),
        "source_multiview_cache_path": str(source_path.resolve()),
        "source_multiview_version": SOURCE_VERSION,
        "source_settings_fingerprint": str(source.get("settings_fingerprint", "")),
    }


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _valid_existing(path: Path, dataset: str, stem: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch_load(path, map_location="cpu")
        score = payload.get("gbsp_abs_minmax_37")
        return (
            payload.get("dataset") == dataset
            and payload.get("stem") == stem
            and payload.get("gbsp_version") == OUTPUT_VERSION
            and payload.get("fusion") == "identity_hflip_continuous_equal_mean"
            and payload.get("post_fusion_minmax_used") is False
            and payload.get("gt_used_for_generation") is False
            and torch.is_tensor(score)
            and tuple(score.shape) == (1, 37, 37)
            and bool(torch.isfinite(score).all())
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False


def materialize(
    source_manifest: str | Path,
    output_root: str | Path,
    max_samples: int = -1,
    force: bool = False,
) -> dict:
    source_path = _resolve(source_manifest)
    output = validate_output_root(output_root)
    rows = read_jsonl(source_path)
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or a positive integer")
    selected = rows if max_samples == -1 else rows[:max_samples]
    identities = [(str(row["dataset"]), str(row["stem"])) for row in selected]
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"duplicate source identities: {source_path}")

    manifest_rows = []
    generated = reused = 0
    for index, row in enumerate(selected, 1):
        dataset, stem = str(row["dataset"]), str(row["stem"])
        cache_path = output / "train" / dataset / f"{stem}.pt"
        if _valid_existing(cache_path, dataset, stem) and not force:
            reused += 1
        else:
            source_cache = Path(row["cache_path"]).resolve()
            source = torch_load(source_cache, map_location="cpu")
            if source.get("dataset") != dataset or source.get("stem") != stem:
                raise RuntimeError(f"source identity mismatch: {source_cache}")
            _atomic_save(build_training_payload(source, source_cache), cache_path)
            generated += 1
        manifest_rows.append(
            {
                "dataset": dataset,
                "stem": stem,
                "image_path": str(row.get("image_path", "")),
                "cache_path": str(cache_path.resolve()),
                "backbone_key": "dinov1-s8",
                "source_dabe_version": "v2",
                "gbsp_version": OUTPUT_VERSION,
                "source_augs": list(FUSED_VIEWS),
                "source_num_views": len(FUSED_VIEWS),
                "shape": [1, 37, 37],
                "fallback_used": False,
                "fallback_reason": "",
            }
        )
        if index % 500 == 0 or index == len(selected):
            print(
                f"[{_now()}] {index}/{len(selected)} generated={generated} reused={reused}",
                flush=True,
            )

    counts = dict(Counter(row["dataset"] for row in manifest_rows))
    formal = max_samples == -1
    if formal and (len(manifest_rows) != 4040 or counts != EXPECTED_COUNTS):
        raise RuntimeError(
            f"formal cache count mismatch: total={len(manifest_rows)}, counts={counts}"
        )
    write_jsonl(output / "manifest_train.jsonl", manifest_rows)
    protocol = {
        "version": OUTPUT_VERSION,
        "created_at": _now(),
        "source_manifest": str(source_path),
        "source_version": SOURCE_VERSION,
        "output_root": str(output),
        "formal_full4040": formal,
        "num_samples": len(manifest_rows),
        "dataset_counts": counts,
        "views": list(FUSED_VIEWS),
        "fusion": "continuous_equal_mean",
        "fusion_weights": [0.5, 0.5],
        "post_fusion_minmax_used": False,
        "threshold": 0.50,
        "gt_used_for_generation": False,
        "training_used": False,
        "generated": generated,
        "reused": reused,
    }
    write_json(output / "protocol.json", protocol)
    return protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = materialize(
        args.source_manifest,
        args.output_root,
        max_samples=args.max_samples,
        force=args.force,
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
