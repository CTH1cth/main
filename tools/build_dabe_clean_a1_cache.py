#!/usr/bin/env python3
"""Build the enriched A1 DABE->Clean cache in one command and one directory."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.cache_dabe_pseudo import _params_from_cfg  # noqa: E402
from common.dabe_clean import DABE_CLEAN_VERSION, build_background_evidence, build_clean_targets  # noqa: E402
from common.dabe_clean_cache import cache_manifest_row, parse_bool  # noqa: E402
from common.dabe_pseudo import generate_dabe_pseudo  # noqa: E402
from common.utils import (  # noqa: E402
    build_image_items,
    ensure_dir,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
    write_jsonl,
)


DEFAULT_CONFIG = (
    "configs/dinov1_s8_dabe_clean_v1_dp_noecst_a1_residual_only_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
SOURCE_KEYS = (
    "residual_pass1_37",
    "residual_37",
    "fg_score_37",
    "fg_core_37",
    "bg_core_37",
    "p_rw_37",
    "evidence_37",
    "p_base_37",
    "p_base_68",
    "bc_map_37",
)


def _prepare_output(path, overwrite):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is non-empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feature(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if str(payload.get("dataset")) != dataset or str(payload.get("stem")) != stem:
        raise RuntimeError(f"Feature identity mismatch for {dataset}/{stem}.")
    tensor = payload.get("tensor")
    if not torch.is_tensor(tensor) or tuple(tensor.shape) != (384, 37, 37):
        raise RuntimeError(f"Invalid DINO feature for {dataset}/{stem}.")
    tensor = tensor.detach().cpu().float()
    if tensor.requires_grad or not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"Feature must be detached and finite for {dataset}/{stem}.")
    return tensor


def _detached_map(result, key, shape):
    value = result.get(key)
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
        raise RuntimeError(f"Invalid {key}: expected {list(shape)}.")
    value = value.detach().cpu().float().clamp(0.0, 1.0).contiguous()
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{key} must be detached and finite.")
    return value


def _build_payload(result, item, params, ablation):
    source = {
        key: _detached_map(
            result,
            key,
            (1, 68, 68) if key.endswith("_68") else (1, 37, 37),
        )
        for key in SOURCE_KEYS
    }
    foreground_37 = source["p_base_37"]
    foreground_68 = source["p_base_68"]
    background_37 = build_background_evidence(
        source["bc_map_37"], source["residual_37"]
    )
    background_68 = F.interpolate(
        background_37.unsqueeze(0),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0).detach()
    clean_37 = build_clean_targets(foreground_37, background_37)
    clean_68 = build_clean_targets(foreground_68, background_68)
    payload = {
        **source,
        "foreground_evidence_37": clean_37["foreground_evidence"],
        "foreground_evidence_68": clean_68["foreground_evidence"],
        "background_evidence_37": clean_37["background_evidence"],
        "background_evidence_68": clean_68["background_evidence"],
        "target_dp_37": clean_37["target_dp"],
        "target_dp_68": clean_68["target_dp"],
        "target_diff_37": clean_37["target_diff"],
        "target_diff_68": clean_68["target_diff"],
        "source_p_base_37": foreground_37,
        "source_p_base_68": foreground_68,
        "source_bc_map_37": source["bc_map_37"],
        "source_residual_37": source["residual_37"],
        "dataset": str(item["dataset"]),
        "stem": str(item["stem"]),
        "image_path": str(item["image_path"]),
        "gt_path": str(item["gt_path"]),
        "backbone_key": "dinov1-s8",
        "version": DABE_CLEAN_VERSION,
        "source_version": "pu_v11",
        "dabe_version": "pu_v11",
        "params": dict(params),
        "ablation": str(ablation),
        "training_gt_read": False,
        "teacher_prediction_read": False,
        "static_weight_map_written": False,
    }
    score_error = float(
        (payload["fg_score_37"] - payload["residual_37"]).abs().max().item()
    )
    if score_error > 1e-6:
        raise RuntimeError(
            f"A1 formula failed for {item['dataset']}/{item['stem']}: {score_error}"
        )
    if not torch.equal(
        payload["target_dp_68"] > 0.5,
        payload["p_base_68"] > 0.5,
    ):
        raise RuntimeError(
            f"Clean-DP direction failed for {item['dataset']}/{item['stem']}."
        )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--datasets", default="TR-CAMO,TR-COD10K")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        parser.error("--max-samples must be -1 or a positive integer.")

    cfg = load_config(args.config)
    if str(getattr(cfg, "BACKBONE_KEY", "")) != "dinov1-s8":
        raise RuntimeError("A1 cache supports only BACKBONE_KEY=dinov1-s8.")
    if float(getattr(cfg, "DABE_BC_LAMBDA", -1.0)) != 0.0:
        raise RuntimeError("A1 cache requires DABE_BC_LAMBDA=0.0.")
    if not bool(getattr(cfg, "USE_DABE_CLEAN", False)):
        raise RuntimeError("A1 config must enable USE_DABE_CLEAN.")
    output_root = _prepare_output(
        args.output_root or str(cfg.DABE_CLEAN_ROOT), args.overwrite
    )
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    items = build_image_items(cfg.DATA_ROOT, datasets, require_gt=True)
    if args.max_samples > 0:
        items = items[: args.max_samples]
    if not items:
        raise RuntimeError("No A1 cache samples selected.")
    feature_manifest = feature_manifest_path(cfg, "train")
    feature_map = manifest_to_map(read_jsonl(feature_manifest), feature_manifest)
    params = _params_from_cfg(cfg)
    if set(params) != {"BC_LAMBDA"} or float(params["BC_LAMBDA"]) != 0.0:
        raise RuntimeError(f"Unexpected A1 DABE config parameters: {params}")
    params["VERSION"] = "pu_v11"
    ablation = str(cfg.DABE_CLEAN_ABLATION)

    manifest_rows = []
    for item in tqdm(items, desc="build enriched A1 Clean cache"):
        dataset, stem = str(item["dataset"]), str(item["stem"])
        key = (dataset, stem)
        if key not in feature_map:
            raise RuntimeError(f"Feature cache is missing {dataset}/{stem}.")
        result = generate_dabe_pseudo(
            _feature(feature_map[key], dataset, stem),
            item["image_path"],
            params=params,
            augs=["identity"],
        )
        payload = _build_payload(result, item, result["params"], ablation)
        output_dir = output_root / dataset
        ensure_dir(output_dir)
        output_path = output_dir / f"{stem}.pt"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite A1 payload: {output_path}")
        torch.save(payload, output_path)
        row = cache_manifest_row(payload, output_path)
        row.update(
            {
                "ablation": ablation,
                "enriched_source_maps": True,
                "bc_lambda": 0.0,
            }
        )
        manifest_rows.append(row)

    manifest = output_root / "manifest_train.jsonl"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {manifest}")
    write_jsonl(manifest, manifest_rows)
    print("TRAIN_GT_READ=False")
    print("TEACHER_PREDICTION_READ=False")
    print("ONE_DIRECTORY=True")
    print("ENRICHED_SOURCE_MAPS=True")
    print(f"samples={len(manifest_rows)}")
    print(f"output_root={output_root}")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
