#!/usr/bin/env python3
"""Build the GT-free DABE retrieval-index cache used by BITC-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import torch

from common.bitc import (
    BITC_CACHE_SCHEMA,
    bitc_cache_fingerprint,
    bitc_cache_protocol_payload,
    build_bitc_retrieval_cache,
)
from common.dabe_pseudo import _load_rgb_grid
from common.utils import load_config, read_jsonl, torch_load


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BITC-v1 DABE background retrieval index/weight cache"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--feature_root", required=True)
    parser.add_argument("--dabe_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max_samples", type=int, default=-1)
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty BITC cache root: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _map(rows: list[dict[str, Any]], manifest: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("stem")))
        if key in result:
            raise RuntimeError(f"Duplicate cache key in {manifest}: {key}")
        cache_path = row.get("cache_path")
        if not cache_path:
            raise RuntimeError(f"Missing cache_path in {manifest}: {row}")
        result[key] = row
    return result


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load_feature(row: dict[str, Any], key: tuple[str, str]) -> torch.Tensor:
    payload = torch_load(row["cache_path"], map_location="cpu")
    if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
        raise RuntimeError(f"Feature provenance mismatch for {key}: {row['cache_path']}")
    feature = payload.get("tensor")
    if not torch.is_tensor(feature) or list(feature.shape) != [384, 37, 37]:
        raise RuntimeError(
            f"BITC feature must be [384,37,37] for {key}, got "
            f"{list(feature.shape) if torch.is_tensor(feature) else type(feature)}"
        )
    if not bool(torch.isfinite(feature).all().item()):
        raise RuntimeError(f"BITC feature contains NaN/Inf for {key}")
    return feature.detach().float()


def _load_dabe(row: dict[str, Any], key: tuple[str, str], cfg) -> dict[str, Any]:
    payload = torch_load(row["cache_path"], map_location="cpu")
    checks = {
        "dataset": key[0],
        "stem": key[1],
        "backbone_key": cfg.BACKBONE_KEY,
        "dabe_version": str(getattr(cfg, "DABE_PU_VERSION", "pu_v11")),
    }
    mismatch = {
        name: (payload.get(name), expected)
        for name, expected in checks.items()
        if payload.get(name) != expected
    }
    if mismatch:
        raise RuntimeError(f"DABE provenance mismatch for {key}: {mismatch}")
    anchor = payload.get("bg_anchor_37")
    params = payload.get("params")
    if not torch.is_tensor(anchor) or list(anchor.shape) != [1, 37, 37]:
        raise RuntimeError(f"DABE bg_anchor_37 is unavailable for {key}")
    if not isinstance(params, dict):
        raise RuntimeError(f"DABE params are unavailable for {key}")
    return payload


def main() -> None:
    args = _parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer")
    cfg_path = _resolve(args.config)
    feature_root = _resolve(args.feature_root)
    dabe_root = _resolve(args.dabe_root)
    output_root = _resolve(args.output_root)
    cfg = load_config(cfg_path)
    if str(getattr(cfg, "DABE_PU_VERSION", "")).lower() != "pu_v11":
        raise RuntimeError("BITC-v1 cache builder requires DABE_PU_VERSION='pu_v11'")
    feature_manifest = feature_root / "manifest_train.jsonl"
    dabe_manifest = dabe_root / "manifest_train.jsonl"
    if not feature_manifest.is_file() or not dabe_manifest.is_file():
        raise FileNotFoundError(
            f"Missing feature/DABE manifest: {feature_manifest} / {dabe_manifest}"
        )
    feature_rows = read_jsonl(feature_manifest)
    dabe_rows = read_jsonl(dabe_manifest)
    feature_map = _map(feature_rows, feature_manifest)
    dabe_map = _map(dabe_rows, dabe_manifest)
    dataset_set = {str(name) for name in args.datasets}
    ordered_keys = [
        (str(row["dataset"]), str(row["stem"]))
        for row in feature_rows
        if str(row["dataset"]) in dataset_set
    ]
    missing_dabe = [key for key in ordered_keys if key not in dabe_map]
    if missing_dabe:
        raise RuntimeError(f"DABE cache misses feature keys: {missing_dabe[:10]}")
    extra_dabe = sorted(
        key for key in dabe_map if key[0] in dataset_set and key not in feature_map
    )
    if extra_dabe:
        raise RuntimeError(f"DABE cache has unmatched keys: {extra_dabe[:10]}")
    if args.max_samples > 0:
        ordered_keys = ordered_keys[: args.max_samples]
    if not ordered_keys:
        raise RuntimeError("No samples matched the requested BITC datasets")
    selected_counts = Counter(dataset for dataset, _ in ordered_keys)
    missing_datasets = sorted(dataset_set.difference(selected_counts))
    if missing_datasets:
        raise RuntimeError(f"Requested BITC datasets are empty: {missing_datasets}")
    if args.max_samples == -1 and dataset_set == {"TR-CAMO", "TR-COD10K"}:
        expected_counts = {"TR-CAMO": 1000, "TR-COD10K": 3040}
        if dict(selected_counts) != expected_counts:
            raise RuntimeError(
                "BITC full-cache cardinality mismatch: "
                f"actual={dict(selected_counts)}, expected={expected_counts}"
            )
    _prepare_output(output_root)

    manifest_rows = []
    fingerprints = set()
    for index, key in enumerate(ordered_keys, start=1):
        feature_row = feature_map[key]
        dabe_row = dabe_map[key]
        feature = _load_feature(feature_row, key)
        dabe = _load_dabe(dabe_row, key, cfg)
        image_path = Path(
            feature_row.get("image_path") or dabe_row.get("image_path") or ""
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"BITC RGB image is unavailable for {key}: {image_path}")
        rgb_grid = _load_rgb_grid(image_path, 37).detach().float()
        retrieval = build_bitc_retrieval_cache(
            feature,
            rgb_grid,
            dabe["bg_anchor_37"],
            dabe["params"],
            seed=args.seed,
            sample_key=f"{key[0]}::{key[1]}",
        )
        protocol = bitc_cache_protocol_payload(
            backbone_key=cfg.BACKBONE_KEY,
            dabe_version=dabe["dabe_version"],
            params=dabe["params"],
            seed=args.seed,
        )
        fingerprint = bitc_cache_fingerprint(protocol)
        fingerprints.add(fingerprint)
        payload = {
            "schema": BITC_CACHE_SCHEMA,
            "dataset": key[0],
            "stem": key[1],
            "backbone_key": cfg.BACKBONE_KEY,
            "dabe_version": dabe["dabe_version"],
            "feature_version": cfg.BACKBONE_KEY,
            "seed": int(args.seed),
            "cache_protocol": protocol,
            "cache_fingerprint": fingerprint,
            "source_feature_cache": str(Path(feature_row["cache_path"]).resolve()),
            "source_dabe_cache": str(Path(dabe_row["cache_path"]).resolve()),
            **retrieval,
        }
        destination = output_root / key[0] / f"{key[1]}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        row = {
            "dataset": key[0],
            "stem": key[1],
            "cache_path": str(destination.resolve()),
            "backbone_key": cfg.BACKBONE_KEY,
            "dabe_version": dabe["dabe_version"],
            "schema": BITC_CACHE_SCHEMA,
            "cache_fingerprint": fingerprint,
            "topk": int(retrieval["topk"]),
            "background_anchor_count": int(retrieval["background_anchor_count"]),
            "sha256": _sha256(destination),
        }
        manifest_rows.append(row)
        print(
            f"[BITC cache] {index}/{len(ordered_keys)} {key[0]}/{key[1]} | "
            f"anchors={row['background_anchor_count']} | topk={row['topk']}"
        )
    if len(fingerprints) != 1:
        raise RuntimeError(f"BITC cache protocol fingerprint changed across samples: {fingerprints}")
    manifest_path = output_root / "manifest_train.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema": BITC_CACHE_SCHEMA,
        "sample_count": len(manifest_rows),
        "dataset_counts": dict(sorted(selected_counts.items())),
        "requested_datasets": list(args.datasets),
        "full_cache": bool(args.max_samples == -1),
        "cache_fingerprint": next(iter(fingerprints)),
        "manifest": str(manifest_path.resolve()),
        "gt_read": False,
    }
    with (output_root / "cache_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
