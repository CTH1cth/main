import argparse
import csv
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.static_weight import (  # noqa: E402
    STATIC_WEIGHT_REGIONS,
    build_static_weight_region_masks,
)
from common.utils import (  # noqa: E402
    build_image_items,
    dabe_pu_manifest_path,
    ensure_dir,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
)


PAYLOAD_FIELDS = {
    "weight": "weight_map_68",
    "fg_core": "fg_core_pu_68",
    "fg_fallback": "fg_core_fallback_68",
    "bg_core": "bg_core_pu_68",
    "extent": "extent_candidate_68",
    "unknown": "unknown_68",
}


def _payload_tensor(payload, key, cache_path, expected_shape):
    tensor = payload.get(key)
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"Missing {key} in DABE-PU payload: {cache_path}")
    tensor = tensor.float()
    if list(tensor.shape) != list(expected_shape):
        raise RuntimeError(
            f"{key} shape mismatch in {cache_path}: "
            f"{list(tensor.shape)} != {list(expected_shape)}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"{key} contains NaN/Inf: {cache_path}")
    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise RuntimeError(
            f"{key} outside [0,1] in {cache_path}: "
            f"{min_value:.8f}/{max_value:.8f}"
        )
    return tensor.clamp(0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the cached DABE-PU static weight map without training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--out",
        default="../workdir/static_weight_cache_audit",
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer")
    return args


def main():
    args = parse_args()
    cfg = load_config(args.config)
    manifest_path = dabe_pu_manifest_path(cfg)
    rows = read_jsonl(manifest_path)
    row_map = manifest_to_map(rows, manifest_path)
    items = build_image_items(
        cfg.DATA_ROOT,
        cfg.TRAIN_DATASETS,
        require_gt=False,
    )
    expected_total = len(items)
    if args.max_samples > 0:
        items = items[: args.max_samples]
    expected_shape = [1, int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)]

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    per_sample_rows = []
    raw_sum = 0.0
    raw_count = 0
    raw_min = None
    raw_max = None
    zero_count = 0
    low_count = 0
    full_count = 0
    region_pixels = {name: 0 for name in STATIC_WEIGHT_REGIONS}
    region_weight = {name: 0.0 for name in STATIC_WEIGHT_REGIONS}

    for item in items:
        key = (item["dataset"], item["stem"])
        if key not in row_map:
            raise RuntimeError(f"DABE-PU manifest missing {key}")
        manifest_row = row_map[key]
        cache_path = Path(manifest_row["cache_path"])
        payload = torch_load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(f"DABE-PU payload must be dict: {cache_path}")
        if payload.get("dataset") != key[0] or payload.get("stem") != key[1]:
            raise RuntimeError(
                f"DABE-PU payload identity mismatch for {key}: {cache_path}"
            )
        tensors = {
            name: _payload_tensor(
                payload,
                payload_key,
                cache_path,
                expected_shape,
            )
            for name, payload_key in PAYLOAD_FIELDS.items()
        }
        weight = tensors["weight"].unsqueeze(0)
        batch = {
            "pu_fg_core": tensors["fg_core"].unsqueeze(0),
            "pu_fg_fallback": tensors["fg_fallback"].unsqueeze(0),
            "pu_bg_core": tensors["bg_core"].unsqueeze(0),
            "pu_extent": tensors["extent"].unsqueeze(0),
            "pu_unknown": tensors["unknown"].unsqueeze(0),
        }
        masks = build_static_weight_region_masks(
            batch,
            torch.device("cpu"),
            expected_shape=weight.shape,
        )

        pixel_count = int(weight.numel())
        weight_sum = float(weight.double().sum().item())
        raw_sum += weight_sum
        raw_count += pixel_count
        image_min = float(weight.min().item())
        image_max = float(weight.max().item())
        raw_min = image_min if raw_min is None else min(raw_min, image_min)
        raw_max = image_max if raw_max is None else max(raw_max, image_max)
        zero_count += int((weight <= 1e-8).sum().item())
        low_count += int((weight <= 0.10).sum().item())
        full_count += int((weight >= 0.99).sum().item())

        sample_row = {
            "dataset": key[0],
            "stem": key[1],
            "cache_path": str(cache_path),
            "weight_min": image_min,
            "weight_mean": weight_sum / pixel_count,
            "weight_max": image_max,
            "weight_zero_ratio": float((weight <= 1e-8).float().mean().item()),
            "weight_le_010_ratio": float((weight <= 0.10).float().mean().item()),
            "weight_ge_099_ratio": float((weight >= 0.99).float().mean().item()),
        }
        for name in STATIC_WEIGHT_REGIONS:
            mask = masks[name]
            count = int(mask.sum().item())
            region_sum = float(weight[mask].double().sum().item())
            region_pixels[name] += count
            region_weight[name] += region_sum
            sample_row[f"{name}_area"] = count / pixel_count
            sample_row[f"{name}_weight_mean"] = (
                region_sum / count if count > 0 else 0.0
            )
            sample_row[f"{name}_weight_mass"] = (
                region_sum / weight_sum if weight_sum > 0.0 else 0.0
            )
        per_sample_rows.append(sample_row)

    if not per_sample_rows or raw_count <= 0:
        raise RuntimeError("DABE-PU static-weight audit selected no samples")
    summary = {
        "schema_version": "dabe_pu_static_weight_audit_v1",
        "config": str(Path(args.config).resolve()),
        "manifest": str(manifest_path.resolve()),
        "uses_gt": False,
        "processed_samples": len(per_sample_rows),
        "expected_full_samples": expected_total,
        "full_audit": len(per_sample_rows) == expected_total,
        "raw_weight_min": raw_min,
        "raw_weight_mean": raw_sum / raw_count,
        "raw_weight_max": raw_max,
        "zero_weight_pixel_ratio": zero_count / raw_count,
        "weight_le_010_ratio": low_count / raw_count,
        "weight_ge_099_ratio": full_count / raw_count,
        "regions": {},
    }
    for name in STATIC_WEIGHT_REGIONS:
        count = region_pixels[name]
        summary["regions"][name] = {
            "area_ratio": count / raw_count,
            "raw_weight_mean": (
                region_weight[name] / count if count > 0 else 0.0
            ),
            "raw_weight_mass": (
                region_weight[name] / raw_sum if raw_sum > 0.0 else 0.0
            ),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    csv_path = out_dir / "per_sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0]))
        writer.writeheader()
        writer.writerows(per_sample_rows)
    print(
        "DABE-PU static-weight audit complete | "
        f"samples={len(per_sample_rows)}/{expected_total} | "
        f"summary={summary_path} | per_sample={csv_path}"
    )


if __name__ == "__main__":
    main()
