import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.cache_features import (  # noqa: E402
    call_dino,
    key_to_feature_tensor,
    load_dino,
    preprocess_image,
    resolve_key_projection_at_layer,
    split_datasets,
)
from common.utils import (  # noqa: E402
    build_image_items,
    ensure_dir,
    feature_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
    write_json,
    write_jsonl,
)


VERSION = "cacd_last3_extra_v1"
FEATURE_TYPE = "attention_key_projection"
LAYER_TO_FIELD = {9: "feature_l10", 10: "feature_l11", 11: "feature_l12"}
EXPECTED_LAYERS = (9, 10, 11)
STORE_LAYERS = (9, 10)
EXPECTED_SHAPE = [384, 37, 37]
BYTES_PER_FEATURE = 384 * 37 * 37 * 4


def _parse_csv_ints(value):
    try:
        return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value!r}") from exc


def _parse_splits(value):
    splits = tuple(part.strip().lower() for part in str(value).split(",") if part.strip())
    if not splits or any(split not in {"train", "val", "test"} for split in splits):
        raise argparse.ArgumentTypeError("--splits must contain train,val,test only.")
    if len(set(splits)) != len(splits):
        raise argparse.ArgumentTypeError(f"Duplicate split in --splits: {value!r}")
    return splits


def _items_for_split(cfg, split, max_samples=-1):
    items = build_image_items(cfg.DATA_ROOT, split_datasets(cfg, split), require_gt=False)
    if int(max_samples) >= 0:
        items = items[: int(max_samples)]
    return items


def _all_items_by_split(cfg, splits):
    return {
        split: build_image_items(cfg.DATA_ROOT, split_datasets(cfg, split), require_gt=False)
        for split in splits
    }


def _unique_items(items_by_split):
    out = {}
    for split, items in items_by_split.items():
        for item in items:
            key = (item["dataset"], item["stem"])
            existing = out.get(key)
            if existing is not None:
                if Path(existing["image_path"]).resolve() != Path(item["image_path"]).resolve():
                    raise RuntimeError(
                        f"Physical duplicate has different image paths for {key}: "
                        f"{existing['image_path']} vs {item['image_path']}"
                    )
                existing["splits"].append(split)
                continue
            out[key] = {**item, "splits": [split]}
    return out


def _normal_feature_maps(cfg, splits):
    maps = {}
    for split in splits:
        path = feature_manifest_path(cfg, split)
        split_map = manifest_to_map(read_jsonl(path), path)
        for key, row in split_map.items():
            if key in maps and Path(maps[key]["cache_path"]).resolve() != Path(row["cache_path"]).resolve():
                left = torch_load(maps[key]["cache_path"], map_location="cpu")["tensor"]
                right = torch_load(row["cache_path"], map_location="cpu")["tensor"]
                if not torch.equal(left, right):
                    raise RuntimeError(f"Normal F12 cache differs across split manifests for {key}.")
            maps[key] = row
    return maps


def _capture_features(model, inputs, holders):
    for holder in holders.values():
        holder["tensor"] = None
    with torch.no_grad():
        call_dino(model, inputs)
    features = {}
    for layer, holder in holders.items():
        key = holder["tensor"]
        if key is None:
            raise RuntimeError(f"DINO key hook produced no output for layer {layer}.")
        feature = key_to_feature_tensor(key)
        if list(feature.shape) != EXPECTED_SHAPE:
            raise RuntimeError(
                f"CACD layer {layer} feature shape mismatch: {list(feature.shape)} != {EXPECTED_SHAPE}"
            )
        if feature.dtype != torch.float32 or not bool(torch.isfinite(feature).all().item()):
            raise RuntimeError(f"CACD layer {layer} feature is not finite float32.")
        features[layer] = feature
    return features


def audit_f12_equivalence(cfg, model, holders, all_unique_items, splits, device, audit_samples=32):
    if int(audit_samples) < 32:
        raise ValueError("CACD fixed F12 audit requires at least 32 samples.")
    normal_maps = _normal_feature_maps(cfg, splits)
    keys = sorted(all_unique_items)[: min(int(audit_samples), len(all_unique_items))]
    if len(keys) < 32:
        raise RuntimeError(f"CACD F12 audit needs 32 unique samples, found {len(keys)}.")

    max_diffs = []
    mean_diffs = []
    pair_cosines = {"l10_l11": [], "l11_l12": [], "l10_l12": []}
    pair_max_diffs = {"l10_l11": [], "l11_l12": [], "l10_l12": []}
    layer_stats = {str(layer): {"mean": [], "std": [], "abs_mean": [], "min": [], "max": []} for layer in EXPECTED_LAYERS}
    for key in tqdm(keys, desc="CACD F12 fixed audit"):
        item = all_unique_items[key]
        inputs, _ = preprocess_image(item["image_path"], int(cfg.DINO["feature_input_size"]))
        features = _capture_features(model, inputs.to(device), holders)
        if key not in normal_maps:
            raise RuntimeError(f"Normal F12 cache missing during audit for {key}.")
        normal_payload = torch_load(normal_maps[key]["cache_path"], map_location="cpu")
        normal = normal_payload.get("tensor") if isinstance(normal_payload, dict) else None
        if not torch.is_tensor(normal) or list(normal.shape) != EXPECTED_SHAPE:
            raise RuntimeError(f"Invalid normal F12 cache during audit for {key}.")
        diff = (features[11] - normal.float()).abs()
        max_diffs.append(float(diff.max().item()))
        mean_diffs.append(float(diff.mean().item()))
        for layer, feature in features.items():
            stats = layer_stats[str(layer)]
            stats["mean"].append(float(feature.mean().item()))
            stats["std"].append(float(feature.std().item()))
            stats["abs_mean"].append(float(feature.abs().mean().item()))
            stats["min"].append(float(feature.min().item()))
            stats["max"].append(float(feature.max().item()))
        pair_cosines["l10_l11"].append(float(F.cosine_similarity(features[9], features[10], dim=0).mean().item()))
        pair_cosines["l11_l12"].append(float(F.cosine_similarity(features[10], features[11], dim=0).mean().item()))
        pair_cosines["l10_l12"].append(float(F.cosine_similarity(features[9], features[11], dim=0).mean().item()))
        pair_max_diffs["l10_l11"].append(float((features[9] - features[10]).abs().max().item()))
        pair_max_diffs["l11_l12"].append(float((features[10] - features[11]).abs().max().item()))
        pair_max_diffs["l10_l12"].append(float((features[9] - features[11]).abs().max().item()))

    max_diff = max(max_diffs)
    mean_diff = sum(mean_diffs) / len(mean_diffs)
    if max_diff > 1e-6 or mean_diff > 1e-7:
        raise RuntimeError(
            f"CACD F12 equivalence audit failed: max_diff={max_diff:.9g}, mean_diff={mean_diff:.9g}."
        )
    for name, values in pair_cosines.items():
        if not values or not all(torch.isfinite(torch.tensor(values))):
            raise RuntimeError(f"CACD layer cosine audit is non-finite for {name}.")
    for name, values in pair_max_diffs.items():
        if min(values) == 0.0:
            raise RuntimeError(
                f"CACD layer hook/index error: {name} is bitwise identical on at least one audit sample."
            )

    def aggregate(stats):
        return {
            "mean": sum(stats["mean"]) / len(stats["mean"]),
            "std": sum(stats["std"]) / len(stats["std"]),
            "abs_mean": sum(stats["abs_mean"]) / len(stats["abs_mean"]),
            "min": min(stats["min"]),
            "max": max(stats["max"]),
        }

    return {
        "version": VERSION,
        "audit_samples": len(keys),
        "audit_keys": [list(key) for key in keys],
        "f12_max_abs_diff": max_diff,
        "f12_mean_abs_diff": mean_diff,
        "pairwise_token_cosine_mean": {
            name: sum(values) / len(values) for name, values in pair_cosines.items()
        },
        "pairwise_max_abs_diff": {
            name: max(values) for name, values in pair_max_diffs.items()
        },
        "pairwise_min_of_sample_max_abs_diff": {
            name: min(values) for name, values in pair_max_diffs.items()
        },
        "layer_stats": {layer: aggregate(stats) for layer, stats in layer_stats.items()},
    }


def generate_cacd_cache(cfg, splits, out_root, max_samples=-1, overwrite=False, audit_samples=32):
    if str(cfg.BACKBONE_KEY) != "dinov1-s8":
        raise ValueError(f"CACD-v1-Base supports dinov1-s8 only, got {cfg.BACKBONE_KEY!r}.")
    if int(cfg.DINO["feature_input_size"]) != 296 or int(cfg.DINO["patch_size"]) != 8:
        raise ValueError("CACD-v1-Base requires DINO input_size=296 and patch_size=8.")
    out_root = Path(out_root)
    ensure_dir(out_root)

    full_items_by_split = _all_items_by_split(cfg, splits)
    full_unique = _unique_items(full_items_by_split)
    selected_by_split = {
        split: items[: int(max_samples)] if int(max_samples) >= 0 else items
        for split, items in full_items_by_split.items()
    }
    selected_unique = _unique_items(selected_by_split)
    required_bytes = len(full_unique) * len(STORE_LAYERS) * BYTES_PER_FEATURE
    required_with_margin = int(required_bytes * 1.20)
    available = shutil.disk_usage(out_root).free
    print(f"requested layer indices 0-based = {list(EXPECTED_LAYERS)}")
    print("requested layers 1-based = [10, 11, 12]")
    print(f"estimated_sample_count = {len(full_unique)}")
    print(f"manifest_rows_full = {sum(len(v) for v in full_items_by_split.values())}")
    print(f"unique_samples_full = {len(full_unique)}")
    print(f"estimated_extra_gib = {required_bytes / 2**30:.2f}")
    print(f"required_with_20pct_margin_gib = {required_with_margin / 2**30:.2f}")
    print(f"available_gib = {available / 2**30:.2f}")
    if available < required_with_margin:
        raise RuntimeError(
            "Insufficient disk for full CACD cache before generation: "
            f"available={available / 2**30:.2f} GiB, required={required_with_margin / 2**30:.2f} GiB."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(cfg, device)
    holders = {layer: {"tensor": None} for layer in EXPECTED_LAYERS}
    key_paths = {}
    handles = []
    for layer in EXPECTED_LAYERS:
        module, path = resolve_key_projection_at_layer(model, layer)
        key_paths[layer] = path

        def hook_fn(_module, _inputs, output, *, layer_index=layer):
            holders[layer_index]["tensor"] = output.detach()

        handles.append(module.register_forward_hook(hook_fn))
    print(f"device = {device}")
    print(f"dino_model_class = {model.__class__.__module__}.{model.__class__.__name__}")
    print("feature_stage = block layernorm_before -> attention key linear (before final model norm)")
    print(f"key_projection_paths = {json.dumps(key_paths, sort_keys=True)}")

    try:
        audit = audit_f12_equivalence(
            cfg, model, holders, full_unique, splits, device, audit_samples=audit_samples
        )
        audit.update(
            {
                "manifest_rows_full": sum(len(v) for v in full_items_by_split.values()),
                "unique_samples_full": len(full_unique),
                "selected_manifest_rows": sum(len(v) for v in selected_by_split.values()),
                "selected_unique_samples": len(selected_unique),
                "estimated_extra_bytes": required_bytes,
                "required_with_20pct_margin_bytes": required_with_margin,
                "available_bytes": available,
                "full_split_counts": {split: len(items) for split, items in full_items_by_split.items()},
                "selected_split_counts": {split: len(items) for split, items in selected_by_split.items()},
                "dino_model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                "feature_stage": "block layernorm_before -> attention key linear (before final model norm)",
                "post_key_normalization": "none",
                "key_projection_paths": {str(key): value for key, value in key_paths.items()},
            }
        )
        write_json(out_root / "audit_report.json", audit)
        print(f"f12_max_abs_diff = {audit['f12_max_abs_diff']:.9g}")
        print(f"f12_mean_abs_diff = {audit['f12_mean_abs_diff']:.9g}")

        rows_by_key = {}
        for key, item in tqdm(sorted(selected_unique.items()), desc="cache CACD F10/F11"):
            dataset, stem = key
            out_dir = out_root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"CACD cache exists; pass --overwrite: {out_path}")
            inputs, original_size = preprocess_image(
                item["image_path"], int(cfg.DINO["feature_input_size"])
            )
            features = _capture_features(model, inputs.to(device), holders)
            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "original_size": original_size,
                "backbone": cfg.BACKBONE_KEY,
                "backbone_key": cfg.BACKBONE_KEY,
                "model_key": cfg.DINO["model_name"],
                "model_path": str(Path(cfg.DINO["model_path"]).resolve()),
                "version": VERSION,
                "feature_type": FEATURE_TYPE,
                "feature_layer_definition": "encoder.layer[i].attention.attention.key output",
                "feature_stage": "block layernorm_before -> attention key linear (before final model norm)",
                "post_key_normalization": "none",
                "input_size": int(cfg.DINO["feature_input_size"]),
                "patch_size": int(cfg.DINO["patch_size"]),
                "dtype": "float32",
                "layer_indices": list(EXPECTED_LAYERS),
                "layer_indices_0based": list(STORE_LAYERS),
                "layer_indices_1based": [10, 11],
                "stored_layer_indices": list(STORE_LAYERS),
                "feature_shape": EXPECTED_SHAPE,
                "key_projection_paths": {str(k): v for k, v in key_paths.items()},
                "feature_l10": features[9],
                "feature_l11": features[10],
            }
            torch.save(payload, out_path)
            rows_by_key[key] = {
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "shape_l10": EXPECTED_SHAPE,
                "shape_l11": EXPECTED_SHAPE,
                "dtype": "float32",
                "version": VERSION,
                "feature_type": FEATURE_TYPE,
            }

        for split, items in selected_by_split.items():
            rows = [rows_by_key[(item["dataset"], item["stem"])] for item in items]
            manifest_path = out_root / f"manifest_{split}.jsonl"
            if manifest_path.exists() and not overwrite:
                raise FileExistsError(f"CACD manifest exists; pass --overwrite: {manifest_path}")
            write_jsonl(manifest_path, rows)
            counts = Counter(row["dataset"] for row in rows)
            print(f"wrote_manifest = {manifest_path} | rows={len(rows)} | counts={dict(counts)}")
    finally:
        for handle in handles:
            handle.remove()


def audit_saved_cache(cfg, splits, out_root, max_samples=-1):
    out_root = Path(out_root)
    for split in splits:
        manifest_path = out_root / f"manifest_{split}.jsonl"
        rows = read_jsonl(manifest_path)
        row_map = manifest_to_map(rows, manifest_path)
        expected = _items_for_split(cfg, split, max_samples=max_samples)
        expected_keys = [(item["dataset"], item["stem"]) for item in expected]
        if set(row_map) != set(expected_keys):
            missing = sorted(set(expected_keys) - set(row_map))
            extra = sorted(set(row_map) - set(expected_keys))
            raise RuntimeError(
                f"CACD {split} manifest mismatch: missing={missing[:10]}, extra={extra[:10]}"
            )
        for key in expected_keys:
            payload = torch_load(row_map[key]["cache_path"], map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError(f"CACD payload must be dict: {row_map[key]['cache_path']}")
            for field, expected_value in (
                ("dataset", key[0]),
                ("stem", key[1]),
                ("backbone", cfg.BACKBONE_KEY),
                ("backbone_key", cfg.BACKBONE_KEY),
                ("model_key", cfg.DINO["model_name"]),
                ("version", VERSION),
                ("feature_type", FEATURE_TYPE),
                ("dtype", "float32"),
                ("input_size", 296),
                ("patch_size", 8),
                ("layer_indices_0based", [9, 10]),
                ("layer_indices_1based", [10, 11]),
                ("feature_shape", EXPECTED_SHAPE),
            ):
                if payload.get(field) != expected_value:
                    raise RuntimeError(
                        f"CACD payload metadata mismatch {field} for {key}: "
                        f"{payload.get(field)!r} != {expected_value!r}"
                    )
            if any(field in payload for field in ("feature_l12", "tensor", "feature")):
                raise RuntimeError(f"CACD payload unexpectedly duplicates F12 for {key}.")
            for field in ("feature_l10", "feature_l11"):
                value = payload.get(field)
                if (
                    not torch.is_tensor(value)
                    or value.dtype != torch.float32
                    or list(value.shape) != EXPECTED_SHAPE
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise RuntimeError(f"Invalid CACD {field} for {key}.")
        print(f"audit_saved_{split} = ok | rows={len(rows)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CACD F10/F11 attention-key caches and audit F12 equivalence."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", type=_parse_splits, default=("train", "val", "test"))
    parser.add_argument("--layers", type=_parse_csv_ints, default=EXPECTED_LAYERS)
    parser.add_argument("--store-layers", type=_parse_csv_ints, default=STORE_LAYERS)
    parser.add_argument("--dtype", default="float32", choices=["float32"])
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--audit-samples", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if tuple(args.layers) != EXPECTED_LAYERS:
        raise ValueError(f"CACD-v1-Base requires --layers 9,10,11, got {args.layers}.")
    if tuple(args.store_layers) != STORE_LAYERS:
        raise ValueError(f"CACD-v1-Base requires --store-layers 9,10, got {args.store_layers}.")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max-samples must be -1 or a positive integer.")
    cfg = load_config(args.config)
    if not args.audit_only:
        generate_cacd_cache(
            cfg,
            args.splits,
            args.out,
            max_samples=args.max_samples,
            overwrite=args.overwrite,
            audit_samples=args.audit_samples,
        )
    audit_saved_cache(cfg, args.splits, args.out, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
