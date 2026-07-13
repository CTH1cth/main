import argparse
import sys
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
    resolve_key_projection,
)
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


VERSION = "cssd_hr_feature_v1"
FEATURE_LAYER = "final_attention_key"


def expected_items(cfg, max_samples=-1):
    items = build_image_items(cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False)
    return items[: int(max_samples)] if int(max_samples) >= 0 else items


def generate_cache(cfg, out_root, input_size, max_samples=-1, overwrite=False):
    if int(input_size) != 384:
        raise ValueError(f"CSSD-v1a requires input_size=384, got {input_size}")
    if cfg.BACKBONE_KEY != "dinov1-s8":
        raise ValueError(f"CSSD-v1a supports only dinov1-s8, got {cfg.BACKBONE_KEY}")
    out_root = Path(out_root)
    manifest_path = out_root / "manifest_train.jsonl"
    ensure_dir(out_root)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(cfg, device)
    key_holder = {"tensor": None}

    def hook_fn(_module, _inputs, output):
        key_holder["tensor"] = output.detach()

    key_module, key_path = resolve_key_projection(model)
    handle = key_module.register_forward_hook(hook_fn)
    rows = []
    items = expected_items(cfg, max_samples=max_samples)
    print(f"device = {device}")
    print(f"model_key = {cfg.DINO['model_name']}")
    print(f"model_path = {cfg.DINO['model_path']}")
    print(f"feature_layer = {FEATURE_LAYER}")
    print(f"key_projection_path = {key_path}")
    print(f"input_size = {input_size}")
    print(f"num_items = {len(items)}")
    try:
        for item in tqdm(items, desc="cache CSSD HR train features"):
            dataset = item["dataset"]
            stem = item["stem"]
            out_dir = out_root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")
            inputs, _original_size = preprocess_image(item["image_path"], int(input_size))
            key_holder["tensor"] = None
            with torch.no_grad():
                call_dino(model, inputs.to(device))
            key = key_holder["tensor"]
            if key is None:
                raise RuntimeError(f"DINO key hook produced no tensor for {dataset}/{stem}")
            feature = key_to_feature_tensor(key).float()
            expected_shape = [384, 48, 48]
            if list(feature.shape) != expected_shape or not bool(torch.isfinite(feature).all().item()):
                raise RuntimeError(
                    f"Invalid CSSD HR feature for {dataset}/{stem}: "
                    f"shape={list(feature.shape)}, finite={bool(torch.isfinite(feature).all().item())}"
                )
            payload = {
                "feature": feature,
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "model_key": cfg.DINO["model_name"],
                "backbone_key": cfg.BACKBONE_KEY,
                "feature_layer": FEATURE_LAYER,
                "key_projection_path": key_path,
                "input_size": int(input_size),
                "patch_size": int(cfg.DINO["patch_size"]),
                "feature_shape": expected_shape,
                "dtype": "float32",
                "version": VERSION,
            }
            torch.save(payload, out_path)
            rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "image_path": item["image_path"],
                    "cache_path": str(out_path.resolve()),
                    "shape": expected_shape,
                    "dtype": "float32",
                    "input_size": int(input_size),
                    "version": VERSION,
                }
            )
    finally:
        handle.remove()
    write_jsonl(manifest_path, rows)
    print(f"wrote_manifest = {manifest_path}")
    return manifest_path


def audit_cache(cfg, out_root, max_samples=-1, cosine_samples=32):
    out_root = Path(out_root)
    manifest_path = out_root / "manifest_train.jsonl"
    rows = read_jsonl(manifest_path)
    row_map = manifest_to_map(rows, manifest_path)
    items = expected_items(cfg, max_samples=max_samples)
    expected_keys = [(item["dataset"], item["stem"]) for item in items]
    expected_item_map = {(item["dataset"], item["stem"]): item for item in items}
    actual_keys = set(row_map)
    expected_set = set(expected_keys)
    missing = sorted(expected_set - actual_keys)
    extra = sorted(actual_keys - expected_set)
    counts = {}
    for dataset, _stem in row_map:
        counts[dataset] = counts.get(dataset, 0) + 1
    if int(max_samples) < 0:
        expected_counts = {"TR-CAMO": 1000, "TR-COD10K": 3040}
        if len(expected_keys) != 4040 or counts != expected_counts:
            raise RuntimeError(
                f"CSSD full cache count mismatch: total={len(rows)}, counts={counts}, expected={expected_counts}"
            )

    nonfinite_count = 0
    shape_mismatch_count = 0
    means = []
    stds = []
    abs_means = []
    norm_means = []
    minima = []
    maxima = []
    valid_features = {}
    for key in expected_keys:
        if key not in row_map:
            continue
        payload = torch_load(row_map[key]["cache_path"], map_location="cpu")
        feature = payload.get("feature") if isinstance(payload, dict) else None
        metadata_ok = isinstance(payload, dict) and all(
            (
                payload.get("dataset") == key[0],
                payload.get("stem") == key[1],
                payload.get("backbone_key") == cfg.BACKBONE_KEY,
                payload.get("model_key") == cfg.DINO["model_name"],
                payload.get("feature_layer") == FEATURE_LAYER,
                payload.get("input_size") == 384,
                payload.get("patch_size") == 8,
                payload.get("dtype") == "float32",
                payload.get("version") == VERSION,
                isinstance(payload.get("key_projection_path"), str),
                bool(payload.get("key_projection_path")),
                Path(payload.get("image_path", "")).resolve()
                == Path(expected_item_map[key]["image_path"]).resolve(),
            )
        )
        if (
            not torch.is_tensor(feature)
            or feature.dtype != torch.float32
            or list(feature.shape) != [384, 48, 48]
            or not metadata_ok
        ):
            shape_mismatch_count += 1
            continue
        feature = feature.float()
        if not bool(torch.isfinite(feature).all().item()):
            nonfinite_count += 1
            continue
        means.append(float(feature.mean().item()))
        stds.append(float(feature.std().item()))
        abs_means.append(float(feature.abs().mean().item()))
        norm_means.append(float(torch.linalg.vector_norm(feature, dim=0).mean().item()))
        minima.append(float(feature.min().item()))
        maxima.append(float(feature.max().item()))
        valid_features[key] = feature

    cosine_values = []
    normal_manifest = feature_manifest_path(cfg, "train")
    normal_map = manifest_to_map(read_jsonl(normal_manifest), normal_manifest)
    for key in sorted(valid_features)[: min(int(cosine_samples), len(valid_features))]:
        normal_payload = torch_load(normal_map[key]["cache_path"], map_location="cpu")
        normal = normal_payload.get("tensor") if isinstance(normal_payload, dict) else None
        if not torch.is_tensor(normal) or list(normal.shape) != [384, 37, 37]:
            raise RuntimeError(f"Invalid normal feature for cross-scale audit: {key}")
        high_down = F.interpolate(
            valid_features[key].unsqueeze(0),
            size=(37, 37),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        cosine = F.cosine_similarity(high_down.float(), normal.float(), dim=0)
        cosine_values.extend(cosine.flatten().tolist())

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    report = {
        "num_samples": len(rows),
        "missing_count": len(missing),
        "duplicate_count": len(rows) - len(row_map),
        "extra_count": len(extra),
        "nonfinite_count": nonfinite_count,
        "shape_mismatch_count": shape_mismatch_count,
        "feature_mean": avg(means),
        "feature_std": avg(stds),
        "feature_abs_mean": avg(abs_means),
        "feature_norm_mean": avg(norm_means),
        "feature_min": min(minima) if minima else 0.0,
        "feature_max": max(maxima) if maxima else 0.0,
        "mean_cross_scale_cosine": avg(cosine_values),
        "min_cross_scale_cosine": min(cosine_values) if cosine_values else 0.0,
        "max_cross_scale_cosine": max(cosine_values) if cosine_values else 0.0,
    }
    for key, value in report.items():
        print(f"{key} = {value}")
    if missing or extra or report["duplicate_count"] or nonfinite_count or shape_mismatch_count:
        raise RuntimeError(f"CSSD HR cache audit failed: {report}")
    if report["mean_cross_scale_cosine"] < 0.20:
        raise RuntimeError("CSSD cross-scale cosine < 0.20; cache representation is likely wrong.")
    if report["mean_cross_scale_cosine"] < 0.50:
        print("[CSSD WARNING] mean cross-scale cosine < 0.50")
    return report


def main():
    parser = argparse.ArgumentParser(description="Generate/audit train-only CSSD 384px DINO key features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="float32", choices=["float32"])
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if int(args.input_size) != 384:
        raise ValueError(f"CSSD-v1a only permits --input-size 384, got {args.input_size}.")
    if int(args.max_samples) == 0 or int(args.max_samples) < -1:
        raise ValueError("--max-samples must be -1 or a positive integer.")
    cfg = load_config(args.config)
    if args.audit_only:
        audit_cache(cfg, args.out, max_samples=args.max_samples)
        return
    generate_cache(
        cfg,
        args.out,
        input_size=args.input_size,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
    )
    audit_cache(cfg, args.out, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
