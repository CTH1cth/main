import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.utils import ensure_dir, manifest_to_map, read_jsonl, torch_load, write_jsonl  # noqa: E402


def validate_map(name, root):
    manifest = Path(root).expanduser() / "manifest_train.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{name} manifest not found: {manifest}")
    return manifest_to_map(read_jsonl(manifest), manifest), manifest


def as_tensor(payload, key, path, shape=(1, 68, 68)):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise KeyError(f"Missing tensor {key}: {path}")
    value = value.float()
    if tuple(value.shape) != tuple(shape):
        raise RuntimeError(f"{key} shape mismatch: {list(value.shape)} != {list(shape)} | {path}")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{key} contains NaN/Inf: {path}")
    min_value = float(value.min().item())
    max_value = float(value.max().item())
    if min_value < -1e-6 or max_value > 1.0 + 1e-6:
        raise RuntimeError(f"{key} out of [0,1]: min={min_value:.6f}, max={max_value:.6f} | {path}")
    return value.clamp(0.0, 1.0)


def load_rgb_68(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((68, 68), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def sobel_norm(image_68):
    gray = 0.299 * image_68[0:1] + 0.587 * image_68[1:2] + 0.114 * image_68[2:3]
    gray = gray.unsqueeze(0)
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=torch.float32,
    ).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=torch.float32,
    ).unsqueeze(0)
    dx = F.conv2d(gray, sobel_x, padding=1)
    dy = F.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(dx * dx + dy * dy + 1e-6)
    edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return edge[0].clamp(0.0, 1.0)


def compute_dino_margin_68(feature, fg_core, bg_core):
    if feature.ndim != 3 or int(feature.shape[0]) != 384 or tuple(feature.shape[-2:]) != (37, 37):
        raise RuntimeError(f"DABE-PU++ expects cached DINO feature [384,37,37], got {list(feature.shape)}")
    feat = F.normalize(feature.float(), dim=0)
    fg_37 = F.interpolate(fg_core.unsqueeze(0).float(), size=(37, 37), mode="nearest")[0] > 0.5
    bg_37 = F.interpolate(bg_core.unsqueeze(0).float(), size=(37, 37), mode="nearest")[0] > 0.5
    skipped_no_fg = int(fg_37.sum().item() <= 0)
    skipped_no_bg = int(bg_37.sum().item() <= 0)
    margin_37 = torch.zeros(1, 37, 37, dtype=torch.float32)
    if not skipped_no_fg and not skipped_no_bg:
        fg_proto = F.normalize(feat[:, fg_37[0]].mean(dim=1), dim=0).detach()
        bg_proto = F.normalize(feat[:, bg_37[0]].mean(dim=1), dim=0).detach()
        sim_fg = (feat * fg_proto.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        sim_bg = (feat * bg_proto.view(-1, 1, 1)).sum(dim=0, keepdim=True)
        margin_37 = (sim_fg - sim_bg).detach()
    margin_68 = F.interpolate(margin_37.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False)[0]
    return margin_68, skipped_no_fg, skipped_no_bg


def cap_mask_by_score_per_image(mask, score, max_ratio, min_pixels):
    mask = mask.bool()
    if int(mask.sum().item()) <= 0:
        return mask
    num_pixels = int(mask.numel())
    k = int(max(1, round(float(max_ratio) * num_pixels)))
    valid_scores = score[mask]
    if int(valid_scores.numel()) < int(min_pixels):
        return torch.zeros_like(mask)
    k = min(k, int(valid_scores.numel()))
    top_values = torch.topk(valid_scores.flatten(), k=k, largest=True).values
    thresh = top_values[-1]
    capped = mask & (score >= thresh)
    if int(capped.sum().item()) < int(min_pixels):
        return torch.zeros_like(mask)
    return capped


def apply_max_target_delta(base_target, target_v12, max_delta):
    delta = target_v12 - base_target
    current_delta_mean = float(delta.mean().item())
    if current_delta_mean <= float(max_delta):
        return target_v12, 1.0, current_delta_mean
    positive_delta = delta.clamp_min(0.0)
    negative_delta = delta.clamp_max(0.0)
    pos_mean = float(positive_delta.mean().item())
    neg_mean = float(negative_delta.mean().item())
    if pos_mean <= 1e-12:
        return target_v12, 1.0, current_delta_mean
    scale = max(0.0, min(1.0, (float(max_delta) - neg_mean) / pos_mean))
    adjusted = (base_target + negative_delta + scale * positive_delta).clamp(0.0, 1.0)
    return adjusted, scale, float((adjusted - base_target).mean().item())


def process_one(args, key, base_row, clean_row, cover_row, feature_row):
    dataset_name, stem = key
    base_path = base_row["cache_path"]
    clean_path = clean_row["cache_path"]
    cover_path = cover_row["cache_path"]
    feature_path = feature_row["cache_path"]
    base = torch_load(base_path, map_location="cpu")
    clean = torch_load(clean_path, map_location="cpu")
    cover = torch_load(cover_path, map_location="cpu")
    feature_payload = torch_load(feature_path, map_location="cpu")
    for name, payload, path in (
        ("base", base, base_path),
        ("clean", clean, clean_path),
        ("cover", cover, cover_path),
        ("feature", feature_payload, feature_path),
    ):
        if payload.get("dataset") != dataset_name or payload.get("stem") != stem:
            raise RuntimeError(f"{name} key mismatch for {dataset_name}/{stem}: {path}")

    base_target = as_tensor(base, "target_soft_68", base_path)
    base_weight = as_tensor(base, "weight_map_68", base_path)
    fg_core = as_tensor(base, "fg_core_pu_68", base_path) > 0.5
    fg_fallback = as_tensor(base, "fg_core_fallback_68", base_path) > 0.5
    bg_core = as_tensor(base, "bg_core_pu_68", base_path) > 0.5
    extent = as_tensor(base, "extent_candidate_68", base_path) > 0.5
    unknown = as_tensor(base, "unknown_68", base_path) > 0.5
    p_clean = as_tensor(clean, "prob_68", clean_path)
    b_clean = as_tensor(clean, "binary_68", clean_path)
    c_clean = as_tensor(clean, "conf_68", clean_path)
    p_cover = as_tensor(cover, "prob_68", cover_path)
    b_cover = as_tensor(cover, "binary_68", cover_path)
    c_cover = as_tensor(cover, "conf_68", cover_path)
    feature = feature_payload["tensor"].float()

    image_path = base.get("image_path") or base_row.get("image_path") or feature_row.get("image_path")
    if not image_path:
        raise RuntimeError(f"Missing image_path for {dataset_name}/{stem}")
    image_68 = load_rgb_68(image_path)
    edge_norm = sobel_norm(image_68)
    edge_thresh = torch.quantile(edge_norm.flatten(), float(args.edge_q))
    edge_support = edge_norm >= edge_thresh
    margin_68, skipped_no_fg, skipped_no_bg = compute_dino_margin_68(feature, fg_core.float(), bg_core.float())

    target_v12 = base_target.clone()
    weight_v12 = base_weight.clone()

    low_target_bg = (base_target < 0.15) & (base_weight > 0.50)
    stable_clean_bg = (p_clean <= 0.15) & (p_cover <= 0.25) & (c_clean >= 0.60)
    bg_lock = bg_core | low_target_bg | stable_clean_bg
    target_v12 = torch.where(bg_lock, torch.minimum(target_v12, torch.full_like(target_v12, 0.05)), target_v12)
    weight_v12 = torch.where(bg_lock, torch.maximum(weight_v12, torch.full_like(weight_v12, 0.90)), weight_v12)

    fg_core_keep = fg_core & (~bg_lock)
    target_v12 = torch.where(fg_core_keep, torch.maximum(target_v12, torch.full_like(target_v12, 0.85)), target_v12)
    weight_v12 = torch.where(fg_core_keep, torch.maximum(weight_v12, torch.full_like(weight_v12, 0.90)), weight_v12)

    extent_agree_fg = extent & (~bg_lock) & (p_clean >= 0.50) & (p_cover >= 0.40) & (margin_68 >= -0.05)
    target_v12 = torch.where(extent_agree_fg, torch.maximum(target_v12, torch.full_like(target_v12, 0.65)), target_v12)
    weight_v12 = torch.where(extent_agree_fg, torch.maximum(weight_v12, torch.full_like(weight_v12, 0.60)), weight_v12)

    near_seed = ((p_clean >= 0.45) | (p_cover >= 0.40) | fg_core).float().unsqueeze(0)
    near_fg = F.max_pool2d(near_seed, kernel_size=7, stride=1, padding=3)[0] > 0.5
    lost_extent_raw = (
        extent
        & (~bg_lock)
        & (p_cover >= 0.40)
        & (p_clean < 0.50)
        & (margin_68 >= -0.02)
        & near_fg
    )
    lost_score = 0.40 * p_cover + 0.20 * (1.0 - p_clean) + 0.20 * margin_68.clamp_min(0.0) + 0.20 * edge_norm
    lost_extent = cap_mask_by_score_per_image(
        lost_extent_raw,
        lost_score,
        float(args.lost_extent_max_ratio),
        int(args.min_pixels),
    )
    target_v12 = torch.where(lost_extent, torch.maximum(target_v12, torch.full_like(target_v12, 0.55)), target_v12)
    weight_v12 = torch.where(lost_extent, torch.maximum(weight_v12, torch.full_like(weight_v12, 0.50)), weight_v12)

    new_boundary_raw = (
        extent
        & (~bg_lock)
        & (p_cover < 0.40)
        & (p_clean < 0.50)
        & (margin_68 >= 0.10)
        & edge_support
        & near_fg
    )
    new_score = 0.35 * margin_68.clamp_min(0.0) + 0.35 * edge_norm + 0.20 * p_cover + 0.10 * base_target
    new_boundary = cap_mask_by_score_per_image(
        new_boundary_raw,
        new_score,
        float(args.new_boundary_max_ratio),
        int(args.min_pixels),
    )
    target_v12 = torch.where(new_boundary, torch.maximum(target_v12, torch.full_like(target_v12, 0.45)), target_v12)
    weight_v12 = torch.where(new_boundary, torch.maximum(weight_v12, torch.full_like(weight_v12, 0.35)), weight_v12)

    unknown_only = unknown & (~fg_core) & (~bg_core) & (~extent)
    weight_v12 = torch.where(unknown_only, torch.minimum(weight_v12, torch.full_like(weight_v12, 0.50)), weight_v12)

    target_v12 = target_v12.clamp(0.0, 1.0)
    target_v12, delta_scale, target_delta_mean = apply_max_target_delta(
        base_target,
        target_v12,
        float(args.max_target_delta_mean),
    )
    weight_v12 = weight_v12.clamp(0.0, 1.0)

    stats = {
        "dataset": dataset_name,
        "stem": stem,
        "target_base_mean": float(base_target.mean().item()),
        "target_v12_mean": float(target_v12.mean().item()),
        "target_delta_mean": float(target_delta_mean),
        "weight_base_mean": float(base_weight.mean().item()),
        "weight_v12_mean": float(weight_v12.mean().item()),
        "bg_lock_ratio": float(bg_lock.float().mean().item()),
        "fg_core_ratio": float(fg_core.float().mean().item()),
        "extent_agree_fg_ratio": float(extent_agree_fg.float().mean().item()),
        "lost_extent_ratio": float(lost_extent.float().mean().item()),
        "new_boundary_ratio": float(new_boundary.float().mean().item()),
        "unknown_ratio": float(unknown.float().mean().item()),
        "lost_extent_valid": int(lost_extent.float().sum().item() > 0),
        "new_boundary_valid": int(new_boundary.float().sum().item() > 0),
        "skipped_no_fg_proto": int(skipped_no_fg),
        "skipped_no_bg_proto": int(skipped_no_bg),
        "target_delta_scale": float(delta_scale),
    }
    payload = {
        "dataset": dataset_name,
        "stem": stem,
        "image_path": str(image_path),
        "gt_path": base.get("gt_path", base_row.get("gt_path", "")),
        "backbone_key": "dinov1-s8",
        "dabe_version": "pu_v12_shape_complete",
        "version": "dabe_pu_v12_shape_complete_v1",
        "target_soft": target_v12,
        "weight_map": weight_v12,
        "target_soft_68": target_v12,
        "weight_map_68": weight_v12,
        "target_hard": (target_v12 >= 0.5).float(),
        "target_hard_68": (target_v12 >= 0.5).float(),
        "fg_core": fg_core.float(),
        "fg_fallback": fg_fallback.float(),
        "bg_core": bg_core.float(),
        "extent": extent.float(),
        "unknown": unknown.float(),
        "fg_core_pu_68": fg_core.float(),
        "fg_core_fallback_68": fg_fallback.float(),
        "bg_core_pu_68": bg_core.float(),
        "extent_candidate_68": extent.float(),
        "unknown_68": unknown.float(),
        "target_soft_base": base_target,
        "weight_map_base": base_weight,
        "target_soft_base_68": base_target,
        "weight_map_base_68": base_weight,
        "sc_bg_lock": bg_lock.float(),
        "sc_extent_agree_fg": extent_agree_fg.float(),
        "sc_lost_extent": lost_extent.float(),
        "sc_new_boundary": new_boundary.float(),
        "sc_bg_lock_68": bg_lock.float(),
        "sc_extent_agree_fg_68": extent_agree_fg.float(),
        "sc_lost_extent_68": lost_extent.float(),
        "sc_new_boundary_68": new_boundary.float(),
        "clean_prob": p_clean,
        "clean_binary": b_clean,
        "clean_conf": c_clean,
        "cover_prob": p_cover,
        "cover_binary": b_cover,
        "cover_conf": c_cover,
        "clean_prob_68": p_clean,
        "clean_binary_68": b_clean,
        "clean_conf_68": c_clean,
        "cover_prob_68": p_cover,
        "cover_binary_68": b_cover,
        "cover_conf_68": c_cover,
        "dino_margin_68": margin_68,
        "edge_norm_68": edge_norm,
        "source_base": "dabe_pu_v11",
        "source_clean_teacher": "long45_epoch045",
        "source_cover_teacher": "lceg_epoch035",
        "source_clean_cache": str(clean_path),
        "source_cover_cache": str(cover_path),
        "source_base_cache": str(base_path),
        "shape_68": [1, 68, 68],
        "target_soft_area": float(target_v12.mean().item()),
        "weight_mean": float(weight_v12.mean().item()),
        "fg_core_pu_area": float(fg_core.float().mean().item()),
        "fg_core_fallback_area": float(fg_fallback.float().mean().item()),
        "bg_core_pu_area": float(bg_core.float().mean().item()),
        "extent_area": float(extent.float().mean().item()),
        "unknown_area": float(unknown.float().mean().item()),
        "target_base_mean": stats["target_base_mean"],
        "weight_base_mean": stats["weight_base_mean"],
        "target_delta_mean": stats["target_delta_mean"],
        "target_delta_scale": stats["target_delta_scale"],
        "sc_bg_lock_ratio": stats["bg_lock_ratio"],
        "sc_extent_agree_fg_ratio": stats["extent_agree_fg_ratio"],
        "sc_lost_extent_ratio": stats["lost_extent_ratio"],
        "sc_new_boundary_ratio": stats["new_boundary_ratio"],
        "skipped_no_fg_proto": int(skipped_no_fg),
        "skipped_no_bg_proto": int(skipped_no_bg),
        "params": vars(args),
    }
    return payload, stats


def summarize(rows):
    if not rows:
        return {}
    keys = [
        "target_base_mean",
        "target_v12_mean",
        "target_delta_mean",
        "weight_base_mean",
        "weight_v12_mean",
        "bg_lock_ratio",
        "fg_core_ratio",
        "extent_agree_fg_ratio",
        "lost_extent_ratio",
        "new_boundary_ratio",
        "unknown_ratio",
        "lost_extent_valid",
        "new_boundary_valid",
        "skipped_no_fg_proto",
        "skipped_no_bg_proto",
    ]
    summary = {"num_samples": len(rows)}
    for key in keys:
        summary[key] = float(sum(float(row[key]) for row in rows) / len(rows))
    summary["lost_extent_valid_image_ratio"] = summary.pop("lost_extent_valid")
    summary["new_boundary_valid_image_ratio"] = summary.pop("new_boundary_valid")
    summary["skipped_no_fg_proto"] = int(sum(int(row["skipped_no_fg_proto"]) for row in rows))
    summary["skipped_no_bg_proto"] = int(sum(int(row["skipped_no_bg_proto"]) for row in rows))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dabe-root", required=True)
    parser.add_argument("--clean-pred-root", required=True)
    parser.add_argument("--cover-pred-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--edge_q", type=float, default=0.60)
    parser.add_argument("--lost_extent_max_ratio", type=float, default=0.020)
    parser.add_argument("--new_boundary_max_ratio", type=float, default=0.005)
    parser.add_argument("--min_pixels", type=int, default=4)
    parser.add_argument("--max_target_delta_mean", type=float, default=0.030)
    args = parser.parse_args()

    base_map, base_manifest = validate_map("base DABE-PU", args.base_dabe_root)
    clean_map, _ = validate_map("clean teacher pred", args.clean_pred_root)
    cover_map, _ = validate_map("coverage teacher pred", args.cover_pred_root)
    feature_manifest = Path(args.feature_root).expanduser() / f"manifest_{args.split}.jsonl"
    feature_map = manifest_to_map(read_jsonl(feature_manifest), feature_manifest)
    keys = list(base_map.keys())
    if int(args.max_samples) >= 0:
        keys = keys[: int(args.max_samples)]
    for name, mapping in (("clean", clean_map), ("cover", cover_map), ("feature", feature_map)):
        missing = sorted(set(keys) - set(mapping))
        if missing:
            raise RuntimeError(f"{name} cache missing first 10: {missing[:10]}")

    out_root = Path(args.out).expanduser()
    ensure_dir(out_root)
    manifest_rows = []
    audit_rows = []
    written = 0
    for key in keys:
        out_path = out_root / key[0] / f"{key[1]}.pt"
        ensure_dir(out_path.parent)
        if out_path.exists() and not args.overwrite:
            payload = torch_load(out_path, map_location="cpu")
            stats = {
                "dataset": key[0],
                "stem": key[1],
                "target_base_mean": float(payload.get("target_base_mean", payload["target_soft_base_68"].mean().item())),
                "target_v12_mean": float(payload["target_soft_68"].mean().item()),
                "target_delta_mean": float(payload.get("target_delta_mean", payload["target_soft_68"].mean().item() - payload["target_soft_base_68"].mean().item())),
                "weight_base_mean": float(payload.get("weight_base_mean", payload["weight_map_base_68"].mean().item())),
                "weight_v12_mean": float(payload["weight_map_68"].mean().item()),
                "bg_lock_ratio": float(payload["sc_bg_lock_68"].float().mean().item()),
                "fg_core_ratio": float(payload["fg_core_pu_68"].float().mean().item()),
                "extent_agree_fg_ratio": float(payload["sc_extent_agree_fg_68"].float().mean().item()),
                "lost_extent_ratio": float(payload["sc_lost_extent_68"].float().mean().item()),
                "new_boundary_ratio": float(payload["sc_new_boundary_68"].float().mean().item()),
                "unknown_ratio": float(payload["unknown_68"].float().mean().item()),
                "lost_extent_valid": int(payload["sc_lost_extent_68"].float().sum().item() > 0),
                "new_boundary_valid": int(payload["sc_new_boundary_68"].float().sum().item() > 0),
                "skipped_no_fg_proto": int(payload.get("skipped_no_fg_proto", 0)),
                "skipped_no_bg_proto": int(payload.get("skipped_no_bg_proto", 0)),
                "target_delta_scale": float(payload.get("target_delta_scale", 1.0)),
            }
        else:
            payload, stats = process_one(
                args,
                key,
                base_map[key],
                clean_map[key],
                cover_map[key],
                feature_map[key],
            )
            torch.save(payload, out_path)
            written += 1
        audit_rows.append(stats)
        manifest_rows.append(
            {
                "dataset": key[0],
                "stem": key[1],
                "cache_path": str(out_path.resolve()),
                "image_path": base_map[key].get("image_path", ""),
                "gt_path": base_map[key].get("gt_path", ""),
                "backbone_key": "dinov1-s8",
                "dabe_version": "pu_v12_shape_complete",
                "version": "dabe_pu_v12_shape_complete_v1",
                "shape_68": [1, 68, 68],
                "target_delta_mean": stats["target_delta_mean"],
                "lost_extent_ratio": stats["lost_extent_ratio"],
                "new_boundary_ratio": stats["new_boundary_ratio"],
                "bg_lock_ratio": stats["bg_lock_ratio"],
            }
        )
        print(
            f"[DABE-PU++] {key[0]}/{key[1]} | "
            f"target_delta={stats['target_delta_mean']:.6f} | "
            f"bg_lock={stats['bg_lock_ratio']:.6f} | "
            f"lost={stats['lost_extent_ratio']:.6f} | "
            f"new={stats['new_boundary_ratio']:.6f} | path={out_path}",
            flush=True,
        )

    write_jsonl(out_root / f"manifest_{args.split}.jsonl", manifest_rows)
    summary = summarize(audit_rows)
    with (out_root / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (out_root / "audit_per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()) if audit_rows else ["dataset", "stem"])
        writer.writeheader()
        writer.writerows(audit_rows)
    print(
        f"[DABE-PU++] done | rows={len(manifest_rows)} | written={written} | "
        f"manifest={out_root / f'manifest_{args.split}.jsonl'} | "
        f"audit_summary={out_root / 'audit_summary.json'} | base_manifest={base_manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
