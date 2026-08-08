import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.dabe_pseudo import (  # noqa: E402
    DABE_V1_DEFAULT_PARAMS,
    DABE_V2_DEFAULT_PARAMS,
    DABE_V3_DEFAULT_PARAMS,
    DABE_V31_DEFAULT_PARAMS,
    DABE_GC_DEFAULT_PARAMS,
    DABE_RAC_DEFAULT_PARAMS,
    DABE_RAC_SAFE_DEFAULT_PARAMS,
    DABE_PU_DEFAULT_PARAMS,
    DABE_PU_V11_DEFAULT_PARAMS,
    generate_dabe_pseudo,
)
from common.utils import (  # noqa: E402
    build_image_items,
    despl_light_cache_manifest_path,
    ensure_dir,
    feature_manifest_path,
    check_exact_keys,
    load_config,
    manifest_to_map,
    pseudo_manifest_path,
    read_jsonl,
    split_dataset_names,
    torch_load,
    write_jsonl,
)


try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


def _params_from_cfg(cfg):
    params = {}
    known_keys = (
        set(DABE_V1_DEFAULT_PARAMS)
        | set(DABE_V2_DEFAULT_PARAMS)
        | set(DABE_V3_DEFAULT_PARAMS)
        | set(DABE_V31_DEFAULT_PARAMS)
        | set(DABE_GC_DEFAULT_PARAMS)
        | set(DABE_RAC_DEFAULT_PARAMS)
        | set(DABE_RAC_SAFE_DEFAULT_PARAMS)
        | set(DABE_PU_DEFAULT_PARAMS)
        | set(DABE_PU_V11_DEFAULT_PARAMS)
    )
    for key in known_keys:
        cfg_key = f"DABE_{key}"
        if hasattr(cfg, cfg_key):
            params[key] = getattr(cfg, cfg_key)
    return params


def _expected_feature_shape(cfg):
    embed_dim = int(cfg.DINO["embed_dim"])
    input_size = int(cfg.DINO["feature_input_size"])
    patch_size = int(cfg.DINO["patch_size"])
    if input_size % patch_size != 0:
        raise RuntimeError(
            f"feature_input_size={input_size} is not divisible by "
            f"patch_size={patch_size}"
        )
    grid = input_size // patch_size
    return [embed_dim, grid, grid]


def _load_feature(row, dataset, stem, cfg):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Feature payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Feature cache key mismatch: {row['cache_path']}")
    tensor = payload.get("tensor")
    if not torch.is_tensor(tensor):
        raise TypeError(f"Feature payload missing tensor: {row['cache_path']}")
    tensor = tensor.detach().cpu().float().contiguous()
    expected_shape = _expected_feature_shape(cfg)
    if list(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Expected feature {expected_shape}, got {list(tensor.shape)}: "
            f"{row['cache_path']}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"Feature tensor contains NaN/Inf: {row['cache_path']}")
    return tensor


def _single_channel(payload, name, cache_path, required=True):
    if name not in payload:
        if required:
            raise KeyError(f"{name} missing from {cache_path}")
        return None
    tensor = payload[name]
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor: {cache_path}")
    tensor = tensor.float()
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise RuntimeError(f"{name} must be [1,H,W], got {list(tensor.shape)}: {cache_path}")
    return tensor.clamp(0.0, 1.0)


def _safe_manifest_map(path, name, logger):
    path = Path(path)
    if not path.exists():
        logger(f"[Warn] {name} manifest missing: {path}")
        return None
    return manifest_to_map(read_jsonl(path), path)


def _load_light_payload(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DESPL light payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"DESPL light key mismatch: {row['cache_path']}")
    return payload


def _load_fixed_payload(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Fixed pseudo payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"Fixed pseudo key mismatch: {row['cache_path']}")
    return _single_channel(payload, "tensor", row["cache_path"], required=True)


def _load_dabe_payload(row, dataset, stem):
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"DABE payload must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(f"DABE cache key mismatch: {row['cache_path']}")
    return payload


def _resize_single(tensor, size):
    if list(tensor.shape[-2:]) == [int(size), int(size)]:
        return tensor.float().clamp(0.0, 1.0)
    return F.interpolate(
        tensor.unsqueeze(0).float(),
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)


def _load_vis_comparison_tensors(key, dataset, stem, light_map, fixed_map, loss_size):
    fixed = None
    despl = None
    if light_map is not None and key in light_map:
        light_payload = _load_light_payload(light_map[key], dataset, stem)
        fixed = _single_channel(light_payload, "p_fixed_68", light_map[key]["cache_path"], required=False)
        despl = _single_channel(light_payload, "p_despl_68", light_map[key]["cache_path"], required=False)
    if fixed is None and fixed_map is not None and key in fixed_map:
        fixed = _load_fixed_payload(fixed_map[key], dataset, stem)
    if fixed is None:
        raise RuntimeError(f"Fixed pseudo missing for visualization: {dataset}/{stem}")
    if despl is None:
        raise RuntimeError(f"DESPL light p_despl_68 missing for visualization: {dataset}/{stem}")
    return _resize_single(fixed, loss_size), _resize_single(despl, loss_size)


def _tensor_payload(result):
    keys = [
        "p_dabe_37",
        "p_dabe_68",
        "p_dabe_v3_37",
        "p_dabe_v3_68",
        "p_dabe_v31_37",
        "p_dabe_v31_68",
        "p_dabe_gc_37",
        "p_dabe_gc_68",
        "p_dabe_rac_37",
        "p_dabe_rac_68",
        "p_dabe_rac_safe_37",
        "p_dabe_rac_safe_68",
        "p_base_37",
        "p_base_68",
        "base_mask_37",
        "candidate_band_37",
        "local_band_37",
        "fg_seed_37",
        "bg_seed_37",
        "region_id_map_37",
        "p_region_37",
        "p_region_support_37",
        "strong_region_mask_37",
        "weak_region_mask_37",
        "rejected_region_mask_37",
        "kept_region_mask_37",
        "p_rac_before_area_37",
        "region_aff_degree_37",
        "graph_degree_37",
        "edge_37",
        "p_gc_37",
        "delta_pos_37",
        "p_safe_before_budget_37",
        "top_delta_mask_37",
        "delta_pos_budgeted_37",
        "p_safe_after_core_lock_37",
        "gc_conf_37",
        "p_rw_37",
        "p_rw_evid_37",
        "evidence_37",
        "evidence_soft_37",
        "core_affinity_37",
        "color_affinity_37",
        "residual_norm_37",
        "fg_score_norm_37",
        "aff_norm_37",
        "pixel_gate_37",
        "pixel_completion_37",
        "fg_core_candidate_37",
        "fg_core_pu_37",
        "fg_core_fallback_37",
        "fg_reliability_37",
        "fg_core_weight_37",
        "bg_core_pu_37",
        "extent_band_37",
        "extent_candidate_37",
        "extent_score_37",
        "valid_extent_37",
        "extent_removed_mask_37",
        "unknown_37",
        "target_soft_37",
        "weight_map_37",
        "fg_core_pu_68",
        "fg_core_fallback_68",
        "bg_core_pu_68",
        "extent_candidate_68",
        "unknown_68",
        "target_soft_68",
        "weight_map_68",
        "expand_score_37",
        "valid_expand_37",
        "p_expand_37",
        "residual_37",
        "residual_pass1_37",
        "bc_map_37",
        "fg_score_37",
        "fg_core_37",
        "fg_core_68",
        "bg_anchor_37",
        "bg_core_37",
        "bg_core_68",
        "uncertain_37",
        "uncertain_68",
        "component_keep_mask_37",
        "component_weak_mask_37",
        "component_removed_mask_37",
        "view_agreement_37",
    ]
    return {
        key: result[key].detach().cpu().float().contiguous()
        for key in keys
        if key in result and torch.is_tensor(result[key])
    }


def _image_panel(image, title, size=160):
    panel = image.resize((size, size), RESAMPLE_BICUBIC).convert("RGB")
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(panel, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def _tensor_panel(tensor, title, size=160, nearest=False):
    array = tensor.detach().cpu().float().squeeze().numpy()
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="L")
    image = image.resize((size, size), RESAMPLE_NEAREST if nearest else RESAMPLE_BICUBIC).convert("RGB")
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def _overlay_panel(image, pseudo, title, size=160):
    rgb = image.resize((size, size), RESAMPLE_BICUBIC).convert("RGB")
    pseudo_array = pseudo.detach().cpu().float().squeeze().numpy()
    pseudo_array = np.clip(np.nan_to_num(pseudo_array, nan=0.0), 0.0, 1.0)
    pseudo_img = Image.fromarray(np.rint(pseudo_array * 255.0).astype(np.uint8), mode="L")
    pseudo_img = pseudo_img.resize((size, size), RESAMPLE_BICUBIC)
    rgb_np = np.asarray(rgb, dtype=np.float32)
    p_np = np.asarray(pseudo_img, dtype=np.float32) / 255.0
    heat = np.zeros_like(rgb_np)
    heat[..., 0] = 255.0 * p_np
    overlay = np.clip(0.62 * rgb_np + 0.38 * heat, 0.0, 255.0).astype(np.uint8)
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(Image.fromarray(overlay, mode="RGB"), (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def _dabe_display_name(version):
    version = str(version).lower()
    if version == "pu_v11":
        return "DABE-PU-v1.1"
    if version == "pu":
        return "DABE-PU"
    if version == "rac_safe":
        return "DABE-RAC-Safe"
    if version == "rac":
        return "DABE-RAC"
    if version == "gc":
        return "DABE-GC"
    if version == "v3_1":
        return "DABE-v3.1"
    if version == "v3":
        return "DABE-v3"
    if version == "v2":
        return "DABE-v2"
    if version == "v1":
        return "DABE-v1"
    return "DABE"


def _make_panels(
    image,
    gt,
    fixed,
    despl,
    payload,
    mode,
    dabe_v2=None,
    dabe_v31=None,
    dabe_gc=None,
    dabe_rac=None,
    dabe_rac_safe=None,
    dabe_pu=None,
):
    dabe_name = _dabe_display_name(payload.get("dabe_version", "v2"))
    if mode == "compact":
        panels = [
            _image_panel(image, "RGB"),
            _image_panel(gt.convert("RGB"), "GT"),
            _tensor_panel(fixed, "fixed"),
            _tensor_panel(despl, "DESPL"),
        ]
        if dabe_v2 is not None:
            panels.append(_tensor_panel(dabe_v2["p_dabe_68"], "DABE-v2"))
        if dabe_rac is not None:
            panels.append(_tensor_panel(dabe_rac["p_dabe_68"], "DABE-RAC"))
        if dabe_gc is not None:
            panels.append(_tensor_panel(dabe_gc["p_dabe_68"], "DABE-GC"))
        if dabe_rac_safe is not None:
            panels.append(_tensor_panel(dabe_rac_safe["p_dabe_68"], "RAC-Safe"))
        if dabe_v31 is not None:
            panels.append(_tensor_panel(dabe_v31["p_dabe_68"], "DABE-v3.1"))
        if dabe_pu is not None:
            panels.append(_tensor_panel(dabe_pu["p_dabe_68"], "DABE-PU-v1"))
        payload_version = str(payload.get("dabe_version", "")).lower()
        if payload_version in {"pu", "pu_v11"}:
            suffix = "_v11" if payload_version == "pu_v11" else ""
            panels.extend(
                [
                    _tensor_panel(payload["target_soft_68"], f"target_soft{suffix}"),
                    _tensor_panel(payload["weight_map_68"], f"weight_map{suffix}"),
                    _tensor_panel(payload["fg_core_pu_68"], f"fg_core{suffix}", nearest=True),
                    _tensor_panel(payload["bg_core_pu_68"], f"bg_core{suffix}", nearest=True),
                    _tensor_panel(payload["extent_candidate_68"], f"extent{suffix}"),
                    _tensor_panel(payload["unknown_68"], f"unknown{suffix}", nearest=True),
                    _overlay_panel(image, payload["target_soft_68"], "overlay"),
                ]
            )
            return panels
        panels.extend(
            [
                _tensor_panel(payload["p_dabe_68"], dabe_name),
                _overlay_panel(image, payload["p_dabe_68"], "overlay"),
            ]
        )
        return panels
    if mode == "debug":
        panels = [
            _image_panel(image, "RGB"),
            _image_panel(gt.convert("RGB"), "GT"),
            _tensor_panel(fixed, "fixed"),
            _tensor_panel(despl, "DESPL"),
        ]
        if dabe_v2 is not None:
            panels.append(_tensor_panel(dabe_v2["p_dabe_68"], "DABE-v2"))
        if dabe_rac is not None:
            panels.append(_tensor_panel(dabe_rac["p_dabe_68"], "DABE-RAC"))
        if dabe_gc is not None:
            panels.append(_tensor_panel(dabe_gc["p_dabe_68"], "DABE-GC"))
        if dabe_rac_safe is not None:
            panels.append(_tensor_panel(dabe_rac_safe["p_dabe_68"], "RAC-Safe"))
        if dabe_v31 is not None:
            panels.append(_tensor_panel(dabe_v31["p_dabe_68"], "DABE-v3.1"))
        if dabe_pu is not None:
            panels.append(_tensor_panel(dabe_pu["p_dabe_68"], "DABE-PU-v1"))
        debug_keys = [
            ("p_dabe_68", dabe_name, False),
            ("p_base_37", "p_base", False),
            ("target_soft_68", "target_soft", False),
            ("weight_map_68", "weight_map", False),
            ("fg_core_candidate_37", "fg_candidate", True),
            ("fg_core_pu_37", "fg_core_pu", True),
            ("fg_core_fallback_37", "fg_fallback", True),
            ("fg_reliability_37", "fg_reliability", False),
            ("fg_core_weight_37", "fg_weight", False),
            ("bg_core_pu_37", "bg_core_pu", True),
            ("extent_candidate_37", "extent", False),
            ("extent_score_37", "extent_score", False),
            ("unknown_37", "unknown_pu", True),
            ("extent_band_37", "extent_band", True),
            ("valid_extent_37", "valid_extent", True),
            ("extent_removed_mask_37", "extent_removed", True),
            ("fg_seed_37", "fg_seed", True),
            ("bg_seed_37", "bg_seed", True),
            ("region_id_map_37", "region_id_map", False),
            ("p_region_37", "p_region", False),
            ("p_region_support_37", "p_region_support", False),
            ("strong_region_mask_37", "strong_region", True),
            ("weak_region_mask_37", "weak_region", True),
            ("rejected_region_mask_37", "rejected_region", True),
            ("kept_region_mask_37", "kept_region", True),
            ("candidate_band_37", "candidate_band", True),
            ("local_band_37", "local_band", True),
            ("bc_map_37", "bc_map", False),
            ("residual_37", "residual", False),
            ("residual_norm_37", "residual_norm", False),
            ("fg_score_37", "fg_score", False),
            ("fg_score_norm_37", "fg_score_norm", False),
            ("p_rw_37", "p_rw", False),
            ("evidence_37", "evidence", False),
            ("evidence_soft_37", "evidence_soft", False),
            ("core_affinity_37", "core_aff", False),
            ("valid_expand_37", "valid_expand", True),
            ("p_expand_37", "p_expand", False),
            ("graph_degree_37", "graph_degree", False),
            ("region_aff_degree_37", "region_aff_degree", False),
            ("aff_norm_37", "aff_norm", False),
            ("edge_37", "edge", False),
            ("pixel_gate_37", "pixel_gate", False),
            ("pixel_completion_37", "pixel_completion", False),
            ("p_gc_37", "p_gc", False),
            ("delta_pos_37", "delta_pos", False),
            ("top_delta_mask_37", "top_delta_mask", True),
            ("delta_pos_budgeted_37", "delta_budgeted", False),
            ("gc_conf_37", "gc_conf", False),
            ("fg_core_37", "fg_core", True),
            ("bg_core_37", "bg_core", True),
            ("uncertain_37", "uncertain", True),
            ("component_keep_mask_37", "component_keep", True),
            ("component_removed_mask_37", "component_removed", True),
        ]
        for key, title, nearest in debug_keys:
            if key in payload and torch.is_tensor(payload[key]):
                panels.append(_tensor_panel(payload[key], title, nearest=nearest))
        overlay_tensor = payload["target_soft_68"] if "target_soft_68" in payload else payload["p_dabe_68"]
        panels.append(_overlay_panel(image, overlay_tensor, "overlay"))
        return panels
    raise ValueError(f"Unsupported DABE vis mode: {mode}")


def _save_visualization(
    path,
    image_path,
    gt_path,
    fixed,
    despl,
    payload,
    mode,
    dabe_v2=None,
    dabe_v31=None,
    dabe_gc=None,
    dabe_rac=None,
    dabe_rac_safe=None,
    dabe_pu=None,
):
    image = Image.open(image_path).convert("RGB")
    gt = Image.open(gt_path).convert("L")
    panels = _make_panels(
        image,
        gt,
        fixed,
        despl,
        payload,
        mode,
        dabe_v2=dabe_v2,
        dabe_v31=dabe_v31,
        dabe_gc=dabe_gc,
        dabe_rac=dabe_rac,
        dabe_rac_safe=dabe_rac_safe,
        dabe_pu=dabe_pu,
    )
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    ensure_dir(Path(path).parent)
    canvas.save(path)


def _safe_reason(reason):
    reason = str(reason or "none")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", reason).strip("_") or "fallback"


def _selected_items(cfg, split, max_samples):
    datasets = split_dataset_names(cfg, split)
    items = build_image_items(cfg.DATA_ROOT, datasets, require_gt=True)
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    if not items:
        raise RuntimeError(f"No {split} images selected for DABE cache.")
    return items


def generate_dabe_cache(
    cfg,
    out_root=None,
    split="train",
    max_samples=-1,
    augs="identity,hflip,vflip,rot180",
    dabe_version="v2",
    save_vis=False,
    vis_mode="compact",
    vis_dir=None,
    vis_limit=-1,
    compare_dabe_v2_root=None,
    compare_dabe_v31_root=None,
    compare_dabe_gc_root=None,
    compare_dabe_rac_root=None,
    compare_dabe_rac_safe_root=None,
    compare_dabe_pu_root=None,
    overwrite=False,
    logger=print,
):
    if getattr(cfg, "BACKBONE_KEY", None) not in getattr(cfg, "DINO_CONFIGS", {}):
        raise RuntimeError(f"Unknown DABE BACKBONE_KEY={getattr(cfg, 'BACKBONE_KEY', None)!r}.")
    dabe_version = str(dabe_version).lower()
    if dabe_version not in {"v1", "v2", "v3", "v3_1", "gc", "rac", "rac_safe", "pu", "pu_v11"}:
        raise ValueError(f"Unsupported --dabe_version: {dabe_version}")
    vis_mode = str(vis_mode).lower()
    if vis_mode not in {"compact", "debug", "both"}:
        raise ValueError(f"Unsupported --vis_mode: {vis_mode}")

    if out_root is None:
        if dabe_version == "rac_safe":
            out_root = "../datasets/cache/dabe_rac_safe_pseudo_cache/dinov1-s8"
        elif dabe_version == "pu":
            out_root = "../datasets/cache/dabe_pu_pseudo_cache/dinov1-s8"
        elif dabe_version == "pu_v11":
            out_root = "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8"
        else:
            out_root = "../datasets/cache/dabe_v2_pseudo_cache/dinov1-s8"
    out_root = Path(out_root).expanduser()
    manifest_path = out_root / f"manifest_{split}.jsonl"
    ensure_dir(out_root)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to regenerate: {manifest_path}")

    feature_manifest = feature_manifest_path(cfg, split)
    feature_rows = read_jsonl(feature_manifest)
    feature_map = manifest_to_map(feature_rows, feature_manifest)
    all_items = _selected_items(cfg, split, -1)
    all_item_map = {(item["dataset"], item["stem"]): item for item in all_items}
    check_exact_keys("DABE feature manifest", feature_map, all_item_map)
    expected_feature_shape = _expected_feature_shape(cfg)
    for row in feature_rows:
        if "shape" in row and list(row["shape"]) != expected_feature_shape:
            raise RuntimeError(
                f"Feature manifest shape mismatch: expected {expected_feature_shape}, "
                f"got {row['shape']} for {row.get('dataset')}/{row.get('stem')}"
            )
    items = all_items
    if max_samples is not None and int(max_samples) >= 0:
        items = items[: int(max_samples)]
    params = _params_from_cfg(cfg)
    params["VERSION"] = dabe_version
    if dabe_version == "v1":
        effective_params = {**DABE_V1_DEFAULT_PARAMS, **params}
    elif dabe_version == "gc":
        effective_params = {**DABE_GC_DEFAULT_PARAMS, **params}
    elif dabe_version == "rac":
        effective_params = {**DABE_RAC_DEFAULT_PARAMS, **params}
    elif dabe_version == "rac_safe":
        effective_params = {**DABE_RAC_SAFE_DEFAULT_PARAMS, **params}
    elif dabe_version == "pu":
        effective_params = {**DABE_PU_DEFAULT_PARAMS, **params}
    elif dabe_version == "pu_v11":
        effective_params = {**DABE_PU_V11_DEFAULT_PARAMS, **params}
    elif dabe_version == "v3_1":
        effective_params = {**DABE_V31_DEFAULT_PARAMS, **params}
    elif dabe_version == "v3":
        effective_params = {**DABE_V3_DEFAULT_PARAMS, **params}
    else:
        effective_params = {**DABE_V2_DEFAULT_PARAMS, **params}
    vis_dir = Path(vis_dir or "../workdir/dabe_v2_offline_eval/vis").expanduser()
    vis_limit = int(vis_limit)
    light_map = None
    fixed_map = None
    dabe_v2_map = None
    dabe_v31_map = None
    dabe_gc_map = None
    dabe_rac_map = None
    dabe_rac_safe_map = None
    dabe_pu_map = None
    if save_vis:
        light_map = _safe_manifest_map(despl_light_cache_manifest_path(cfg), "DESPL light", logger)
        fixed_map = _safe_manifest_map(pseudo_manifest_path(cfg), "fixed pseudo", logger)
        if compare_dabe_v2_root:
            dabe_v2_manifest = Path(compare_dabe_v2_root).expanduser() / f"manifest_{split}.jsonl"
            dabe_v2_map = _safe_manifest_map(dabe_v2_manifest, "DABE-v2", logger)
        if compare_dabe_v31_root:
            dabe_v31_manifest = Path(compare_dabe_v31_root).expanduser() / f"manifest_{split}.jsonl"
            dabe_v31_map = _safe_manifest_map(dabe_v31_manifest, "DABE-v3.1", logger)
        if compare_dabe_gc_root:
            dabe_gc_manifest = Path(compare_dabe_gc_root).expanduser() / f"manifest_{split}.jsonl"
            dabe_gc_map = _safe_manifest_map(dabe_gc_manifest, "DABE-GC", logger)
        if compare_dabe_rac_root:
            dabe_rac_manifest = Path(compare_dabe_rac_root).expanduser() / f"manifest_{split}.jsonl"
            dabe_rac_map = _safe_manifest_map(dabe_rac_manifest, "DABE-RAC", logger)
        if compare_dabe_rac_safe_root:
            dabe_rac_safe_manifest = Path(compare_dabe_rac_safe_root).expanduser() / f"manifest_{split}.jsonl"
            dabe_rac_safe_map = _safe_manifest_map(dabe_rac_safe_manifest, "DABE-RAC-Safe", logger)
        if compare_dabe_pu_root:
            dabe_pu_manifest = Path(compare_dabe_pu_root).expanduser() / f"manifest_{split}.jsonl"
            dabe_pu_map = _safe_manifest_map(dabe_pu_manifest, "DABE-PU-v1", logger)

    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"feature_manifest = {feature_manifest}")
    logger(f"expected_feature_shape = {expected_feature_shape}")
    logger(f"feature_manifest_num_samples = {len(feature_rows)}")
    logger(f"dabe_out_root = {out_root}")
    logger(f"manifest_path = {manifest_path}")
    logger(f"split = {split}")
    logger(f"dabe_version = {dabe_version}")
    logger(f"num_items = {len(items)}")
    logger(f"augs = {augs}")
    logger("gt_used_for_generation = false")
    logger("dino_forward_used = false")
    logger("training_used = false")
    logger(f"save_vis = {bool(save_vis)}")
    if save_vis:
        logger(f"vis_mode = {vis_mode}")
        logger(f"vis_dir = {vis_dir}")
        logger(f"vis_limit = {vis_limit}")
        logger(f"compare_dabe_v2_root = {compare_dabe_v2_root or ''}")
        logger(f"compare_dabe_v31_root = {compare_dabe_v31_root or ''}")
        logger(f"compare_dabe_gc_root = {compare_dabe_gc_root or ''}")
        logger(f"compare_dabe_rac_root = {compare_dabe_rac_root or ''}")
        logger(f"compare_dabe_rac_safe_root = {compare_dabe_rac_safe_root or ''}")
        logger(f"compare_dabe_pu_root = {compare_dabe_pu_root or ''}")
    if dabe_version == "gc":
        logger(
            "DABE key params | "
            f"BASE_THRESH={effective_params.get('DABE_GC_BASE_THRESH', 'NA')} | "
            f"DIFFUSE_ITERS={effective_params.get('DABE_GC_DIFFUSE_ITERS', 'NA')} | "
            f"DIFFUSE_MU={effective_params.get('DABE_GC_DIFFUSE_MU', 'NA')} | "
            f"BLEND_WEIGHT={effective_params.get('DABE_GC_BLEND_WEIGHT', 'NA')} | "
            f"GRAPH_RADIUS={effective_params.get('DABE_GC_GRAPH_RADIUS', 'NA')} | "
            f"AREA_LOW={effective_params.get('DABE_GC_AREA_LOW', 'NA')} | "
            f"AREA_HIGH={effective_params.get('DABE_GC_AREA_HIGH', 'NA')}"
        )
    elif dabe_version == "rac":
        logger(
            "DABE key params | "
            f"BASE_MASK_THRESH={effective_params.get('DABE_RAC_BASE_MASK_THRESH', 'NA')} | "
            f"REGION_RADIUS={effective_params.get('DABE_RAC_REGION_RADIUS', 'NA')} | "
            f"MAX_REGION_STEPS={effective_params.get('DABE_RAC_MAX_REGION_STEPS', 'NA')} | "
            f"REGION_TOPK={effective_params.get('DABE_RAC_REGION_TOPK', 'NA')} | "
            f"KEEP_THRESH={effective_params.get('DABE_RAC_REGION_KEEP_THRESH', 'NA')} | "
            f"STRONG_KEEP_THRESH={effective_params.get('DABE_RAC_REGION_STRONG_KEEP_THRESH', 'NA')} | "
            f"COMPLETION_WEIGHT={effective_params.get('DABE_RAC_COMPLETION_WEIGHT', 'NA')} | "
            f"AREA_LOW={effective_params.get('DABE_RAC_AREA_LOW', 'NA')} | "
            f"AREA_HIGH={effective_params.get('DABE_RAC_AREA_HIGH', 'NA')}"
        )
    elif dabe_version == "rac_safe":
        logger(
            "DABE key params | "
            f"BASE_MASK_THRESH={effective_params.get('DABE_RAC_SAFE_BASE_MASK_THRESH', 'NA')} | "
            f"LOCAL_BAND_RADIUS={effective_params.get('DABE_RAC_SAFE_LOCAL_BAND_RADIUS', 'NA')} | "
            f"REGION_RADIUS={effective_params.get('DABE_RAC_SAFE_REGION_RADIUS', 'NA')} | "
            f"MAX_REGION_STEPS={effective_params.get('DABE_RAC_SAFE_MAX_REGION_STEPS', 'NA')} | "
            f"KEEP_THRESH={effective_params.get('DABE_RAC_SAFE_REGION_KEEP_THRESH', 'NA')} | "
            f"PIXEL_AFF_MIN={effective_params.get('DABE_RAC_SAFE_PIXEL_AFF_MIN', 'NA')} | "
            f"DELTA_BUDGET_ABS={effective_params.get('DABE_RAC_SAFE_DELTA_BUDGET_ABS', 'NA')} | "
            f"DELTA_BUDGET_REL={effective_params.get('DABE_RAC_SAFE_DELTA_BUDGET_REL', 'NA')} | "
            f"TARGET_AREA_LOW={effective_params.get('DABE_RAC_SAFE_TARGET_AREA_LOW', 'NA')} | "
            f"TARGET_AREA_HIGH={effective_params.get('DABE_RAC_SAFE_TARGET_AREA_HIGH', 'NA')}"
        )
    elif dabe_version == "pu":
        logger(
            "DABE key params | "
            f"BASE_FG_THRESH={effective_params.get('DABE_PU_BASE_FG_THRESH', 'NA')} | "
            f"FG_CORE_P_BASE_THRESH={effective_params.get('DABE_PU_FG_CORE_P_BASE_THRESH', 'NA')} | "
            f"BG_CORE_BC_MIN={effective_params.get('DABE_PU_BG_CORE_BC_MIN', 'NA')} | "
            f"EXTENT_BAND_RADIUS={effective_params.get('DABE_PU_EXTENT_BAND_RADIUS', 'NA')} | "
            f"EXTENT_EVIDENCE_MIN={effective_params.get('DABE_PU_EXTENT_EVIDENCE_MIN', 'NA')} | "
            f"CORE_AFF_MIN={effective_params.get('DABE_PU_CORE_AFF_MIN', 'NA')} | "
            f"TARGET_EXTENT={effective_params.get('DABE_PU_TARGET_EXTENT_MIN', 'NA')}-{effective_params.get('DABE_PU_TARGET_EXTENT_MAX', 'NA')} | "
            f"WEIGHT_EXTENT={effective_params.get('DABE_PU_WEIGHT_EXTENT', 'NA')} | "
            f"WEIGHT_UNKNOWN={effective_params.get('DABE_PU_WEIGHT_UNKNOWN', 'NA')}"
        )
    elif dabe_version == "pu_v11":
        logger(
            "DABE key params | "
            f"BASE_FG_THRESH={effective_params.get('DABE_PU_V11_BASE_FG_THRESH', 'NA')} | "
            f"FG_CORE_P_BASE_THRESH={effective_params.get('DABE_PU_V11_FG_CORE_P_BASE_THRESH', 'NA')} | "
            f"FG_CORE_EVIDENCE_MIN={effective_params.get('DABE_PU_V11_FG_CORE_EVIDENCE_MIN', 'NA')} | "
            f"FG_CORE_WEIGHT={effective_params.get('DABE_PU_V11_FG_CORE_WEIGHT_MIN', 'NA')}-"
            f"{effective_params.get('DABE_PU_V11_FG_CORE_WEIGHT_MAX', 'NA')} | "
            f"FG_FALLBACK_WEIGHT={effective_params.get('DABE_PU_V11_FG_CORE_FALLBACK_WEIGHT', 'NA')} | "
            f"BG_CORE_BC_MIN={effective_params.get('DABE_PU_V11_BG_CORE_BC_MIN', 'NA')} | "
            f"EXTENT_BAND_RADIUS={effective_params.get('DABE_PU_V11_EXTENT_BAND_RADIUS', 'NA')} | "
            f"EXTENT_TARGET={effective_params.get('DABE_PU_V11_TARGET_EXTENT', 'NA')} | "
            f"WEIGHT_EXTENT={effective_params.get('DABE_PU_V11_WEIGHT_EXTENT', 'NA')} | "
            f"WEIGHT_UNKNOWN={effective_params.get('DABE_PU_V11_WEIGHT_UNKNOWN', 'NA')} | "
            f"WEIGHT_SOFT_BASE={effective_params.get('DABE_PU_V11_WEIGHT_SOFT_BASE', 'NA')}"
        )
    elif dabe_version == "v3_1":
        logger(
            "DABE key params | "
            f"BASE_MASK_THRESH={effective_params.get('DABE_V31_BASE_MASK_THRESH', 'NA')} | "
            f"CORE_AFF_MIN={effective_params.get('DABE_V31_CORE_AFF_MIN', 'NA')} | "
            f"RESIDUAL_PERCENTILE={effective_params.get('DABE_V31_RESIDUAL_PERCENTILE', 'NA')} | "
            f"EVIDENCE_MIN={effective_params.get('DABE_V31_EVIDENCE_MIN', 'NA')} | "
            f"EXPAND_PERCENTILE={effective_params.get('DABE_V31_EXPAND_PERCENTILE', 'NA')} | "
            f"EXPAND_TAU={effective_params.get('DABE_V31_EXPAND_TAU', 'NA')} | "
            f"ADAPTIVE_EXPAND={effective_params.get('DABE_V31_USE_ADAPTIVE_EXPAND', 'NA')}"
        )
    else:
        logger(
            "DABE key params | "
            f"EVIDENCE_FLOOR={effective_params.get('DABE_V3_EVIDENCE_FLOOR', 'NA')} | "
            f"EXPAND_WEIGHT={effective_params.get('DABE_V3_EXPAND_WEIGHT', 'NA')} | "
            f"EXPAND_PERCENTILE={effective_params.get('DABE_V3_EXPAND_PERCENTILE', 'NA')} | "
            f"CORE_AFF_MIN={effective_params.get('DABE_V3_CORE_AFF_MIN', 'NA')} | "
            f"AREA_PRIOR_LOW={effective_params.get('DABE_V3_AREA_PRIOR_LOW', 'NA')} | "
            f"AREA_PRIOR_HIGH={effective_params.get('DABE_V3_AREA_PRIOR_HIGH', 'NA')}"
        )

    rows = []
    num_vis = 0
    all_feature_finite = True
    all_r1_finite = True
    all_r1_in_unit_range = True
    for item in tqdm(items, desc=f"cache DABE pseudo {split}"):
        dataset = item["dataset"]
        stem = item["stem"]
        key = (dataset, stem)
        if key not in feature_map:
            raise RuntimeError(f"Missing feature cache for {dataset}/{stem}")
        out_dir = out_root / dataset
        ensure_dir(out_dir)
        out_path = out_dir / f"{stem}.pt"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Cache exists; pass --overwrite to regenerate: {out_path}")

        feature = _load_feature(feature_map[key], dataset, stem, cfg)
        all_feature_finite = all_feature_finite and bool(torch.isfinite(feature).all().item())
        result = generate_dabe_pseudo(feature, item["image_path"], params=params, augs=augs)
        r1 = result.get("residual_pass1_37")
        if not torch.is_tensor(r1) or tuple(r1.shape) != (1, 37, 37):
            raise RuntimeError(
                f"residual_pass1_37 must be Tensor[1,37,37]: {dataset}/{stem}"
            )
        r1_finite = bool(torch.isfinite(r1).all().item())
        r1_unit = r1_finite and float(r1.min()) >= 0.0 and float(r1.max()) <= 1.0
        if not r1_finite:
            raise RuntimeError(f"R1 contains NaN/Inf: {dataset}/{stem}")
        if not r1_unit:
            raise RuntimeError(f"R1 outside [0,1]: {dataset}/{stem}")
        all_r1_finite = all_r1_finite and r1_finite
        all_r1_in_unit_range = all_r1_in_unit_range and r1_unit
        payload = {
            "dataset": dataset,
            "stem": stem,
            "image_path": item["image_path"],
            "gt_path": item["gt_path"],
            "backbone_key": str(cfg.BACKBONE_KEY),
            "dabe_version": str(result.get("dabe_version", dabe_version)),
            **_tensor_payload(result),
            "area": float(result["area"]),
            "base_area": float(result.get("base_area", 0.0)),
            "safe_area": float(result.get("safe_area", 0.0)),
            "gc_area": float(result.get("gc_area", 0.0)),
            "region_area": float(result.get("region_area", 0.0)),
            "local_band_area": float(result.get("local_band_area", 0.0)),
            "expand_area": float(result.get("expand_area", 0.0)),
            "valid_expand_area": float(result.get("valid_expand_area", 0.0)),
            "fg_core_pu_area": float(result.get("fg_core_pu_area", 0.0)),
            "fg_core_fallback_area": float(result.get("fg_core_fallback_area", 0.0)),
            "fg_core_reliability_mean": float(result.get("fg_core_reliability_mean", 0.0)),
            "bg_core_pu_area": float(result.get("bg_core_pu_area", 0.0)),
            "extent_area": float(result.get("extent_area", 0.0)),
            "unknown_area": float(result.get("unknown_area", 0.0)),
            "target_soft_area": float(result.get("target_soft_area", 0.0)),
            "weight_mean": float(result.get("weight_mean", 0.0)),
            "fg_seed_area": float(result.get("fg_seed_area", 0.0)),
            "bg_seed_area": float(result.get("bg_seed_area", 0.0)),
            "fg_core_area": float(result["fg_core_area"]),
            "bg_core_area": float(result["bg_core_area"]),
            "num_components": int(result["num_components"]),
            "num_kept_components": int(result.get("num_kept_components", 0)),
            "num_weak_components": int(result.get("num_weak_components", 0)),
            "num_removed_components": int(result.get("num_removed_components", 0)),
            "num_candidate_regions": int(result.get("num_candidate_regions", 0)),
            "num_kept_regions": int(result.get("num_kept_regions", 0)),
            "num_strong_regions": int(result.get("num_strong_regions", 0)),
            "num_weak_regions": int(result.get("num_weak_regions", 0)),
            "num_rejected_regions": int(result.get("num_rejected_regions", 0)),
            "region_stats": list(result.get("region_stats", [])),
            "large_area_flag": bool(result["large_area_flag"]),
            "small_area_flag": bool(result.get("small_area_flag", False)),
            "uncertain_area": float(result.get("uncertain_area", 0.0)),
            "area_before_area_prior": float(result.get("area_before_area_prior", result["area"])),
            "area_after_area_prior": float(result.get("area_after_area_prior", result["area"])),
            "area_before_area_control": float(result.get("area_before_area_control", result["area"])),
            "area_after_area_control": float(result.get("area_after_area_control", result["area"])),
            "area_suppress_factor": float(result.get("area_suppress_factor", 1.0)),
            "delta_budget": float(result.get("delta_budget", 0.0)),
            "max_final_area": float(result.get("max_final_area", 0.0)),
            "area_before_budget": float(result.get("area_before_budget", result["area"])),
            "area_after_budget": float(result.get("area_after_budget", result["area"])),
            "adaptive_radius": int(result.get("adaptive_radius", 0)),
            "adaptive_expand_weight": float(result.get("adaptive_expand_weight", 0.0)),
            "expand_thr": float(result.get("expand_thr", 0.0)),
            "gc_conf_source": str(result.get("gc_conf_source", "")),
            "residual_mean": float(result["residual_mean"]),
            "fg_score_mean": float(result["fg_score_mean"]),
            "bc_mean": float(result["bc_mean"]),
            "fallback_flag": bool(result["fallback_flag"]),
            "fallback_reason": str(result["fallback_reason"]),
            "params": result["params"],
            "augs": result["augs"],
            "num_views": int(result.get("num_views", len([item for item in str(augs).split(",") if item.strip()]))),
        }
        torch.save(payload, out_path)

        if save_vis and (vis_limit < 0 or num_vis < vis_limit):
            fixed_vis, despl_vis = _load_vis_comparison_tensors(
                key,
                dataset,
                stem,
                light_map,
                fixed_map,
                int(payload["p_dabe_68"].shape[-1]),
            )
            dabe_v2_payload = None
            if dabe_v2_map is not None and key in dabe_v2_map:
                dabe_v2_payload = _load_dabe_payload(dabe_v2_map[key], dataset, stem)
            dabe_v31_payload = None
            if dabe_v31_map is not None and key in dabe_v31_map:
                dabe_v31_payload = _load_dabe_payload(dabe_v31_map[key], dataset, stem)
            dabe_gc_payload = None
            if dabe_gc_map is not None and key in dabe_gc_map:
                dabe_gc_payload = _load_dabe_payload(dabe_gc_map[key], dataset, stem)
            dabe_rac_payload = None
            if dabe_rac_map is not None and key in dabe_rac_map:
                dabe_rac_payload = _load_dabe_payload(dabe_rac_map[key], dataset, stem)
            dabe_rac_safe_payload = None
            if dabe_rac_safe_map is not None and key in dabe_rac_safe_map:
                dabe_rac_safe_payload = _load_dabe_payload(dabe_rac_safe_map[key], dataset, stem)
            dabe_pu_payload = None
            if dabe_pu_map is not None and key in dabe_pu_map:
                dabe_pu_payload = _load_dabe_payload(dabe_pu_map[key], dataset, stem)
            file_name = (
                f"{dataset}_{stem}_area{payload['area']:.3f}_"
                f"fb{int(payload['fallback_flag'])}_{_safe_reason(payload['fallback_reason'])}_"
                f"small{int(payload['small_area_flag'])}_large{int(payload['large_area_flag'])}.png"
            )
            modes = ["compact", "debug"] if vis_mode == "both" else [vis_mode]
            for mode in modes:
                mode_dir = vis_dir / mode if vis_mode == "both" else vis_dir
                vis_path = mode_dir / dataset / file_name
                _save_visualization(
                    vis_path,
                    item["image_path"],
                    item["gt_path"],
                    fixed_vis,
                    despl_vis,
                    payload,
                    mode,
                    dabe_v2=dabe_v2_payload,
                    dabe_v31=dabe_v31_payload,
                    dabe_gc=dabe_gc_payload,
                    dabe_rac=dabe_rac_payload,
                    dabe_rac_safe=dabe_rac_safe_payload,
                    dabe_pu=dabe_pu_payload,
                )
            num_vis += 1

        row = {
            "dataset": dataset,
            "stem": stem,
            "cache_path": str(out_path.resolve()),
            "image_path": item["image_path"],
            "gt_path": item["gt_path"],
            "backbone_key": str(cfg.BACKBONE_KEY),
            "dabe_version": payload["dabe_version"],
            "shape_37": list(payload["p_dabe_37"].shape),
            "shape_68": list(payload["p_dabe_68"].shape),
            "fg_core_area": payload["fg_core_area"],
            "bg_core_area": payload["bg_core_area"],
            "num_components": payload["num_components"],
            "num_kept_components": payload["num_kept_components"],
            "num_weak_components": payload["num_weak_components"],
            "num_removed_components": payload["num_removed_components"],
            "num_candidate_regions": payload["num_candidate_regions"],
            "num_kept_regions": payload["num_kept_regions"],
            "num_strong_regions": payload["num_strong_regions"],
            "num_weak_regions": payload["num_weak_regions"],
            "num_rejected_regions": payload["num_rejected_regions"],
            "fallback_flag": payload["fallback_flag"],
            "fallback_reason": payload["fallback_reason"],
            "small_area_flag": payload["small_area_flag"],
            "large_area_flag": payload["large_area_flag"],
            "area": payload["area"],
            "base_area": payload["base_area"],
            "safe_area": payload["safe_area"],
            "gc_area": payload["gc_area"],
            "region_area": payload["region_area"],
            "local_band_area": payload["local_band_area"],
            "expand_area": payload["expand_area"],
            "valid_expand_area": payload["valid_expand_area"],
            "fg_core_pu_area": payload["fg_core_pu_area"],
            "fg_core_fallback_area": payload["fg_core_fallback_area"],
            "fg_core_reliability_mean": payload["fg_core_reliability_mean"],
            "bg_core_pu_area": payload["bg_core_pu_area"],
            "extent_area": payload["extent_area"],
            "unknown_area": payload["unknown_area"],
            "target_soft_area": payload["target_soft_area"],
            "weight_mean": payload["weight_mean"],
            "fg_seed_area": payload["fg_seed_area"],
            "bg_seed_area": payload["bg_seed_area"],
            "area_before_area_prior": payload["area_before_area_prior"],
            "area_after_area_prior": payload["area_after_area_prior"],
            "area_before_area_control": payload["area_before_area_control"],
            "area_after_area_control": payload["area_after_area_control"],
            "area_suppress_factor": payload["area_suppress_factor"],
            "delta_budget": payload["delta_budget"],
            "max_final_area": payload["max_final_area"],
            "area_before_budget": payload["area_before_budget"],
            "area_after_budget": payload["area_after_budget"],
            "adaptive_radius": payload["adaptive_radius"],
            "adaptive_expand_weight": payload["adaptive_expand_weight"],
            "expand_thr": payload["expand_thr"],
            "gc_conf_source": payload["gc_conf_source"],
        }
        rows.append(row)
        p_dabe_37 = payload["p_dabe_37"].float()
        p_dabe_68 = payload["p_dabe_68"].float()
        tqdm.write(
            "[DABE] "
            f"{dataset}/{stem} | "
            f"p_dabe_37_shape={list(p_dabe_37.shape)} | "
            f"p_dabe_68_shape={list(p_dabe_68.shape)} | "
            f"p_min={float(p_dabe_68.min()):.6f} | "
            f"p_max={float(p_dabe_68.max()):.6f} | "
            f"p_mean={float(p_dabe_68.mean()):.6f} | "
            f"area={payload['area']:.6f} | "
            f"base_area={payload['base_area']:.6f} | "
            f"safe_area={payload['safe_area']:.6f} | "
            f"gc_area={payload['gc_area']:.6f} | "
            f"region_area={payload['region_area']:.6f} | "
            f"local_band_area={payload['local_band_area']:.6f} | "
            f"expand_area={payload['expand_area']:.6f} | "
            f"valid_expand_area={payload['valid_expand_area']:.6f} | "
            f"fg_core_pu_area={payload['fg_core_pu_area']:.6f} | "
            f"fg_core_fallback_area={payload['fg_core_fallback_area']:.6f} | "
            f"fg_core_reliability_mean={payload['fg_core_reliability_mean']:.6f} | "
            f"bg_core_pu_area={payload['bg_core_pu_area']:.6f} | "
            f"extent_area={payload['extent_area']:.6f} | "
            f"unknown_area={payload['unknown_area']:.6f} | "
            f"target_soft_area={payload['target_soft_area']:.6f} | "
            f"weight_mean={payload['weight_mean']:.6f} | "
            f"fg_seed_area={payload['fg_seed_area']:.6f} | "
            f"bg_seed_area={payload['bg_seed_area']:.6f} | "
            f"fallback_flag={payload['fallback_flag']} | "
            f"fallback_reason={payload['fallback_reason']} | "
            f"large_area_flag={payload['large_area_flag']} | "
            f"small_area_flag={payload['small_area_flag']} | "
            f"fg_core_area={payload['fg_core_area']:.6f} | "
            f"bg_core_area={payload['bg_core_area']:.6f} | "
            f"num_components={payload['num_components']} | "
            f"num_kept={payload['num_kept_components']} | "
            f"num_weak={payload['num_weak_components']} | "
            f"num_removed={payload['num_removed_components']} | "
            f"num_candidate_regions={payload['num_candidate_regions']} | "
            f"num_kept_regions={payload['num_kept_regions']} | "
            f"num_strong_regions={payload['num_strong_regions']} | "
            f"num_weak_regions={payload['num_weak_regions']} | "
            f"num_rejected_regions={payload['num_rejected_regions']} | "
            f"delta_budget={payload['delta_budget']:.6f} | "
            f"max_final_area={payload['max_final_area']:.6f} | "
            f"area_before_budget={payload['area_before_budget']:.6f} | "
            f"area_after_budget={payload['area_after_budget']:.6f} | "
            f"adaptive_radius={payload['adaptive_radius']} | "
            f"adaptive_expand_weight={payload['adaptive_expand_weight']:.6f} | "
            f"area_suppress_factor={payload['area_suppress_factor']:.6f} | "
            f"residual_mean={payload['residual_mean']:.6f} | "
            f"fg_score_mean={payload['fg_score_mean']:.6f} | "
            f"bc_mean={payload['bc_mean']:.6f}"
        )

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_rows = {len(rows)}")
    logger(f"all_feature_finite = {str(bool(all_feature_finite)).lower()}")
    logger(f"all_r1_finite = {str(bool(all_r1_finite)).lower()}")
    logger(f"all_r1_in_unit_range = {str(bool(all_r1_in_unit_range)).lower()}")
    if save_vis:
        logger(f"num_visualizations = {num_vis}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate offline DABE pseudo-label cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--augs", default="identity,hflip,vflip,rot180")
    parser.add_argument("--dabe_version", default="v2", choices=["v1", "v2", "v3", "v3_1", "gc", "rac", "rac_safe", "pu", "pu_v11"])
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--vis_mode", default="compact", choices=["compact", "debug", "both"])
    parser.add_argument("--vis_dir", default="../workdir/dabe_v2_offline_eval/vis")
    parser.add_argument("--vis_limit", type=int, default=-1)
    parser.add_argument("--compare_dabe_v2_root", default="")
    parser.add_argument("--compare_dabe_v31_root", default="")
    parser.add_argument("--compare_dabe_gc_root", default="")
    parser.add_argument("--compare_dabe_rac_root", default="")
    parser.add_argument("--compare_dabe_rac_safe_root", default="")
    parser.add_argument("--compare_dabe_pu_root", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")
    if args.vis_limit < -1:
        raise ValueError("--vis_limit must be -1 or a non-negative integer.")

    cfg = load_config(args.config)
    generate_dabe_cache(
        cfg,
        out_root=args.out_root,
        split=args.split,
        max_samples=args.max_samples,
        augs=args.augs,
        dabe_version=args.dabe_version,
        save_vis=args.save_vis,
        vis_mode=args.vis_mode,
        vis_dir=args.vis_dir,
        vis_limit=args.vis_limit,
        compare_dabe_v2_root=args.compare_dabe_v2_root,
        compare_dabe_v31_root=args.compare_dabe_v31_root,
        compare_dabe_gc_root=args.compare_dabe_gc_root,
        compare_dabe_rac_root=args.compare_dabe_rac_root,
        compare_dabe_rac_safe_root=args.compare_dabe_rac_safe_root,
        compare_dabe_pu_root=args.compare_dabe_pu_root,
        overwrite=args.overwrite,
        logger=print,
    )


if __name__ == "__main__":
    main()
