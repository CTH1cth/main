#!/usr/bin/env python3
"""Render a detailed, single-sample DABE-v2 -> Clean-DP -> contrec pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import common.dabe_pseudo as dabe  # noqa: E402
from common.utils import (  # noqa: E402
    dabe_clean_manifest_path,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
)


DISPLAY_SIZE = (256, 256)


def _resolve_from_main(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (MAIN_ROOT / path).resolve()


def _prepare_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tensor(payload, key):
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Missing tensor {key!r}.")
    value = value.detach().cpu().float()
    if value.ndim == 3 and int(value.shape[0]) == 1:
        value = value.squeeze(0)
    if value.ndim != 2:
        raise RuntimeError(f"{key} must reduce to [H,W], got {list(value.shape)}.")
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{key} contains NaN/Inf.")
    return value


def _resize(value, *, nearest=False, size=DISPLAY_SIZE):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    value = value.detach().cpu().float()
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value[None]
    mode = "nearest" if nearest else "bilinear"
    kwargs = {} if nearest else {"align_corners": False}
    value = F.interpolate(value, size=size, mode=mode, **kwargs)
    return value.squeeze().numpy()


def _prob_panel(title, value, *, nearest=False, subtitle=True):
    tensor = value.detach().cpu().float()
    stats = ""
    if subtitle:
        stats = (
            f"\nmin/mean/max="
            f"{float(tensor.min()):.3f}/{float(tensor.mean()):.3f}/"
            f"{float(tensor.max()):.3f}"
        )
    return {
        "title": title + stats,
        "image": _resize(tensor, nearest=nearest),
        "cmap": "gray",
        "vmin": 0.0,
        "vmax": 1.0,
        "nearest": nearest,
    }


def _mask_panel(title, value):
    mask = value.detach().cpu().bool()
    return {
        "title": f"{title}\narea={float(mask.float().mean()):.3f}",
        "image": _resize(mask.float(), nearest=True),
        "cmap": "gray",
        "vmin": 0.0,
        "vmax": 1.0,
        "nearest": True,
    }


def _signed_panel(title, value, limit=1.0):
    tensor = value.detach().cpu().float()
    return {
        "title": (
            f"{title}\nmin/mean/max="
            f"{float(tensor.min()):.3f}/{float(tensor.mean()):.3f}/"
            f"{float(tensor.max()):.3f}"
        ),
        "image": _resize(tensor),
        "cmap": "coolwarm",
        "vmin": -float(limit),
        "vmax": float(limit),
        "nearest": False,
    }


def _rgb_panel(title, image):
    return {
        "title": title,
        "image": image,
        "cmap": None,
        "vmin": None,
        "vmax": None,
        "nearest": False,
    }


def _render_page(path, suptitle, panels, rows=4, cols=4):
    if len(panels) > rows * cols:
        raise RuntimeError(f"Too many panels for {path.name}: {len(panels)}")
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=(5.0 * cols, 4.6 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, panel in zip(axes, panels):
        axis.imshow(
            panel["image"],
            cmap=panel["cmap"],
            vmin=panel["vmin"],
            vmax=panel["vmax"],
            interpolation="nearest" if panel["nearest"] else "bilinear",
        )
        axis.set_title(panel["title"], fontsize=10)
        axis.axis("off")
    for axis in axes[len(panels) :]:
        axis.axis("off")
    figure.suptitle(suptitle, fontsize=16)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _load_rgb(path):
    return np.asarray(
        Image.open(path).convert("RGB").resize(
            DISPLAY_SIZE, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0


def _load_gt(path):
    return np.asarray(
        Image.open(path).convert("L").resize(
            DISPLAY_SIZE, Image.Resampling.NEAREST
        ),
        dtype=np.float32,
    ) / 255.0


def _pca_rgb(feature):
    channels, height, width = feature.shape
    tokens = feature.detach().cpu().float().permute(1, 2, 0).reshape(-1, channels)
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(tokens, full_matrices=False)
    projected = tokens @ vh[:3].T
    output = []
    for channel in projected.T:
        low = torch.quantile(channel, 0.01)
        high = torch.quantile(channel, 0.99)
        output.append(((channel - low) / (high - low + 1e-12)).clamp(0.0, 1.0))
    rgb = torch.stack(output, dim=-1).reshape(height, width, 3).numpy()
    return np.asarray(
        Image.fromarray((rgb * 255.0).astype(np.uint8)).resize(
            DISPLAY_SIZE, Image.Resampling.NEAREST
        ),
        dtype=np.float32,
    ) / 255.0


def _overlay(rgb, value, *, cmap="turbo", alpha=0.45):
    mapped = plt.get_cmap(cmap)(_resize(value))[..., :3]
    return np.clip((1.0 - alpha) * rgb + alpha * mapped, 0.0, 1.0)


def _hard_overlay(rgb, mask, color=(1.0, 0.1, 0.1), alpha=0.45):
    mask_np = _resize(mask.float(), nearest=True)[..., None]
    color_np = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(rgb * (1.0 - alpha * mask_np) + color_np * alpha * mask_np, 0, 1)


def _stats(value):
    tensor = value.detach().cpu().float()
    return {
        "min": float(tensor.min().item()),
        "mean": float(tensor.mean().item()),
        "max": float(tensor.max().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = _prepare_output(args.output_root)
    cfg = load_config(args.config)
    key = (str(args.dataset), str(args.stem))

    clean_manifest = _resolve_from_main(dabe_clean_manifest_path(cfg))
    clean_rows = manifest_to_map(read_jsonl(clean_manifest), clean_manifest)
    if key not in clean_rows:
        raise KeyError(f"Clean cache manifest is missing {key}.")
    clean_path = Path(clean_rows[key]["cache_path"])
    clean = torch_load(clean_path, map_location="cpu")

    source_root = _resolve_from_main(cfg.DABE_CLEAN_SOURCE_ROOT)
    source_manifest = source_root / "manifest_train.jsonl"
    source_rows = manifest_to_map(read_jsonl(source_manifest), source_manifest)
    if key not in source_rows:
        raise KeyError(f"Source cache manifest is missing {key}.")
    source_path = Path(source_rows[key]["cache_path"])
    source = torch_load(source_path, map_location="cpu")
    params = dict(source["params"])
    grid = int(params["GRID"])
    if grid != 37:
        raise RuntimeError(f"This visualizer expects GRID=37, got {grid}.")

    feature_path = (
        _resolve_from_main(cfg.CACHE_ROOT)
        / "features_cache"
        / str(cfg.BACKBONE_KEY)
        / "train"
        / key[0]
        / f"{key[1]}.pt"
    )
    feature_payload = torch_load(feature_path, map_location="cpu")
    feature = feature_payload["tensor"].detach().cpu().float()
    if tuple(feature.shape) != (384, grid, grid):
        raise RuntimeError(f"Unexpected feature shape: {list(feature.shape)}")

    image_path = Path(source["image_path"])
    gt_path = Path(source["gt_path"])
    rgb = _load_rgb(image_path)
    gt = _load_gt(gt_path)
    gt_panel = _rgb_panel("GT (audit/visualization only)", np.repeat(gt[..., None], 3, axis=2))
    rgb_panel = _rgb_panel("RGB", rgb)
    pca_panel = _rgb_panel("DINO token PCA (diagnostic)", _pca_rgb(feature))

    bc = _tensor(source, "bc_map_37")
    bg_anchor = _tensor(source, "bg_anchor_37")
    residual_pass1 = _tensor(source, "residual_pass1_37")
    residual_final = _tensor(source, "residual_37")
    edge = _tensor(source, "edge_37")
    fg_score_final = _tensor(source, "fg_score_37")
    fg_core_final_cache = _tensor(source, "fg_core_37") > 0.5
    bg_core_final_cache = _tensor(source, "bg_core_37") > 0.5
    p_rw = _tensor(source, "p_rw_37")
    evidence = _tensor(source, "evidence_37")
    p_base = _tensor(source, "p_base_37")

    feat_n = F.normalize(
        feature.permute(1, 2, 0).reshape(grid * grid, -1), dim=1, p=2
    )
    score_pass1 = dabe._weak_bc_fg_score(
        residual_pass1.reshape(-1), bc.reshape(-1), params
    ).reshape(grid, grid)
    fg_threshold_pass1 = torch.quantile(
        score_pass1, float(params["FG_PERCENTILE"]) / 100.0
    )
    fg_candidate_pass1 = (
        (score_pass1 > fg_threshold_pass1)
        & (bc < float(params["FG_BC_MAX"]))
    )
    fg_core_pass1_flat, _, _, _ = dabe._select_fg_core_v2(
        feat_n,
        score_pass1.reshape(-1),
        bc.reshape(-1),
        grid,
        params,
    )
    fg_core_pass1 = fg_core_pass1_flat.reshape(grid, grid)

    residual_pass1_algorithm_threshold = torch.quantile(
        residual_pass1,
        float(params["BG_RESIDUAL_PERCENTILE"]) / 100.0,
    )
    bg_core_pass2_flat = dabe._build_bg_core_pass2(
        bc.reshape(-1),
        residual_pass1.reshape(-1),
        fg_core_pass1_flat,
        bg_anchor.reshape(-1).bool(),
        params,
    )
    bg_core_pass2 = bg_core_pass2_flat.reshape(grid, grid)

    fg_threshold_final = torch.quantile(
        fg_score_final, float(params["FG_PERCENTILE"]) / 100.0
    )
    fg_candidate_final = (
        (fg_score_final > fg_threshold_final)
        & (bc < float(params["FG_BC_MAX"]))
    )
    fg_core_final_flat, _, _, _ = dabe._select_fg_core_v2(
        feat_n,
        fg_score_final.reshape(-1),
        bc.reshape(-1),
        grid,
        params,
    )
    border_width = int(params["BORDER_WIDTH"])
    border = torch.zeros((grid, grid), dtype=torch.bool)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    bg_core_final_flat = dabe._select_bg_core_v2(
        bc.reshape(-1),
        fg_score_final.reshape(-1),
        fg_core_final_flat,
        bg_core_pass2_flat,
        bg_anchor.reshape(-1).bool(),
        border.reshape(-1),
        params,
    )
    fg_core_final = fg_core_final_flat.reshape(grid, grid)
    bg_core_final = bg_core_final_flat.reshape(grid, grid)
    if not torch.equal(fg_core_final, fg_core_final_cache):
        raise RuntimeError("Recomputed final foreground core differs from source cache.")
    if not torch.equal(bg_core_final, bg_core_final_cache):
        raise RuntimeError("Recomputed final background core differs from source cache.")

    evidence_threshold = torch.quantile(
        fg_score_final,
        float(params["EVIDENCE_PERCENTILE"]) / 100.0,
    )
    evidence_recomputed = torch.sigmoid(
        (fg_score_final - evidence_threshold) / float(params["EVIDENCE_TAU"])
    ).clamp(0.0, 1.0)
    p_base_recomputed = (p_rw * evidence).clamp(0.0, 1.0)

    background = _tensor(clean, "background_evidence_37")
    clean_dp_37 = _tensor(clean, "target_dp_37")
    foreground = _tensor(clean, "foreground_evidence_37")
    confidence = torch.maximum(foreground, background)
    clean_dp_recomputed = 0.5 + confidence * (foreground - 0.5)
    p_base_68 = _tensor(source, "p_base_68")
    background_68 = _tensor(clean, "background_evidence_68")
    clean_dp_68 = _tensor(clean, "target_dp_68")

    semantic = _tensor(clean, "semantic_fg_tendency_37")
    latent = _tensor(clean, "latent_rw_37")
    recovery = _tensor(clean, "recoverability_37")
    recovery_68 = _tensor(clean, "recoverability_68")
    recovery_product = (latent * (1.0 - background) * semantic).clamp(0.0, 1.0)

    normalized = F.normalize(feature, dim=0, eps=1e-12)
    eps = 1e-6
    global_proto = F.normalize(normalized.mean(dim=(1, 2)), dim=0, eps=1e-12)

    def prototype(weight):
        weight_sum = weight.sum()
        if float(weight_sum.item()) < eps:
            return global_proto
        weighted = (normalized * weight.unsqueeze(0)).sum(dim=(1, 2)) / (
            weight_sum + eps
        )
        return F.normalize(weighted, dim=0, eps=1e-12)

    fg_proto = prototype(foreground)
    bg_proto = prototype(background)
    similarity_fg = (normalized * fg_proto[:, None, None]).sum(dim=0)
    similarity_bg = (normalized * bg_proto[:, None, None]).sum(dim=0)
    similarity_margin = similarity_fg - similarity_bg
    semantic_recomputed = torch.sigmoid(
        similarity_margin / float(getattr(cfg, "ECST_CLEAN_MARGIN_TAU", 0.05))
    )

    regression = {
        "evidence_gate_max_abs_error": float(
            (evidence_recomputed - evidence).abs().max().item()
        ),
        "p_base_equals_p_rw_times_evidence_max_abs_error": float(
            (p_base_recomputed - p_base).abs().max().item()
        ),
        "clean_foreground_equals_p_base_max_abs_error": float(
            (foreground - p_base).abs().max().item()
        ),
        "background_formula_max_abs_error": float(
            (
                background
                - _tensor(source, "bc_map_37")
                * (1.0 - _tensor(source, "residual_37"))
            )
            .abs()
            .max()
            .item()
        ),
        "clean_dp_formula_max_abs_error": float(
            (clean_dp_recomputed - clean_dp_37).abs().max().item()
        ),
        "semantic_formula_max_abs_error": float(
            (semantic_recomputed - semantic).abs().max().item()
        ),
        "latent_formula_max_abs_error": float(
            ((p_rw - foreground).clamp(0.0, 1.0) - latent).abs().max().item()
        ),
        "recoverability_formula_max_abs_error": float(
            (recovery_product.pow(1.0 / 3.0) - recovery).abs().max().item()
        ),
        "fg_core_exact_match": bool(torch.equal(fg_core_final, fg_core_final_cache)),
        "bg_core_exact_match": bool(torch.equal(bg_core_final, bg_core_final_cache)),
    }
    for name, value in regression.items():
        if name.endswith("max_abs_error") and float(value) >= 1e-5:
            raise RuntimeError(f"Pipeline regression failed: {name}={value}")

    page1 = output_root / "01_dabe_v2_pass1_background_reconstruction.png"
    _render_page(
        page1,
        f"{key[0]} / {key[1]} | DABE-v2 pass 1: background connectivity and reconstruction",
        [
            rgb_panel,
            gt_panel,
            pca_panel,
            _prob_panel("Sobel edge", edge),
            _prob_panel("BC score / bc_map", bc),
            _mask_panel("BC > 0.5 (direct diagnostic)", bc > 0.5),
            _mask_panel("Border mask used by DABE", border),
            _mask_panel("Initial background anchor", bg_anchor > 0.5),
            _prob_panel("Pass-1 reconstruction residual", residual_pass1),
            _mask_panel(
                "Pass-1 residual > 0.5 (direct diagnostic)",
                residual_pass1 > 0.5,
            ),
            _mask_panel(
                f"Pass-1 low residual < q30={float(residual_pass1_algorithm_threshold):.3f}",
                residual_pass1 < residual_pass1_algorithm_threshold,
            ),
            _prob_panel("Pass-1 FG score = residual × weak-BC penalty", score_pass1),
            _mask_panel(
                f"Pass-1 FG score > q92={float(fg_threshold_pass1):.3f}",
                score_pass1 > fg_threshold_pass1,
            ),
            _mask_panel("Pass-1 candidate: score q92 + BC<0.4", fg_candidate_pass1),
            _mask_panel("Pass-1 foreground core after CC filtering", fg_core_pass1),
            _mask_panel("Pass-2 background core candidate", bg_core_pass2),
        ],
    )

    page2 = output_root / "02_dabe_v2_pass2_random_walk_and_pbase.png"
    _render_page(
        page2,
        f"{key[0]} / {key[1]} | DABE-v2 pass 2: final evidence and p_base",
        [
            rgb_panel,
            gt_panel,
            _mask_panel("Pass-2 background core used for reconstruction", bg_core_pass2),
            _prob_panel("Final reconstruction residual", residual_final),
            _mask_panel(
                "Final residual > 0.5 (direct diagnostic)", residual_final > 0.5
            ),
            _prob_panel("Final FG score", fg_score_final),
            _mask_panel(
                f"Final FG score > q92={float(fg_threshold_final):.3f}",
                fg_score_final > fg_threshold_final,
            ),
            _mask_panel("Final FG candidate: score q92 + BC<0.4", fg_candidate_final),
            _mask_panel("Final foreground core", fg_core_final),
            _mask_panel("Final background core", bg_core_final),
            _prob_panel("Random-walk foreground p_rw", p_rw),
            _prob_panel(
                f"Evidence gate E (q70={float(evidence_threshold):.3f})", evidence
            ),
            _prob_panel("p_base = p_rw × E", p_base),
            _mask_panel("p_base > 0.5", p_base > 0.5),
            _prob_panel("Suppressed RW mass p_rw - p_base", latent),
            _rgb_panel("p_base soft overlay", _overlay(rgb, p_base)),
        ],
    )

    page3 = output_root / "03_clean_dp_target_construction.png"
    _render_page(
        page3,
        f"{key[0]} / {key[1]} | Clean-DP target construction",
        [
            rgb_panel,
            gt_panel,
            _prob_panel("Foreground evidence F = p_base (37)", foreground),
            _mask_panel("F > 0.5", foreground > 0.5),
            _prob_panel("BC score", bc),
            _prob_panel("1 - final residual", 1.0 - residual_final),
            _prob_panel("Background evidence B = BC × (1-residual)", background),
            _prob_panel("Confidence C = max(F,B)", confidence),
            _signed_panel("Direction F - 0.5", foreground - 0.5, limit=0.5),
            _prob_panel("Clean-DP 37 = 0.5 + C(F-0.5)", clean_dp_37),
            _mask_panel("Clean-DP 37 > 0.5", clean_dp_37 > 0.5),
            _prob_panel("Foreground evidence F68", p_base_68),
            _prob_panel("Background evidence B68", background_68),
            _prob_panel("Clean-DP 68 (actual static target)", clean_dp_68),
            _mask_panel("Clean-DP 68 > 0.5 (audit mask)", clean_dp_68 > 0.5),
            _rgb_panel("Clean-DP soft overlay", _overlay(rgb, clean_dp_68)),
        ],
    )

    page4 = output_root / "04_v5_continuous_recoverability.png"
    _render_page(
        page4,
        f"{key[0]} / {key[1]} | Clean-ECST v5 continuous recoverability",
        [
            rgb_panel,
            gt_panel,
            pca_panel,
            _signed_panel("DINO similarity to FG prototype", similarity_fg),
            _signed_panel("DINO similarity to BG prototype", similarity_bg),
            _signed_panel("DINO margin sim_fg - sim_bg", similarity_margin),
            _prob_panel("Semantic FG tendency H = sigmoid(margin/tau)", semantic),
            _prob_panel("Random-walk p_rw", p_rw),
            _prob_panel("Foreground evidence F", foreground),
            _prob_panel("latent_rw = clamp(p_rw-F,0,1)", latent),
            _prob_panel("1 - Background evidence B", 1.0 - background),
            _prob_panel("Product latent × (1-B) × H", recovery_product),
            _prob_panel("A_rec = cube-root(product)", recovery),
            _mask_panel("A_rec > 0.1", recovery > 0.1),
            _mask_panel("A_rec > 0.3", recovery > 0.3),
            _rgb_panel("A_rec 68 overlay", _overlay(rgb, recovery_68)),
        ],
    )

    page5 = output_root / "05_end_to_end_and_legacy_source_context.png"
    target_soft = _tensor(source, "target_soft_68")
    weight_map = _tensor(source, "weight_map_68")
    _render_page(
        page5,
        f"{key[0]} / {key[1]} | End-to-end summary and unused legacy PU fields",
        [
            rgb_panel,
            gt_panel,
            _prob_panel("DABE-v2 p_base soft (Clean source)", p_base_68),
            _mask_panel("DABE-v2 p_base > 0.5", p_base_68 > 0.5),
            _prob_panel("Clean-DP soft (actual static target)", clean_dp_68),
            _mask_panel("Clean-DP > 0.5 (audit only)", clean_dp_68 > 0.5),
            _prob_panel("Recoverability A_rec 68 (Teacher route evidence)", recovery_68),
            _rgb_panel("Recoverability overlay", _overlay(rgb, recovery_68)),
            _prob_panel("PU-v1.1 target_soft (NOT used by Clean-DP)", target_soft),
            _prob_panel("PU-v1.1 weight_map (NOT used by Clean-DP)", weight_map),
            _mask_panel("PU fg_core (NOT used by v5 route)", _tensor(source, "fg_core_pu_68") > 0.5),
            _mask_panel("PU bg_core (NOT used by v5 route)", _tensor(source, "bg_core_pu_68") > 0.5),
            _prob_panel("PU extent (NOT used by v5 route)", _tensor(source, "extent_candidate_68")),
            _mask_panel("PU unknown (NOT used by v5 route)", _tensor(source, "unknown_68") > 0.5),
            _rgb_panel("Clean-DP hard overlay", _hard_overlay(rgb, clean_dp_68 > 0.5)),
            _rgb_panel("p_base hard overlay", _hard_overlay(rgb, p_base_68 > 0.5, color=(0.1, 1.0, 0.1))),
        ],
    )

    maps = {
        "bc_map_37": bc,
        "residual_pass1_37": residual_pass1,
        "fg_score_pass1_37": score_pass1,
        "residual_final_37": residual_final,
        "fg_score_final_37": fg_score_final,
        "p_rw_37": p_rw,
        "evidence_gate_37": evidence,
        "p_base_37": p_base,
        "background_evidence_37": background,
        "target_dp_37": clean_dp_37,
        "target_dp_68": clean_dp_68,
        "semantic_fg_tendency_37": semantic,
        "latent_rw_37": latent,
        "recoverability_product_37": recovery_product,
        "recoverability_37": recovery,
        "recoverability_68": recovery_68,
    }
    summary = {
        "schema": "clean_dp_full_pipeline_visualization_v1",
        "dataset": key[0],
        "stem": key[1],
        "config": str(Path(args.config).resolve()),
        "source_cache": str(source_path),
        "clean_cache": str(clean_path),
        "feature_cache": str(feature_path),
        "image_path": str(image_path),
        "gt_path": str(gt_path),
        "gt_usage": "offline_visualization_only",
        "thresholds": {
            "bc_direct_diagnostic": 0.5,
            "residual_direct_diagnostic": 0.5,
            "pass1_residual_algorithm_q30": float(
                residual_pass1_algorithm_threshold.item()
            ),
            "pass1_fg_score_q92": float(fg_threshold_pass1.item()),
            "final_fg_score_q92": float(fg_threshold_final.item()),
            "fg_bc_max": float(params["FG_BC_MAX"]),
            "pass2_bg_bc_min": float(params["BG_BC_MIN"]),
            "evidence_score_q70": float(evidence_threshold.item()),
            "evidence_tau": float(params["EVIDENCE_TAU"]),
            "hard_target_threshold": 0.5,
        },
        "important_note": (
            "BC>0.5 and residual>0.5 panels are direct diagnostic binaries only. "
            "The actual DABE-v2 algorithm uses quantiles, BC constraints, connected "
            "components, seeds, and graph propagation."
        ),
        "regression": regression,
        "map_statistics": {name: _stats(value) for name, value in maps.items()},
        "outputs": [str(page) for page in (page1, page2, page3, page4, page5)],
    }
    summary_path = output_root / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# {key[0]} / {key[1]} Clean-DP 全流程可视化",
                "",
                "1. `01_dabe_v2_pass1_background_reconstruction.png`：第一次 BC、背景锚点、第一次重构误差及首轮核心。",
                "2. `02_dabe_v2_pass2_random_walk_and_pbase.png`：第二次背景重构、随机游走、证据门和 p_base。",
                "3. `03_clean_dp_target_construction.png`：F/B/C 到 Clean-DP 37/68 的完整构造。",
                "4. `04_v5_continuous_recoverability.png`：DINO 原型倾向、latent 和 A_rec。",
                "5. `05_end_to_end_and_legacy_source_context.png`：端到端总结，并明确旧 PU 字段不参与当前 Clean-DP/v5。",
                "",
                "> 注意：图中的 `BC>0.5` 和 `residual>0.5` 是按用户要求添加的直接二值诊断，不是 DABE-v2 实际采用的完整筛选规则。",
                "",
                "GT 仅用于本次离线对照图，不参与 cache 或训练。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
