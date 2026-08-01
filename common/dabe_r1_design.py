"""Frozen formulas for the DABE-TF R1-Design-v1 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image

from common.dabe_pseudo import (
    EPS,
    RESAMPLE_BICUBIC,
    _background_anchor,
    _background_connectivity,
    _background_residual,
    _build_local_graph,
    _load_rgb_grid,
    _minmax,
    _sobel_magnitude,
    _validate_feature,
)
from common.dabe_rank_calibration import average_percentile_rank


DABE_R1_DESIGN_VERSION = "dabe_r1_design_v1"
EDGE_MODES = ("node_sobel", "none", "hr_interface")
RESIDUAL_MODES = ("prototype", "support_consistent")
PROBABILITY_FIELDS = (
    "m0_bw2_node_proto_37", "m1_bw1_node_proto_37",
    "m2_bw2_noedge_proto_37", "m3_bw2_hrinterface_proto_37",
    "m4_bw2_node_sc_37", "m5_bw1_node_sc_37",
    "bc_m0_37", "bc_m1_37", "bc_m2_37", "bc_m3_37",
    "anchor_m0_37", "anchor_m1_37", "anchor_m2_37", "anchor_m3_37",
)
NONNEGATIVE_MAP_FIELDS = (
    "raw_m0_37", "raw_m1_37", "raw_m2_37", "raw_m3_37", "raw_m4_37", "raw_m5_37",
    "color_dispersion_m0_37", "color_dispersion_m1_37",
    "semantic_m0_37", "semantic_m1_37", "semantic_m4_37", "semantic_m5_37",
    "color_m0_37", "color_m1_37", "color_m4_37", "color_m5_37",
    "support_norm_m0_37", "support_norm_m1_37",
    "weight_entropy_m0_37", "weight_entropy_m1_37",
)


@dataclass(frozen=True)
class ReconstructionDetails:
    raw_residual: torch.Tensor
    normalized_residual: torch.Tensor
    semantic_residual: torch.Tensor
    color_residual: torch.Tensor
    topk_anchor_local_index: torch.Tensor
    topk_anchor_global_index: torch.Tensor
    topk_score: torch.Tensor
    topk_weight: torch.Tensor
    reconstructed_feature_raw: torch.Tensor
    reconstructed_feature_normalized: torch.Tensor
    reconstructed_rgb: torch.Tensor
    support_norm: torch.Tensor
    weight_entropy: torch.Tensor
    color_support_dispersion: torch.Tensor


@dataclass(frozen=True)
class GraphDetails:
    neigh_idx: torch.Tensor
    neigh_weight: torch.Tensor
    edge_cue: torch.Tensor
    edge_mode: str


def border_source_mask(grid: int, width: int) -> torch.Tensor:
    if int(grid) <= 0 or int(width) <= 0 or int(width) * 2 > int(grid):
        raise ValueError("grid/width do not define a valid border")
    mask = torch.zeros((int(grid), int(grid)), dtype=torch.bool)
    mask[:width] = True
    mask[-width:] = True
    mask[:, :width] = True
    mask[:, -width:] = True
    return mask.reshape(-1)


def _load_rgb_296(image_path: str | Path, size: int) -> torch.Tensor:
    if int(size) != 296:
        raise ValueError("R1-Design-v1 requires feature_input_size=296")
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((size, size), RESAMPLE_BICUBIC),
            dtype=np.float32,
        ).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _neighbor_layout(grid: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = grid * grid
    idx = torch.zeros((count, 8), dtype=torch.long)
    valid = torch.zeros((count, 8), dtype=torch.bool)
    for y in range(grid):
        for x in range(grid):
            src, slot = y * grid + x, 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < grid and 0 <= nx < grid:
                        idx[src, slot] = ny * grid + nx
                        valid[src, slot] = True
                        slot += 1
    return idx, valid


def _hr_interface_cue(image_rgb_296: torch.Tensor, grid: int) -> torch.Tensor:
    if tuple(image_rgb_296.shape) != (3, 296, 296) or grid != 37:
        raise ValueError("HR interface requires RGB[3,296,296] and grid=37")
    edge = _sobel_magnitude(image_rgb_296)
    idx, valid = _neighbor_layout(grid)
    cue = torch.zeros((grid * grid, 8), dtype=torch.float32)
    for src in range(grid * grid):
        y, x = divmod(src, grid)
        for slot in torch.where(valid[src])[0].tolist():
            dst = int(idx[src, slot])
            ny, nx = divmod(dst, grid)
            dy, dx = ny - y, nx - x
            if dy == 0:
                xb = max(x, nx) * 8
                region = edge[y * 8 : (y + 1) * 8, xb - 1 : xb + 1]
            elif dx == 0:
                yb = max(y, ny) * 8
                region = edge[yb - 1 : yb + 1, x * 8 : (x + 1) * 8]
            else:
                yb, xb = max(y, ny) * 8, max(x, nx) * 8
                region = edge[yb - 1 : yb + 1, xb - 1 : xb + 1]
            if region.numel() == 0:
                raise RuntimeError(f"Empty HR interface region at {src}->{dst}")
            cue[src, slot] = region.mean()
    return cue.clamp(0.0, 1.0)


def build_local_graph_variant(
    *,
    feat_n: torch.Tensor,
    rgb_37: torch.Tensor,
    image_rgb_296: torch.Tensor,
    grid: int,
    params: dict,
    edge_mode: str,
) -> GraphDetails:
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"edge_mode must be one of {EDGE_MODES}")
    grid = int(grid)
    count = grid * grid
    if tuple(feat_n.shape) != (count, 384) or tuple(rgb_37.shape) != (count, 3):
        raise ValueError("feat_n/rgb_37 shape mismatch")
    idx, valid = _neighbor_layout(grid)
    if edge_mode == "node_sobel":
        rgb_chw = rgb_37.reshape(grid, grid, 3).permute(2, 0, 1).contiguous()
        edge_nodes = _sobel_magnitude(rgb_chw).reshape(-1)
        weights_idx, weights = _build_local_graph(
            feat_n, rgb_37, edge_nodes, grid, params
        )
        if not torch.equal(weights_idx, idx):
            raise RuntimeError("Current graph neighbor layout changed")
        cue = torch.zeros_like(weights)
        src_edge = edge_nodes[:, None].expand_as(cue)
        dst_edge = edge_nodes[idx]
        cue[valid] = torch.maximum(src_edge, dst_edge)[valid]
        return GraphDetails(idx, weights, cue, edge_mode)

    cue = (
        torch.zeros((count, 8), dtype=torch.float32)
        if edge_mode == "none"
        else _hr_interface_cue(image_rgb_296, grid)
    )
    src = torch.arange(count)[:, None].expand_as(idx)
    df = 1.0 - (feat_n[src] * feat_n[idx]).sum(dim=-1)
    dc = torch.square(rgb_37[src] - rgb_37[idx]).sum(dim=-1)
    exponent = -df / float(params["SIGMA_F"]) - dc / float(params["SIGMA_C"])
    if edge_mode == "hr_interface":
        exponent = exponent - cue / float(params["SIGMA_E"])
    weights = torch.zeros((count, 8), dtype=torch.float32)
    weights[valid] = torch.exp(exponent[valid]).clamp_min(EPS).float()
    cue[~valid] = 0.0
    return GraphDetails(idx, weights, cue.float().contiguous(), edge_mode)


def reconstruct_background(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    anchor: torch.Tensor,
    params: dict,
    residual_mode: str = "prototype",
) -> ReconstructionDetails:
    if residual_mode not in RESIDUAL_MODES:
        raise ValueError(f"residual_mode must be one of {RESIDUAL_MODES}")
    if feat_n.ndim != 2 or rgb_n.ndim != 2 or anchor.ndim != 1:
        raise ValueError("feat_n, rgb_n and anchor must be flattened")
    if feat_n.shape[0] != rgb_n.shape[0] or feat_n.shape[0] != anchor.numel():
        raise ValueError("query counts differ")
    anchor_idx = torch.where(anchor.bool())[0]
    if anchor_idx.numel() == 0:
        raise ValueError("background anchor set must not be empty")
    feat_anchor, rgb_anchor = feat_n[anchor_idx], rgb_n[anchor_idx]
    k = min(int(params["K_RECON"]), int(anchor_idx.numel()))
    outputs = {name: [] for name in (
        "raw", "norm_sem", "color", "local", "global", "score", "weight",
        "feat_raw", "feat_norm", "rgb_hat", "support_norm", "entropy", "dispersion",
    )}
    for start in range(0, feat_n.shape[0], 512):
        end = min(start + 512, feat_n.shape[0])
        fq, cq = feat_n[start:end], rgb_n[start:end]
        sim_feat = fq @ feat_anchor.t()
        color_dist2 = torch.cdist(cq, rgb_anchor, p=2.0).square()
        sim_color = torch.exp(-color_dist2 / float(params["SIGMA_COLOR_RECON"]))
        similarity = sim_feat + float(params["LAMBDA_COLOR_RECON"]) * sim_color
        top_score, top_local = torch.topk(similarity, k=k, dim=1)
        weight = torch.softmax(top_score / float(params["TAU_RECON"]), dim=1)
        feat_top, rgb_top = feat_anchor[top_local], rgb_anchor[top_local]
        feat_raw = (weight.unsqueeze(-1) * feat_top).sum(dim=1)
        feat_norm = torch_f.normalize(feat_raw, dim=1, p=2)
        rgb_hat = (weight.unsqueeze(-1) * rgb_top).sum(dim=1)
        if residual_mode == "prototype":
            semantic = (1.0 - (fq * feat_norm).sum(dim=1)).clamp_min(0.0)
            color = torch.linalg.norm(cq - rgb_hat, dim=1)
        else:
            individual_sem = (1.0 - (fq[:, None, :] * feat_top).sum(dim=-1)).clamp_min(0.0)
            semantic = (weight * individual_sem).sum(dim=1)
            individual_color = torch.linalg.norm(cq[:, None, :] - rgb_top, dim=-1)
            color = (weight * individual_color).sum(dim=1)
        raw = (semantic + 0.2 * color).clamp_min(0.0)
        entropy = (
            -(weight * torch.log(weight + EPS)).sum(dim=1) / np.log(k)
            if k > 1 else torch.zeros(weight.shape[0])
        )
        dispersion = (
            weight * torch.linalg.norm(rgb_top - rgb_hat[:, None, :], dim=-1)
        ).sum(dim=1)
        for name, value in {
            "raw": raw, "norm_sem": semantic, "color": color,
            "local": top_local, "global": anchor_idx[top_local], "score": top_score,
            "weight": weight, "feat_raw": feat_raw, "feat_norm": feat_norm,
            "rgb_hat": rgb_hat, "support_norm": torch.linalg.norm(feat_raw, dim=1),
            "entropy": entropy, "dispersion": dispersion,
        }.items():
            outputs[name].append(value)
    values = {name: torch.cat(parts, dim=0).detach().cpu().contiguous() for name, parts in outputs.items()}
    return ReconstructionDetails(
        raw_residual=values["raw"].float(),
        normalized_residual=_minmax(values["raw"]).float().contiguous(),
        semantic_residual=values["norm_sem"].float(),
        color_residual=values["color"].float(),
        topk_anchor_local_index=values["local"].long(),
        topk_anchor_global_index=values["global"].long(),
        topk_score=values["score"].float(),
        topk_weight=values["weight"].float(),
        reconstructed_feature_raw=values["feat_raw"].float(),
        reconstructed_feature_normalized=values["feat_norm"].float(),
        reconstructed_rgb=values["rgb_hat"].float(),
        support_norm=values["support_norm"].float(),
        weight_entropy=values["entropy"].float(),
        color_support_dispersion=values["dispersion"].float(),
    )


def support_consistent_from_details(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    anchor: torch.Tensor,
    prototype: ReconstructionDetails,
) -> ReconstructionDetails:
    anchor_idx = torch.where(anchor.bool())[0]
    feat_top = feat_n[anchor_idx[prototype.topk_anchor_local_index]]
    rgb_top = rgb_n[anchor_idx[prototype.topk_anchor_local_index]]
    weight = prototype.topk_weight
    semantic = (
        weight * (1.0 - (feat_n[:, None, :] * feat_top).sum(dim=-1)).clamp_min(0.0)
    ).sum(dim=1)
    color = (
        weight * torch.linalg.norm(rgb_n[:, None, :] - rgb_top, dim=-1)
    ).sum(dim=1)
    raw = (semantic + 0.2 * color).clamp_min(0.0)
    return ReconstructionDetails(
        raw, _minmax(raw).float().contiguous(), semantic, color,
        prototype.topk_anchor_local_index, prototype.topk_anchor_global_index,
        prototype.topk_score, prototype.topk_weight,
        prototype.reconstructed_feature_raw, prototype.reconstructed_feature_normalized,
        prototype.reconstructed_rgb, prototype.support_norm,
        prototype.weight_entropy, prototype.color_support_dispersion,
    )


def _pearson(left: torch.Tensor, right: torch.Tensor, rank: bool = False) -> float:
    if rank:
        left, right = average_percentile_rank(left), average_percentile_rank(right)
    x, y = left.reshape(-1).double(), right.reshape(-1).double()
    x, y = x - x.mean(), y - y.mean()
    denominator = float(torch.linalg.norm(x) * torch.linalg.norm(y))
    return float(torch.dot(x, y) / denominator) if denominator > 0 else float("nan")


def _map(value: torch.Tensor, grid: int) -> torch.Tensor:
    return value.detach().cpu().float().reshape(1, grid, grid).contiguous()


def build_r1_design_candidates(
    *,
    feature_37: torch.Tensor,
    image_path: str,
    cached_r1_37: torch.Tensor,
    effective_params: dict,
    feature_input_size: int,
) -> dict:
    grid = int(effective_params["GRID"])
    if grid != 37 or int(feature_input_size) != 296:
        raise ValueError("Frozen protocol requires grid=37 and feature_input_size=296")
    feature = _validate_feature(feature_37, grid)
    cached = cached_r1_37.detach().cpu().float().contiguous()
    if tuple(cached.shape) != (1, 37, 37) or not torch.isfinite(cached).all():
        raise ValueError("cached_r1_37 must be finite Tensor[1,37,37]")
    rgb_chw = _load_rgb_grid(image_path, grid)
    rgb_n = rgb_chw.permute(1, 2, 0).reshape(grid * grid, 3)
    feat_n = torch_f.normalize(
        feature.permute(1, 2, 0).reshape(grid * grid, 384), dim=1, p=2
    )
    rgb296 = _load_rgb_296(image_path, feature_input_size)

    graph_node = build_local_graph_variant(
        feat_n=feat_n, rgb_37=rgb_n, image_rgb_296=rgb296,
        grid=grid, params=effective_params, edge_mode="node_sobel",
    )
    graph_none = build_local_graph_variant(
        feat_n=feat_n, rgb_37=rgb_n, image_rgb_296=rgb296,
        grid=grid, params=effective_params, edge_mode="none",
    )
    graph_hr = build_local_graph_variant(
        feat_n=feat_n, rgb_37=rgb_n, image_rgb_296=rgb296,
        grid=grid, params=effective_params, edge_mode="hr_interface",
    )
    params_bw2, params_bw1 = dict(effective_params), dict(effective_params)
    params_bw2["BORDER_WIDTH"], params_bw1["BORDER_WIDTH"] = 2, 1
    bc0, border2 = _background_connectivity(graph_node.neigh_idx, graph_node.neigh_weight, grid, params_bw2)
    bc1, border1 = _background_connectivity(graph_node.neigh_idx, graph_node.neigh_weight, grid, params_bw1)
    bc2, _ = _background_connectivity(graph_none.neigh_idx, graph_none.neigh_weight, grid, params_bw2)
    bc3, _ = _background_connectivity(graph_hr.neigh_idx, graph_hr.neigh_weight, grid, params_bw2)
    anchors = [
        _background_anchor(bc0, border2, params_bw2),
        _background_anchor(bc1, border1, params_bw1),
        _background_anchor(bc2, border2, params_bw2),
        _background_anchor(bc3, border2, params_bw2),
    ]
    current = _background_residual(feat_n, rgb_n, anchors[0], params_bw2).reshape(1, grid, grid)
    details = [
        reconstruct_background(feat_n, rgb_n, anchors[0], params_bw2, "prototype"),
        reconstruct_background(feat_n, rgb_n, anchors[1], params_bw1, "prototype"),
        reconstruct_background(feat_n, rgb_n, anchors[2], params_bw2, "prototype"),
        reconstruct_background(feat_n, rgb_n, anchors[3], params_bw2, "prototype"),
    ]
    sc0 = support_consistent_from_details(feat_n, rgb_n, anchors[0], details[0])
    sc1 = support_consistent_from_details(feat_n, rgb_n, anchors[1], details[1])
    cached_error = float(torch.abs(details[0].normalized_residual.reshape_as(cached) - cached).max())
    current_error = float(torch.abs(current - cached).max())
    if cached_error > 1e-6 or current_error > 1e-6:
        raise RuntimeError(
            f"M0 baseline mismatch: detailed={cached_error}, current={current_error}"
        )
    valid = graph_node.neigh_weight > 0
    diagnostics = {
        "m0_cached_r1_max_abs": cached_error,
        "m0_current_function_max_abs": current_error,
        "border_source_count_bw2": int(border2.sum()),
        "border_source_count_bw1": int(border1.sum()),
        "border_source_ratio_bw2": float(border2.float().mean()),
        "border_source_ratio_bw1": float(border1.float().mean()),
    }
    for index, (bc, anchor) in enumerate(zip((bc0, bc1, bc2, bc3), anchors)):
        diagnostics.update({
            f"anchor_count_m{index}": int(anchor.sum()),
            f"anchor_ratio_m{index}": float(anchor.float().mean()),
            f"bc_mean_m{index}": float(bc.mean()),
            f"bc_std_m{index}": float(bc.std(unbiased=False)),
        })
    all_details = [*details, sc0, sc1]
    for index, detail in enumerate(all_details):
        diagnostics[f"m{index}_area_gt_05"] = float((detail.normalized_residual > 0.5).float().mean())
        diagnostics[f"raw_mean_m{index}"] = float(detail.raw_residual.mean())
        diagnostics[f"raw_std_m{index}"] = float(detail.raw_residual.std(unbiased=False))
    for index, detail in enumerate(details[:2]):
        diagnostics.update({
            f"support_norm_mean_m{index}": float(detail.support_norm.mean()),
            f"support_norm_std_m{index}": float(detail.support_norm.std(unbiased=False)),
            f"weight_entropy_mean_m{index}": float(detail.weight_entropy.mean()),
            f"color_dispersion_mean_m{index}": float(detail.color_support_dispersion.mean()),
        })
    diagnostics.update({
        "proto_sc_spearman_bw2": _pearson(details[0].raw_residual, sc0.raw_residual, rank=True),
        "proto_sc_spearman_bw1": _pearson(details[1].raw_residual, sc1.raw_residual, rank=True),
        "node_hr_edge_pearson": _pearson(graph_node.edge_cue[valid], graph_hr.edge_cue[valid]),
        "node_hr_edge_spearman": _pearson(graph_node.edge_cue[valid], graph_hr.edge_cue[valid], rank=True),
    })
    result = {
        "m0_bw2_node_proto_37": _map(details[0].normalized_residual, grid),
        "m1_bw1_node_proto_37": _map(details[1].normalized_residual, grid),
        "m2_bw2_noedge_proto_37": _map(details[2].normalized_residual, grid),
        "m3_bw2_hrinterface_proto_37": _map(details[3].normalized_residual, grid),
        "m4_bw2_node_sc_37": _map(sc0.normalized_residual, grid),
        "m5_bw1_node_sc_37": _map(sc1.normalized_residual, grid),
        **{f"bc_m{i}_37": _map(value, grid) for i, value in enumerate((bc0, bc1, bc2, bc3))},
        **{f"anchor_m{i}_37": _map(value.float(), grid) for i, value in enumerate(anchors)},
        **{f"raw_m{i}_37": _map(value.raw_residual, grid) for i, value in enumerate(all_details)},
        "support_norm_m0_37": _map(details[0].support_norm, grid),
        "support_norm_m1_37": _map(details[1].support_norm, grid),
        "weight_entropy_m0_37": _map(details[0].weight_entropy, grid),
        "weight_entropy_m1_37": _map(details[1].weight_entropy, grid),
        "color_dispersion_m0_37": _map(details[0].color_support_dispersion, grid),
        "color_dispersion_m1_37": _map(details[1].color_support_dispersion, grid),
        "semantic_m0_37": _map(details[0].semantic_residual, grid),
        "semantic_m1_37": _map(details[1].semantic_residual, grid),
        "semantic_m4_37": _map(sc0.semantic_residual, grid),
        "semantic_m5_37": _map(sc1.semantic_residual, grid),
        "color_m0_37": _map(details[0].color_residual, grid),
        "color_m1_37": _map(details[1].color_residual, grid),
        "color_m4_37": _map(sc0.color_residual, grid),
        "color_m5_37": _map(sc1.color_residual, grid),
        "neigh_idx_37": graph_node.neigh_idx,
        "neigh_valid_37": valid,
        "edge_cue_node": graph_node.edge_cue,
        "edge_cue_hr": graph_hr.edge_cue,
        "graph_weight_node": graph_node.neigh_weight,
        "graph_weight_hr": graph_hr.neigh_weight,
        "diagnostics": diagnostics,
    }
    return result
