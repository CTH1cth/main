"""Frozen formulas for DABE-TF Reliability-Prior Reconstruction (RPR-v1).

RPR keeps the B0/BW2 anchor set and the original B0 retrieval ranking fixed.
The only experimental operation is a post-TopK reliability prior inside the
reconstruction softmax.  This module is deliberately GT-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as torch_f

from common.dabe_pseudo import _load_rgb_grid, _minmax, _validate_feature


DABE_RPR_VERSION = "dabe_rpr_v1"
RPR_Q_MIN = 1e-6
RPR_LOW_Q_DIAGNOSTIC_THRESHOLD = 0.5


PROBABILITY_FIELDS = (
    "p1_rpr_secondring_37",
    "p2_rpr_allborder_37",
    "atom_prior_p1_37",
    "atom_prior_p2_37",
    "u_base_p1_37",
    "u_rpr_p1_37",
    "u_reduction_p1_37",
    "u_base_p2_37",
    "u_rpr_p2_37",
    "u_reduction_p2_37",
    "weight_l1_shift_p1_37",
    "weight_l1_shift_p2_37",
    "max_weight_base_37",
    "max_weight_p1_37",
    "max_weight_p2_37",
    "topk_prior_min_p1_37",
    "topk_prior_mean_p1_37",
    "topk_low_q_ratio_p1_37",
    "topk_prior_min_p2_37",
    "topk_prior_mean_p2_37",
    "topk_low_q_ratio_p2_37",
)

NONNEGATIVE_FIELDS = (
    "raw_b0_recomputed_37",
    "raw_p1_37",
    "raw_p2_37",
)

EFFECTIVE_ATOM_FIELDS = (
    "effective_atoms_base_37",
    "effective_atoms_p1_37",
    "effective_atoms_p2_37",
)

SIGNED_FIELDS = (
    "delta_raw_p1_vs_b0_37",
    "delta_raw_p2_vs_b0_37",
)


@dataclass(frozen=True)
class RPRReconstructionDetails:
    normalized_residual: torch.Tensor
    raw_residual: torch.Tensor

    feature_residual: torch.Tensor
    color_residual: torch.Tensor

    topk_anchor_local_index: torch.Tensor
    topk_anchor_global_index: torch.Tensor
    topk_score: torch.Tensor

    base_weight: torch.Tensor
    rpr_weight: torch.Tensor
    topk_prior: torch.Tensor

    unreliable_mass_base: torch.Tensor
    unreliable_mass_rpr: torch.Tensor
    weight_l1_shift: torch.Tensor

    effective_atoms_base: torch.Tensor
    effective_atoms_rpr: torch.Tensor

    max_weight_base: torch.Tensor
    max_weight_rpr: torch.Tensor

    topk_prior_min: torch.Tensor
    topk_prior_mean: torch.Tensor
    topk_low_q_ratio: torch.Tensor

    # Explicit B0/unit-prior controls used by regression tests and cache gates.
    base_normalized_residual: torch.Tensor
    base_raw_residual: torch.Tensor
    base_feature_residual: torch.Tensor
    base_color_residual: torch.Tensor
    reconstructed_feature_base: torch.Tensor
    reconstructed_feature_rpr: torch.Tensor
    reconstructed_rgb_base: torch.Tensor
    reconstructed_rgb_rpr: torch.Tensor


def _validate_inputs(
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    anchor_mask: torch.Tensor,
    atom_prior: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not all(
        torch.is_tensor(value)
        for value in (feat_n, rgb_n, anchor_mask, atom_prior)
    ):
        raise TypeError("feat_n, rgb_n, anchor_mask and atom_prior must be tensors")
    if any(
        value.device.type != "cpu"
        for value in (feat_n, rgb_n, anchor_mask, atom_prior)
    ):
        raise ValueError("RPR reconstruction requires CPU tensors")
    if tuple(feat_n.shape) != (1369, 384):
        raise ValueError(f"feat_n must be [1369,384], got {tuple(feat_n.shape)}")
    if tuple(rgb_n.shape) != (1369, 3):
        raise ValueError(f"rgb_n must be [1369,3], got {tuple(rgb_n.shape)}")
    if anchor_mask.numel() != 1369 or atom_prior.numel() != 1369:
        raise ValueError("anchor_mask and atom_prior must contain 1369 elements")
    if int(params.get("GRID", 0)) != 37:
        raise ValueError("RPR-v1 requires GRID=37")
    for key in ("K_RECON", "LAMBDA_COLOR_RECON", "SIGMA_COLOR_RECON", "TAU_RECON"):
        if key not in params:
            raise KeyError(f"missing reconstruction parameter: {key}")
    if not torch.isfinite(feat_n).all() or not torch.isfinite(rgb_n).all():
        raise ValueError("feature/RGB contains NaN or Inf")
    if not torch.isfinite(atom_prior).all():
        raise ValueError("atom_prior contains NaN or Inf")

    # Detach/copy views only; no input tensor is mutated.
    feature = feat_n.detach().cpu().float().contiguous()
    rgb = rgb_n.detach().cpu().float().contiguous()
    anchor = anchor_mask.detach().cpu().bool().reshape(-1).contiguous()
    prior = atom_prior.detach().cpu().float().reshape(-1).contiguous()
    if not bool(anchor.any()):
        raise ValueError("background anchor set must not be empty")
    anchor_prior = prior[anchor]
    q_min_float32 = float(torch.tensor(RPR_Q_MIN, dtype=anchor_prior.dtype))
    if float(anchor_prior.min()) < q_min_float32 or float(anchor_prior.max()) > 1.0:
        raise ValueError("anchor atom prior must be in [1e-6,1]")

    k = min(int(params["K_RECON"]), int(anchor.sum()))
    if k <= 0:
        raise ValueError("K_RECON and anchor count must be positive")
    if float(params["TAU_RECON"]) <= 0 or float(params["SIGMA_COLOR_RECON"]) <= 0:
        raise ValueError("TAU_RECON and SIGMA_COLOR_RECON must be positive")
    return feature, rgb, anchor, prior, k


def _cat(parts: list[torch.Tensor], *, dtype: torch.dtype | None = torch.float32):
    value = torch.cat(parts, dim=0).detach().cpu()
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.contiguous()


def reconstruct_with_reliability_prior(
    *,
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    anchor_mask: torch.Tensor,
    atom_prior: torch.Tensor,
    params: dict,
) -> RPRReconstructionDetails:
    """Reconstruct every query with a fixed B0 TopK and post-TopK prior.

    The selected atom indices depend only on the original B0 retrieval score.
    ``atom_prior`` is introduced only as ``log(q)`` after TopK selection.
    """
    feat_n, rgb_n, anchor_mask, atom_prior, k = _validate_inputs(
        feat_n, rgb_n, anchor_mask, atom_prior, params
    )
    anchor_global = torch.where(anchor_mask)[0]
    feat_anchor = feat_n.index_select(0, anchor_global)
    rgb_anchor = rgb_n.index_select(0, anchor_global)
    prior_anchor = atom_prior.index_select(0, anchor_global)

    tau = float(params["TAU_RECON"])
    sigma_color = float(params["SIGMA_COLOR_RECON"])
    lambda_color = float(params["LAMBDA_COLOR_RECON"])
    parts: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "local",
            "global",
            "score",
            "base_weight",
            "rpr_weight",
            "prior",
            "feature_base",
            "feature_rpr",
            "rgb_base",
            "rgb_rpr",
            "u_base",
            "u_rpr",
            "shift",
            "neff_base",
            "neff_rpr",
            "max_base",
            "max_rpr",
            "prior_min",
            "prior_mean",
            "low_q_ratio",
        )
    }

    for start in range(0, feat_n.shape[0], 512):
        end = min(start + 512, feat_n.shape[0])
        feat_chunk = feat_n[start:end]
        rgb_chunk = rgb_n[start:end]
        sim_feat = feat_chunk @ feat_anchor.t()
        color_dist2 = torch.cdist(rgb_chunk, rgb_anchor, p=2.0).square()
        similarity = sim_feat + lambda_color * torch.exp(-color_dist2 / sigma_color)

        top_score, top_local = torch.topk(similarity, k=k, dim=1)
        top_global = anchor_global.index_select(0, top_local.reshape(-1)).reshape_as(top_local)
        top_prior = prior_anchor[top_local]
        logits_base = top_score / tau
        base_weight = torch.softmax(logits_base, dim=1)
        logits_rpr = logits_base + torch.log(top_prior.clamp_min(RPR_Q_MIN))
        rpr_weight = torch.softmax(logits_rpr, dim=1)
        # A common positive prior is analytically cancelled by Softmax.  Reuse
        # the base row exactly in that case so the required scaling invariant
        # is not obscured by float32 addition/Softmax rounding.
        uniform_prior = (top_prior == top_prior[:, :1]).all(dim=1)
        if uniform_prior.any():
            rpr_weight[uniform_prior] = base_weight[uniform_prior]

        feat_top = feat_anchor[top_local]
        rgb_top = rgb_anchor[top_local]
        feature_base = torch_f.normalize(
            (base_weight.unsqueeze(-1) * feat_top).sum(dim=1), dim=1, p=2
        )
        feature_rpr = torch_f.normalize(
            (rpr_weight.unsqueeze(-1) * feat_top).sum(dim=1), dim=1, p=2
        )
        rgb_base = (base_weight.unsqueeze(-1) * rgb_top).sum(dim=1)
        rgb_rpr = (rpr_weight.unsqueeze(-1) * rgb_top).sum(dim=1)
        unreliability = 1.0 - top_prior

        parts["local"].append(top_local)
        parts["global"].append(top_global)
        parts["score"].append(top_score)
        parts["base_weight"].append(base_weight)
        parts["rpr_weight"].append(rpr_weight)
        parts["prior"].append(top_prior)
        parts["feature_base"].append(feature_base)
        parts["feature_rpr"].append(feature_rpr)
        parts["rgb_base"].append(rgb_base)
        parts["rgb_rpr"].append(rgb_rpr)
        parts["u_base"].append((base_weight * unreliability).sum(dim=1))
        parts["u_rpr"].append((rpr_weight * unreliability).sum(dim=1))
        parts["shift"].append(0.5 * (rpr_weight - base_weight).abs().sum(dim=1))
        parts["neff_base"].append(1.0 / base_weight.square().sum(dim=1))
        parts["neff_rpr"].append(1.0 / rpr_weight.square().sum(dim=1))
        parts["max_base"].append(base_weight.max(dim=1).values)
        parts["max_rpr"].append(rpr_weight.max(dim=1).values)
        parts["prior_min"].append(top_prior.min(dim=1).values)
        parts["prior_mean"].append(top_prior.mean(dim=1))
        parts["low_q_ratio"].append(
            (top_prior < RPR_LOW_Q_DIAGNOSTIC_THRESHOLD).float().mean(dim=1)
        )

    local = _cat(parts["local"], dtype=torch.long)
    global_index = _cat(parts["global"], dtype=torch.long)
    score = _cat(parts["score"])
    base_weight = _cat(parts["base_weight"])
    rpr_weight = _cat(parts["rpr_weight"])
    top_prior = _cat(parts["prior"])
    feature_base = _cat(parts["feature_base"])
    feature_rpr = _cat(parts["feature_rpr"])
    rgb_base = _cat(parts["rgb_base"])
    rgb_rpr = _cat(parts["rgb_rpr"])

    base_feature_residual = (1.0 - (feat_n * feature_base).sum(dim=1)).clamp_min(0.0)
    feature_residual = (1.0 - (feat_n * feature_rpr).sum(dim=1)).clamp_min(0.0)
    base_color_residual = torch.linalg.norm(rgb_n - rgb_base, dim=1)
    color_residual = torch.linalg.norm(rgb_n - rgb_rpr, dim=1)
    base_raw = (base_feature_residual + 0.2 * base_color_residual).float().contiguous()
    raw = (feature_residual + 0.2 * color_residual).float().contiguous()
    base_normalized = _minmax(base_raw).detach().cpu().float().contiguous()
    normalized = _minmax(raw).detach().cpu().float().contiguous()

    finite_tensors = (
        score,
        base_weight,
        rpr_weight,
        top_prior,
        feature_base,
        feature_rpr,
        rgb_base,
        rgb_rpr,
        base_raw,
        raw,
        base_normalized,
        normalized,
    )
    if not all(torch.isfinite(value).all() for value in finite_tensors):
        raise RuntimeError("RPR reconstruction produced NaN or Inf")
    for name, weight in (("base", base_weight), ("rpr", rpr_weight)):
        error = float((weight.sum(dim=1) - 1.0).abs().max())
        if error > 1e-6:
            raise RuntimeError(f"{name} reconstruction weights do not sum to one: {error}")
    u_base = _cat(parts["u_base"])
    u_rpr = _cat(parts["u_rpr"])
    # Extremely low and nearly equal q values can produce a few float32 U
    # inversions around 1e-7.  They are diagnostic-only: candidate weights and
    # residuals remain valid.  The caller records their count/max instead of
    # aborting a full cache run.

    return RPRReconstructionDetails(
        normalized_residual=normalized,
        raw_residual=raw,
        feature_residual=feature_residual.detach().cpu().float().contiguous(),
        color_residual=color_residual.detach().cpu().float().contiguous(),
        topk_anchor_local_index=local,
        topk_anchor_global_index=global_index,
        topk_score=score,
        base_weight=base_weight,
        rpr_weight=rpr_weight,
        topk_prior=top_prior,
        unreliable_mass_base=u_base,
        unreliable_mass_rpr=u_rpr,
        weight_l1_shift=_cat(parts["shift"]),
        effective_atoms_base=_cat(parts["neff_base"]),
        effective_atoms_rpr=_cat(parts["neff_rpr"]),
        max_weight_base=_cat(parts["max_base"]),
        max_weight_rpr=_cat(parts["max_rpr"]),
        topk_prior_min=_cat(parts["prior_min"]),
        topk_prior_mean=_cat(parts["prior_mean"]),
        topk_low_q_ratio=_cat(parts["low_q_ratio"]),
        base_normalized_residual=base_normalized,
        base_raw_residual=base_raw,
        base_feature_residual=base_feature_residual.detach().cpu().float().contiguous(),
        base_color_residual=base_color_residual.detach().cpu().float().contiguous(),
        reconstructed_feature_base=feature_base,
        reconstructed_feature_rpr=feature_rpr,
        reconstructed_rgb_base=rgb_base,
        reconstructed_rgb_rpr=rgb_rpr,
    )


def _reconstruct_from_fixed_topk(
    *,
    feat_n: torch.Tensor,
    rgb_n: torch.Tensor,
    anchor_mask: torch.Tensor,
    atom_prior: torch.Tensor,
    params: dict,
    base: RPRReconstructionDetails,
) -> RPRReconstructionDetails:
    """Apply a new prior to an already frozen B0 TopK retrieval.

    This is an exact computational reuse, not a new candidate: indices,
    scores and base weights are copied from the unit-prior B0 pass.  It avoids
    recomputing the same query/atom similarity matrix for P1 and P2.
    """
    feat_n, rgb_n, anchor_mask, atom_prior, k = _validate_inputs(
        feat_n, rgb_n, anchor_mask, atom_prior, params
    )
    expected = (1369, k)
    if tuple(base.topk_anchor_global_index.shape) != expected:
        raise ValueError("base TopK shape does not match current anchor/K contract")
    if tuple(base.topk_score.shape) != expected or tuple(base.base_weight.shape) != expected:
        raise ValueError("base score/weight shape mismatch")
    if not torch.isfinite(base.topk_score).all() or not torch.isfinite(base.base_weight).all():
        raise ValueError("base TopK details contain NaN or Inf")

    tau = float(params["TAU_RECON"])
    parts: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "rpr_weight",
            "prior",
            "feature_rpr",
            "rgb_rpr",
            "u_base",
            "u_rpr",
            "shift",
            "neff_rpr",
            "max_rpr",
            "prior_min",
            "prior_mean",
            "low_q_ratio",
        )
    }
    for start in range(0, 1369, 512):
        end = min(start + 512, 1369)
        global_index = base.topk_anchor_global_index[start:end]
        score = base.topk_score[start:end]
        base_weight = base.base_weight[start:end]
        top_prior = atom_prior[global_index]
        logits_rpr = score / tau + torch.log(top_prior.clamp_min(RPR_Q_MIN))
        rpr_weight = torch.softmax(logits_rpr, dim=1)
        uniform_prior = (top_prior == top_prior[:, :1]).all(dim=1)
        if uniform_prior.any():
            rpr_weight[uniform_prior] = base_weight[uniform_prior]
        feat_top = feat_n[global_index]
        rgb_top = rgb_n[global_index]
        feature_rpr = torch_f.normalize(
            (rpr_weight.unsqueeze(-1) * feat_top).sum(dim=1), dim=1, p=2
        )
        rgb_rpr = (rpr_weight.unsqueeze(-1) * rgb_top).sum(dim=1)
        unreliability = 1.0 - top_prior

        parts["rpr_weight"].append(rpr_weight)
        parts["prior"].append(top_prior)
        parts["feature_rpr"].append(feature_rpr)
        parts["rgb_rpr"].append(rgb_rpr)
        parts["u_base"].append((base_weight * unreliability).sum(dim=1))
        parts["u_rpr"].append((rpr_weight * unreliability).sum(dim=1))
        parts["shift"].append(0.5 * (rpr_weight - base_weight).abs().sum(dim=1))
        parts["neff_rpr"].append(1.0 / rpr_weight.square().sum(dim=1))
        parts["max_rpr"].append(rpr_weight.max(dim=1).values)
        parts["prior_min"].append(top_prior.min(dim=1).values)
        parts["prior_mean"].append(top_prior.mean(dim=1))
        parts["low_q_ratio"].append(
            (top_prior < RPR_LOW_Q_DIAGNOSTIC_THRESHOLD).float().mean(dim=1)
        )

    rpr_weight = _cat(parts["rpr_weight"])
    top_prior = _cat(parts["prior"])
    feature_rpr = _cat(parts["feature_rpr"])
    rgb_rpr = _cat(parts["rgb_rpr"])
    feature_residual = (1.0 - (feat_n * feature_rpr).sum(dim=1)).clamp_min(0.0)
    color_residual = torch.linalg.norm(rgb_n - rgb_rpr, dim=1)
    raw = (feature_residual + 0.2 * color_residual).float().contiguous()
    normalized = _minmax(raw).detach().cpu().float().contiguous()
    u_base, u_rpr = _cat(parts["u_base"]), _cat(parts["u_rpr"])
    if not all(
        torch.isfinite(value).all()
        for value in (rpr_weight, top_prior, feature_rpr, rgb_rpr, raw, normalized)
    ):
        raise RuntimeError("fixed-TopK RPR reconstruction produced NaN or Inf")
    if float((rpr_weight.sum(dim=1) - 1.0).abs().max()) > 1e-6:
        raise RuntimeError("fixed-TopK RPR weights do not sum to one")
    # See the public path above: rare float32-only U inversions are surfaced
    # through diagnostics and never change the RPR candidate response.

    return RPRReconstructionDetails(
        normalized_residual=normalized,
        raw_residual=raw,
        feature_residual=feature_residual.detach().cpu().float().contiguous(),
        color_residual=color_residual.detach().cpu().float().contiguous(),
        topk_anchor_local_index=base.topk_anchor_local_index,
        topk_anchor_global_index=base.topk_anchor_global_index,
        topk_score=base.topk_score,
        base_weight=base.base_weight,
        rpr_weight=rpr_weight,
        topk_prior=top_prior,
        unreliable_mass_base=u_base,
        unreliable_mass_rpr=u_rpr,
        weight_l1_shift=_cat(parts["shift"]),
        effective_atoms_base=base.effective_atoms_base,
        effective_atoms_rpr=_cat(parts["neff_rpr"]),
        max_weight_base=base.max_weight_base,
        max_weight_rpr=_cat(parts["max_rpr"]),
        topk_prior_min=_cat(parts["prior_min"]),
        topk_prior_mean=_cat(parts["prior_mean"]),
        topk_low_q_ratio=_cat(parts["low_q_ratio"]),
        base_normalized_residual=base.base_normalized_residual,
        base_raw_residual=base.base_raw_residual,
        base_feature_residual=base.base_feature_residual,
        base_color_residual=base.base_color_residual,
        reconstructed_feature_base=base.reconstructed_feature_base,
        reconstructed_feature_rpr=feature_rpr,
        reconstructed_rgb_base=base.reconstructed_rgb_base,
        reconstructed_rgb_rpr=rgb_rpr,
    )


def _map(value: torch.Tensor, grid: int = 37) -> torch.Tensor:
    return value.detach().cpu().float().reshape(1, grid, grid).contiguous()


def _probability(payload: dict, field: str) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be Tensor[1,37,37]")
    value = value.detach().cpu().float().contiguous()
    if not torch.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"{field} must be finite in [0,1]")
    return value


def _nonnegative(payload: dict, field: str) -> torch.Tensor:
    value = payload.get(field)
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 37, 37):
        raise ValueError(f"{field} must be Tensor[1,37,37]")
    value = value.detach().cpu().float().contiguous()
    if not torch.isfinite(value).all() or float(value.min()) < -1e-7:
        raise ValueError(f"{field} must be finite and nonnegative")
    return value


def _prior_stats(prior: torch.Tensor, target_atoms: torch.Tensor, prefix: str) -> dict:
    values = prior[target_atoms]
    modified = target_atoms & (prior < 1.0)
    modified_values = prior[modified]
    return {
        f"{prefix}_prior_modified_atom_count": int(modified.sum()),
        f"{prefix}_prior_target_atom_count": int(target_atoms.sum()),
        f"{prefix}_prior_min": float(values.min()) if values.numel() else 1.0,
        f"{prefix}_prior_mean_on_modified_atoms": (
            float(modified_values.mean()) if modified_values.numel() else 1.0
        ),
        f"{prefix}_prior_below_09_count": int((values < 0.9).sum()),
        f"{prefix}_prior_below_05_count": int((values < 0.5).sum()),
        f"{prefix}_prior_below_01_count": int((values < 0.1).sum()),
    }


def _query_stats(details: RPRReconstructionDetails, prefix: str) -> dict:
    exposure = details.unreliable_mass_base
    raw_delta = details.raw_residual - details.base_raw_residual
    return {
        f"{prefix}_query_exposure_mean": float(exposure.mean()),
        f"{prefix}_query_exposure_max": float(exposure.max()),
        f"{prefix}_query_exposure_nonzero_ratio": float((exposure > 0).float().mean()),
        f"{prefix}_u_reduction_mean": float(
            (details.unreliable_mass_base - details.unreliable_mass_rpr).mean()
        ),
        f"{prefix}_weight_shift_mean": float(details.weight_l1_shift.mean()),
        f"{prefix}_weight_shift_max": float(details.weight_l1_shift.max()),
        f"raw_delta_{prefix}_mean": float(raw_delta.mean()),
        f"raw_delta_{prefix}_std": float(raw_delta.std(unbiased=False)),
    }


def build_rpr_candidates(
    *,
    feature_37: torch.Tensor,
    image_path: str,
    cached_r1_37: torch.Tensor,
    cvbr_payload: dict,
    effective_params: dict,
) -> dict:
    """Build the frozen P1/P2 candidates without reading GT."""
    if not isinstance(cvbr_payload, dict):
        raise TypeError("cvbr_payload must be a dict")
    if cvbr_payload.get("cvbr_version") != "dabe_cvbr_v1":
        raise ValueError("cvbr_payload must use dabe_cvbr_v1")
    if cvbr_payload.get("source_augs") != ["identity"] or int(
        cvbr_payload.get("source_num_views", 0)
    ) != 1:
        raise ValueError("RPR-v1 requires identity single-view CVBR payloads")
    if int(effective_params.get("GRID", 0)) != 37:
        raise ValueError("RPR-v1 requires GRID=37")

    feature = _validate_feature(feature_37, 37)
    rgb_chw = _load_rgb_grid(image_path, 37)
    rgb_n = rgb_chw.permute(1, 2, 0).reshape(1369, 3).float()
    feat_n = torch_f.normalize(
        feature.permute(1, 2, 0).reshape(1369, 384), dim=1, p=2
    )
    cached = cached_r1_37.detach().cpu().float().contiguous()
    if tuple(cached.shape) != (1, 37, 37) or not torch.isfinite(cached).all():
        raise ValueError("cached_r1_37 must be finite Tensor[1,37,37]")

    cvbr_b0 = _probability(cvbr_payload, "b0_r1_bw2_37")
    _probability(cvbr_payload, "bc_b0_37")  # Protocol guard; never replaced by V1/V2 BC.
    anchor_b0 = _probability(cvbr_payload, "anchor_b0_37").reshape(-1).bool()
    ring1 = _probability(cvbr_payload, "border_ring1_37").reshape(-1).bool()
    ring2_only = _probability(cvbr_payload, "border_ring2_only_37").reshape(-1).bool()
    ring2_full = _probability(cvbr_payload, "border_ring2_full_37").reshape(-1).bool()
    q_v1 = _probability(cvbr_payload, "source_q_v1_37").reshape(-1)
    q_v2 = _probability(cvbr_payload, "source_q_v2_37").reshape(-1)
    if bool((ring1 & ring2_only).any()) or not torch.equal(ring1 | ring2_only, ring2_full):
        raise RuntimeError("invalid CVBR border-ring partition")
    cached_error = float((cvbr_b0 - cached).abs().max())
    if cached_error > 1e-6:
        raise RuntimeError(f"CVBR B0/cached R1 mismatch: {cached_error}")

    prior_unit = torch.ones(1369, dtype=torch.float32)
    prior_p1 = prior_unit.clone()
    prior_p2 = prior_unit.clone()
    target_p1 = anchor_b0 & ring2_only
    target_p2 = anchor_b0 & ring2_full
    prior_p1[target_p1] = q_v1[target_p1]
    prior_p2[target_p2] = q_v2[target_p2]

    unit = reconstruct_with_reliability_prior(
        feat_n=feat_n,
        rgb_n=rgb_n,
        anchor_mask=anchor_b0,
        atom_prior=prior_unit,
        params=effective_params,
    )
    p1 = _reconstruct_from_fixed_topk(
        feat_n=feat_n,
        rgb_n=rgb_n,
        anchor_mask=anchor_b0,
        atom_prior=prior_p1,
        params=effective_params,
        base=unit,
    )
    p2 = _reconstruct_from_fixed_topk(
        feat_n=feat_n,
        rgb_n=rgb_n,
        anchor_mask=anchor_b0,
        atom_prior=prior_p2,
        params=effective_params,
        base=unit,
    )

    p1_topk_mismatch = int(
        (p1.topk_anchor_global_index != unit.topk_anchor_global_index).sum()
    )
    p2_topk_mismatch = int(
        (p2.topk_anchor_global_index != unit.topk_anchor_global_index).sum()
    )
    topk_mismatch = p1_topk_mismatch + p2_topk_mismatch
    unit_weight_error = float((unit.rpr_weight - unit.base_weight).abs().max())
    unit_raw_self_error = float((unit.raw_residual - unit.base_raw_residual).abs().max())
    unit_normalized_self_error = float(
        (unit.normalized_residual - unit.base_normalized_residual).abs().max()
    )
    recomputed_error = float(
        (unit.base_normalized_residual.reshape_as(cached) - cached).abs().max()
    )
    raw_source = cvbr_payload.get("raw_b0_37")
    if raw_source is not None:
        raw_source = _nonnegative(cvbr_payload, "raw_b0_37")
        raw_source_error = float(
            (unit.base_raw_residual.reshape_as(raw_source) - raw_source).abs().max()
        )
    else:
        raw_source_error = unit_raw_self_error

    if topk_mismatch:
        raise RuntimeError(f"RPR changed B0 TopK indices: {topk_mismatch}")
    if unit_weight_error > 1e-7:
        raise RuntimeError(f"unit-prior weight regression failed: {unit_weight_error}")
    if unit_raw_self_error > 1e-6 or raw_source_error > 1e-6:
        raise RuntimeError(
            "unit-prior raw residual regression failed: "
            f"self={unit_raw_self_error}, source={raw_source_error}"
        )
    if unit_normalized_self_error > 1e-6 or recomputed_error > 1e-6:
        raise RuntimeError(
            "unit-prior normalized residual regression failed: "
            f"self={unit_normalized_self_error}, cached={recomputed_error}"
        )

    u_reduction_p1 = (p1.unreliable_mass_base - p1.unreliable_mass_rpr).clamp(0.0, 1.0)
    u_reduction_p2 = (p2.unreliable_mass_base - p2.unreliable_mass_rpr).clamp(0.0, 1.0)
    diagnostics = {
        "b0_cached_r1_max_abs": cached_error,
        "b0_recomputed_cached_r1_max_abs": recomputed_error,
        "unit_prior_topk_mismatch_count": topk_mismatch,
        "p1_topk_mismatch_count": p1_topk_mismatch,
        "p2_topk_mismatch_count": p2_topk_mismatch,
        "unit_prior_weight_max_abs": unit_weight_error,
        "unit_prior_raw_self_max_abs": unit_raw_self_error,
        "unit_prior_raw_max_abs": raw_source_error,
        "unit_prior_normalized_self_max_abs": unit_normalized_self_error,
        "unit_prior_normalized_max_abs": recomputed_error,
        "b0_anchor_count": int(anchor_b0.sum()),
        **_prior_stats(prior_p1, target_p1, "p1"),
        **_prior_stats(prior_p2, target_p2, "p2"),
        **_query_stats(p1, "p1"),
        **_query_stats(p2, "p2"),
        "effective_atoms_base_mean": float(unit.effective_atoms_base.mean()),
        "effective_atoms_p1_mean": float(p1.effective_atoms_rpr.mean()),
        "effective_atoms_p2_mean": float(p2.effective_atoms_rpr.mean()),
        "max_weight_base_mean": float(unit.max_weight_base.mean()),
        "max_weight_p1_mean": float(p1.max_weight_rpr.mean()),
        "max_weight_p2_mean": float(p2.max_weight_rpr.mean()),
        "p1_area_gt_05": float((p1.normalized_residual > 0.5).float().mean()),
        "p2_area_gt_05": float((p2.normalized_residual > 0.5).float().mean()),
        "cross_fallback_count_from_source": int(
            cvbr_payload.get("diagnostics", {}).get("cross_fallback_count", 0)
        ),
        "p1_u_monotonicity_violation_count": int(
            ((p1.unreliable_mass_rpr - p1.unreliable_mass_base) > 1e-7).sum()
        ),
        "p1_u_monotonicity_max_violation": float(
            (p1.unreliable_mass_rpr - p1.unreliable_mass_base).clamp_min(0).max()
        ),
        "p2_u_monotonicity_violation_count": int(
            ((p2.unreliable_mass_rpr - p2.unreliable_mass_base) > 1e-7).sum()
        ),
        "p2_u_monotonicity_max_violation": float(
            (p2.unreliable_mass_rpr - p2.unreliable_mass_base).clamp_min(0).max()
        ),
    }

    result = {
        "rpr_version": DABE_RPR_VERSION,
        "p1_rpr_secondring_37": _map(p1.normalized_residual),
        "p2_rpr_allborder_37": _map(p2.normalized_residual),
        "raw_b0_recomputed_37": _map(unit.base_raw_residual),
        "raw_p1_37": _map(p1.raw_residual),
        "raw_p2_37": _map(p2.raw_residual),
        "atom_prior_p1_37": _map(prior_p1),
        "atom_prior_p2_37": _map(prior_p2),
        "u_base_p1_37": _map(p1.unreliable_mass_base),
        "u_rpr_p1_37": _map(p1.unreliable_mass_rpr),
        "u_reduction_p1_37": _map(u_reduction_p1),
        "u_base_p2_37": _map(p2.unreliable_mass_base),
        "u_rpr_p2_37": _map(p2.unreliable_mass_rpr),
        "u_reduction_p2_37": _map(u_reduction_p2),
        "weight_l1_shift_p1_37": _map(p1.weight_l1_shift),
        "weight_l1_shift_p2_37": _map(p2.weight_l1_shift),
        "effective_atoms_base_37": _map(unit.effective_atoms_base),
        "effective_atoms_p1_37": _map(p1.effective_atoms_rpr),
        "effective_atoms_p2_37": _map(p2.effective_atoms_rpr),
        "max_weight_base_37": _map(unit.max_weight_base),
        "max_weight_p1_37": _map(p1.max_weight_rpr),
        "max_weight_p2_37": _map(p2.max_weight_rpr),
        "topk_prior_min_p1_37": _map(p1.topk_prior_min),
        "topk_prior_mean_p1_37": _map(p1.topk_prior_mean),
        "topk_low_q_ratio_p1_37": _map(p1.topk_low_q_ratio),
        "topk_prior_min_p2_37": _map(p2.topk_prior_min),
        "topk_prior_mean_p2_37": _map(p2.topk_prior_mean),
        "topk_low_q_ratio_p2_37": _map(p2.topk_low_q_ratio),
        "delta_raw_p1_vs_b0_37": _map(p1.raw_residual - unit.base_raw_residual),
        "delta_raw_p2_vs_b0_37": _map(p2.raw_residual - unit.base_raw_residual),
        "diagnostics": diagnostics,
    }
    return result


def build_rpr_p1_candidate(
    *,
    feature_37: torch.Tensor,
    image_path: str,
    cached_r1_37: torch.Tensor,
    cvbr_payload: dict,
    effective_params: dict,
) -> dict:
    """Build only P1-RPR using the frozen B0 Top-K and q_v1 reweighting."""

    if not isinstance(cvbr_payload, dict):
        raise TypeError("cvbr_payload must be a dict")
    if cvbr_payload.get("cvbr_version") != "dabe_cvbr_v1":
        raise ValueError("cvbr_payload must use dabe_cvbr_v1")
    if cvbr_payload.get("source_augs") != ["identity"] or int(
        cvbr_payload.get("source_num_views", 0)
    ) != 1:
        raise ValueError("RPR-v1 requires identity single-view CVBR payloads")
    if int(effective_params.get("GRID", 0)) != 37:
        raise ValueError("RPR-v1 requires GRID=37")

    feature = _validate_feature(feature_37, 37)
    rgb_chw = _load_rgb_grid(image_path, 37)
    rgb_n = rgb_chw.permute(1, 2, 0).reshape(1369, 3).float()
    feat_n = torch_f.normalize(
        feature.permute(1, 2, 0).reshape(1369, 384), dim=1, p=2
    )
    cached = cached_r1_37.detach().cpu().float().contiguous()
    if tuple(cached.shape) != (1, 37, 37) or not torch.isfinite(cached).all():
        raise ValueError("cached_r1_37 must be finite Tensor[1,37,37]")

    cvbr_b0 = _probability(cvbr_payload, "b0_r1_bw2_37")
    _probability(cvbr_payload, "bc_b0_37")
    anchor_b0 = _probability(cvbr_payload, "anchor_b0_37").reshape(-1).bool()
    ring1 = _probability(cvbr_payload, "border_ring1_37").reshape(-1).bool()
    ring2_only = _probability(
        cvbr_payload, "border_ring2_only_37"
    ).reshape(-1).bool()
    ring2_full = _probability(
        cvbr_payload, "border_ring2_full_37"
    ).reshape(-1).bool()
    q_v1 = _probability(cvbr_payload, "source_q_v1_37").reshape(-1)
    if bool((ring1 & ring2_only).any()) or not torch.equal(
        ring1 | ring2_only, ring2_full
    ):
        raise RuntimeError("invalid CVBR border-ring partition")
    cached_error = float((cvbr_b0 - cached).abs().max())
    if cached_error > 1e-6:
        raise RuntimeError(f"CVBR B0/cached R1 mismatch: {cached_error}")

    prior_unit = torch.ones(1369, dtype=torch.float32)
    prior_p1 = prior_unit.clone()
    target_p1 = anchor_b0 & ring2_only
    prior_p1[target_p1] = q_v1[target_p1]

    unit = reconstruct_with_reliability_prior(
        feat_n=feat_n,
        rgb_n=rgb_n,
        anchor_mask=anchor_b0,
        atom_prior=prior_unit,
        params=effective_params,
    )
    p1 = _reconstruct_from_fixed_topk(
        feat_n=feat_n,
        rgb_n=rgb_n,
        anchor_mask=anchor_b0,
        atom_prior=prior_p1,
        params=effective_params,
        base=unit,
    )

    p1_topk_mismatch = int(
        (p1.topk_anchor_global_index != unit.topk_anchor_global_index).sum()
    )
    unit_weight_error = float((unit.rpr_weight - unit.base_weight).abs().max())
    unit_raw_error = float(
        (unit.raw_residual - unit.base_raw_residual).abs().max()
    )
    unit_normalized_error = float(
        (unit.normalized_residual - unit.base_normalized_residual).abs().max()
    )
    recomputed_error = float(
        (unit.base_normalized_residual.reshape_as(cached) - cached).abs().max()
    )
    if p1_topk_mismatch:
        raise RuntimeError(f"RPR changed B0 TopK indices: {p1_topk_mismatch}")
    if unit_weight_error > 1e-7:
        raise RuntimeError(f"unit-prior weight regression failed: {unit_weight_error}")
    if unit_raw_error > 1e-6:
        raise RuntimeError(f"unit-prior raw regression failed: {unit_raw_error}")
    if unit_normalized_error > 1e-6 or recomputed_error > 1e-6:
        raise RuntimeError(
            "unit-prior normalized regression failed: "
            f"self={unit_normalized_error}, cached={recomputed_error}"
        )

    u_reduction = (
        p1.unreliable_mass_base - p1.unreliable_mass_rpr
    ).clamp(0.0, 1.0)
    difference = p1.unreliable_mass_rpr - p1.unreliable_mass_base
    diagnostics = {
        "generation_mode": "rpr_p1_only",
        "b0_cached_r1_max_abs": cached_error,
        "b0_recomputed_cached_r1_max_abs": recomputed_error,
        "unit_prior_topk_mismatch_count": p1_topk_mismatch,
        "p1_topk_mismatch_count": p1_topk_mismatch,
        "unit_prior_weight_max_abs": unit_weight_error,
        "unit_prior_raw_max_abs": unit_raw_error,
        "unit_prior_normalized_max_abs": recomputed_error,
        "b0_anchor_count": int(anchor_b0.sum()),
        **_prior_stats(prior_p1, target_p1, "p1"),
        **_query_stats(p1, "p1"),
        "effective_atoms_base_mean": float(unit.effective_atoms_base.mean()),
        "effective_atoms_p1_mean": float(p1.effective_atoms_rpr.mean()),
        "max_weight_base_mean": float(unit.max_weight_base.mean()),
        "max_weight_p1_mean": float(p1.max_weight_rpr.mean()),
        "p1_area_gt_05": float((p1.normalized_residual > 0.5).float().mean()),
        "cross_fallback_count_from_source": int(
            cvbr_payload.get("diagnostics", {}).get("cross_fallback_count", 0)
        ),
        "p1_u_monotonicity_violation_count": int((difference > 1e-7).sum()),
        "p1_u_monotonicity_max_violation": float(
            difference.clamp_min(0).max()
        ),
    }
    return {
        "rpr_version": DABE_RPR_VERSION,
        "p1_rpr_secondring_37": _map(p1.normalized_residual),
        "u_reduction_p1_37": _map(u_reduction),
        "diagnostics": diagnostics,
    }
