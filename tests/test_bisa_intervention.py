from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from analysis.bisa.intervention import (
    apply_group_interventions,
    assemble_group_responses,
    build_dabe_background_substitutes,
    coordinate_group_ids,
)
from analysis.bisa.metrics import (
    classify_teacher_errors,
    downsample_gt_occupancy,
    strict_gt_labels,
)
from analysis.bisa.statistics import (
    grouped_logistic_cv,
    image_bootstrap_class_difference,
    image_bootstrap_mode_auc_difference,
    image_bootstrap_oof,
)


def test_coordinate_group_coverage_and_spacing() -> None:
    for group_size in (3, 4):
        group_ids = coordinate_group_ids(37, 37, group_size)
        assert group_ids.shape == (37, 37)
        assert set(torch.unique(group_ids).tolist()) == set(range(group_size**2))
        for group in range(group_size**2):
            coordinates = torch.nonzero(group_ids == group)
            for coordinate in coordinates[:20]:
                distance = (coordinates - coordinate).abs().sum(dim=1)
                nonzero = distance[distance > 0]
                assert nonzero.numel() == 0 or int(nonzero.min()) >= group_size


def test_each_token_is_replaced_exactly_once() -> None:
    feature = torch.zeros(2, 37, 37)
    substitute = torch.ones_like(feature)
    group_ids = coordinate_group_ids(37, 37, 4)
    counterfactual = apply_group_interventions(feature, substitute, group_ids)
    assert counterfactual.shape == (16, 2, 37, 37)
    replacement_count = (counterfactual[:, 0] == 1).sum(dim=0)
    assert torch.equal(replacement_count, torch.ones_like(replacement_count))
    for group in range(16):
        assert torch.equal(counterfactual[group, 0] == 1, group_ids == group)
    assert not counterfactual.requires_grad


def test_identity_response_is_exactly_zero() -> None:
    generator = torch.Generator().manual_seed(7)
    feature = torch.randn(5, 37, 37, generator=generator)
    group_ids = coordinate_group_ids(37, 37, 4)
    counterfactual = apply_group_interventions(feature, feature, group_ids)
    original_logits = feature.sum(dim=0)
    counterfactual_logits = counterfactual.sum(dim=1)
    responses = assemble_group_responses(original_logits, counterfactual_logits, group_ids)
    for name in ("c_self_raw", "c_local_raw", "c_self_centered", "c_local_centered"):
        assert torch.count_nonzero(responses[name]) == 0


def test_centered_and_local_responses_match_analytic_values() -> None:
    group_ids = coordinate_group_ids(3, 3, 3)
    original = torch.zeros(3, 3)
    differences = torch.zeros(9, 3, 3)
    for group in range(9):
        differences[group].fill_(float(group))
        y, x = torch.nonzero(group_ids == group)[0]
        differences[group, y, x] += 9.0
    counterfactual = original.unsqueeze(0) - differences
    responses = assemble_group_responses(original, counterfactual, group_ids)
    assert torch.allclose(responses["group_median"], torch.arange(9.0))
    assert torch.allclose(
        responses["c_self_centered"], torch.full((3, 3), 9.0)
    )
    # At the centre, the 3x3 mean contains the single +9 impulse.
    assert torch.isclose(responses["c_local_centered"][1, 1], torch.tensor(1.0))
    # At a corner, the boundary-aware 2x2 denominator gives 9 / 4.
    assert torch.isclose(responses["c_local_centered"][0, 0], torch.tensor(2.25))


def test_dabe_substitutes_are_detached_and_random_is_reproducible() -> None:
    generator = torch.Generator().manual_seed(17)
    feature = torch.randn(6, 7, 7, generator=generator)
    rgb = torch.rand(3, 7, 7, generator=generator)
    anchor = torch.zeros(7, 7)
    anchor[0] = 1
    anchor[-1] = 1
    params = {
        "K_RECON": 5,
        "TAU_RECON": 0.07,
        "SIGMA_COLOR_RECON": 0.05,
        "LAMBDA_COLOR_RECON": 0.2,
    }
    modes = ("weighted_bg", "nearest_bg", "random_bg", "identity")
    first = build_dabe_background_substitutes(
        feature, rgb, anchor, params, modes, seed=123, sample_key="TR-CAMO/a"
    )
    second = build_dabe_background_substitutes(
        feature, rgb, anchor, params, modes, seed=123, sample_key="TR-CAMO/a"
    )
    matched = build_dabe_background_substitutes(
        feature,
        rgb,
        anchor,
        params,
        modes,
        seed=123,
        sample_key="TR-CAMO/a",
        norm_mode="query_match",
    )
    assert torch.equal(first.substitutes["random_bg"], second.substitutes["random_bg"])
    assert torch.equal(first.substitutes["identity"], feature)
    anchor_indices = set(torch.nonzero(anchor.reshape(-1) > 0.5).reshape(-1).tolist())
    assert set(first.diagnostics["nearest_anchor_indices"].tolist()).issubset(anchor_indices)
    assert set(first.diagnostics["random_anchor_indices"].tolist()).issubset(anchor_indices)
    for mode in ("weighted_bg", "nearest_bg", "random_bg"):
        flattened = first.substitutes[mode].permute(1, 2, 0).reshape(-1, 6)
        assert torch.allclose(
            torch.linalg.vector_norm(flattened, dim=1),
            torch.ones(49),
            atol=1e-5,
        )
        matched_flattened = (
            matched.substitutes[mode].permute(1, 2, 0).reshape(-1, 6)
        )
        original_flattened = feature.permute(1, 2, 0).reshape(-1, 6)
        assert torch.allclose(
            torch.linalg.vector_norm(matched_flattened, dim=1),
            torch.linalg.vector_norm(original_flattened, dim=1),
            atol=1e-5,
            rtol=1e-5,
        )
    for value in first.substitutes.values():
        assert value.shape == feature.shape
        assert torch.isfinite(value).all()
        assert not value.requires_grad


def test_gt_pooling_strict_labels_and_error_encoding() -> None:
    gt = torch.zeros(74, 74)
    gt[:37, :37] = 1.0
    occupancy = downsample_gt_occupancy(gt, 37, 37)
    assert occupancy.shape == (37, 37)
    assert float(occupancy[0, 0]) == 1.0
    assert float(occupancy[-1, -1]) == 0.0
    labels = strict_gt_labels(torch.tensor([[0.0, 0.5, 1.0]]))
    assert labels.tolist() == [[0, -1, 1]]
    probability = torch.tensor([[0.1, 0.9, 0.1, 0.9]])
    gt_labels = torch.tensor([[0, 0, 1, 1]], dtype=torch.int8)
    assert classify_teacher_errors(probability, gt_labels).tolist() == [[0, 1, 2, 3]]


def _classification_frame() -> pd.DataFrame:
    rows = []
    for image_index in range(10):
        for error_type, label in (("FP", 0), ("TP", 1), ("TN", 0), ("FN", 1)):
            for token_index in range(4):
                signal = float(label) + 0.01 * image_index + 0.001 * token_index
                rows.append(
                    {
                        "image_id": f"image-{image_index}",
                        "teacher_error_type": error_type,
                        "teacher_logit": signal,
                        "teacher_entropy": 0.6 - 0.1 * signal,
                        "dabe_target_soft": 0.2 + 0.6 * signal,
                        "dabe_background_residual": 0.1 + 0.2 * signal,
                        "dabe_background_connectivity": 0.8 - 0.3 * signal,
                        "bg_feature_l2_distance": 0.4 + 0.2 * signal,
                        "c_self_centered": signal,
                        "c_local_centered": signal + 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_groupkfold_is_image_grouped_and_bootstrap_is_deterministic() -> None:
    frame = _classification_frame()
    result, oof = grouped_logistic_cv(frame, "tp_fp", seed=9)
    assert result["status"] == "ok"
    assert sorted(oof["fold"].unique().tolist()) == [0, 1, 2, 3, 4]
    assert int(oof.groupby("image_id")["fold"].nunique().max()) == 1
    first = image_bootstrap_oof(oof, seed=11, replicates=20)
    second = image_bootstrap_oof(oof, seed=11, replicates=20)
    assert first == second
    class_ci = image_bootstrap_class_difference(
        frame, "tp_fp", "c_local_centered", seed=12, replicates=20
    )
    assert class_ci["ci_low"] > 0.0
    weighted = frame.assign(substitution_mode="weighted_bg")
    random = frame.assign(
        substitution_mode="random_bg",
        c_local_centered=-frame["c_local_centered"],
    )
    mode_ci = image_bootstrap_mode_auc_difference(
        pd.concat([weighted, random], ignore_index=True),
        "tp_fp",
        "c_local_centered",
        seed=13,
        replicates=20,
    )
    assert mode_ci["estimate"] > 0.0
