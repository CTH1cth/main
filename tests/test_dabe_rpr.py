from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn.functional as torch_f
from PIL import Image

from common.cache_dabe_r1_design import _manifest_map, effective_dabe_v2_params
from common.cache_dabe_rpr import _validate_cvbr_protocol, build_rpr_cache
from common.dabe_cvbr import border_ring_masks
from common.dabe_pseudo import DABE_V2_DEFAULT_PARAMS, _load_rgb_grid
from common.dabe_rpr import (
    DABE_RPR_VERSION,
    PROBABILITY_FIELDS,
    _reconstruct_from_fixed_topk,
    build_rpr_candidates,
    reconstruct_with_reliability_prior,
)
from common.utils import load_config, read_jsonl, torch_load


MAIN_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MAIN_ROOT.parent
CONFIG = MAIN_ROOT / "configs" / "dinov1_s8_dabev2_dagp_uncgate_ndr_lrfloor_2e5.py"
DABE_ROOT = BASELINE_ROOT / "workdir" / "dabe_v2_direct_test_cache_identity" / "dinov1-s8"
CVBR_ROOT = BASELINE_ROOT / "workdir" / "dabe_cvbr_v1_identity" / "dinov1-s8"
torch.set_num_threads(1)


@contextmanager
def cth_temporary_directory():
    root = BASELINE_ROOT / "workdir" / "pytest_rpr_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _params(k=4):
    return {**DABE_V2_DEFAULT_PARAMS, "GRID": 37, "K_RECON": k}


def _inputs(seed=7, k=4):
    generator = torch.Generator().manual_seed(seed)
    feat = torch_f.normalize(torch.rand((1369, 384), generator=generator), dim=1)
    rgb = torch.rand((1369, 3), generator=generator)
    anchor = torch.zeros(1369, dtype=torch.bool)
    anchor[torch.tensor([0, 36, 1332, 1368, 684, 650])] = True
    prior = torch.ones(1369)
    return feat, rgb, anchor, prior, _params(k)


def test_unit_prior_exact_reconstruction_regression_and_input_immutability():
    feat, rgb, anchor, prior, params = _inputs()
    copies = tuple(value.clone() for value in (feat, rgb, anchor, prior))
    result = reconstruct_with_reliability_prior(
        feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=prior, params=params
    )
    assert torch.equal(result.topk_anchor_global_index, result.topk_anchor_global_index)
    assert torch.equal(result.base_weight, result.rpr_weight)
    assert torch.equal(result.reconstructed_feature_base, result.reconstructed_feature_rpr)
    assert torch.equal(result.reconstructed_rgb_base, result.reconstructed_rgb_rpr)
    assert torch.equal(result.base_raw_residual, result.raw_residual)
    assert torch.equal(result.base_normalized_residual, result.normalized_residual)
    assert torch.equal(result.weight_l1_shift, torch.zeros_like(result.weight_l1_shift))
    assert torch.equal(result.unreliable_mass_base, torch.zeros_like(result.unreliable_mass_base))
    assert torch.allclose(result.base_weight.sum(1), torch.ones(1369), atol=1e-7)
    for value, copy in zip((feat, rgb, anchor, prior), copies):
        assert torch.equal(value, copy)


def test_post_topk_prior_does_not_change_selection_and_downweights_low_q_atom():
    feat, rgb, anchor, prior, params = _inputs(k=6)
    unit = reconstruct_with_reliability_prior(
        feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=prior, params=params
    )
    prior[anchor] = torch.tensor([1.0, 0.1, 0.7, 0.3, 1.0, 0.2])
    rpr = reconstruct_with_reliability_prior(
        feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=prior, params=params
    )
    assert torch.equal(unit.topk_anchor_global_index, rpr.topk_anchor_global_index)
    assert torch.equal(unit.topk_score, rpr.topk_score)
    query = 0
    selected_prior = rpr.topk_prior[query]
    low = int(torch.argmin(selected_prior))
    same_score = torch.isclose(rpr.topk_score[query], rpr.topk_score[query, low])
    if int(same_score.sum()) > 1:
        peer = int(torch.where(same_score & (selected_prior > selected_prior[low]))[0][0])
        assert rpr.rpr_weight[query, peer] > rpr.rpr_weight[query, low]
    assert float((rpr.rpr_weight - rpr.base_weight).abs().max()) > 0


def test_equal_scores_low_reliability_atom_is_downweighted():
    feat = torch_f.normalize(torch.ones((1369, 384)), dim=1)
    rgb = torch.zeros((1369, 3))
    anchor = torch.zeros(1369, dtype=torch.bool)
    anchor[:2] = True
    prior = torch.ones(1369)
    prior[1] = 0.1
    result = reconstruct_with_reliability_prior(
        feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=prior, params=_params(k=2)
    )
    assert torch.allclose(result.base_weight[:, 0], result.base_weight[:, 1])
    q = result.topk_prior[0]
    high, low = int(torch.argmax(q)), int(torch.argmin(q))
    assert result.rpr_weight[0, high] > result.rpr_weight[0, low]


@pytest.mark.parametrize("constant", [0.7, 0.01, 1e-6])
def test_constant_prior_scaling_invariant_and_all_low_set_is_finite(constant):
    feat, rgb, anchor, prior, params = _inputs()
    prior[anchor] = constant
    result = reconstruct_with_reliability_prior(
        feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=prior, params=params
    )
    assert torch.allclose(result.base_weight, result.rpr_weight, atol=1e-7)
    assert torch.isfinite(result.raw_residual).all()
    assert torch.allclose(result.rpr_weight.sum(1), torch.ones(1369), atol=1e-6)


def test_unreliable_mass_shift_and_effective_atom_contracts():
    feat, rgb, anchor, prior, params = _inputs(k=6)
    prior[anchor] = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.1, 1e-6])
    result = reconstruct_with_reliability_prior(
        feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=prior, params=params
    )
    expected = (result.base_weight * (1 - result.topk_prior)).sum(1)
    assert torch.allclose(result.unreliable_mass_base, expected)
    assert torch.all(result.unreliable_mass_rpr <= result.unreliable_mass_base + 1e-7)
    assert float(result.weight_l1_shift.min()) >= 0
    assert float(result.weight_l1_shift.max()) <= 1
    assert float(result.effective_atoms_base.min()) >= 1
    assert float(result.effective_atoms_rpr.min()) >= 1
    assert float(result.effective_atoms_base.max()) <= 6 + 1e-5
    assert float(result.effective_atoms_rpr.max()) <= 6 + 1e-5


def test_fixed_topk_reuse_is_bit_exact_to_independent_retrieval():
    feat, rgb, anchor, unit_prior, params = _inputs(k=6)
    prior = unit_prior.clone()
    prior[anchor] = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.1, 1e-6])
    unit = reconstruct_with_reliability_prior(
        feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=unit_prior, params=params
    )
    independent = reconstruct_with_reliability_prior(
        feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=prior, params=params
    )
    reused = _reconstruct_from_fixed_topk(
        feat_n=feat,
        rgb_n=rgb,
        anchor_mask=anchor,
        atom_prior=prior,
        params=params,
        base=unit,
    )
    for field in (
        "normalized_residual",
        "raw_residual",
        "feature_residual",
        "color_residual",
        "topk_anchor_global_index",
        "topk_score",
        "base_weight",
        "rpr_weight",
        "topk_prior",
        "unreliable_mass_base",
        "unreliable_mass_rpr",
        "weight_l1_shift",
        "effective_atoms_rpr",
        "max_weight_rpr",
        "reconstructed_feature_rpr",
        "reconstructed_rgb_rpr",
    ):
        assert torch.equal(getattr(independent, field), getattr(reused, field)), field


@pytest.mark.parametrize(
    "mutation,exception",
    [
        ("non_tensor", TypeError),
        ("feature_shape", ValueError),
        ("rgb_shape", ValueError),
        ("nan_feature", ValueError),
        ("inf_prior", ValueError),
        ("low_prior", ValueError),
        ("high_prior", ValueError),
        ("empty_anchor", ValueError),
    ],
)
def test_input_validation(mutation, exception):
    feat, rgb, anchor, prior, params = _inputs()
    if mutation == "non_tensor":
        feat = []
    elif mutation == "feature_shape":
        feat = feat[:-1]
    elif mutation == "rgb_shape":
        rgb = rgb[:, :2]
    elif mutation == "nan_feature":
        feat[0, 0] = float("nan")
    elif mutation == "inf_prior":
        prior[0] = float("inf")
    elif mutation == "low_prior":
        prior[anchor][0] = 0.0
        prior[torch.where(anchor)[0][0]] = 0.0
    elif mutation == "high_prior":
        prior[torch.where(anchor)[0][0]] = 1.01
    else:
        anchor[:] = False
    with pytest.raises(exception):
        reconstruct_with_reliability_prior(
            feat_n=feat, rgb_n=rgb, anchor_mask=anchor, atom_prior=prior, params=params
        )


def _synthetic_candidate_case(case: Path):
    generator = torch.Generator().manual_seed(41)
    image_path = case / "image.png"
    Image.fromarray(
        (torch.rand((37, 37, 3), generator=generator) * 255).byte().numpy(), mode="RGB"
    ).save(image_path)
    feature = torch.rand((384, 37, 37), generator=generator)
    feat_n = torch_f.normalize(feature.permute(1, 2, 0).reshape(1369, 384), dim=1)
    rgb_chw = _load_rgb_grid(image_path, 37)
    rgb_n = rgb_chw.permute(1, 2, 0).reshape(1369, 3)
    ring1, ring2_only, ring2_full = border_ring_masks(37)
    anchor = ring2_full.clone()
    anchor[684] = True
    params = _params(k=8)
    base = reconstruct_with_reliability_prior(
        feat_n=feat_n,
        rgb_n=rgb_n,
        anchor_mask=anchor,
        atom_prior=torch.ones(1369),
        params=params,
    )
    q1 = torch.zeros(1369)
    q1[ring1] = 1
    q1[ring2_only] = 0.2
    q2 = torch.zeros(1369)
    q2[ring2_full] = 0.1
    cvbr = {
        "dataset": "CHAMELEON",
        "stem": "synthetic",
        "cvbr_version": "dabe_cvbr_v1",
        "source_augs": ["identity"],
        "source_num_views": 1,
        "b0_r1_bw2_37": base.base_normalized_residual.reshape(1, 37, 37),
        "raw_b0_37": base.base_raw_residual.reshape(1, 37, 37),
        "bc_b0_37": torch.ones((1, 37, 37)),
        "anchor_b0_37": anchor.float().reshape(1, 37, 37),
        "border_ring1_37": ring1.float().reshape(1, 37, 37),
        "border_ring2_only_37": ring2_only.float().reshape(1, 37, 37),
        "border_ring2_full_37": ring2_full.float().reshape(1, 37, 37),
        "source_q_v1_37": q1.reshape(1, 37, 37),
        "source_q_v2_37": q2.reshape(1, 37, 37),
        "diagnostics": {"cross_fallback_count": 0},
    }
    return image_path, feature, base, cvbr, params, ring1, ring2_only, ring2_full, anchor


def test_atom_prior_construction_p1_p2_and_payload_contract():
    with cth_temporary_directory() as case:
        image_path, feature, base, cvbr, params, ring1, ring2_only, ring2_full, anchor = _synthetic_candidate_case(case)
        output = build_rpr_candidates(
            feature_37=feature,
            image_path=str(image_path),
            cached_r1_37=base.base_normalized_residual.reshape(1, 37, 37),
            cvbr_payload=cvbr,
            effective_params=params,
        )
        p1, p2 = output["atom_prior_p1_37"].reshape(-1), output["atom_prior_p2_37"].reshape(-1)
        assert torch.all(p1[anchor & ring2_only] == 0.2)
        assert torch.all(p1[anchor & ring1] == 1)
        assert p1[684] == 1
        assert torch.all(p2[anchor & ring2_full] == 0.1)
        assert p2[684] == 1
        for field in PROBABILITY_FIELDS:
            value = output[field]
            assert tuple(value.shape) == (1, 37, 37)
            assert value.dtype == torch.float32 and value.device.type == "cpu"
            assert torch.isfinite(value).all() and 0 <= float(value.min()) <= float(value.max()) <= 1
        diagnostics = output["diagnostics"]
        assert diagnostics["unit_prior_topk_mismatch_count"] == 0
        assert diagnostics["unit_prior_weight_max_abs"] <= 1e-7
        assert diagnostics["unit_prior_raw_max_abs"] <= 1e-6
        assert diagnostics["unit_prior_normalized_max_abs"] <= 1e-6
        assert output["rpr_version"] == DABE_RPR_VERSION


def test_manifest_duplicate_and_missing_cache_are_rejected():
    with cth_temporary_directory() as case:
        cache = case / "x.pt"
        torch.save({}, cache)
        row = {"dataset": "D", "stem": "x", "cache_path": str(cache)}
        manifest = case / "manifest.jsonl"
        manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError):
            _manifest_map(manifest)
        row["stem"] = "missing"
        row["cache_path"] = str(case / "missing.pt")
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            _manifest_map(manifest)


def test_actual_cvbr_protocol_is_frozen_and_matches_dabe_manifest():
    protocol = _validate_cvbr_protocol(
        CVBR_ROOT / "protocol.json", DABE_ROOT / "manifest_test.jsonl"
    )
    assert protocol["cvbr_version"] == "dabe_cvbr_v1"
    assert protocol["source_augs"] == ["identity"]
    assert protocol["num_samples"] == 6473


def test_cache_one_sample_contract_source_readonly_and_no_gt_access():
    with cth_temporary_directory() as case:
        watched = (
            DABE_ROOT / "manifest_test.jsonl",
            CVBR_ROOT / "manifest_test.jsonl",
            CVBR_ROOT / "protocol.json",
        )
        before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in watched}
        out = case / "rpr"
        protocol = build_rpr_cache(
            CONFIG,
            DABE_ROOT,
            CVBR_ROOT,
            out,
            max_samples=1,
            workers=1,
            torch_threads=1,
        )
        assert protocol["num_samples"] == 1
        assert protocol["gt_used_for_generation"] is False
        assert protocol["touch_metadata_used_for_generation"] is False
        assert protocol["unit_prior_topk_mismatch_count"] == 0
        assert protocol["unit_prior_weight_max_abs"] <= 1e-7
        assert protocol["unit_prior_raw_max_abs"] <= 1e-6
        assert protocol["unit_prior_normalized_max_abs"] <= 1e-6
        assert {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in watched} == before
        row = read_jsonl(out / "manifest_test.jsonl")[0]
        payload = torch_load(row["cache_path"], map_location="cpu")
        assert payload["rpr_version"] == DABE_RPR_VERSION
        assert payload["source_augs"] == ["identity"]
        assert payload["p1_rpr_secondring_37"].shape == (1, 37, 37)
        source_text = (MAIN_ROOT / "common" / "cache_dabe_rpr.py").read_text(encoding="utf-8")
        assert "gt_path" not in source_text


def test_effective_config_has_frozen_rpr_reconstruction_parameters():
    cfg = load_config(CONFIG)
    params = effective_dabe_v2_params(cfg)
    assert params["K_RECON"] == 32
    assert params["LAMBDA_COLOR_RECON"] == pytest.approx(0.2)
    assert params["SIGMA_COLOR_RECON"] == pytest.approx(0.05)
    assert params["TAU_RECON"] == pytest.approx(0.07)
