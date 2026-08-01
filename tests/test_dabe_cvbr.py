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

from common.cache_dabe_cvbr import build_cvbr_cache
from common.dabe_background_null import reconstruct_from_background_atoms
from common.dabe_cvbr import (
    CVBR_Q_MIN,
    background_connectivity_with_source_reliability,
    border_ring_masks,
    build_cvbr_candidates,
    cross_reconstruct_boundary,
    cvbr_reliability,
)
from common.dabe_pseudo import (
    DABE_V2_DEFAULT_PARAMS,
    _background_anchor,
    _background_connectivity,
    _background_residual,
    _build_local_graph,
    _load_rgb_grid,
    _sobel_magnitude,
    _validate_feature,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MAIN_ROOT.parent
torch.set_num_threads(1)


@contextmanager
def cth_temporary_directory():
    root = BASELINE_ROOT / "workdir" / "pytest_cvbr_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _synthetic(seed=11):
    generator = torch.Generator().manual_seed(seed)
    feature = torch.rand((384, 37, 37), generator=generator)
    feat_n = torch_f.normalize(feature.permute(1, 2, 0).reshape(1369, 384), dim=1)
    rgb_chw = torch.rand((3, 37, 37), generator=generator)
    rgb_n = rgb_chw.permute(1, 2, 0).reshape(1369, 3)
    return feature, feat_n, rgb_chw, rgb_n


def test_border_masks_exact_partition():
    ring1, ring2_only, ring2_full = border_ring_masks(37)
    assert int(ring1.sum()) == 144
    assert int(ring2_only.sum()) == 136
    assert int(ring2_full.sum()) == 280
    assert not bool((ring1 & ring2_only).any())
    assert torch.equal(ring1 | ring2_only, ring2_full)
    assert torch.all(~ring1 | ring2_full)


def test_b0_graph_bc_anchor_residual_and_weighted_dijkstra_regression():
    _, feat_n, rgb_chw, rgb_n = _synthetic()
    params = dict(DABE_V2_DEFAULT_PARAMS); params["BORDER_WIDTH"] = 2
    edge = _sobel_magnitude(rgb_chw).reshape(-1)
    idx, weight = _build_local_graph(feat_n, rgb_n, edge, 37, params)
    current_bc, border = _background_connectivity(idx, weight, 37, params)
    weighted = background_connectivity_with_source_reliability(
        idx, weight, border, torch.ones(1369), 37, params["TAU_BC"]
    )
    assert torch.allclose(weighted, current_bc, atol=1e-6)
    anchor = _background_anchor(current_bc, border, params)
    current_r1 = _background_residual(feat_n, rgb_n, anchor, params)
    details = reconstruct_from_background_atoms(
        feat_n, rgb_n, anchor, params, exclude_chebyshev_radius=None
    )
    assert torch.allclose(details.normalized_residual, current_r1, atol=1e-6)


def test_weighted_dijkstra_low_q_source_loses_local_dominance():
    grid, count = 2, 4
    idx = torch.zeros((count, 8), dtype=torch.long)
    weight = torch.zeros((count, 8), dtype=torch.float32)
    edge_weight = float(torch.exp(torch.tensor(-0.1)))
    for left, right in ((0, 1), (1, 2), (2, 3)):
        slot_left = int((weight[left] > 0).sum()); slot_right = int((weight[right] > 0).sum())
        idx[left, slot_left] = right; weight[left, slot_left] = edge_weight
        idx[right, slot_right] = left; weight[right, slot_right] = edge_weight
    source = torch.tensor([True, False, False, True])
    unit = background_connectivity_with_source_reliability(
        idx, weight, source, torch.ones(count), grid, .3
    )
    q = torch.tensor([1.0, 0.0, 0.0, 0.1])
    downweighted = background_connectivity_with_source_reliability(
        idx, weight, source, q, grid, .3
    )
    q_at_frozen_minimum = torch.tensor([1.0, 0.0, 0.0, CVBR_Q_MIN])
    minimum = background_connectivity_with_source_reliability(
        idx, weight, source, q_at_frozen_minimum, grid, .3
    )
    assert float(unit[3]) == pytest.approx(1.0)
    assert float(downweighted[3]) < float(downweighted[0])
    assert float(downweighted[2]) > float(downweighted[3])
    assert torch.isfinite(minimum).all()


def test_cross_reconstruction_excludes_radius_and_fixed_fallback():
    generator = torch.Generator().manual_seed(5)
    grid, count = 7, 49
    feat = torch_f.normalize(torch.rand((count, 12), generator=generator), dim=1)
    rgb = torch.rand((count, 3), generator=generator)
    params = {**DABE_V2_DEFAULT_PARAMS, "GRID": grid, "K_RECON": 8}
    anchor = torch.zeros(count, dtype=torch.bool)
    anchor[[24, 0, 6, 42, 48]] = True
    boundary = torch.ones(count, dtype=torch.bool)
    details = cross_reconstruct_boundary(feat, rgb, anchor, boundary, params)
    center = 24
    positive = details.reconstruction.topk_weight[center] > 0
    selected = details.reconstruction.topk_anchor_global_index[center][positive]
    cy, cx = divmod(center, grid)
    for value in selected.tolist():
        y, x = divmod(value, grid)
        assert max(abs(y - cy), abs(x - cx)) > 1
    assert torch.allclose(details.reconstruction.topk_weight[center].sum(), torch.tensor(1.0), atol=1e-6)
    only_local = torch.zeros(count, dtype=torch.bool); only_local[center] = True
    fallback = cross_reconstruct_boundary(feat, rgb, only_local, boundary, params)
    assert bool(fallback.fallback_mask[center])
    assert int(fallback.reconstruction.valid_anchor_count[center]) == 0
    assert torch.allclose(fallback.reconstruction.topk_weight[center].sum(), torch.tensor(1.0), atol=1e-6)


def test_robust_median_mad_and_reliability_rules():
    values = torch.tensor([1.0, 1.0, 1.0, 2.0, 100.0])
    mask = torch.ones(5, dtype=torch.bool)
    q, stats = cvbr_reliability(values, mask)
    assert stats["reference_median"] == pytest.approx(1.0)
    assert stats["reference_mad"] == pytest.approx(0.0)
    assert stats["reference_scale"] == pytest.approx(1e-6)
    assert torch.all(q[:3] == 1)
    assert float(q[3]) == pytest.approx(CVBR_Q_MIN)
    constant, constant_stats = cvbr_reliability(torch.ones(8), torch.ones(8, dtype=torch.bool))
    assert constant_stats["reference_scale"] == pytest.approx(1e-6)
    assert torch.equal(constant, torch.ones_like(constant))
    regular = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])
    q_regular, regular_stats = cvbr_reliability(regular, torch.ones(5, dtype=torch.bool))
    assert regular_stats["reference_median"] == pytest.approx(1.5)
    assert regular_stats["reference_mad"] == pytest.approx(0.5)
    assert torch.all(q_regular[:3] == 1)
    assert torch.all(q_regular[3:] < 1)
    assert float(q_regular.min()) >= CVBR_Q_MIN and float(q_regular.max()) <= 1


def _prepare_case(case: Path):
    generator = torch.Generator().manual_seed(31)
    image_path = case / "image.png"
    Image.fromarray(
        (torch.rand((296, 296, 3), generator=generator) * 255).byte().numpy(), mode="RGB"
    ).save(image_path)
    feature = torch.rand((384, 37, 37), generator=generator)
    params = dict(DABE_V2_DEFAULT_PARAMS); params["BORDER_WIDTH"] = 2
    validated = _validate_feature(feature, 37)
    rgb_chw = _load_rgb_grid(image_path, 37)
    feat_n = torch_f.normalize(validated.permute(1, 2, 0).reshape(1369, 384), dim=1)
    rgb_n = rgb_chw.permute(1, 2, 0).reshape(1369, 3)
    edge = _sobel_magnitude(rgb_chw).reshape(-1)
    idx, weight = _build_local_graph(feat_n, rgb_n, edge, 37, params)
    bc, border = _background_connectivity(idx, weight, 37, params)
    anchor = _background_anchor(bc, border, params)
    r1 = _background_residual(feat_n, rgb_n, anchor, params).reshape(1, 37, 37)
    dabe_root = case / "dabe"; dabe_path = dabe_root / "CHAMELEON" / "synthetic.pt"
    dabe_path.parent.mkdir(parents=True)
    torch.save({
        "dataset":"CHAMELEON", "stem":"synthetic", "image_path":str(image_path.resolve()),
        "gt_path":str((case/"must_not_be_read.png").resolve()), "dabe_version":"v2",
        "augs":["identity"], "num_views":1, "residual_pass1_37":r1,
    }, dabe_path)
    dabe_manifest = dabe_root / "manifest_test.jsonl"
    drow = {
        "dataset":"CHAMELEON", "stem":"synthetic", "cache_path":str(dabe_path.resolve()),
        "image_path":str(image_path.resolve()), "gt_path":str((case/"must_not_be_read.png").resolve()),
    }
    _write_jsonl(dabe_manifest, [drow])
    cache_root = case / "datasets" / "cache"
    feature_root = cache_root / "features_cache" / "dinov1-s8"
    feature_path = feature_root / "test" / "CHAMELEON" / "synthetic.pt"
    feature_path.parent.mkdir(parents=True)
    torch.save({
        "dataset":"CHAMELEON", "stem":"synthetic", "image_path":str(image_path.resolve()),
        "tensor":feature,
    }, feature_path)
    feature_manifest = feature_root / "manifest_test.jsonl"
    frow = {
        "dataset":"CHAMELEON", "stem":"synthetic", "cache_path":str(feature_path.resolve()),
        "image_path":str(image_path.resolve()),
    }
    _write_jsonl(feature_manifest, [frow])
    config = case / "config.py"
    config.write_text(
        f"BACKBONE_KEY='dinov1-s8'\nCACHE_ROOT={str(cache_root)!r}\nDINO={{'feature_input_size':296}}\n",
        encoding="utf-8",
    )
    return locals()


def test_build_candidates_v1_v2_reliability_and_prototype_contract():
    with cth_temporary_directory() as case:
        p = _prepare_case(case)
        result = build_cvbr_candidates(
            feature_37=p["feature"], image_path=str(p["image_path"]),
            cached_r1_37=p["r1"], effective_params=p["params"],
        )
        diagnostics = result["diagnostics"]
        assert diagnostics["b0_cached_r1_max_abs"] <= 1e-6
        assert diagnostics["weighted_dijkstra_unit_reliability_max_abs"] <= 1e-6
        assert diagnostics["ring1_count"] == 144 and diagnostics["ring2_only_count"] == 136
        ring1, ring2_only, ring2_full = border_ring_masks()
        q1, q2 = result["source_q_v1_37"].reshape(-1), result["source_q_v2_37"].reshape(-1)
        assert torch.all(q1[ring1] == 1)
        assert torch.equal(q1[ring2_only], q2[ring2_only])
        assert torch.all(q1[~ring2_full] == 0) and torch.all(q2[~ring2_full] == 0)
        assert torch.equal(result["b0_r1_bw2_37"], p["r1"])
        for field in ("b0_r1_bw2_37","b1_r1_bw1_37","v1_cvbr_second_ring_37","v2_cvbr_all_border_37"):
            assert tuple(result[field].shape) == (1,37,37)
            assert torch.isfinite(result[field]).all()


def test_cache_io_payload_source_readonly_and_generator_never_reads_gt():
    with cth_temporary_directory() as case:
        p = _prepare_case(case)
        source = [p["dabe_path"], p["dabe_manifest"], p["feature_path"], p["feature_manifest"]]
        before = {path:path.read_bytes() for path in source}
        out = case / "out"
        protocol = build_cvbr_cache(p["config"], p["dabe_root"], out, max_samples=1)
        assert protocol["num_samples"] == 1
        assert protocol["baseline_recompute_max_abs"] <= 1e-6
        assert protocol["gt_used_for_generation"] is False
        for path, data in before.items(): assert path.read_bytes() == data
        row = json.loads((out/"manifest_test.jsonl").read_text().strip())
        payload = torch.load(row["cache_path"], map_location="cpu", weights_only=False)
        assert payload["cvbr_version"] == "dabe_cvbr_v1"
        assert payload["v2_cvbr_all_border_37"].shape == (1,37,37)


@pytest.mark.parametrize("mutation", ["missing","duplicate","nonidentity","bad_feature","bad_r1","backbone"])
def test_cache_rejections(mutation):
    with cth_temporary_directory() as case:
        p = _prepare_case(case)
        if mutation == "missing": _write_jsonl(p["feature_manifest"], [])
        elif mutation == "duplicate": _write_jsonl(p["dabe_manifest"], [p["drow"],p["drow"]])
        elif mutation in {"nonidentity","bad_r1"}:
            value = torch.load(p["dabe_path"], map_location="cpu", weights_only=False)
            if mutation == "nonidentity": value["augs"] = ["hflip"]
            else: value["residual_pass1_37"] = torch.zeros((1,37,37))
            torch.save(value, p["dabe_path"])
        elif mutation == "bad_feature":
            value = torch.load(p["feature_path"], map_location="cpu", weights_only=False)
            value["tensor"] = torch.zeros((384,36,37)); torch.save(value,p["feature_path"])
        else:
            p["config"].write_text("BACKBONE_KEY='dinov2-b14'\nDINO={'feature_input_size':296}\n",encoding="utf-8")
        with pytest.raises((ValueError,RuntimeError)):
            build_cvbr_cache(p["config"], p["dabe_root"], case/"out", max_samples=1)
