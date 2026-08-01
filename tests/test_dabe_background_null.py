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

from common.cache_dabe_background_null import build_background_null_cache
from common.dabe_background_null import (
    make_anchor_cross_error_map,
    reconstruct_from_background_atoms,
    retrieval_control_from_regular_top1,
    weighted_local_null_score,
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
from common.dabe_rank_calibration import rank_transport


MAIN_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MAIN_ROOT.parent


@contextmanager
def cth_temporary_directory():
    root = BASELINE_ROOT / "workdir" / "pytest_bgnull_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def small_inputs(grid=5, dimension=8, k=4):
    generator = torch.Generator().manual_seed(17)
    feature = torch_f.normalize(
        torch.rand((grid * grid, dimension), generator=generator), dim=1
    )
    rgb = torch.rand((grid * grid, 3), generator=generator)
    anchor = torch.zeros(grid * grid, dtype=torch.bool)
    anchor[::2] = True
    params = {**DABE_V2_DEFAULT_PARAMS, "GRID": grid, "K_RECON": k}
    return feature, rgb, anchor, params


def test_regular_reconstruction_contract_alignment_and_determinism():
    feature, rgb, anchor, params = small_inputs()
    feature_copy, rgb_copy, anchor_copy = feature.clone(), rgb.clone(), anchor.clone()
    first = reconstruct_from_background_atoms(
        feature, rgb, anchor, params, exclude_chebyshev_radius=None
    )
    second = reconstruct_from_background_atoms(
        feature, rgb, anchor, params, exclude_chebyshev_radius=None
    )
    n, k = 25, 4
    assert first.raw_residual.shape == (n,)
    assert first.normalized_residual.shape == (n,)
    assert first.topk_weight.shape == (n, k)
    assert first.reconstructed_feature.shape == (n, 8)
    assert first.reconstructed_rgb.shape == (n, 3)
    assert float(first.raw_residual.min()) >= 0
    assert float(first.normalized_residual.min()) >= 0
    assert float(first.normalized_residual.max()) <= 1
    assert torch.allclose(first.topk_weight.sum(1), torch.ones(n), atol=1e-6)
    assert torch.allclose(
        torch.linalg.norm(first.reconstructed_feature, dim=1), torch.ones(n), atol=1e-6
    )
    assert torch.allclose(first.normalized_residual, _background_residual(feature, rgb, anchor, params), atol=1e-6)
    assert torch.equal(first.raw_residual, second.raw_residual)
    assert torch.equal(feature, feature_copy)
    assert torch.equal(rgb, rgb_copy)
    assert torch.equal(anchor, anchor_copy)


def test_cross_exclusion_masked_topk_and_weight_renormalization():
    grid = 5
    feature, rgb, _, params = small_inputs(grid=grid, k=25)
    anchor = torch.ones(grid * grid, dtype=torch.bool)
    result = reconstruct_from_background_atoms(
        feature, rgb, anchor, params, exclude_chebyshev_radius=1
    )
    queries = torch.arange(grid * grid)
    query_y, query_x = queries // grid, queries % grid
    chosen = result.topk_anchor_global_index
    chosen_y, chosen_x = chosen // grid, chosen % grid
    distances = torch.maximum(
        (chosen_y - query_y[:, None]).abs(), (chosen_x - query_x[:, None]).abs()
    )
    positive = result.topk_weight > 0
    assert torch.all(distances[positive] > 1)
    assert torch.all(result.topk_weight[~positive] == 0)
    assert torch.allclose(result.topk_weight.sum(1), torch.ones(grid * grid), atol=1e-6)
    assert not result.fallback_mask.any()
    assert result.local_excluded_weight_mass is not None


def test_cross_no_valid_anchor_uses_explicit_fallback_without_radius_change():
    grid = 5
    feature, rgb, _, params = small_inputs(grid=grid, k=9)
    anchor = torch.zeros(grid * grid, dtype=torch.bool)
    for y in range(1, 4):
        for x in range(1, 4):
            anchor[y * grid + x] = True
    regular = reconstruct_from_background_atoms(
        feature, rgb, anchor, params, exclude_chebyshev_radius=None
    )
    cross = reconstruct_from_background_atoms(
        feature, rgb, anchor, params, exclude_chebyshev_radius=1
    )
    center = 2 * grid + 2
    assert cross.valid_anchor_count[center] == 0
    assert cross.fallback_mask[center]
    assert torch.equal(
        cross.topk_anchor_global_index[center], regular.topk_anchor_global_index[center]
    )
    assert torch.allclose(cross.topk_weight[center], regular.topk_weight[center])


def test_anchor_cross_error_map_preserves_raw_values_only_on_anchors():
    raw = torch.tensor([0.0, 0.2, 0.0, 0.7, 0.4])
    anchor = torch.tensor([True, False, True, False, True])
    errors, error_map = make_anchor_cross_error_map(raw, anchor)
    assert errors.tolist() == pytest.approx([0.0, 0.0, 0.4])
    assert error_map.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.4])
    assert torch.equal(error_map[~anchor], torch.zeros(2))


@pytest.mark.parametrize(
    "errors,expected",
    [
        ([0.1, 0.2], 1.0),
        ([0.8, 0.9], 0.0),
        ([0.5, 0.5], 0.5),
        ([0.2, 0.5], 0.65),
    ],
)
def test_local_null_basic_cases(errors, expected):
    query = torch.tensor([0.5])
    weights = torch.tensor([[0.3, 0.7]])
    output = weighted_local_null_score(query, torch.tensor([errors]), weights)
    assert float(output) == pytest.approx(expected)
    permutation = torch.tensor([1, 0])
    permuted = weighted_local_null_score(
        query, torch.tensor([errors])[:, permutation], weights[:, permutation]
    )
    assert torch.equal(output, permuted)
    assert 0 <= float(output) <= 1


def test_local_null_rejects_unnormalized_weights():
    with pytest.raises(ValueError):
        weighted_local_null_score(
            torch.tensor([0.5]), torch.tensor([[0.1, 0.2]]), torch.tensor([[0.2, 0.2]])
        )


def test_n2_rank_transport_distribution_order_area_and_constant_fallback():
    r1 = torch.tensor([[[0.1, 0.8], [0.3, 0.6]]])
    local = torch.tensor([[[0.9, 0.2], [0.7, 0.4]]])
    n2 = rank_transport(local, r1)
    assert torch.equal(torch.argsort(n2.reshape(-1)), torch.argsort(local.reshape(-1)))
    assert torch.allclose(torch.sort(n2.reshape(-1)).values, torch.sort(r1.reshape(-1)).values)
    assert int((n2 > 0.5).sum()) == int((r1 > 0.5).sum())
    assert torch.equal(rank_transport(torch.ones_like(local), r1), r1)


def test_retrieval_control_uses_regular_combined_similarity_top1_and_formula():
    feature, rgb, anchor, params = small_inputs(k=4)
    regular = reconstruct_from_background_atoms(
        feature, rgb, anchor, params, exclude_chebyshev_radius=None
    )
    raw, normalized = retrieval_control_from_regular_top1(feature, rgb, regular)
    top1 = regular.topk_anchor_global_index[:, 0]
    expected = (
        (1 - (feature * feature[top1]).sum(1)).clamp_min(0)
        + 0.2 * torch.linalg.norm(rgb - rgb[top1], dim=1)
    )
    assert torch.allclose(raw, expected)
    assert float(normalized.min()) == 0
    assert float(normalized.max()) == pytest.approx(1.0, abs=1e-6)
    constant_feature = torch_f.normalize(torch.ones_like(feature), dim=1)
    constant_rgb = torch.zeros_like(rgb)
    constant_details = reconstruct_from_background_atoms(
        constant_feature, constant_rgb, anchor, params, exclude_chebyshev_radius=None
    )
    _, constant_normalized = retrieval_control_from_regular_top1(
        constant_feature, constant_rgb, constant_details
    )
    assert torch.equal(constant_normalized, torch.zeros_like(constant_normalized))


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _prepare_cache_case(case: Path):
    generator = torch.Generator().manual_seed(29)
    image_path = case / "image.png"
    image = (torch.rand((37, 37, 3), generator=generator) * 255).byte().numpy()
    Image.fromarray(image, mode="RGB").save(image_path)
    feature = torch.rand((384, 37, 37), generator=generator)
    params = dict(DABE_V2_DEFAULT_PARAMS)
    validated = _validate_feature(feature, 37)
    rgb = _load_rgb_grid(image_path, 37)
    feat_n = torch_f.normalize(validated.permute(1, 2, 0).reshape(37 * 37, -1), dim=1)
    rgb_n = rgb.permute(1, 2, 0).reshape(37 * 37, 3)
    edge = _sobel_magnitude(rgb).reshape(-1)
    neigh_idx, neigh_weight = _build_local_graph(feat_n, rgb_n, edge, 37, params)
    bc, border = _background_connectivity(neigh_idx, neigh_weight, 37, params)
    anchor = _background_anchor(bc, border, params)
    r1 = _background_residual(feat_n, rgb_n, anchor, params).reshape(1, 37, 37)

    dabe_root = case / "dabe"
    dabe_path = dabe_root / "CHAMELEON" / "synthetic.pt"
    dabe_path.parent.mkdir(parents=True)
    dabe_payload = {
        "dataset": "CHAMELEON",
        "stem": "synthetic",
        "dabe_version": "v2",
        "augs": ["identity"],
        "num_views": 1,
        "residual_pass1_37": r1,
        "bg_anchor_37": anchor.float().reshape(1, 37, 37),
    }
    torch.save(dabe_payload, dabe_path)
    dabe_manifest = dabe_root / "manifest_test.jsonl"
    dabe_row = {
        "dataset": "CHAMELEON",
        "stem": "synthetic",
        "cache_path": str(dabe_path.resolve()),
        "image_path": str(image_path.resolve()),
    }
    _write_jsonl(dabe_manifest, [dabe_row])

    cache_root = case / "datasets" / "cache"
    feature_root = cache_root / "features_cache" / "dinov1-s8"
    feature_path = feature_root / "test" / "CHAMELEON" / "synthetic.pt"
    feature_path.parent.mkdir(parents=True)
    torch.save(
        {
            "dataset": "CHAMELEON",
            "stem": "synthetic",
            "image_path": str(image_path.resolve()),
            "tensor": feature,
        },
        feature_path,
    )
    feature_manifest = feature_root / "manifest_test.jsonl"
    feature_row = {
        "dataset": "CHAMELEON",
        "stem": "synthetic",
        "cache_path": str(feature_path.resolve()),
        "image_path": str(image_path.resolve()),
    }
    _write_jsonl(feature_manifest, [feature_row])
    config_path = case / "config.py"
    config_path.write_text(
        f"BACKBONE_KEY = 'dinov1-s8'\nCACHE_ROOT = {str(cache_root)!r}\n",
        encoding="utf-8",
    )
    return {
        "config": config_path,
        "dabe_root": dabe_root,
        "dabe_path": dabe_path,
        "dabe_manifest": dabe_manifest,
        "feature_path": feature_path,
        "feature_manifest": feature_manifest,
    }


def test_cache_io_contract_join_and_source_unchanged():
    with cth_temporary_directory() as case:
        paths = _prepare_cache_case(case)
        source_bytes = {
            name: paths[name].read_bytes()
            for name in ("dabe_path", "dabe_manifest", "feature_path", "feature_manifest")
        }
        output = case / "derived"
        protocol = build_background_null_cache(
            paths["config"], paths["dabe_root"], output, max_samples=1
        )
        assert protocol["num_samples"] == 1
        assert protocol["r1_recompute_global_max_abs"] <= 1e-6
        assert protocol["gt_used_for_generation"] is False
        for name, contents in source_bytes.items():
            assert paths[name].read_bytes() == contents
        rows = [json.loads(line) for line in (output / "manifest_test.jsonl").read_text().splitlines()]
        assert [(row["dataset"], row["stem"]) for row in rows] == [("CHAMELEON", "synthetic")]
        payload = torch.load(rows[0]["cache_path"], map_location="cpu", weights_only=False)
        assert payload["bgnull_version"] == "dabe_bgnull_v1"
        for field in (
            "n0_r1_37",
            "n1_cross_r1_37",
            "n2_local_null_raw_37",
            "n2_local_null_r1dist_37",
            "rc_nn_minmax_37",
        ):
            assert payload[field].shape == (1, 37, 37)
            assert payload[field].dtype == torch.float32
            assert torch.isfinite(payload[field]).all()


def test_cache_rejects_nonidentity_nonv2_bad_feature_and_bad_r1():
    mutations = ("nonidentity", "nonv2", "bad_feature", "bad_r1")
    for mutation in mutations:
        with cth_temporary_directory() as case:
            paths = _prepare_cache_case(case)
            if mutation in {"nonidentity", "nonv2", "bad_r1"}:
                payload = torch.load(paths["dabe_path"], map_location="cpu", weights_only=False)
                if mutation == "nonidentity":
                    payload["augs"] = ["hflip"]
                elif mutation == "nonv2":
                    payload["dabe_version"] = "v1"
                else:
                    payload["residual_pass1_37"] = torch.zeros(1, 37, 37)
                torch.save(payload, paths["dabe_path"])
            else:
                payload = torch.load(paths["feature_path"], map_location="cpu", weights_only=False)
                payload["tensor"] = torch.zeros(384, 36, 37)
                torch.save(payload, paths["feature_path"])
            with pytest.raises((ValueError, RuntimeError)):
                build_background_null_cache(
                    paths["config"], paths["dabe_root"], case / "derived", max_samples=1
                )


def test_cache_rejects_missing_and_duplicate_manifest_keys():
    with cth_temporary_directory() as case:
        paths = _prepare_cache_case(case)
        row = json.loads(paths["feature_manifest"].read_text().strip())
        row["stem"] = "different"
        _write_jsonl(paths["feature_manifest"], [row])
        with pytest.raises(RuntimeError):
            build_background_null_cache(
                paths["config"], paths["dabe_root"], case / "derived", max_samples=1
            )
    with cth_temporary_directory() as case:
        paths = _prepare_cache_case(case)
        row = json.loads(paths["dabe_manifest"].read_text().strip())
        _write_jsonl(paths["dabe_manifest"], [row, row])
        with pytest.raises(RuntimeError):
            build_background_null_cache(
                paths["config"], paths["dabe_root"], case / "derived", max_samples=1
            )
