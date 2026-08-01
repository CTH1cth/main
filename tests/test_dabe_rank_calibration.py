from __future__ import annotations

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from common.cache_dabe_rank_calibration import build_rank_calibration_cache
from common.dabe_rank_calibration import (
    average_percentile_rank,
    build_f_minmax,
    build_f_rank_to_r1,
    build_median_rank_to_r1,
    empirical_quantile,
    minmax_per_image,
    rank_transport,
)


MAIN_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MAIN_ROOT.parent
CONFIG_PATH = MAIN_ROOT / "configs" / "dinov1_s8_dabev2_dagp_uncgate_ndr_lrfloor_2e5.py"


@contextmanager
def cth_temporary_directory():
    root = BASELINE_ROOT / "workdir" / "pytest_rankcal_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_minmax_normal_constant_and_input_unchanged():
    value = torch.tensor([[[2.0, 4.0], [3.0, 8.0]]])
    original = value.clone()
    output = minmax_per_image(value)
    assert float(output.min()) == 0.0
    assert float(output.max()) == pytest.approx(1.0, abs=1e-6)
    assert torch.isfinite(output).all()
    assert torch.equal(value, original)
    assert torch.equal(minmax_per_image(torch.full((1, 2, 2), 0.3)), torch.zeros(1, 2, 2))


def test_average_percentile_rank_increasing_decreasing_ties_constant_and_deterministic():
    increasing = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
    assert torch.allclose(average_percentile_rank(increasing), expected)
    assert torch.allclose(average_percentile_rank(increasing.flip(0)), expected.flip(0))
    tied = average_percentile_rank(torch.tensor([1.0, 1.0, 3.0, 5.0]))
    assert tied[0] == tied[1] == pytest.approx(1 / 6)
    assert tied.tolist() == pytest.approx([1 / 6, 1 / 6, 2 / 3, 1.0])
    constant = average_percentile_rank(torch.ones(8))
    assert torch.equal(constant, torch.full((8,), 0.5))
    assert float(tied.min()) >= 0 and float(tied.max()) <= 1
    assert torch.equal(tied, average_percentile_rank(torch.tensor([1.0, 1.0, 3.0, 5.0])))


def test_empirical_quantile_endpoints_interpolation_and_constant():
    reference = torch.tensor([10.0, 0.0])
    output = empirical_quantile(reference, torch.tensor([0.0, 0.25, 0.5, 1.0]))
    assert output.tolist() == pytest.approx([0.0, 2.5, 5.0, 10.0])
    constant = empirical_quantile(torch.full((5,), 0.4), torch.tensor([-1.0, 0.5, 2.0]))
    assert constant.tolist() == pytest.approx([0.4, 0.4, 0.4])
    assert float(output.min()) >= float(reference.min())
    assert float(output.max()) <= float(reference.max())


def test_rank_transport_contract_and_fallbacks():
    with pytest.raises(ValueError):
        rank_transport(torch.zeros(2), torch.zeros(3))
    reference = torch.tensor([0.2, 0.8, 0.5, 0.1])
    assert torch.equal(rank_transport(torch.ones(4), reference), reference)
    constant_reference = torch.full((4,), 0.25)
    assert torch.equal(rank_transport(torch.arange(4.0), constant_reference), constant_reference)

    source = torch.tensor([0.6, 0.1, 0.9, 0.2])
    source_copy, reference_copy = source.clone(), reference.clone()
    output = rank_transport(source, reference)
    assert torch.allclose(torch.sort(output).values, torch.sort(reference).values)
    assert torch.equal(torch.argsort(source), torch.argsort(output))
    assert torch.equal(source, source_copy) and torch.equal(reference, reference_copy)
    assert torch.isfinite(output).all() and float(output.min()) >= 0 and float(output.max()) <= 1

    ties = rank_transport(torch.tensor([0.0, 0.0, 1.0, 2.0]), reference)
    assert ties[0] == ties[1]


def test_c1_spatial_order_from_f_and_value_distribution_from_r1():
    foreground = torch.tensor([[[0.9, 0.1], [0.4, 0.7]]])
    r1 = torch.tensor([[[0.05, 0.2], [0.6, 0.95]]])
    c1 = build_f_rank_to_r1(foreground, r1)
    assert torch.equal(torch.argsort(c1.reshape(-1)), torch.argsort(foreground.reshape(-1)))
    assert torch.allclose(torch.sort(c1.reshape(-1)).values, torch.sort(r1.reshape(-1)).values)


def test_c2_median_rank_invariance_identity_and_constant_fallback():
    r1 = torch.tensor([[[0.1, 0.7], [0.4, 0.9]]])
    residual = torch.tensor([[[0.2, 0.8], [0.3, 0.6]]])
    foreground = torch.tensor([[[0.6, 0.1], [0.9, 0.4]]])
    c2 = build_median_rank_to_r1(r1, residual, foreground)
    ranks = torch.stack(
        [average_percentile_rank(x) for x in (r1, residual, foreground)], dim=0
    )
    expected = empirical_quantile(r1, torch.median(ranks, dim=0).values)
    assert torch.allclose(c2, expected)

    transformed = build_median_rank_to_r1(
        r1.square(), residual.square(), foreground * 0.5 + 0.1
    )
    assert torch.equal(torch.argsort(c2.reshape(-1)), torch.argsort(transformed.reshape(-1)))
    assert torch.allclose(build_median_rank_to_r1(r1, r1, r1), r1)
    constants = build_median_rank_to_r1(
        torch.full((1, 2, 2), 0.3),
        torch.full((1, 2, 2), 0.4),
        torch.full((1, 2, 2), 0.5),
    )
    assert torch.equal(constants, torch.full((1, 2, 2), 0.3))
    assert torch.equal(c2, build_median_rank_to_r1(r1, residual, foreground))


def test_build_f_minmax_output_contract():
    output = build_f_minmax(torch.rand(1, 37, 37))
    assert output.shape == (1, 37, 37)
    assert output.dtype == torch.float32
    assert not output.requires_grad and output.is_contiguous()


def _synthetic_payload(**updates):
    generator = torch.Generator().manual_seed(7)
    payload = {
        "dataset": "CHAMELEON",
        "stem": "synthetic",
        "dabe_version": "v2",
        "augs": ["identity"],
        "num_views": 1,
        "residual_pass1_37": torch.rand((1, 37, 37), generator=generator),
        "residual_37": torch.rand((1, 37, 37), generator=generator),
        "fg_score_37": torch.rand((1, 37, 37), generator=generator),
    }
    payload.update(updates)
    return payload


def _prepare_source(case: Path, payload: dict):
    source_root = case / "source"
    source_root.mkdir()
    cache_path = source_root / "CHAMELEON" / "synthetic.pt"
    cache_path.parent.mkdir()
    torch.save(payload, cache_path)
    manifest = source_root / "manifest_test.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "CHAMELEON",
                "stem": "synthetic",
                "cache_path": str(cache_path.resolve()),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return source_root, cache_path, manifest


def test_cache_io_contract_and_source_unchanged():
    with cth_temporary_directory() as case:
        source_root, cache_path, manifest = _prepare_source(case, _synthetic_payload())
        source_bytes, manifest_bytes = cache_path.read_bytes(), manifest.read_bytes()
        output_root = case / "derived"
        protocol = build_rank_calibration_cache(
            CONFIG_PATH, source_root, output_root, max_samples=1
        )
        assert protocol["num_samples"] == 1
        assert protocol["gt_used_for_generation"] is False
        assert cache_path.read_bytes() == source_bytes and manifest.read_bytes() == manifest_bytes
        rows = [json.loads(line) for line in (output_root / "manifest_test.jsonl").read_text().splitlines()]
        assert [(row["dataset"], row["stem"]) for row in rows] == [("CHAMELEON", "synthetic")]
        derived = torch.load(rows[0]["cache_path"], map_location="cpu", weights_only=False)
        assert derived["rankcal_version"] == "dabe_rankcal_v1"
        assert derived["source_augs"] == ["identity"] and derived["source_num_views"] == 1
        for field in ("c0_f_minmax_37", "c1_f_rank_r1_37", "c2_median_rank_r1_37"):
            assert derived[field].shape == (1, 37, 37)
            assert derived[field].dtype == torch.float32
            assert torch.isfinite(derived[field]).all()
        assert "c1_sorted_l1_vs_r1" in derived["diagnostics"]


@pytest.mark.parametrize(
    "update,error_type",
    [
        ({"augs": ["hflip"]}, ValueError),
        ({"fg_score_37": None}, TypeError),
        ({"residual_37": torch.zeros(1, 36, 37)}, ValueError),
        ({"residual_pass1_37": torch.full((1, 37, 37), float("nan"))}, ValueError),
    ],
)
def test_cache_rejects_invalid_payload(update, error_type):
    with cth_temporary_directory() as case:
        source_root, _, _ = _prepare_source(case, _synthetic_payload(**update))
        with pytest.raises(error_type):
            build_rank_calibration_cache(CONFIG_PATH, source_root, case / "derived", max_samples=1)


def test_cache_rejects_missing_field():
    with cth_temporary_directory() as case:
        payload = _synthetic_payload()
        del payload["fg_score_37"]
        source_root, _, _ = _prepare_source(case, payload)
        with pytest.raises(KeyError):
            build_rank_calibration_cache(CONFIG_PATH, source_root, case / "derived", max_samples=1)
