from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from common.cache_dabe_crossrank import build_crossrank_cache
from common.dabe_crossrank import build_crossrank_r1dist, crossrank_diagnostics
from common.dabe_rank_calibration import rank_transport


MAIN_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = MAIN_ROOT.parent


@contextmanager
def cth_temporary_directory():
    root = BASELINE_ROOT / "workdir" / "pytest_crossrank_tmp"
    root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="case-", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _values(seed=7):
    generator = torch.Generator().manual_seed(seed)
    r1 = torch.rand((1, 37, 37), generator=generator)
    cross = torch.rand((1, 37, 37), generator=generator)
    return r1, cross


@pytest.mark.parametrize(
    "r1,cross,error",
    [
        (torch.zeros(1, 37, 37), "bad", TypeError),
        (torch.zeros(1, 37, 37), torch.zeros(1, 36, 37), ValueError),
        (torch.zeros(37, 37), torch.zeros(37, 37), ValueError),
        (torch.full((1, 37, 37), float("nan")), torch.zeros(1, 37, 37), ValueError),
        (torch.zeros(1, 37, 37), torch.full((1, 37, 37), float("inf")), ValueError),
        (torch.full((1, 37, 37), -0.01), torch.zeros(1, 37, 37), ValueError),
        (torch.zeros(1, 37, 37), torch.full((1, 37, 37), 1.01), ValueError),
    ],
)
def test_basic_interface_rejects_invalid_inputs(r1, cross, error):
    with pytest.raises(error):
        build_crossrank_r1dist(r1, cross)


def test_output_contract_input_unchanged_and_exact_rank_transport():
    r1, cross = _values()
    r1.requires_grad_(True)
    cross.requires_grad_(True)
    r1_copy, cross_copy = r1.detach().clone(), cross.detach().clone()
    output = build_crossrank_r1dist(r1, cross)
    expected = rank_transport(cross.detach().cpu().float(), r1.detach().cpu().float())
    assert output.shape == (1, 37, 37)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert output.is_contiguous()
    assert not output.requires_grad
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0 and float(output.max()) <= 1
    assert torch.equal(output, expected)
    assert torch.equal(r1.detach(), r1_copy)
    assert torch.equal(cross.detach(), cross_copy)


def test_no_tie_order_distribution_monotonicity_and_hard_area():
    count = 37 * 37
    r1 = torch.linspace(0.0, 1.0, count).reshape(1, 37, 37).flip(-1)
    cross = torch.randperm(count).float().reshape(1, 37, 37) / (count - 1)
    output = build_crossrank_r1dist(r1, cross)
    diagnostics = crossrank_diagnostics(r1, cross, output)
    assert torch.equal(torch.argsort(output.reshape(-1)), torch.argsort(cross.reshape(-1)))
    assert torch.allclose(
        torch.sort(output.reshape(-1)).values,
        torch.sort(r1.reshape(-1)).values,
        atol=1e-6,
    )
    assert diagnostics["crossrank_spearman_vs_cross"] >= 1 - 1e-6
    assert diagnostics["crossrank_monotonic_violation_count"] == 0
    assert diagnostics["crossrank_sorted_max_abs_vs_r1"] <= 1e-6
    assert int((output > 0.5).sum()) == int((r1 > 0.5).sum())


def test_constant_source_or_reference_falls_back_to_r1():
    r1, cross = _values()
    source_constant = torch.full_like(cross, 0.4)
    output = build_crossrank_r1dist(r1, source_constant)
    assert torch.equal(output, r1)
    assert crossrank_diagnostics(r1, source_constant, output)[
        "crossrank_constant_source_fallback"
    ]
    reference_constant = torch.full_like(r1, 0.7)
    output = build_crossrank_r1dist(reference_constant, cross)
    assert torch.equal(output, reference_constant)
    assert crossrank_diagnostics(reference_constant, cross, output)[
        "crossrank_constant_source_fallback"
    ]


def _prepare_cache_case(case: Path, count=2):
    dabe_root = case / "dabe"
    bgnull_root = case / "bgnull"
    dabe_rows, bgnull_rows = [], []
    for index in range(count):
        dataset, stem = "CHAMELEON", f"sample-{index}"
        r1, cross = _values(100 + index)
        dabe_path = dabe_root / dataset / f"{stem}.pt"
        bgnull_path = bgnull_root / "test" / dataset / f"{stem}.pt"
        dabe_path.parent.mkdir(parents=True, exist_ok=True)
        bgnull_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dataset": dataset,
                "stem": stem,
                "dabe_version": "v2",
                "augs": ["identity"],
                "num_views": 1,
                "residual_pass1_37": r1,
            },
            dabe_path,
        )
        torch.save(
            {
                "dataset": dataset,
                "stem": stem,
                "bgnull_version": "dabe_bgnull_v1",
                "source_dabe_version": "v2",
                "source_augs": ["identity"],
                "source_num_views": 1,
                "n0_r1_37": r1.clone(),
                "n1_cross_r1_37": cross,
            },
            bgnull_path,
        )
        dabe_rows.append(
            {"dataset": dataset, "stem": stem, "cache_path": str(dabe_path.resolve())}
        )
        bgnull_rows.append(
            {"dataset": dataset, "stem": stem, "cache_path": str(bgnull_path.resolve())}
        )
    dabe_manifest = dabe_root / "manifest_test.jsonl"
    bgnull_manifest = bgnull_root / "manifest_test.jsonl"
    _write_jsonl(dabe_manifest, dabe_rows)
    _write_jsonl(bgnull_manifest, bgnull_rows)
    protocol = {
        "bgnull_version": "dabe_bgnull_v1",
        "split": "test",
        "backbone": "DINOv1-S/8",
        "source_augs": ["identity"],
        "gt_used_for_generation": False,
        "dino_forward_used": False,
        "cross_exclusion_metric": "chebyshev",
        "cross_exclusion_radius": 1,
        "k_recon": 32,
        "source_dabe_manifest_sha256": _sha256(dabe_manifest),
    }
    protocol_path = bgnull_root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    config = case / "config.py"
    config.write_text("BACKBONE_KEY = 'dinov1-s8'\n", encoding="utf-8")
    return {
        "config": config,
        "dabe_root": dabe_root,
        "bgnull_root": bgnull_root,
        "dabe_manifest": dabe_manifest,
        "bgnull_manifest": bgnull_manifest,
        "bgnull_protocol": protocol_path,
        "dabe_rows": dabe_rows,
        "bgnull_rows": bgnull_rows,
    }


def test_cache_io_join_contract_and_sources_unchanged():
    with cth_temporary_directory() as case:
        paths = _prepare_cache_case(case)
        source_paths = [
            paths["dabe_manifest"], paths["bgnull_manifest"], paths["bgnull_protocol"],
            *[Path(row["cache_path"]) for row in paths["dabe_rows"]],
            *[Path(row["cache_path"]) for row in paths["bgnull_rows"]],
        ]
        before = {path: path.read_bytes() for path in source_paths}
        output = case / "derived"
        protocol = build_crossrank_cache(
            paths["config"], paths["dabe_root"], paths["bgnull_root"], output,
            max_samples=2,
        )
        assert protocol["num_samples"] == 2
        assert protocol["r1_source_max_abs"] <= 1e-6
        assert protocol["source_dabe_cache_modified"] is False
        assert protocol["source_bgnull_cache_modified"] is False
        assert protocol["gt_used_for_generation"] is False
        assert protocol["background_reconstruction_rerun"] is False
        assert protocol["bgnull_reconstruction_rerun"] is False
        for path, contents in before.items():
            assert path.read_bytes() == contents
        rows = [json.loads(line) for line in (output / "manifest_test.jsonl").read_text().splitlines()]
        assert len(rows) == 2
        assert {(row["dataset"], row["stem"]) for row in rows} == {
            ("CHAMELEON", "sample-0"), ("CHAMELEON", "sample-1")
        }
        payload = torch.load(rows[0]["cache_path"], map_location="cpu", weights_only=False)
        assert payload["crossrank_version"] == "dabe_crossrank_v1"
        value = payload["x1_crossrank_r1dist_37"]
        assert value.shape == (1, 37, 37) and value.dtype == torch.float32
        assert value.device.type == "cpu" and value.is_contiguous()
        assert torch.isfinite(value).all()
        assert isinstance(payload["diagnostics"], dict)


@pytest.mark.parametrize(
    "field,value",
    [
        ("bgnull_version", "wrong"),
        ("cross_exclusion_radius", 2),
        ("k_recon", 31),
        ("source_augs", ["hflip"]),
    ],
)
def test_cache_rejects_bad_bgnull_protocol(field, value):
    with cth_temporary_directory() as case:
        paths = _prepare_cache_case(case)
        protocol = json.loads(paths["bgnull_protocol"].read_text())
        protocol[field] = value
        paths["bgnull_protocol"].write_text(json.dumps(protocol), encoding="utf-8")
        with pytest.raises((ValueError, RuntimeError)):
            build_crossrank_cache(
                paths["config"], paths["dabe_root"], paths["bgnull_root"],
                case / "derived", max_samples=2,
            )


def test_cache_rejects_r1_mismatch_and_nonidentity_payload():
    for mutation in ("r1", "identity"):
        with cth_temporary_directory() as case:
            paths = _prepare_cache_case(case)
            path = Path(paths["bgnull_rows"][0]["cache_path"] if mutation == "r1" else paths["dabe_rows"][0]["cache_path"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if mutation == "r1":
                payload["n0_r1_37"] = torch.zeros(1, 37, 37)
            else:
                payload["augs"] = ["hflip"]
            torch.save(payload, path)
            with pytest.raises((ValueError, RuntimeError)):
                build_crossrank_cache(
                    paths["config"], paths["dabe_root"], paths["bgnull_root"],
                    case / "derived", max_samples=2,
                )


def test_cache_rejects_missing_and_duplicate_manifest_keys():
    with cth_temporary_directory() as case:
        paths = _prepare_cache_case(case)
        rows = paths["bgnull_rows"][:1]
        _write_jsonl(paths["bgnull_manifest"], rows)
        with pytest.raises(RuntimeError):
            build_crossrank_cache(
                paths["config"], paths["dabe_root"], paths["bgnull_root"],
                case / "derived", max_samples=2,
            )
    with cth_temporary_directory() as case:
        paths = _prepare_cache_case(case)
        row = paths["dabe_rows"][0]
        _write_jsonl(paths["dabe_manifest"], [row, row])
        protocol = json.loads(paths["bgnull_protocol"].read_text())
        protocol["source_dabe_manifest_sha256"] = _sha256(paths["dabe_manifest"])
        paths["bgnull_protocol"].write_text(json.dumps(protocol), encoding="utf-8")
        with pytest.raises(RuntimeError):
            build_crossrank_cache(
                paths["config"], paths["dabe_root"], paths["bgnull_root"],
                case / "derived", max_samples=2,
            )
