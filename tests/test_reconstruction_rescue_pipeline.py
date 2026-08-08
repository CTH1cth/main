from __future__ import annotations

from types import SimpleNamespace

import torch

from tools.cache_reconstruction_scores import audit_existing
from tools.gbsp_knn_lsr_common import score_from_payload
from tools.reconstruction_rescue_common import apply_feature_manifest_override, load_core_rows


def test_score_loader_prefers_exact_new_name_before_historical_alias() -> None:
    exact = torch.arange(1369, dtype=torch.float32).reshape(1, 37, 37)
    historical = torch.zeros(1, 37, 37)
    payload = {"scores": {"knn8": exact, "knn8_cos": historical}}
    assert torch.equal(score_from_payload(payload, "knn8"), exact.reshape(-1))


def test_score_loader_keeps_historical_alias_compatibility() -> None:
    historical = torch.arange(1369, dtype=torch.float32).reshape(1, 37, 37)
    payload = {"scores": {"knn8_cos": historical}}
    assert torch.equal(score_from_payload(payload, "knn8"), historical.reshape(-1))


def test_feature_manifest_override_rebases_machine_paths(tmp_path) -> None:
    feature = tmp_path / "feature.pt"; feature.touch()
    image = tmp_path / "CHAMELEON" / "im" / "sample.jpg"
    gt = tmp_path / "CHAMELEON" / "gt" / "sample.png"
    image.parent.mkdir(parents=True); gt.parent.mkdir(parents=True)
    image.touch(); gt.touch()
    manifest = tmp_path / "manifest_test.jsonl"
    manifest.write_text(
        '{"dataset":"CHAMELEON","stem":"sample","cache_path":"'
        + str(feature) + '","image_path":"' + str(image) + '"}\n',
        encoding="utf-8",
    )
    rows = apply_feature_manifest_override(
        [{"dataset": "CHAMELEON", "stem": "sample", "cache_path": "/old/core.pt"}], manifest
    )
    assert rows[0]["source_feature_path_override"] == str(feature)
    assert rows[0]["image_path"] == str(image)
    assert rows[0]["gt_path"] == str(gt)


def test_core_manifest_rebase_deduplicates_same_dataset_directory(tmp_path) -> None:
    payload = tmp_path / "test" / "CHAMELEON" / "animal-1.pt"
    payload.parent.mkdir(parents=True); payload.touch()
    (tmp_path / "manifest_test.jsonl").write_text(
        '{"dataset":"CHAMELEON","stem":"animal-1",'
        '"cache_path":"/old/machine/full_rank/test/CHAMELEON/animal-1.pt"}\n',
        encoding="utf-8",
    )
    rows = load_core_rows(tmp_path, split="test", max_samples=-1)
    assert rows[0]["cache_path"] == str(payload)


def test_resume_replaces_device_global_l2_by_frozen_core_response(tmp_path) -> None:
    formal = torch.arange(1369, dtype=torch.float32).reshape(1, 37, 37)
    core_path = tmp_path / "core.pt"
    torch.save({"results": {"r8": {"absolute_raw": formal}}}, core_path)
    score_path = tmp_path / "score.pt"
    torch.save({
        "version": "test_version", "dataset": "CHAMELEON", "stem": "sample",
        "background_indices": torch.arange(40), "self_match_violation_count": 0,
        "scores": {"knn8": torch.zeros_like(formal), "global_pca_l2": formal + 1e-3},
        "diagnostics": {"global_pca_l2": {}},
    }, score_path)
    row = {"dataset": "CHAMELEON", "stem": "sample", "cache_path": str(core_path),
           "image_path": "/image.jpg", "gt_path": "/gt.png"}
    _, audit = audit_existing(
        row, score_path, ("global_pca_l2",),
        SimpleNamespace(
            RECONSTRUCTION_RESCUE_VERSION="test_version",
            RECONSTRUCTION_BASELINE_TOLERANCE=1e-4,
        ),
    )
    repaired = torch.load(score_path, map_location="cpu", weights_only=False)
    assert audit["global_l2_reference_patched"] is True
    assert torch.equal(repaired["scores"]["global_pca_l2"], formal)
    assert repaired["global_l2_response_source"] == "frozen_core_r8_absolute_raw"
