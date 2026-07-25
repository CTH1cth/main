import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from common.dabe_clean import build_clean_targets
from common.dataset import CachedTrainDataset


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _build_fixture(root, legacy=False):
    dataset, stem = "TR-SYN", "sample"
    image_dir = root / "data" / dataset / "im"
    image_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((12, 15, 3), dtype=np.uint8)).save(
        image_dir / f"{stem}.png"
    )
    # Intentionally do not create a gt/ directory.

    feature_path = root / "cache" / "feature.pt"
    feature_path.parent.mkdir(parents=True)
    torch.save(
        {"dataset": dataset, "stem": stem, "tensor": torch.zeros(384, 37, 37)},
        feature_path,
    )
    _write_jsonl(
        root / "cache" / "features_cache" / "dinov1-s8" / "manifest_train.jsonl",
        [{"dataset": dataset, "stem": stem, "cache_path": str(feature_path)}],
    )

    foreground_37 = torch.linspace(0.0, 1.0, 37 * 37).reshape(1, 37, 37)
    background_37 = 1.0 - foreground_37
    foreground_68 = torch.nn.functional.interpolate(
        foreground_37.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    )[0]
    background_68 = torch.nn.functional.interpolate(
        background_37.unsqueeze(0), size=(68, 68), mode="bilinear", align_corners=False
    )[0]
    clean_37 = build_clean_targets(foreground_37, background_37)
    clean_68 = build_clean_targets(foreground_68, background_68)
    clean_payload = {
        "dataset": dataset,
        "stem": stem,
        "backbone_key": "dinov1-s8",
        "version": "dabe_clean_v1",
        "source_version": "pu_v11",
        "foreground_evidence_37": foreground_37,
        "foreground_evidence_68": foreground_68,
        "background_evidence_37": background_37,
        "background_evidence_68": background_68,
        "target_dp_37": clean_37["target_dp"],
        "target_dp_68": clean_68["target_dp"],
        "target_diff_37": clean_37["target_diff"],
        "target_diff_68": clean_68["target_diff"],
    }
    clean_path = root / "clean" / dataset / f"{stem}.pt"
    clean_path.parent.mkdir(parents=True)
    torch.save(clean_payload, clean_path)
    _write_jsonl(
        root / "clean" / "manifest_train.jsonl",
        [{"dataset": dataset, "stem": stem, "cache_path": str(clean_path)}],
    )

    legacy_root = root / "legacy"
    if legacy:
        legacy_payload = {
            "dataset": dataset,
            "stem": stem,
            "backbone_key": "dinov1-s8",
        }
        for key in (
            "fg_core_pu_68",
            "fg_core_fallback_68",
            "bg_core_pu_68",
            "extent_candidate_68",
            "unknown_68",
        ):
            legacy_payload[key] = torch.zeros(1, 68, 68)
        # No target_soft or weight_map is present: routing-only loading must work.
        legacy_path = legacy_root / dataset / f"{stem}.pt"
        legacy_path.parent.mkdir(parents=True)
        torch.save(legacy_payload, legacy_path)
        _write_jsonl(
            legacy_root / "manifest_train.jsonl",
            [{"dataset": dataset, "stem": stem, "cache_path": str(legacy_path)}],
        )

    cfg = SimpleNamespace(
        DATA_ROOT=str(root / "data"),
        TRAIN_DATASETS=[dataset],
        CACHE_ROOT=str(root / "cache"),
        BACKBONE_KEY="dinov1-s8",
        LOSS_SIZE=68,
        USE_DABE_CLEAN=True,
        USE_DABE_PU=False,
        USE_DABE_PSEUDO=False,
        USE_DESPL_PSEUDO=False,
        USE_DREPP=False,
        USE_QRA=False,
        USE_CCR=False,
        USE_DABE_CLEAN_DESPL_SCHEDULE=True,
        DABE_CLEAN_TARGET_MODE="dp",
        DABE_CLEAN_STATIC_WEIGHT_MODE="ones",
        DABE_CLEAN_ROOT=str(root / "clean"),
        DABE_CLEAN_USE_LEGACY_ECST_REGIONS=legacy,
        DABE_CLEAN_LEGACY_REGION_ROOT=str(legacy_root),
        P_INIT_MODE="dabe_clean_v1_desplsched",
        PSEUDO_CACHE_OVERRIDE=None,
        USE_NDR_BRANCH=False,
    )
    return cfg


class DABECleanDatasetTest(unittest.TestCase):
    def test_clean_dataset_has_no_gt_or_static_weight_map(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = CachedTrainDataset(
                _build_fixture(Path(directory), legacy=False)
            )
            sample = dataset[0]
            self.assertEqual(sample["dataset_name"], "TR-SYN")
            self.assertEqual(sample["stem"], "sample")
            self.assertEqual(sample["sample_index"], 0)
            self.assertTrue(
                torch.equal(sample["pseudo"], sample["dabe_clean_target_68"])
            )
            self.assertFalse(
                any("weight_map" in key or key == "gt" for key in sample)
            )
            self.assertFalse(
                any(key.startswith("legacy_ecst_") for key in sample)
            )

    def test_legacy_loader_exposes_only_namespaced_routing_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = CachedTrainDataset(
                _build_fixture(Path(directory), legacy=True)
            )
            sample = dataset[0]
            legacy_keys = {
                key for key in sample if key.startswith("legacy_ecst_")
            }
            self.assertEqual(
                legacy_keys,
                {
                    "legacy_ecst_fg_core",
                    "legacy_ecst_fg_fallback",
                    "legacy_ecst_bg_core",
                    "legacy_ecst_extent",
                    "legacy_ecst_unknown",
                },
            )
            self.assertNotIn("pu_target_soft", sample)
            self.assertNotIn("pu_weight_map", sample)


if __name__ == "__main__":
    unittest.main()
