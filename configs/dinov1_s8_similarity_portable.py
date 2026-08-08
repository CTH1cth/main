"""Portable from-scratch configuration for the GBSP similarity audit.

Defaults are derived from this repository instead of a user-specific /home path.
Every external root may be overridden through an environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

from configs.dinov1_s8_gbsp_core_ablation import *  # noqa: F401,F403


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _root(env_name: str, default: Path) -> str:
    return str(Path(os.environ.get(env_name, str(default))).expanduser().resolve())


# Expected default layout:
# MY-baseline/datasets/COD/{CHAMELEON,TE-CAMO,TE-COD10K,NC4K}/{im,gt}
DATA_ROOT = _root("COD_DATA_ROOT", PROJECT_ROOT / "datasets/COD")
CACHE_ROOT = _root("CTH_CACHE_ROOT", PROJECT_ROOT / "datasets/cache")
WORK_ROOT = _root("CTH_WORK_ROOT", PROJECT_ROOT / "workdir")

# cache_features.py deliberately uses local_files_only=True. Put the downloaded
# HuggingFace snapshot here, or override DINO_V1_S8_MODEL_PATH.
DINO = dict(DINO)
DINO["model_path"] = _root(
    "DINO_V1_S8_MODEL_PATH",
    PROJECT_ROOT / "weights/huggingface/facebook-dino-vits8",
)

MBSP_FEATURE_MANIFEST = str(
    Path(CACHE_ROOT) / "features_cache/dinov1-s8/manifest_test.jsonl"
)
MBSP_DABE_MANIFEST = str(
    Path(CACHE_ROOT)
    / "gbsp_fullbc_identity/dinov1-s8/manifest_test.jsonl"
)
MBSP_OUT_ROOT = str(Path(WORK_ROOT) / "mbsp_pca_v1/dinov1-s8")
MBSP_EVAL_ROOT = str(Path(WORK_ROOT) / "mbsp_pca_v1_eval")

GBSP_CORE_SOURCE_ROOT = str(Path(WORK_ROOT) / "mbsp_pca_v1/ablations/M1_full6473")
GBSP_CORE_OUTPUT_ROOT = str(Path(WORK_ROOT) / "gbsp_core_optimization")
GBSP_CORE_TEST20_LIST = str(PROJECT_ROOT / "main/configs/gbsp_similarity_test20_ids.txt")
