"""Configuration for Similarity Ambiguity vs Background Variation Consistency."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = PROJECT_ROOT / "workdir/gbsp_core_optimization/full_rank"
CORE_MANIFEST = CORE_ROOT / "manifest_test.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "workdir/gbsp_similarity_variation"
TEST20_IDS = PROJECT_ROOT / "workdir/gbsp_threshold_v2/test20_ids.txt"

GRID_SIZE = 37
KNN_K = 8
METHODS = ("mean_l2", "proto_cos", "nn_cos", "knn8_cos", "gbsp_r8")
HIGH_SIMILARITY_QUANTILES = {"H20": 0.80, "H30": 0.70, "H40": 0.60}
SIMILARITY_BINS = 5
PATCH_FG_THRESHOLD = 0.50
STRICT_FG_THRESHOLD = 0.80
STRICT_BG_THRESHOLD = 0.20
MIN_CLASS_PATCHES = 3
MATCH_TOLERANCE = 0.01
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260807
EXPECTED_DATASETS = ("CHAMELEON", "CAMO", "COD10K", "NC4K")
EXPECTED_TOTAL = 6473

