"""Frozen protocol for Boundary Background Memory reconstruction rescue."""

from configs.dinov1_s8_gbsp_core_ablation import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_reconstruction_rescue"
RECONSTRUCTION_RESCUE_VERSION = "reconstruction_rescue_v1"
RECONSTRUCTION_CORE_ROOT = "../workdir/gbsp_core_optimization/full_rank"
RECONSTRUCTION_OUTPUT_ROOT = "../workdir/reconstruction_rescue_20260807"

RETRIEVAL_FEATURE = "l2_key"
RECONSTRUCTION_FEATURE = "l2_key"
BACKGROUND_SOURCE = "current_full_bc"

RECONSTRUCTION_GRID = 37
RECONSTRUCTION_FEATURE_DIM = 384
RECONSTRUCTION_GLOBAL_PCA_RANK = 8
RECONSTRUCTION_KNN_MAX_K = 32
RECONSTRUCTION_KNN_BASELINE_K = 8

RECONSTRUCTION_CONVEX_SOLVER = "projected_gradient_simplex"
RECONSTRUCTION_CONVEX_MAX_ITER = 64
RECONSTRUCTION_CONVEX_TOL = 1e-6
RECONSTRUCTION_QUERY_BATCH_SIZE = 128

RECONSTRUCTION_LOCAL_PCA_K = 16
RECONSTRUCTION_LOCAL_PCA_RANK = 4
RECONSTRUCTION_AFFINE_RCOND = 1e-6

RECONSTRUCTION_PRIMARY_VARIANTS = (
    "global_pca_l2",
    "global_pca_raw",
    "lcbr_l2_k8",
    "lcbr_l2_k16",
    "lcbr_l2_k32",
    "lcbr_raw_k16",
    "lsr_l2_k16_r4",
    "lsr_raw_k16_r4",
)
RECONSTRUCTION_SECONDARY_LSR_VARIANTS = (
    "lsr_l2_k8_r2",
    "lsr_l2_k32_r8",
)
RECONSTRUCTION_CONDITIONAL_VARIANTS = ("lar_l2_k16",)

RECONSTRUCTION_FIXED_THRESHOLDS = (0.50, 0.58)
RECONSTRUCTION_THRESHOLD_START = 0.30
RECONSTRUCTION_THRESHOLD_END = 0.70
RECONSTRUCTION_THRESHOLD_STEP = 0.01
RECONSTRUCTION_ADAPTIVE_THRESHOLDS = ("otsu", "multi_otsu_3")
RECONSTRUCTION_BOOTSTRAP_REPETITIONS = 2000
RECONSTRUCTION_BOOTSTRAP_SEED = 20260807

# Frozen values are read from the formal previous-stage deliverable. They are
# audit gates only and are never used to generate a response.
RECONSTRUCTION_BASELINE_REFERENCE = {
    "knn8": {"dataset_macro_AP": 0.7716604706496237, "dataset_macro_AUROC": 0.9529828398382176},
    "global_pca_l2": {"dataset_macro_AP": 0.7726448926517061, "dataset_macro_AUROC": 0.9524009945102647},
}
RECONSTRUCTION_BASELINE_TOLERANCE = 1e-4
# Per-pixel CUDA-vs-frozen-core numerical audit only.  Exceeding this value
# is recorded but never changes the published frozen Global-PCA-L2 response.
RECONSTRUCTION_DEVICE_REPRODUCTION_WARNING_TOLERANCE = 2e-4

RECONSTRUCTION_EXPECTED_COUNTS = {
    "CHAMELEON": 76,
    "CAMO": 250,
    "COD10K": 2026,
    "NC4K": 4121,
}
