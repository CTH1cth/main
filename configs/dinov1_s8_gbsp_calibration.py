"""DINOv1-S/8 single-global-PCA residual calibration protocol."""

from configs.dinov1_s8_mbsp import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_calibration"

# Formal M=1 GBSP test cache.  Despite the historical directory name, this
# manifest contains all 6473 images and no generation fallback.
GBSP_CALIBRATION_GBSP_ROOT = (
    "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
)
GBSP_CALIBRATION_R1_ROOT = (
    "../datasets/cache/dabe_v2_direct_test_cache_identity/dinov1-s8"
)
GBSP_CALIBRATION_FEATURE_MANIFEST = (
    "../datasets/cache/features_cache/dinov1-s8/manifest_test.jsonl"
)

# Reuse the exact identities from the original 200-image MBSP pilot.  The
# stage-A evaluator materializes these identities into pilot200_ids.txt once.
GBSP_CALIBRATION_PILOT_SOURCE_MANIFEST = (
    "../workdir/mbsp_pca_v1/dinov1-s8/manifest_test.jsonl"
)
GBSP_CALIBRATION_PILOT_LIST = (
    "../workdir/gbsp_calibration/pilot200_ids.txt"
)
GBSP_CALIBRATION_ROOT = "../workdir/gbsp_calibration"

GBSP_CALIBRATION_GRID = 37
GBSP_CALIBRATION_LOSS_SIZE = 68
GBSP_CALIBRATION_R1_THRESHOLD = 0.5
GBSP_CALIBRATION_MINMAX_THRESHOLD = 0.5
GBSP_CALIBRATION_RESIZE_MODE = "bilinear_37_to_68_to_original"

# Reserved for stage B.  These are protocol constants, not searched values.
GBSP_CALIBRATION_NUM_FOLDS = 5
GBSP_CALIBRATION_FOLD_SEED = 0
GBSP_CALIBRATION_CF_ALPHA = 0.05
GBSP_CALIBRATION_Q_ALPHA = 0.05
GBSP_CALIBRATION_RMAD_KAPPA = 3.0
GBSP_CALIBRATION_EPS = 1e-12
