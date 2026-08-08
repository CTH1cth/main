"""CF-BRC-HC calibration for formal DINOv1-S/8 single-PCA GBSP."""

from configs.dinov1_s8_gbsp_calibration import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_cf_brc_hc"
GBSP_THRESHOLD_VERSION = "gbsp_cf_brc_hc_v1"

# Formal, complete 6473-image M=1 cache.  The directory name is historical.
GBSP_THRESHOLD_TEST_GBSP_ROOT = (
    "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
)
GBSP_THRESHOLD_TEST_R1_ROOT = (
    "../datasets/cache/dabe_v2_direct_test_cache_identity/dinov1-s8"
)
GBSP_THRESHOLD_PILOT_LIST = "../workdir/gbsp_calibration/pilot200_ids.txt"
GBSP_THRESHOLD_ROOT = "../workdir/gbsp_threshold"

# Existing immutable 4040-image continuous GBSP training cache.  Its legacy
# name contains t063, but thresholding is performed only by the consumers.
GBSP_THRESHOLD_TRAIN_GBSP_ROOT = (
    "../workdir/gbsp_absmm_t063_train_identity/dinov1-s8"
)
GBSP_THRESHOLD_TRAIN_OUT_ROOT = (
    "../workdir/gbsp_threshold/crossfit_train4040"
)

GBSP_THRESHOLD_GRID = 37
GBSP_THRESHOLD_FEATURE_DIM = 384
GBSP_THRESHOLD_NUM_FOLDS = 5
GBSP_THRESHOLD_EPS = 1e-12
GBSP_THRESHOLD_SCALE_FLOOR = 1e-6
GBSP_THRESHOLD_HC_P_MAX = 0.20
GBSP_THRESHOLD_HC_FALLBACK_P = 0.05
GBSP_THRESHOLD_BQ_ALPHA = 0.05
GBSP_THRESHOLD_RMAD_KAPPA = 3.0
GBSP_THRESHOLD_RMAD_SENSITIVITY = (2.5, 3.0, 3.5)
GBSP_THRESHOLD_FIXED_BASELINES = (0.50, 0.58)

# HC-failure branch: Background-Tail Quantile with Global Shrinkage.
# The OOF background z-tail is mapped back through the fold log-MAD scales and
# into the image-wise GBSP Min-Max domain before shrinkage.  No parameter scan.
GBSP_THRESHOLD_BTS_BACKGROUND_QUANTILE = 0.95
GBSP_THRESHOLD_BTS_GLOBAL_MINMAX = 0.58
GBSP_THRESHOLD_BTS_LAMBDA = 0.50
GBSP_THRESHOLD_BTS_BASELINE_LAMBDA = 0.0
GBSP_THRESHOLD_BTS_DOMAIN_MAPPING = (
    "oof_z_q95_to_fold_raw_median_to_image_minmax"
)

# The only PCA difference from formal GBSP is fitting on four folds.
GBSP_THRESHOLD_PCA_ENERGY = 0.90
GBSP_THRESHOLD_PCA_MAX_RANK = 8
GBSP_THRESHOLD_PCA_MIN_RANK = 1
GBSP_THRESHOLD_MIN_CLUSTER_SIZE = 16

GBSP_THRESHOLD_FALLBACK_GATE = 0.01
GBSP_THRESHOLD_EMPTY_GATE = 0.01
GBSP_THRESHOLD_LARGE_AREA_GATE = 0.01
GBSP_THRESHOLD_LARGE_AREA = 0.50
