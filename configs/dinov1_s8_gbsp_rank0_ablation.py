"""DINOv1-S/8 GBSP rank-0 background-mean ablation."""

from configs.dinov1_s8_mbsp import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_rank0_ablation"

# Rank-0 keeps the formal M=1 background candidates and mean, but fits no PCA
# direction.  cache_mbsp_pseudo.py has an explicit no-SVD path for this case.
MBSP_NUM_SUBSPACES = 1
MBSP_PCA_MAX_RANK = 0
MBSP_PCA_MIN_RANK = 0
MBSP_OUT_ROOT = "../workdir/gbsp_rank0_ablation/r0"

GBSP_RANK0_ROOT = MBSP_OUT_ROOT
GBSP_RANKR_ROOT = "../workdir/mbsp_pca_v1/ablations/M1_pilot200"
GBSP_RANK0_EVAL_ROOT = "../workdir/gbsp_rank0_ablation/eval"
# Frozen formal rank-r reference settings.  These are checked per cache item;
# the inherited MBSP_PCA_* values above intentionally describe rank-0.
GBSP_RANKR_PCA_ENERGY = 0.90
GBSP_RANKR_PCA_MAX_RANK = 8
GBSP_RANKR_PCA_MIN_RANK = 1
GBSP_RANK0_THRESHOLD = 0.5
GBSP_RANK0_CALIBRATION = "image_minmax_then_fixed_0.5"
GBSP_RANK0_RESIZE = "bilinear_37_to_68_to_original"
