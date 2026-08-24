"""Fixed-R16 t0.50 pseudo-label ablation for the matched 1x1 control."""

from configs.dinov1_s8_gbsp_r32_linear_dice005_t050_seed2027 import *  # noqa: F401,F403


EXP_NAME = "33-gbsp-r16-1x1-dice005-t050-seed2027"

# Keep the decoder, optimizer, loss, threshold, and seed unchanged; replace
# only the fixed-rank continuous GBSP cache and its audited metadata.
GBSP_R32_LINEAR_DICE005_T050_SEED2027 = False
GBSP_R16_LINEAR_DICE005_T050_SEED2027 = True

GBSP_VERSION = "gbsp_pca_absmm_r16_v1"
GBSP_PCA_RANK_MODE = "fixed"
GBSP_FIXED_PCA_RANK = 16
GBSP_THRESHOLD_SELECTION_SPLIT = "train4040_oracle_gt68_r16_uniform050"

DABE_CLEAN_DABE_V2_ROOT = (
    "../datasets/cache/gbsp_pca_absmm_r16_pseudo_cache/dinov1-s8"
)
DABE_CLEAN_GBSP_VERSION = "gbsp_pca_absmm_r16_v1"
DABE_CLEAN_GBSP_PCA_RANK_MODE = "fixed"
DABE_CLEAN_GBSP_FIXED_PCA_RANK = 16
