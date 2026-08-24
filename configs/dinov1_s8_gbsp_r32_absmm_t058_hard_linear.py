"""Fixed-R32 GBSP Absolute-MinMax t=0.58 with the established 1x1 head.

This is the rank-only successor to the numbered 23 GBSP experiment.  It keeps
the decoder, optimization, loss, threshold, data splits, and training schedule
unchanged while replacing the legacy EV90-capped-at-R8 continuous response
cache with the independently generated fixed-R32 response cache.

The 0.58 threshold remains an oracle diagnostic informed by the evenly sampled
training-GT pilot.  It must not be presented as GT-free model selection.  GT
was not used to generate the fixed-R32 continuous response cache itself.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_gbsp_absmm_t058_hard_linear_staticonly_purestudent_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "25-gbsp-r32-absmm-t058-1×1"

GBSP_VERSION = "gbsp_pca_absmm_r32_v1"
GBSP_PCA_RANK_MODE = "fixed"
GBSP_FIXED_PCA_RANK = 32
GBSP_THRESHOLD_SELECTION_SPLIT = "train4040_pilot200_even_oracle_gt68"

DABE_CLEAN_DABE_V2_ROOT = (
    "../datasets/cache/gbsp_pca_absmm_r32_pseudo_cache/dinov1-s8"
)
DABE_CLEAN_GBSP_VERSION = "gbsp_pca_absmm_r32_v1"
DABE_CLEAN_GBSP_PCA_RANK_MODE = "fixed"
DABE_CLEAN_GBSP_FIXED_PCA_RANK = 32

# Suppress compatibility-only Teacher/DABE routing diagnostics.  The formal log
# retains protocol, cache, loss/LR, pseudo area, student area, validation, and
# checkpoint records only.
LEAN_PURE_STUDENT_LOGGING = True
