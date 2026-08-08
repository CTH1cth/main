"""Frozen DINOv1-S/8 MBSP offline experiment configuration."""

from configs.dinov1_s8_dabev2_dagp_uncgate_ndr_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_mbsp"

MBSP_VERSION = "mbsp_pca_v1"
MBSP_GRID = 37
MBSP_FEATURE_DIM = 384
MBSP_NUM_SUBSPACES = 4
MBSP_MIN_CLUSTER_SIZE = 16
MBSP_PCA_ENERGY = 0.90
MBSP_PCA_MAX_RANK = 8
MBSP_PCA_MIN_RANK = 1
MBSP_KMEANS_SEED = 0
MBSP_KMEANS_N_INIT = 10
MBSP_KMEANS_MAX_ITER = 100
MBSP_EPS = 1e-8

MBSP_FEATURE_MANIFEST = (
    "../datasets/cache/features_cache/dinov1-s8/manifest_test.jsonl"
)
MBSP_DABE_MANIFEST = (
    "../datasets/cache/dabe_v2_direct_test_cache_identity/dinov1-s8/manifest_test.jsonl"
)
MBSP_OUT_ROOT = "../workdir/mbsp_pca_v1/dinov1-s8"
MBSP_EVAL_ROOT = "../workdir/mbsp_pca_v1_eval"
