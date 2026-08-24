"""Mechanism-only 512 audit: coarse 32x32 background, native 64x64 queries."""

from configs.dinov1_s8_gbsp_resolution_512_native64_bw2 import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_gbsp_resolution_512_coarsebg32_finequery64"
GBSP_RESOLUTION_VARIANT = "512-CoarseBG32-FineQ64"
GBSP_BACKGROUND_POOL = 2
GBSP_BACKGROUND_GRID = 32
GBSP_QUERY_GRID = 64

