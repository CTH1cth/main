from configs.dinov1_s8_dabev2_dagp_uncgate_ndr_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = "dinov2_b14_dabe_r1_direct"

BACKBONE_KEY = "dinov2-b14"

DINO = dict(DINO_CONFIGS[BACKBONE_KEY])
DINO["embed_dim"] = 768
