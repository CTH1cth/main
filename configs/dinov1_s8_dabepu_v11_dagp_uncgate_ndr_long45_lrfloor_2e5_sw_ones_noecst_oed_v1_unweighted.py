from copy import deepcopy

from configs.dinov1_s8_dabepu_v11_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones_noecst_oed_v1 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_long45_"
    "lrfloor_2e5_sw_ones_noecst_oed_v1_unweighted"
)

OED = deepcopy(OED)
OED["rank_gap_power"] = 0.0
