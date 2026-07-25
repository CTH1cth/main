from configs.dinov1_s8_dabepu_v11_cvsa_v1_hflip_router_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_cvsa_v1_hflip_direct_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

CVSA = {
    **CVSA,
    "version": "cvsa_v1_hflip_direct",
    "route_mode": "direct",
}
