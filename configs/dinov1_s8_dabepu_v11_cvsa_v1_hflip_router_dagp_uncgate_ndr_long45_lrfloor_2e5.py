from configs.dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones_noecst import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_cvsa_v1_hflip_router_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

USE_CVSA = True
USE_AP_STCR = False
SUPERVISION_MODE = "cvsa"

CVSA = dict(
    enabled=True,
    version="cvsa_v1_hflip_router",
    route_mode="learnable",
    view_type="hflip",
    feature_cache_hflip_root=(
        "../datasets/cache/features_cache_hflip_cvsa_v1/dinov1-s8"
    ),
    fixed_cache_hflip_root=(
        "../datasets/cache/dabe_pu_v11_pseudo_cache_hflip_cvsa_v1/dinov1-s8"
    ),
    evidence_resolution=37,
    loss_resolution=68,
    risk_eq_weight=0.50,
    risk_sem_weight=0.50,
    semantic_temperature=0.20,
    route_temperature=0.10,
    min_proto_mass_patches=4.0,
    eps=1e-6,
    router_feature_dim=32,
    router_hidden_dim=32,
    router_gn_groups=4,
    router_init_teacher_prob=0.05,
    router_loss_weight=1.0,
    use_global_schedule=False,
    use_epoch_ratio=False,
    use_temporal_history=False,
    use_future_teacher=False,
    log_statistics=True,
    export_visualization=True,
    visualization_interval=1,
    visualization_sample_indices=[0],
)
