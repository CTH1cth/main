from configs.dinov1_s8_dabepu_v11_dagp_uncgate_ndr_long50_lrfloor_2e5_sw_ones_noecst_linear30 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_apstcr_dagp_uncgate_ndr_"
    "long50_lrfloor_2e5"
)

USE_AP_STCR = True
SUPERVISION_MODE = "ap_stcr"

# AP-STCR replaces the two global loss groups with one mixed target, while
# preserving the protected Linear30 schedule and raw teacher route identity.
TEACHER_FUSION_MODE = "dabe_pu_despl_sched"
TEACHER_ROUTING_MODE = "none"
STATIC_WEIGHT_MODE = "ones"
USE_ECST = False

AP_STCR = {
    "enabled": True,
    "version": "ap_stcr_v1_full_pixel_37_to_68",
    "fg_anchor_ratio": 0.20,
    "bg_anchor_ratio": 0.20,
    "min_fg_anchors": 4,
    "min_bg_anchors": 4,
    "max_fg_anchors": 64,
    "max_bg_anchors": 64,
    "prefer_dabe_background_seed": True,
    "background_source": "dabe_seed:bg_anchor_37",
    "tau_delta": 0.25,
    "tau_margin": 0.50,
    "temporal_window": 3,
    "tau_temporal": 0.20,
    "history_dtype": "float16",
    "clear_history_on_reset": True,
    "lambda_semantic": 1.0,
    "lambda_temporal": 1.0,
    "evidence_resolution": 37,
    "loss_resolution": 68,
    "teacher_binary_threshold": 0.5,
    "teacher_binary_comparison": "strict_gt",
    "eps": 1e-6,
    "log_statistics": True,
    "export_visualization": True,
    "visualization_interval": 1,
    "visualization_sample_indices": [0],
}
