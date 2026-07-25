from configs.dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones_noecst import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_apstcr_v2_conflictpass_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

USE_AP_STCR = True
SUPERVISION_MODE = "ap_stcr"

# Preserve the real Long45 No-ECST protocol. AP-STCR replaces the two loss
# groups with one detached mixed target but does not alter the global schedule.
TEACHER_FUSION_MODE = "dabe_pu_despl_sched"
TEACHER_ROUTING_MODE = "none"
STATIC_WEIGHT_MODE = "ones"
USE_ECST = False

AP_STCR = {
    "enabled": True,
    "version": "ap_stcr_v2_conflict_only_pass_through",
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
    "temporal_empty_support": 0.50,
    "history_dtype": "float16",
    "conflict_only": True,
    "rejection_max": 0.35,
    "support_neutral_point": 0.50,
    "evidence_resolution": 37,
    "loss_resolution": 68,
    "teacher_binary_threshold": 0.5,
    "teacher_binary_comparison": "strict_gt",
    "clear_history_on_teacher_reset": True,
    "eps": 1e-6,
    "log_statistics": True,
    "log_interval_epoch": 1,
    "export_visualization": True,
    "visualization_interval": 1,
    "visualization_sample_indices": [0],
}
