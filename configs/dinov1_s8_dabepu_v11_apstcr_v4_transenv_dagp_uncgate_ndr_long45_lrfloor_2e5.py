from configs.dinov1_s8_dabepu_v11_apstcr_v3_softdev_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_apstcr_v4_transenv_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

AP_STCR = {
    "enabled": True,
    "version": "ap_stcr_v4_transition_envelope_non_compensatory",
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
    # Shared temporal-interface sentinel only. V4 masks temporal instability
    # to zero whenever the past-only history is invalid.
    "temporal_empty_support": 0.50,
    "history_dtype": "float16",
    "clear_history_on_teacher_reset": True,
    "use_soft_deviation": True,
    "use_soft_correction_for_semantic": True,
    "evidence_fusion": "negative_soft_or",
    "use_transition_envelope": True,
    "evidence_rejection_strength": 0.35,
    "min_local_acceptance": 0.65,
    # These keys describe the protected global supervision schedule inherited
    # from v3; they are not an AP-STCR-local activation window.
    "post_reset_teacher_continuation": True,
    "post_reset_teacher_start_ratio": 0.95,
    "post_reset_teacher_end_ratio": 1.00,
    "post_reset_teacher_start_epoch": 21,
    "post_reset_teacher_end_epoch": 25,
    "evidence_resolution": 37,
    "loss_resolution": 68,
    "teacher_binary_threshold": 0.5,
    "teacher_binary_comparison": "strict_gt",
    "eps": 1e-6,
    "log_statistics": True,
    "log_interval_epoch": 1,
    "export_visualization": True,
    "visualization_interval": 1,
    "visualization_sample_indices": [0],
}
