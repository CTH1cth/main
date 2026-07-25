from configs.dinov1_s8_dabepu_v11_ecst_dagp_uncgate_ndr_long45_lrfloor_2e5_sw_ones_noecst import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabepu_v11_dagp_uncgate_ndr_long45_"
    "lrfloor_2e5_sw_ones_noecst_oed_v1"
)

OED = dict(
    version="oed_v1_ordinal_evidence_distillation",
    enabled=True,
    mode="oed",
    loss_weight=0.05,
    num_pair_rounds=2,
    logit_temperature=1.0,
    rank_gap_power=1.0,
    pair_gap_eps=1e-8,
    rank_tie_mode="average",
    use_soft_fixed_only=True,
    apply_to="final_logits",
    deterministic_pairing=True,
    seed=20260722,
    diagnostic_interval=100,
    log_spearman=True,
    log_teacher_conflict=True,
    log_gradient_ratio=True,
    export_debug_vis=False,
)
