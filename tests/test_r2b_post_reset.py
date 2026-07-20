from types import SimpleNamespace

from common.source_arbiter import (
    get_arbiter_apply_scale,
    get_arbiter_train_scale,
)
from common.utils import load_config
from train import (
    evaluate_r2b_exploratory_stop_conditions,
    get_dabe_pu_despl_schedule,
    validate_ecst_config,
    validate_source_arbiter_config,
)


STAGE_CONFIG = (
    "configs/"
    "dinov1_s8_dabepu_v11_egsa_r2b_signaware_dagp_uncgate_ndr_"
    "stage20_lrfloor_2e5.py"
)
LONG_CONFIG = (
    "configs/"
    "dinov1_s8_dabepu_v11_egsa_r2b_signaware_dagp_uncgate_ndr_"
    "long45_lrfloor_2e5.py"
)


def test_r2b_train_and_apply_schedules_across_reset():
    cfg = load_config(STAGE_CONFIG)
    assert get_arbiter_train_scale(6, cfg) == 0.0
    assert get_arbiter_train_scale(7, cfg) == 1.0 / 9.0
    assert get_arbiter_train_scale(15, cfg) == 1.0
    assert get_arbiter_train_scale(20, cfg) == 1.0
    assert get_arbiter_train_scale(21, cfg) == 0.0
    assert get_arbiter_apply_scale(20, cfg) == 1.0
    assert get_arbiter_apply_scale(21, cfg) == 1.0
    assert get_arbiter_apply_scale(45, cfg) == 1.0
    assert get_arbiter_apply_scale(46, cfg) == 0.0


def test_stage20_and_long45_lifecycle_configuration():
    stage = load_config(STAGE_CONFIG)
    long45 = load_config(LONG_CONFIG)
    validate_ecst_config(stage)
    validate_source_arbiter_config(stage)
    validate_ecst_config(long45)
    validate_source_arbiter_config(long45)
    assert stage.SOURCE_ARBITER_SKIP_AFTER_EPOCH_RESET is True
    assert stage.SOURCE_ARBITER_REQUIRED_RESUME_EPOCH == 6
    assert stage.MAX_EPOCH == 20
    assert long45.SOURCE_ARBITER_SKIP_AFTER_EPOCH_RESET is False
    assert long45.SOURCE_ARBITER_REQUIRED_RESUME_EPOCH == 6
    assert long45.MAX_EPOCH == 45
    assert long45.SOURCE_ARBITER_AUDIT_CONCENTRATION_OVERRIDE is True
    assert long45.SOURCE_ARBITER_AUDIT_ALLOWED_FAILURES == [
        "positive:top_1_percent_images_minority_share",
        "positive:top_10_percent_images_minority_share",
    ]
    assert long45.SOURCE_ARBITER_POS_TEACHER_CLASS_WEIGHT == 3.4297335021
    assert long45.SOURCE_ARBITER_POS_DABE_CLASS_WEIGHT == 0.7226316013
    assert long45.SOURCE_ARBITER_NEG_TEACHER_CLASS_WEIGHT == 3.7554045924
    assert long45.SOURCE_ARBITER_NEG_DABE_CLASS_WEIGHT == 0.7199848699
    assert long45.SOURCE_ARBITER_KEEP_TEMPORAL_MEMORY_AFTER_RESET is True
    assert long45.SOURCE_ARBITER_RELEASE_ROUTE_MEMORY_AFTER_FREEZE is True
    assert long45.SOURCE_ARBITER_RELEASE_UTILITY_EVALUATOR_AFTER_FREEZE is True
    assert get_dabe_pu_despl_schedule(21, long45) == (0.05, 0.95)
    assert get_dabe_pu_despl_schedule(45, long45) == (0.05, 0.95)


def test_exploratory_stop_rules_use_required_consecutive_streaks():
    cfg = SimpleNamespace(
        SOURCE_ARBITER_STOP_RESIDUAL_SATURATION_MAX=0.60,
        SOURCE_ARBITER_STOP_RESIDUAL_SATURATION_PATIENCE=2,
        SOURCE_ARBITER_STOP_NEG_CONFLICT_START_EPOCH=11,
        SOURCE_ARBITER_STOP_NEG_CONFLICT_PATIENCE=2,
        SOURCE_ARBITER_STOP_AREA_MAX_EPOCH10=0.135,
        SOURCE_ARBITER_STOP_AREA_MAX_EPOCH20=0.115,
        SOURCE_ARBITER_STOP_AREA_POST_RESET=0.105,
        SOURCE_ARBITER_POST_RESET_MIN_EPOCH20_AREA_RATIO=0.90,
        SOURCE_ARBITER_POST_RESET_AREA_PATIENCE=3,
        SOURCE_ARBITER_STOP_AREA_CEILING=0.190,
        SOURCE_ARBITER_STOP_AREA_CEILING_PATIENCE=2,
    )
    source_sums = {
        "residual_positive_saturation_ratio": 0.61,
        "residual_negative_saturation_ratio": 0.10,
        "gate_negative_teacher_bg_fg_core": 0.60,
        "teacher_prior": 0.60,
    }
    reasons, _, state = evaluate_r2b_exploratory_stop_conditions(
        cfg, 11, source_sums, 1, 0.15, {}
    )
    assert reasons == []
    reasons, _, state = evaluate_r2b_exploratory_stop_conditions(
        cfg, 12, source_sums, 1, 0.15, state
    )
    assert "positive_residual_saturation" in reasons
    assert "negative_conflict_gate_not_below_teacher_prior" in reasons


def test_exploratory_stop_rules_preserve_epoch20_area_reference():
    cfg = SimpleNamespace(
        SOURCE_ARBITER_STOP_RESIDUAL_SATURATION_MAX=0.60,
        SOURCE_ARBITER_STOP_RESIDUAL_SATURATION_PATIENCE=2,
        SOURCE_ARBITER_STOP_NEG_CONFLICT_START_EPOCH=11,
        SOURCE_ARBITER_STOP_NEG_CONFLICT_PATIENCE=2,
        SOURCE_ARBITER_STOP_AREA_MAX_EPOCH10=0.135,
        SOURCE_ARBITER_STOP_AREA_MAX_EPOCH20=0.115,
        SOURCE_ARBITER_STOP_AREA_POST_RESET=0.105,
        SOURCE_ARBITER_POST_RESET_MIN_EPOCH20_AREA_RATIO=0.90,
        SOURCE_ARBITER_POST_RESET_AREA_PATIENCE=3,
        SOURCE_ARBITER_STOP_AREA_CEILING=0.190,
        SOURCE_ARBITER_STOP_AREA_CEILING_PATIENCE=2,
    )
    safe_stats = {
        "residual_positive_saturation_ratio": 0.0,
        "residual_negative_saturation_ratio": 0.0,
        "gate_negative_teacher_bg_fg_core": 0.0,
        "teacher_prior": 0.95,
    }
    _, _, state = evaluate_r2b_exploratory_stop_conditions(
        cfg, 20, safe_stats, 1, 0.15, {}
    )
    for epoch in (21, 22):
        reasons, _, state = evaluate_r2b_exploratory_stop_conditions(
            cfg, epoch, safe_stats, 1, 0.13, state
        )
        assert "post_reset_area_below_epoch20_ratio" not in reasons
    reasons, _, _ = evaluate_r2b_exploratory_stop_conditions(
        cfg, 23, safe_stats, 1, 0.13, state
    )
    assert "post_reset_area_below_epoch20_ratio" in reasons
