import math
from types import SimpleNamespace

from common.source_arbiter import (
    compute_r2b_audit_class_weights,
    evaluate_r2b_audit_concentration_override,
    evaluate_r2b_target_audit_v2_admission,
    summarize_r2b_target_audit_branch_v2,
)


def _cfg():
    return SimpleNamespace(
        SOURCE_ARBITER_AUDIT_POS_MIN_VALID_PIXELS=10000,
        SOURCE_ARBITER_AUDIT_POS_MIN_HARD_MINORITY_PIXELS=500,
        SOURCE_ARBITER_AUDIT_POS_MIN_HARD_MINORITY_RATIO=0.01,
        SOURCE_ARBITER_AUDIT_POS_MIN_SOFT_MINORITY_MASS=2000.0,
        SOURCE_ARBITER_AUDIT_POS_MIN_MINORITY_IMAGES=128,
        SOURCE_ARBITER_AUDIT_POS_MIN_ACTIVE_BATCH_RATIO=0.90,
        SOURCE_ARBITER_AUDIT_NEG_MIN_VALID_PIXELS=50000,
        SOURCE_ARBITER_AUDIT_NEG_MIN_HARD_MINORITY_PIXELS=1000,
        SOURCE_ARBITER_AUDIT_NEG_MIN_HARD_MINORITY_RATIO=0.01,
        SOURCE_ARBITER_AUDIT_NEG_MIN_SOFT_MINORITY_MASS=2000.0,
        SOURCE_ARBITER_AUDIT_NEG_MIN_MINORITY_IMAGES=256,
        SOURCE_ARBITER_AUDIT_NEG_MIN_ACTIVE_BATCH_RATIO=0.99,
        SOURCE_ARBITER_AUDIT_MAX_TOP1P_IMAGE_SHARE=0.25,
        SOURCE_ARBITER_AUDIT_MAX_TOP10P_IMAGE_SHARE=0.60,
    )


def _passing_branch(valid, minority, minority_images, active_ratio):
    return {
        "valid_pixels": valid,
        "teacher_preferred_pixels": minority,
        "dabe_preferred_pixels": valid - minority,
        "hard_minority_pixels": minority,
        "hard_minority_ratio": minority / valid,
        "soft_teacher_mass": 0.25 * valid,
        "soft_dabe_mass": 0.75 * valid,
        "soft_minority_mass": 0.25 * valid,
        "valid_images": 1000,
        "minority_valid_images": minority_images,
        "top_1_percent_images_minority_share": 0.10,
        "top_10_percent_images_minority_share": 0.50,
        "active_batches": round(active_ratio * 100),
        "total_batches": 100,
        "active_batch_ratio": active_ratio,
    }


def _passing_branches():
    return {
        "positive": _passing_branch(15000, 600, 200, 0.95),
        "negative": _passing_branch(90000, 3000, 600, 1.0),
    }


def test_soft_minority_mass_and_minority_image_count():
    result = summarize_r2b_target_audit_branch_v2(
        [
            {
                "valid_pixels": 4,
                "teacher_preferred_pixels": 3,
                "dabe_preferred_pixels": 1,
                "soft_teacher_mass": 2.7,
            },
            {
                "valid_pixels": 6,
                "teacher_preferred_pixels": 5,
                "dabe_preferred_pixels": 1,
                "soft_teacher_mass": 4.2,
            },
            {
                "valid_pixels": 0,
                "teacher_preferred_pixels": 0,
                "dabe_preferred_pixels": 0,
                "soft_teacher_mass": 0.0,
            },
        ],
        active_batches=2,
        total_batches=3,
    )
    assert math.isclose(result["soft_teacher_mass"], 6.9)
    assert math.isclose(result["soft_dabe_mass"], 3.1)
    assert math.isclose(result["soft_minority_mass"], 3.1)
    assert result["hard_minority_class"] == "dabe"
    assert result["hard_minority_pixels"] == 2
    assert result["minority_valid_images"] == 2
    assert math.isclose(result["active_batch_ratio"], 2.0 / 3.0)


def test_top_image_concentration_uses_all_audited_images():
    observations = []
    minority_counts = [10] + [5] * 9 + [0] * 90
    for minority in minority_counts:
        observations.append(
            {
                "valid_pixels": minority + 100,
                "teacher_preferred_pixels": minority,
                "dabe_preferred_pixels": 100,
                "soft_teacher_mass": float(minority),
            }
        )
    result = summarize_r2b_target_audit_branch_v2(
        observations,
        active_batches=10,
        total_batches=10,
    )
    assert result["hard_minority_pixels"] == 55
    assert math.isclose(
        result["top_1_percent_images_minority_share"], 10.0 / 55.0
    )
    assert math.isclose(result["top_10_percent_images_minority_share"], 1.0)


def test_branch_specific_thresholds_are_independent():
    branches = _passing_branches()
    branches["positive"]["hard_minority_pixels"] = 499
    branches["positive"]["hard_minority_ratio"] = 499 / 15000
    result = evaluate_r2b_target_audit_v2_admission(
        branches,
        _cfg(),
        full_audit=True,
        processed_images=4040,
    )
    assert result["passed"] is False
    assert result["failures"] == ["positive:hard_minority_pixels"]
    assert all(
        condition["passed"]
        for condition in result["condition_results"]["negative"].values()
    )


def test_limited_audit_never_authorizes_training():
    result = evaluate_r2b_target_audit_v2_admission(
        _passing_branches(),
        _cfg(),
        full_audit=False,
        processed_images=5,
        diagnostic_only=True,
    )
    assert result["passed"] is False
    assert result["mini_run_authorized"] is False
    assert result["stage20_authorized"] is False
    assert "diagnostic_only" in result["failures"]
    assert "full_4040_sample_audit_required" in result["failures"]


def test_empty_branch_is_finite_and_fails_without_nan():
    empty = summarize_r2b_target_audit_branch_v2(
        [],
        active_batches=0,
        total_batches=0,
    )
    for value in empty.values():
        if isinstance(value, float):
            assert math.isfinite(value)
    branches = _passing_branches()
    branches["positive"] = empty
    result = evaluate_r2b_target_audit_v2_admission(
        branches,
        _cfg(),
        full_audit=True,
        processed_images=4040,
    )
    assert result["passed"] is False
    assert "positive:valid_pixels" in result["failures"]
    assert "positive:active_batch_ratio" in result["failures"]


def test_suggested_class_weights_are_clamped():
    missing = compute_r2b_audit_class_weights(10000, 0, 10000)
    assert missing["teacher"] == 4.0
    assert math.isclose(missing["dabe"], math.sqrt(0.5))
    balanced = compute_r2b_audit_class_weights(10000, 5000, 5000)
    assert balanced == {"teacher": 1.0, "dabe": 1.0}
    assert all(
        0.5 <= value <= 4.0
        for value in (*missing.values(), *balanced.values())
    )


def test_audit_v2_default_output_does_not_overlap_v1():
    from tools.audit_egsa_r2b_targets import DEFAULT_OUT, LEGACY_V1_OUT

    assert DEFAULT_OUT.resolve() != LEGACY_V1_OUT.resolve()


def test_concentration_override_allows_only_the_two_declared_failures():
    failures = [
        "positive:top_1_percent_images_minority_share",
        "positive:top_10_percent_images_minority_share",
    ]
    result = evaluate_r2b_audit_concentration_override(
        failures,
        failures,
        full_audit=True,
        processed_images=4040,
    )
    assert result["strict_pass"] is False
    assert result["override_passed"] is True
    assert result["unexpected_failures"] == []


def test_concentration_override_rejects_unexpected_or_limited_audit():
    allowed = [
        "positive:top_1_percent_images_minority_share",
        "positive:top_10_percent_images_minority_share",
    ]
    unexpected = evaluate_r2b_audit_concentration_override(
        [allowed[0], "negative:hard_minority_pixels"],
        allowed,
        full_audit=True,
        processed_images=4040,
    )
    assert unexpected["override_passed"] is False
    assert unexpected["unexpected_failures"] == [
        "negative:hard_minority_pixels"
    ]

    limited = evaluate_r2b_audit_concentration_override(
        allowed,
        allowed,
        full_audit=False,
        processed_images=5,
    )
    assert limited["override_passed"] is False
