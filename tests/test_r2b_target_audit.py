from types import SimpleNamespace

from common.source_arbiter import evaluate_r2b_target_audit_admission


def _cfg():
    return SimpleNamespace(
        SOURCE_ARBITER_UTILITY_MIN_BRANCH_PIXELS=10000,
        SOURCE_ARBITER_UTILITY_MIN_CLASS_PIXELS=1000,
        SOURCE_ARBITER_UTILITY_MIN_CLASS_RATIO=0.01,
        SOURCE_ARBITER_UTILITY_NEG_DABE_MIN_RATIO=0.02,
    )


def _passing_branches():
    return {
        "positive": {
            "valid_pixels": 20000,
            "teacher_preferred_count": 18000,
            "dabe_preferred_count": 2000,
        },
        "negative": {
            "valid_pixels": 25000,
            "teacher_preferred_count": 23500,
            "dabe_preferred_count": 1500,
        },
    }


def test_full_audit_admission_passes_fixed_thresholds():
    result = evaluate_r2b_target_audit_admission(
        _passing_branches(),
        _cfg(),
        full_audit=True,
        processed_images=4040,
    )
    assert result["passed"] is True
    assert result["failures"] == []
    assert (
        result["normalized_branches"]["negative"]["dabe_preferred_ratio"]
        == 0.06
    )


def test_partial_or_negative_branch_collapse_is_rejected():
    branches = _passing_branches()
    branches["negative"]["dabe_preferred_count"] = 100
    result = evaluate_r2b_target_audit_admission(
        branches,
        _cfg(),
        full_audit=False,
        processed_images=32,
    )
    assert result["passed"] is False
    assert "negative:minority_count" in result["failures"]
    assert "negative:dabe_preferred_ratio" in result["failures"]
    assert "full_4040_sample_audit_required" in result["failures"]
    assert "processed_images:32!=4040" in result["failures"]
