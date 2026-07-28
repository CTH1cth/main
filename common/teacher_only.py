"""Shared identifiers for the no-offline-pseudo Teacher-only ablation."""


TEACHER_ONLY_NO_OFFLINE_PSEUDO_SOURCE = "teacher_only_no_offline_pseudo"


def teacher_only_no_offline_pseudo_enabled(cfg):
    return bool(getattr(cfg, "USE_DABE_CLEAN", False)) and str(
        getattr(cfg, "DABE_CLEAN_TRAINING_TARGET_SOURCE", "")
    ).strip().lower() == TEACHER_ONLY_NO_OFFLINE_PSEUDO_SOURCE
