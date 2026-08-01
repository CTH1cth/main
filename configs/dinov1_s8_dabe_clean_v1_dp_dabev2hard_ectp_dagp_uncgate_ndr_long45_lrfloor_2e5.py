"""DABE-v2-hard static supervision with Evidence-Constrained Teacher Projection."""

from configs.dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ectp_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

USE_ECTP = True
TEACHER_ROUTING_MODE = "ectp"
ECTP_VERSION = "ectp_v1_dual_evidence_overlap_projection"

# User-authorized temporary provenance override.  The formal 4040-sample audit
# remains strict and records FAIL; training logs the first >1e-5 mismatch and
# continues because the observed maximum soft drift is only ~1.052e-3.
ECTP_ALLOW_FOREGROUND_CACHE_DRIFT = True

USE_ECST = False
USE_ECST_MINIMAL = False
USE_ECST_CLEAN = False
DABE_CLEAN_USE_LEGACY_ECST_REGIONS = False

STOP_AFTER_EPOCH = 0
