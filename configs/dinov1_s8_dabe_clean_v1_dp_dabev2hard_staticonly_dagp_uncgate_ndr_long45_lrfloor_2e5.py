"""DABE-v2 Hard static-only Long45 ablation.

This configuration inherits the strict no-ECST control and changes only the
supervision contract: DABE-v2 Hard remains the sole effective target for all
45 epochs, while every Teacher-loss endpoint is fixed to zero.
"""

from configs.dinov1_s8_dabe_clean_v1_dp_dabev2hard_noecst_dagp_uncgate_ndr_long45_lrfloor_2e5 import *  # noqa: F401,F403


EXP_NAME = (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_staticonly_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
)

# Dedicated runtime/audit selector.  The training path must interpret this as
# an exact static-only contract rather than the inherited static-to-Teacher
# handover protocol.
DABEV2HARD_STATIC_ONLY = True

# Explicit endpoints make the intended effective supervision auditable even
# though the parent experiment normally hands supervision to the Teacher.
DABE_PU_DESPL_STATIC_START = 1.0
DABE_PU_DESPL_STATIC_END = 1.0
DABE_PU_DESPL_TEACHER_START = 0.0
DABE_PU_DESPL_TEACHER_END = 0.0

# No Teacher loss is authorized in this ablation.  The EMA shadow may remain
# available for protocol-compatible diagnostics, but cannot supervise Student.
USE_TEACHER_BINARY_FULL_LOSS = False
USE_TEACHER_SOFT_FULL_LOSS = False
USE_TEACHER_CONF_LOSS = False
