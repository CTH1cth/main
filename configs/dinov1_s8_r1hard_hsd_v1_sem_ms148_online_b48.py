"""Batch-48 throughput probe for the formal online R1-HSD semantic model."""

from configs.dinov1_s8_r1hard_hsd_v1_sem_ms148_online import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_hsd_v1_sem_ms148_online_b48"
BATCH_SIZE = 48

# This flag makes the protocol deviation explicit in the startup audit.  The
# optimizer and iteration-based StepLR are otherwise intentionally unchanged.
R1_BATCH_SPEED_PROBE = True
R1_REFERENCE_BATCH_SIZE = 16

