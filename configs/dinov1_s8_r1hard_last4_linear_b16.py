"""Decoder-isolation control: online DINOv1-S/8 Last4 + one 1x1 Conv."""

from configs.dinov1_s8_r1hard_last4_linear_online import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_r1hard_last4_linear_b16"
R1_DECODER_ISOLATION_V1 = True
R1_DECODER_DIRECT_R1_37 = True
R1_DECODER_REFERENCE_CONFIG = "configs/dinov1_s8_r1hard_last4_linear_online.py"
BATCH_SIZE = 16

