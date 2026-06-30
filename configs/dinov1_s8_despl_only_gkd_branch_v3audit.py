from configs.dinov1_s8_despl_only_gkd_branch_v3conservative import *  # noqa: F401,F403


EXP_NAME = "dinov1_s8_despl_only_gkd_branch_v3audit"

# Audit only: compute v3 grading and diagnostics, but backpropagate plain BCE.
GKD_MODE = "audit"
USE_GKD_LITE = False
