from configs.dinov1_s8 import *  # noqa: F401,F403


EXP_NAME = "despl_paper_dinov1_s8"

DINO_MODEL_PATH = "../../workspace/weights/huggingface/facebook-dino-vits8"

DESPL_PAPER_CACHE_ROOT = "../datasets/cache/despl_paper_cache"

DESPL_GRID = 34
DESPL_K = 2
DESPL_LAMBDA_COLOR = 0.1
DESPL_COLOR_SIGMA = 0.1
DESPL_ENTROPY_BINS = 50
DESPL_LOWCONF_ALPHA = 0.1
DESPL_NUM_AUGS = 8

DESPL_AUGS = [
    "identity",
    "hflip",
    "vflip",
    "rot180",
    "bright_up",
    "bright_down",
    "contrast_up",
    "contrast_down",
]

BRIGHT_UP_FACTOR = 1.2
BRIGHT_DOWN_FACTOR = 0.8
CONTRAST_UP_FACTOR = 1.2
CONTRAST_DOWN_FACTOR = 0.8

DESPL_SIGN_MODE = "paper"
ALLOW_FIXED_SIGN_ABLATION = True
SAVE_CURRENT_DESPL_COMPARISON = True
AFFINITY_CLAMP_NONNEG = False
