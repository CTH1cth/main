EXP_NAME = "dinov1_s8_qra"
BACKBONE_KEY = "dinov1-s8"

DATA_ROOT = "/home/dell01/CTH/MY-baseline/datasets/COD"
CACHE_ROOT = "../datasets/cache"
WORK_ROOT = "../workdir"

TRAIN_DATASETS = ["TR-CAMO", "TR-COD10K"]
VAL_DATASETS = ["TE-CAMO"]
TEST_DATASETS = ["CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K"]

MAX_EPOCH = 25
BATCH_SIZE = 16
VAL_BATCH_SIZE = 1
NUM_WORKERS = 4
SEED = 42

SAVE_INTERVAL = 5
BEST_DATASET = "TE-CAMO"
BEST_METRIC = "M"
BEST_MODE = "min"

LOSS_SIZE = 68
EMA_WEIGHT = 0.99
THRESHOLD = 0.5

USE_QRA = True
QRA_CACHE_ROOT = "../datasets/cache/qra_pseudo_cache"
QRA_REPLACE_FIXED = True

QRA_DESPL_GRID = 28
QRA_LAMBDA_COLOR = 0.1
QRA_COLOR_SIGMA = 0.1
QRA_EIG_BINS = 50
QRA_LOWCONF_ALPHA = 0.1
QRA_AUGS = ["identity", "hflip", "vflip", "rot180"]

QRA_TH_FG = 0.65
QRA_TH_BG = 0.25

QRA_Q2_SIM = 0.75
QRA_Q1_SIM = 0.50

QRA_Q2_AREA_MIN = 0.01
QRA_Q2_AREA_MAX = 0.45
QRA_Q1_AREA_MIN = 0.003
QRA_Q1_AREA_MAX = 0.60

QRA_Q2_MAX_CC = 3
QRA_Q1_MAX_CC = 8
QRA_Q2_MAX_EDGE_TOUCH = 2

QRA_LAMBDA_ANCHOR_Q2 = 0.5
QRA_LAMBDA_ANCHOR_Q1 = 0.3
QRA_LAMBDA_ANCHOR_Q0 = 0.05

QRA_LAMBDA_SOFT_Q2 = 0.1
QRA_LAMBDA_SOFT_Q1 = 0.03
QRA_LAMBDA_SOFT_Q0 = 0.0

QRA_LAMBDA_LATE_ANCHOR = 0.05

QRA_FIXED_BLEND_Q2 = 1.0
QRA_FIXED_BLEND_Q1 = 0.5
QRA_FIXED_BLEND_Q0 = 0.0

DINO_CONFIGS = {
    "dinov1-s8": {
        "model_name": "facebook/dino-vits8",
        "model_path": "../../workspace/weights/huggingface/facebook-dino-vits8",
        "patch_size": 8,
        "pseudo_input_size": 224,
        "feature_input_size": 296,
        "bkg_th": 0.3,
        "lr": 6e-4,
    },
    "dinov1-b8": {
        "model_name": "facebook/dino-vitb8",
        "model_path": "../../workspace/weights/huggingface/facebook-dino-vitb8",
        "patch_size": 8,
        "pseudo_input_size": 224,
        "feature_input_size": 296,
        "bkg_th": 0.3,
        "lr": 6e-4,
    },
    "dinov2-b14": {
        "model_name": "facebook/dinov2-base",
        "model_path": "../../workspace/weights/huggingface/facebook-dinov2-vitb14",
        "patch_size": 14,
        "pseudo_input_size": 224,
        "feature_input_size": 518,
        "bkg_th": 0.6,
        "lr": 2e-4,
    },
}

DINO = DINO_CONFIGS[BACKBONE_KEY]
