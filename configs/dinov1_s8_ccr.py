EXP_NAME = "dinov1_s8_ccr"
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

USE_QRA = False
USE_CCR = True
CCR_CACHE_ROOT = "../datasets/cache/ccr_pseudo_cache"
CCR_SOURCE_QRA_ROOT = "../datasets/cache/qra_pseudo_cache"
CCR_USE_QRA_DESPL = True

CCR_LOSS_SIZE = 68
CCR_FIXED_BIN_TH = 0.5
CCR_DESPL_BIN_TH = 0.5

CCR_EXPAND_MAX_DIST = 2
CCR_SHRINK_MAX_DIST = 2

CCR_EXPAND_MAX_RATIO = 0.40
CCR_SHRINK_MAX_RATIO = 0.40
CCR_MIN_REMAIN_RATIO = 0.60

CCR_PROTO_SIM_EXPAND_TH = 0.50
CCR_PROTO_SIM_SHRINK_MARGIN = 0.05

CCR_EDGE_BLOCK_TH = 0.25
CCR_CORE_ERODE_ITER = 1

CCR_CORR_KEEP_FIXED = 0.20

CCR_TH_FG = 0.65
CCR_TH_BG = 0.25

CCR_LATE_OVERRIDE = True
CCR_LATE_RHO_Q2 = 0.15
CCR_LATE_RHO_Q1 = 0.08
CCR_LATE_RHO_Q0 = 0.00

CCR_USE_ANCHOR_LOSS = True
CCR_LAMBDA_ANCHOR_Q2 = 0.30
CCR_LAMBDA_ANCHOR_Q1 = 0.15
CCR_LAMBDA_ANCHOR_Q0 = 0.00

HEAD_TYPE = "context_residual"
CONTEXT_HEAD_HIDDEN = 64
CONTEXT_HEAD_GAMMA_INIT = 0.0

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
