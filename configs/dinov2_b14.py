EXP_NAME = "dinov2_b14_clean"
BACKBONE_KEY = "dinov2-b14"

DATA_ROOT = "/home/dell01/CTH/MY-baseline/datasets/COD"
CACHE_ROOT = "/home/dell01/CTH/MY-baseline/datasets/cache"
WORK_ROOT = "/home/dell01/CTH/MY-baseline/workdir"

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

DINO_CONFIGS = {
    "dinov1-s8": {
        "model_name": "facebook/dino-vits8",
        "model_path": "/home/dell01/CTH/workspace/weights/huggingface/facebook-dino-vits8",
        "patch_size": 8,
        "pseudo_input_size": 224,
        "feature_input_size": 296,
        "bkg_th": 0.3,
        "lr": 6e-4,
    },
    "dinov1-b8": {
        "model_name": "facebook/dino-vitb8",
        "model_path": "/home/dell01/CTH/workspace/weights/huggingface/facebook-dino-vitb8",
        "patch_size": 8,
        "pseudo_input_size": 224,
        "feature_input_size": 296,
        "bkg_th": 0.3,
        "lr": 6e-4,
    },
    "dinov2-b14": {
        "model_name": "facebook/dinov2-base",
        "model_path": "/home/dell01/CTH/workspace/weights/huggingface/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "patch_size": 14,
        "pseudo_input_size": 224,
        "feature_input_size": 518,
        "bkg_th": 0.6,
        "lr": 2e-4,
    },
}

DINO = DINO_CONFIGS[BACKBONE_KEY]
