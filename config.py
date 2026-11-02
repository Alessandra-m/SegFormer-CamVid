import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_ROOT = Path("/home/magic/work_task/data/CamVid")
SPLIT_DIRS = {
    "train": ("train", "trainannot"),
    "val": ("val", "valannot"),
    "test": ("test", "testannot"),
}
CLASS_DICT_CSV = DATA_ROOT / "class_palette.csv"
LABEL_SUFFIX = "_L"
MASK_MODE = "rgb"
IGNORE_CLASS_NAME = "unlabelled"
IGNORE_INDEX = 255
NATIVE_SIZE = (360, 480)

CHECKPOINT = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
HEAD_INIT_FROM = {
    "Sky": "sky",
    "Building": "building",
    "Pole": "pole",
    "Road": "road",
    "Sidewalk": "sidewalk",
    "Tree": "vegetation",
    "SignSymbol": "traffic sign",
    "Fence": "fence",
    "Car": "car",
    "Pedestrian": "person",
    "Bicyclist": "rider",
}

SHORT_SIDE = 360
EVAL_SIZE = (SHORT_SIDE, round(SHORT_SIDE * NATIVE_SIZE[1] / NATIVE_SIZE[0]))
CROP_SIZE = 360
SCALE_RANGE = (0.5, 2.0)
COLOR_JITTER = dict(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
HFLIP_PROB = 0.5

LEARNING_RATE = 6e-5
HEAD_LR_MULT = 10
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 300
LOSS = "ce_lovasz"
NUM_EPOCHS = 50
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
NUM_WORKERS = 4
FP16 = True
SEED = 42

RUN_NAME = "b0_cityscapes_360_e50"
TRAIN_AUG = "scale_crop"

EXPERIMENT = os.environ.get("EXP", "main")
if EXPERIMENT == "old":
    CHECKPOINT = "nvidia/mit-b0"
    HEAD_INIT_FROM = {}
    EVAL_SIZE = (512, 512)
    TRAIN_AUG = "resize"
    LOSS = "ce"
    HEAD_LR_MULT = 1
    WARMUP_STEPS = 0
    RUN_NAME = "b0_mitb0_old"
elif EXPERIMENT == "hires":
    SHORT_SIDE = 512
    CROP_SIZE = 512
    EVAL_SIZE = (SHORT_SIDE, round(SHORT_SIDE * NATIVE_SIZE[1] / NATIVE_SIZE[0]))
    RUN_NAME = "b0_cityscapes_512_e50"
elif EXPERIMENT != "main":
    raise ValueError(f"Unknown EXP={EXPERIMENT!r}, expected main | hires | old")

OUTPUTS_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR = OUTPUTS_DIR / RUN_NAME
N_BEST = 3
N_WORST = 3
