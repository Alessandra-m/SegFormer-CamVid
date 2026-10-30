from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


DATA_ROOT = Path("/home/magic/work_task/data/CamVid")
SPLIT_DIRS = {
    "train": ("train", "trainannot"),
    "val":   ("val",   "valannot"),
    "test":  ("test",  "testannot"),
}
CLASS_DICT_CSV = DATA_ROOT / "class_palette.csv"
LABEL_SUFFIX = "_L"   
MASK_MODE = "rgb"         
IGNORE_CLASS_NAME = "unlabelled"

IGNORE_INDEX = 255


CHECKPOINT = "nvidia/mit-b0"

IMAGE_SIZE = 512

LEARNING_RATE = 6e-5
NUM_EPOCHS = 50
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
NUM_WORKERS = 4
SEED = 42

OUTPUT_DIR = PROJECT_DIR / "outputs" / "segformer-b0-camvid"

COLOR_JITTER = dict(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
HFLIP_PROB = 0.5

N_BEST = 3
N_WORST = 3
VIS_DIR = PROJECT_DIR / "outputs" / "predictions"   