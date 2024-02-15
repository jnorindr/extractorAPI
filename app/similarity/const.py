from pathlib import Path

from app.utils import create_dirs_if_not

# Absolute path to root dir (similarity/)
SIM_DIR = Path(__file__).resolve().parent
DOC_PATH = SIM_DIR / "documents"
MODEL_PATH = SIM_DIR / "models"
SCORES_PATH = SIM_DIR / "scores"
FEATS_PATH = SIM_DIR / "feats"

create_dirs_if_not([DOC_PATH, MODEL_PATH, SCORES_PATH, FEATS_PATH])

MAX_SIZE = 244
MAX_RES = 500

FEAT_NET = 'moco_v2_800ep_pretrain'
FEAT_SET = 'imagenet'
FEAT_LAYER = 'conv4'
COS_TOPK = 20
SEG_TOPK = 10
SEG_STRIDE = 16
FINAL_TOPK = 25
