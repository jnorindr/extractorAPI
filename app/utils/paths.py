import os

import environ
from pathlib import Path

FILE = Path(__file__).resolve()
API_ROOT = FILE.parents[2]  # root directory

APP_DIR = "app"
MEDIA_DIR = "mediafiles"
IMG_DIR = "img"
ANNO_DIR = "annotation"
YOLO_DIR = "yolov5"
MANIFESTS_DIR = "manifests"
LOG_DIR = "logs"
DATA_DIR = "data"
DATASETS_DIR = "datasets"

DEFAULT_MODEL = "best_eida.pt"

def create_if_not(path):
    if not os.path.exists(path):
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(path):
            Path(path).touch(mode=0o666)
    return path

IMG_PATH = create_if_not(Path(f"{API_ROOT}/{APP_DIR}/{MEDIA_DIR}/{IMG_DIR}"))
ANNO_PATH = create_if_not(Path(f"{API_ROOT}/{APP_DIR}/{MEDIA_DIR}/{ANNO_DIR}"))
MANIFESTS_PATH = create_if_not(Path(f"{API_ROOT}/{APP_DIR}/{MEDIA_DIR}/{MANIFESTS_DIR}"))

MODEL_PATH = create_if_not(Path(f"{API_ROOT}/{APP_DIR}/{YOLO_DIR}"))
DATA_PATH = create_if_not(Path(f"{API_ROOT}/{APP_DIR}/{YOLO_DIR}/{DATA_DIR}"))
DATASETS_PATH = create_if_not(Path(f"{DATA_PATH}/{DATASETS_DIR}"))


LOG_PATH = create_if_not(Path(f"{API_ROOT}/{LOG_DIR}/api_logs.log"))
IMG_LOG = create_if_not(Path(f"{API_ROOT}/{LOG_DIR}/img.log"))

ENV = environ.Env()
environ.Env.read_env(env_file=f"{API_ROOT}/.env")
