import os

import environ
from pathlib import Path

from app.utils import create_dirs_if_not, create_files_if_not

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

IMG_PATH = Path(f"{API_ROOT}/{APP_DIR}/{MEDIA_DIR}/{IMG_DIR}")
ANNO_PATH = Path(f"{API_ROOT}/{APP_DIR}/{MEDIA_DIR}/{ANNO_DIR}")
MANIFESTS_PATH = Path(f"{API_ROOT}/{APP_DIR}/{MEDIA_DIR}/{MANIFESTS_DIR}")

MODEL_PATH = Path(f"{API_ROOT}/{APP_DIR}/{YOLO_DIR}")
DATA_PATH = Path(f"{API_ROOT}/{APP_DIR}/{YOLO_DIR}/{DATA_DIR}")
DATASETS_PATH = Path(f"{DATA_PATH}/{DATASETS_DIR}")
LOG_PATH = Path(f"{API_ROOT}/{LOG_DIR}")

create_dirs_if_not([IMG_PATH, ANNO_PATH, MANIFESTS_PATH, MODEL_PATH, DATA_PATH, DATASETS_PATH, LOG_PATH])

APP_LOG = Path(f"{LOG_PATH}/api_logs.log")
IMG_LOG = Path(f"{LOG_PATH}/img.log")
CELERY_LOG = Path(f"{LOG_PATH}/celery.log")
CELERY_ERROR_LOG = Path(f"{LOG_PATH}/celery_error.log")

create_files_if_not([APP_LOG, IMG_LOG])

ENV = environ.Env()
environ.Env.read_env(env_file=f"{API_ROOT}/.env")
