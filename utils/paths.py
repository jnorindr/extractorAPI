import environ
from pathlib import Path

FILE = Path(__file__).resolve()
API_ROOT = FILE.parents[1]  # root directory

IMG_DIR = "img"
ANNO_DIR = "annotation"
YOLO_DIR = "yolov5"
MANIFESTS_DIR = "manifests"
LOG_DIR = "logs"

DEFAULT_MODEL = "yolo_last_sved_vhs_sullivan.pt"

IMG_PATH = Path(f"{API_ROOT}/{IMG_DIR}")
ANNO_PATH = Path(f"{API_ROOT}/{ANNO_DIR}")
MODEL_PATH = Path(f"{API_ROOT}/{YOLO_DIR}")
MANIFESTS_PATH = Path(f"{API_ROOT}/{MANIFESTS_DIR}")
LOG_PATH = Path(f"{API_ROOT}/{LOG_DIR}/api_logs.log")

ENV = environ.Env()
environ.Env.read_env(env_file=f"{API_ROOT}/.env")
