import environ
from pathlib import Path

FILE = Path(__file__).resolve()
API_ROOT = FILE.parents[1]  # root directory

IMG_DIR = "img"
ANNO_DIR = "annotation"
YOLO_DIR = "yolov5"
MANIFESTS_DIR = "manifests"
LOG_DIR = "logs"

IMG_PATH = f"{API_ROOT}/{IMG_DIR}"
ANNO_PATH = f"{API_ROOT}/{ANNO_DIR}"
MODEL_PATH = f"{API_ROOT}/{YOLO_DIR}/best.pt"
MANIFESTS_PATH = f"{API_ROOT}/{MANIFESTS_DIR}"
LOG_PATH = f"{API_ROOT}/{LOG_DIR}/api_logs.log"

ENV = environ.Env()
environ.Env.read_env(env_file=f"{API_ROOT}/.env")
