from pathlib import Path

FILE = Path(__file__).resolve()
API_ROOT = FILE.parents[1]  # root directory

IMG_DIR = "img"
ANNO_DIR = "annotation"
YOLO_DIR = "yolov5"

IMG_PATH = f"{API_ROOT}/{IMG_DIR}"
ANNO_PATH = f"{API_ROOT}/{ANNO_DIR}"
MODEL_PATH = f"{API_ROOT}/{YOLO_DIR}/best.pt"