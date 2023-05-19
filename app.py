from flask import Flask, request, jsonify
import os
from pathlib import Path

from os.path import exists

from utils.path import ANNO_DIR, IMG_DIR, IMG_PATH, ANNO_PATH, MODEL_PATH
from utils.hosts import allow_hosts
from yolov5.utils.general import LOGGER
from yolov5.iiif_downloader.src.iiif_downloader import IIIFDownloader
from yolov5.detect_vhs import run_vhs

app = Flask(__name__)


@app.route('/run_detect', methods=['POST'])
@allow_hosts
def run_detect():
    # Get manifest URL from the request form
    manifest_url = request.form['manifest_url']
    # Path to model for inference

    # Save images from IIIF manifest
    downloader = IIIFDownloader(manifest_url, img_dir=IMG_DIR)
    downloader.run()

    # Directory in which to save annotation files
    if not exists(ANNO_PATH):
        os.mkdir(ANNO_PATH)

    # From img directory create witness id
    for wit_dir in os.listdir(f'{IMG_DIR}'):
        wit_id = wit_dir.replace("ms-", "")
        # If annotations are generated again, empty annotation file
        if exists(f"{ANNO_DIR}/{wit_id}.txt"):
            open(f"{ANNO_DIR}/{wit_id}.txt", 'w').close()
        LOGGER.info(f"\n\n\x1b[38;5;226m\033[1mDETECTING VISUAL ELEMENTS FOR {wit_id} 🕵️\x1b[0m\n\n")
        wit_path = f'{IMG_PATH}/{wit_dir}'
        # For number and images in the witness images directory, run detection
        for i, img in enumerate(sorted(os.listdir(wit_path)), 1):
            LOGGER.info(f"\n\x1b[38;5;226m===> Processing {img} 🔍\x1b[0m\n")
            run_vhs(weights=MODEL_PATH, source=Path(f"{wit_path}/{img}"), anno_file=f"{ANNO_DIR}/{wit_id}.txt", img_nb=i)

    return jsonify({'success': True})


if __name__ == '__main__':
    app.run()
