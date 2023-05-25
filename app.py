from flask import Flask, request
import os
from pathlib import Path

from os.path import exists

from utils.paths import ANNO_DIR, IMG_PATH, ANNO_PATH, MODEL_PATH, MANIFESTS_PATH
from utils.hosts import allow_hosts
from utils.logger import log
from utils.celery_utils import get_celery_app_instance

from iiif.iiif_downloader import IIIFDownloader
from yolov5.detect_vhs import run_vhs

app = Flask(__name__)
celery = get_celery_app_instance(app)


@celery.task
def detect(manifest_url):
    # Save images from IIIF manifest
    downloader = IIIFDownloader(manifest_url)
    downloader.run()

    wit_id = downloader.get_dir_name()
    anno_id = downloader.manifest_id.replace("ms", "").replace("-", "")

    # # Directory in which to save annotation files
    # if not exists(ANNO_PATH):
    #     os.mkdir(ANNO_PATH)
    #
    # # If annotations are generated again, empty annotation file
    # if exists(f"{ANNO_DIR}/{anno_id}.txt"):
    #     open(f"{ANNO_DIR}/{anno_id}.txt", 'w').close()
    #
    # log(f"\n\n\x1b[38;5;226m\033[1mDETECTING VISUAL ELEMENTS FOR {wit_id} 🕵️\x1b[0m\n\n")
    # wit_path = downloader.manifest_dir_path
    #
    # # For number and images in the witness images directory, run detection
    # for i, img in enumerate(sorted(os.listdir(wit_path)), 1):
    #     log(f"\n\x1b[38;5;226m===> Processing {img} 🔍\x1b[0m\n")
    #     run_vhs(weights=MODEL_PATH, source=wit_path / img, anno_file=f"{ANNO_DIR}/{anno_id}.txt", img_nb=i)

    return 'Success'


@app.route('/detect_all', methods=['POST'])
@allow_hosts
def detect_all():
    # Get manifest URL file from the request
    url_file = request.files['url_file']

    if not exists(MANIFESTS_PATH):
        os.mkdir(MANIFESTS_PATH)

    # Save a copy of the file on the GPU
    url_file.save(f'{MANIFESTS_PATH}/{url_file.filename}')

    # Read file to list manifest URLs to be processed
    with open(f'{MANIFESTS_PATH}/{url_file.filename}', mode='r') as f:
        manifest_urls = f.read().splitlines()
    manifest_urls = list(filter(None, manifest_urls))

    for manifest_url in manifest_urls:
        detect.delay(manifest_url)

    return 'Success'


@app.route("/run_detect", methods=['POST'])
@allow_hosts
def run_detect():
    # Get manifest URL from the request form
    manifest_url = request.form['manifest_url']
    # function.delay() is used to trigger function as celery task
    detect.delay(manifest_url)
    return f"Run detect task triggered with Celery! Check terminal to see the logs..."


if __name__ == '__main__':
    app.run()
