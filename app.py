from flask import Flask, request
import os
import requests
from datetime import datetime, timedelta
from celery.schedules import crontab
import shutil

from os.path import exists

from utils.paths import ENV, ANNO_DIR, IMG_PATH, ANNO_PATH, MODEL_PATH, MANIFESTS_PATH, DEFAULT_MODEL
from utils.hosts import allow_hosts
from utils.logger import log
from utils.celery_utils import get_celery_app_instance

from iiif.iiif_downloader import IIIFDownloader
from yolov5.detect_vhs import run_vhs

app = Flask(__name__)
celery = get_celery_app_instance(app)


@celery.task
def detect(manifest_url, model):
    # Save images from IIIF manifest
    downloader = IIIFDownloader(manifest_url)
    downloader.run()

    model = DEFAULT_MODEL if model is None else model

    wit_id = downloader.get_dir_name()
    wit_type = 'manuscript' if 'manuscript' in manifest_url else 'volume'
    anno_id = downloader.manifest_id
    anno_model = model.split('.')[0]

    # Directory in which to save annotation files
    if not exists(ANNO_PATH):
        os.mkdir(ANNO_PATH)

    if not exists(ANNO_PATH / anno_model):
        os.mkdir(ANNO_PATH / anno_model)

    if not exists(ANNO_PATH / anno_model / wit_type):
        os.mkdir(ANNO_PATH / anno_model / wit_type)

    anno_file = f"{ANNO_DIR}/{anno_model}/{wit_type}/{anno_id}.txt"

    # If annotations are generated again, empty annotation file
    if exists(anno_file):
        open(anno_file, 'w').close()

    log(f"\n\n\x1b[38;5;226m\033[1mDETECTING VISUAL ELEMENTS FOR {wit_id} 🕵️\x1b[0m\n\n")
    wit_path = downloader.manifest_dir_path

    # For number and images in the witness images directory, run detection
    for i, img in enumerate(sorted(os.listdir(wit_path)), 1):
        log(f"\n\x1b[38;5;226m===> Processing {img} 🔍\x1b[0m\n")
        run_vhs(
            weights=f"{MODEL_PATH}/{model}",
            source=wit_path / img,
            anno_file=anno_file,
            img_nb=i
        )

    app_endpoint = f"{ENV.str('CLIENT_APP_URL')}/{wit_type}/{anno_id}/annotate/"
    with open(anno_file, 'r') as file:
        annotation_file = file.read()

    files = {"annotation_file": annotation_file}

    requests.post(url=app_endpoint, files=files)

    return 'Success'


@celery.task
def delete_images():
    week_ago = datetime.now() - timedelta(days=7)
    for ms_dir in os.listdir(IMG_PATH):
        dir_path = os.path.join(IMG_PATH, ms_dir)
        if os.path.isdir(dir_path):
            dir_modified_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
            if dir_modified_time < week_ago:
                shutil.rmtree(dir_path, ignore_errors=False, onerror=None)


@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(
        crontab(hour=2, minute=0),
        delete_images.s()
    )


@app.route('/detect_all', methods=['POST'])
# @allow_hosts
def detect_all():
    # Get manifest URL file from the request
    url_file = request.files['url_file']
    model = request.form.get('model')

    if not exists(MANIFESTS_PATH):
        os.mkdir(MANIFESTS_PATH)

    # Save a copy of the file on the GPU
    url_file.save(f'{MANIFESTS_PATH}/{url_file.filename}')

    # Read file to list manifest URLs to be processed
    with open(f'{MANIFESTS_PATH}/{url_file.filename}', mode='r') as f:
        manifest_urls = f.read().splitlines()
    manifest_urls = list(filter(None, manifest_urls))

    for manifest_url in manifest_urls:
        detect.delay(manifest_url, model)

    return 'Success'


@app.route("/run_detect", methods=['POST'])
# @allow_hosts
def run_detect():
    # Get manifest URL from the request form
    manifest_url = request.form['manifest_url']
    model = request.form.get('model')

    # function.delay() is used to trigger function as celery task
    detect.delay(manifest_url, model)
    return f"Run detect task triggered with Celery! Check terminal to see the logs..."


if __name__ == '__main__':
    app.run()
