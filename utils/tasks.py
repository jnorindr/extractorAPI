from app import celery

import os
import requests
import shutil
from datetime import datetime, timedelta
from celery.schedules import crontab
from os.path import exists

from utils.paths import ENV, ANNO_DIR, IMG_PATH, ANNO_PATH, MODEL_PATH, DEFAULT_MODEL
from utils.logger import log
from iiif.iiif_downloader import IIIFDownloader
from yolov5.detect_vhs import run_vhs


@celery.task
def delete_images():
    # Function to delete images after a week
    week_ago = datetime.now() - timedelta(days=7)
    for ms_dir in os.listdir(IMG_PATH):
        dir_path = os.path.join(IMG_PATH, ms_dir)
        if os.path.isdir(dir_path):
            dir_modified_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
            if dir_modified_time < week_ago:
                shutil.rmtree(dir_path, ignore_errors=False, onerror=None)


@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Periodic task setting for image deletion
    sender.add_periodic_task(
        crontab(hour=2, minute=0),
        delete_images.s()
    )


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

    # TODO : renvoyer directement à l'URL qui envoie la requête
    try:
        app_endpoint = f"{ENV.str('CLIENT_APP_URL')}/{wit_type}/{anno_id}/annotate/"
        with open(anno_file, 'r') as file:
            annotation_file = file.read()

        files = {"annotation_file": annotation_file}

        requests.post(url=app_endpoint, files=files)

        return 'Annotations sent to application'
    except Exception as e:
        return f'An error occurred: {e}'