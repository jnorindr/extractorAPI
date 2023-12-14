from app.app import celery

import os
import requests
import time
import shutil
from datetime import datetime, timedelta
from celery.schedules import crontab
from os.path import exists

from app.utils.paths import ENV, IMG_PATH, ANNO_PATH, MODEL_PATH, DEFAULT_MODEL, DATA_PATH
from app.utils.logger import log
from app.iiif.iiif_downloader import IIIFDownloader
from app.yolov5.detect_vhs import run_vhs
from app.yolov5 import val, train


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
def detect(manifest_url, model=None, callback=None):
    # Save images from IIIF manifest
    downloader = IIIFDownloader(manifest_url)
    downloader.run()

    model = DEFAULT_MODEL if model is None else model
    weights = f"{MODEL_PATH}/{model}"

    digit_dir = downloader.get_dir_name()
    digit_ref = downloader.manifest_id  # TODO check if it really "{wit_abbr}{wit_id}_{digit_abbr}{digit_id}
    anno_model = model.split('.')[0]

    anno_dir = ANNO_PATH / anno_model
    if not exists(anno_dir):
        # create all necessary parent directories
        os.makedirs(anno_dir)

    anno_file = f"{anno_dir}/{digit_ref}.txt"

    if exists(anno_file):
        # If annotations are generated again, empty annotation file
        open(anno_file, 'w').close()

    log(f"\n\n\x1b[38;5;226m\033[1mDETECTING VISUAL ELEMENTS FOR {manifest_url} 🕵️\x1b[0m\n\n")
    digit_path = IMG_PATH / digit_dir

    # For number and images in the witness images directory, run detection
    for i, img in enumerate(sorted(os.listdir(digit_path)), 1):
        log(f"\n\x1b[38;5;226m===> Processing {img} 🔍\x1b[0m\n")
        run_vhs(
            weights=weights,
            source=digit_path / img,
            anno_file=anno_file,
            img_nb=i
        )

    try:
        with open(anno_file, 'r') as file:
            annotation_file = file.read()

        requests.post(
            url=f"{callback}/{digit_ref}" if callback else f"{ENV.str('CLIENT_APP_URL')}/annotate/{digit_ref}",
            files={"annotation_file": annotation_file},
            data={"model": f"{anno_model}_{time.strftime('%m_%Y', time.gmtime(os.path.getmtime(weights)))}"}
        )

        return f"Annotations from {anno_model} sent to {callback}"
    except Exception as e:
        return f'An error occurred: {e}'


@celery.task
def validate(model, data, name):
    try:
        val.run(
            weights=f"{MODEL_PATH}/{model}",
            data=f"{DATA_PATH}/{data}.yaml",
            name=f"{name}",
            task='test'
        )

        return f"Validated model {model} with {data} dataset."

    except Exception as e:
        return f'An error occurred: {e}'


@celery.task
def training(model, data, hyp):
    if hyp == 'high':
        param = f"{DATA_PATH}/hyps/hyp.scratch-high.yaml"
    else:
        None

    try:
        train.run(
            weights=f"{MODEL_PATH}/{model}",
            data=f"{DATA_PATH}/{data}.yaml",
            imgsz=320,
            hyp=param,
        )

        return f"Trained model {model} with {data} dataset."

    except Exception as e:
        return f'An error occurred: {e}'
