import cv2
import numpy as np

from app.app import celery

import os
import requests
import time
import shutil
from datetime import datetime, timedelta
from celery.schedules import crontab
from os.path import exists

from app.similarity.const import SCORES_PATH, FEAT_NET
from app.similarity.similarity import compute_seg_pairs
from app.similarity.utils import is_downloaded, download_images, doc_pairs, hash_pair
from app.utils import sanitize_str
from app.utils.paths import ENV, IMG_PATH, ANNO_PATH, MODEL_PATH, DEFAULT_MODEL, DATA_PATH, DATASETS_PATH, APP_LOG, \
    CELERY_LOG, CELERY_ERROR_LOG
from app.utils.logger import console
from app.iiif.iiif_downloader import IIIFDownloader
from app.yolov5.detect_vhs import run_vhs
from app.yolov5.detect import run as run_yolov5
from app.yolov5 import train


@celery.task
def delete_images():
    # Function to delete images after a week
    week_ago = datetime.now() - timedelta(days=7)
    # TODO delete images from similarity as well
    for ms_dir in os.listdir(IMG_PATH):
        dir_path = os.path.join(IMG_PATH, ms_dir)
        if os.path.isdir(dir_path):
            dir_modified_time = datetime.fromtimestamp(os.path.getmtime(dir_path))
            if dir_modified_time < week_ago:
                shutil.rmtree(dir_path, ignore_errors=False, onerror=None)


def empty_log(log_path:str, two_weeks_ago):
    line_nb = 0
    if os.path.exists(log_path):
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()

        for line_nb, line in enumerate(lines):
            try:
                log_date = datetime.strptime(line[1:11], "%Y-%m-%d")
                if log_date > two_weeks_ago:
                    break
            except ValueError:
                pass  # Ignore lines without a date

        with open(log_path, 'w') as log_file:
            log_file.writelines(lines[line_nb:])


@celery.task
def empty_logs():
    two_weeks_ago = datetime.now() - timedelta(weeks=2)
    for log_file in [APP_LOG, CELERY_LOG, CELERY_ERROR_LOG]:
        empty_log(log_file, two_weeks_ago)


@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Periodic task setting for image deletion
    sender.add_periodic_task(
        crontab(hour=2, minute=0),
        delete_images.s()
    )
    sender.add_periodic_task(
        crontab(hour=2, minute=0),
        empty_logs.s(),
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

    anno_dir = ANNO_PATH / anno_model / sanitize_str(manifest_url.split('/')[2])
    if not exists(anno_dir):
        # create all necessary parent directories
        os.makedirs(anno_dir)

    anno_file = f"{anno_dir}/{digit_ref}.txt"

    if exists(anno_file):
        # If annotations are generated again, empty annotation file
        open(anno_file, 'w').close()

    console(f"DETECTING VISUAL ELEMENTS FOR {manifest_url} 🕵️")
    digit_path = IMG_PATH / digit_dir

    # For number and images in the witness images directory, run detection
    for i, img in enumerate(sorted(os.listdir(digit_path)), 1):
        console(f"====> Processing {img} 🔍")
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
        console(f'An error occurred', error=e)


@celery.task
def test(model, dataset, save_dir):
    project = f"{DATASETS_PATH}/{dataset}/{save_dir}"
    name = "annotated_images_auto"
    annotated_img_dir = f"{project}/{name}/"
    annotations_dir = f"{DATASETS_PATH}/{dataset}/labels/test/"
    output_dir = f"{project}/comparative_images/"
    neg_output_dir = f"{project}/comparative_images/false_negatives"

    try:
        run_yolov5(
            weights=f"{MODEL_PATH}/{model}",
            source=f"{DATASETS_PATH}/{dataset}/images/test",
            project=project,
            name=name,
        )

    except Exception as e:
        console(f'An error occurred', error=e)

    try:
        if not os.path.exists(neg_output_dir):
            os.makedirs(neg_output_dir)

        for image_file in os.listdir(annotated_img_dir):
            if image_file.endswith(".jpg") or image_file.endswith(".JPG"):
                image_path = os.path.join(annotated_img_dir, image_file)
                img = cv2.imread(image_path)

                if img is None:
                    console(f"[test_model] Error: Failed to load image {image_path}", color="red")
                    continue

                annotation_file = image_file.replace(".jpg", ".txt").replace(".JPG", ".txt")
                annotation_path = os.path.join(annotations_dir, annotation_file)
                if not os.path.exists(annotation_path):
                    continue

                with open(annotation_path, "r") as f:
                    annotations = f.readlines()

                # Parse the annotations to extract the bounding box coordinates and class labels
                for annotation in annotations:
                    class_label, x, y, w, h = annotation.strip().split()
                    x, y, w, h = map(float, [x, y, w, h])

                    # Convert the normalized coordinates to pixel coordinates
                    x, y, w, h = x * img.shape[1], y * img.shape[0], w * img.shape[1], h * img.shape[0]

                    # Compute the bounding box coordinates
                    xmin, ymin, xmax, ymax = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

                    # Draw the bounding boxes on the image
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Add the class labels to the bounding boxes
                    cv2.putText(img, "ground truth", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

                # Save the annotated image to the output folder
                output_path = os.path.join(output_dir, image_file)
                cv2.imwrite(output_path, img)

        for image_file in os.listdir(annotated_img_dir):
            image_path = os.path.join(annotated_img_dir, image_file)
            annotation_file = image_file.replace(".jpg", ".txt").replace(".JPG", ".txt")
            annotation_path = os.path.join(annotations_dir, annotation_file)

            if os.path.exists(annotation_path) and image_file not in os.listdir(output_dir):
                img = cv2.imread(image_path)

                if img is None:
                    console(f"[test_model_false_neg] Error: Failed to load image {image_path}", color="red")
                    continue

                with open(annotation_path, "r") as f:
                    annotations = f.readlines()

                # Parse the annotations to extract the bounding box coordinates and class labels
                for annotation in annotations:
                    class_label, x, y, w, h = annotation.strip().split()
                    x, y, w, h = map(float, [x, y, w, h])

                    # Convert the normalized coordinates to pixel coordinates
                    x, y, w, h = x * img.shape[1], y * img.shape[0], w * img.shape[1], h * img.shape[0]

                    # Compute the bounding box coordinates
                    xmin, ymin, xmax, ymax = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

                    # Draw the bounding boxes on the image
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    # Add the class labels to the bounding boxes
                    cv2.putText(img, "ground truth", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

                # Save the annotated image to the output folder
                neg_output_path = os.path.join(neg_output_dir, image_file)
                cv2.imwrite(neg_output_path, img)

        return f"Annotations plotted on images and saved to {output_dir}"  #, no gt : {n}"

    except Exception as e:
        console(f'An error occurred', error=e)


@celery.task
def training(model, data, epochs):
    try:
        train.run(
            weights=f"{MODEL_PATH}/{model}",
            data=f"{DATA_PATH}/{data}.yaml",
            imgsz=320,
            epochs=int(epochs),
            name=f"{data}_{epochs}"
        )

        return f"Trained model {model} with {data} dataset."

    except Exception as e:
        console(f'An error occurred', error=e)


@celery.task
def similarity(documents, model=FEAT_NET, callback=None):
    """
    E.g.
    "documents": {
        "wit3_man186_anno181": "https://eida.obspm.fr/eida/wit3_man186_anno181/list/",
        "wit87_img87_anno87": "https://eida.obspm.fr/eida/wit87_img87_anno87/list/",
        "wit2_img2_anno2": "https://eida.obspm.fr/eida/wit2_img2_anno2/list/"
    }
    """
    if len(list(documents.keys())) == 0:
        console(f"[@celery.task.similarity] No documents to compare", color="red")

    console(f"[@celery.task.similarity] Similarity task triggered for {list(documents.keys())} with {model}!")

    doc_ids = []
    for doc_id, url in documents.items():
        console(f"[@celery.task.similarity] Processing {doc_id}...", color="cyan")
        doc_ids.append(doc_id)
        try:
            # TODO check first if features were computed + use of model
            if not is_downloaded(doc_id):
                console(f"[@celery.task.similarity] Downloading {doc_id} images...", color="cyan")
                download_images(url, doc_id)
        except Exception as e:
            console(f"[@celery.task.similarity] Unable to download images for {doc_id}", error=e)
            return

    for doc_pair in doc_pairs(doc_ids):
        hashed_pair = hash_pair(doc_pair)
        score_file = SCORES_PATH / f"{hashed_pair}.npy"
        if not os.path.exists(score_file):
            success = compute_seg_pairs(doc_pair, hashed_pair)
            if not success:
                console('[@celery.task.similarity] Error when computing scores', color="red")
                return

        npy_pairs = {}
        with open(score_file, 'rb') as file:
            npy_pairs["-".join(sorted(doc_pair))] = (f"{'-'.join(sorted(doc_pair))}.npy", file.read())

            try:
                if callback:
                    response = requests.post(
                        url=f"{callback}",
                        files=npy_pairs,
                    )
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                console(f'[@celery.task.similarity] Error in callback request for {doc_pair}', error=e)
            except Exception as e:
                console(f'[@celery.task.similarity] An error occurred for {doc_pair}', error=e)

    return console(f"[@celery.task.similarity] Successfully send scores for {doc_ids}", color="green")
