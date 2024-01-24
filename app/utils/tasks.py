import cv2

from app.app import celery

import os
import requests
import time
import shutil
from datetime import datetime, timedelta
from celery.schedules import crontab
from os.path import exists

from app.utils.paths import ENV, IMG_PATH, ANNO_PATH, MODEL_PATH, DEFAULT_MODEL, DATA_PATH, DATASETS_PATH
from app.utils.logger import log
from app.iiif.iiif_downloader import IIIFDownloader
from app.yolov5.detect_vhs import run_vhs
from app.yolov5.detect import run as run_yolov5
from app.yolov5 import train


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
def test(model, dataset, save_dir):
    project = f"{DATASETS_PATH}/{dataset}/{save_dir}"
    name = "annotated_images_auto"
    annotated_img_dir = f"{project}/{name}/"
    annotations_dir = f"{DATASETS_PATH}/{dataset}/labels/test/"
    images_dir = f"{DATASETS_PATH}/{dataset}/images/test/"
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
        return f'An error occurred: {e}'

    try:
        if not os.path.exists(neg_output_dir):
            os.makedirs(neg_output_dir)

        for image_file in os.listdir(annotated_img_dir):
            if image_file.endswith(".jpg") or image_file.endswith(".JPG"):
                image_path = os.path.join(annotated_img_dir, image_file)
                img = cv2.imread(image_path)

                if img is None:
                    log("[test_model] Error: Failed to load image", image_path)
                    continue

                annotation_file = image_file.replace(".jpg", ".txt").replace(".JPG", ".txt")
                annotation_path = os.path.join(annotations_dir, annotation_file)
                if not os.path.exists(annotation_path):
                    # log("[test_model] Error: Failed to load image", image_path)
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

        for image_file in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_file)
            annotation_file = image_file.replace(".jpg", ".txt").replace(".JPG", ".txt")
            annotation_path = os.path.join(annotations_dir, annotation_file)

            if image_file not in os.listdir(output_dir) and os.path.exists(annotation_path):
                img = cv2.imread(image_path)

                if img is None:
                    log("[test_model_false_neg] Error: Failed to load image", image_path)
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
                    cv2.putText(img, "ground truth", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                1)

                # Save the annotated image to the output folder
                neg_output_path = os.path.join(neg_output_dir, image_file)
                cv2.imwrite(neg_output_path, img)

        return f"Annotations plotted on images and saved to {output_dir}"

    except Exception as e:
        return f'An error occurred: {e}'


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
        return f'An error occurred: {e}'
