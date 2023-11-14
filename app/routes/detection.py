import os
import shutil
from flask import request, jsonify
from os.path import exists

from app.app import app
from app.utils.tasks import detect
from app.utils.security import key_required
from app.utils.paths import MANIFESTS_PATH, IMG_PATH, MODEL_PATH, DATA_PATH, DATASETS_PATH
from app.iiif.iiif_downloader import IIIFDownloader


@app.route("/run_detect", methods=['POST'])
@key_required
def run_detect():
    """
    To download images from a IIIF manifest and launch detection
    """
    # Get manifest URL from the request form
    manifest_url = request.form['manifest_url']
    model = request.form.get('model')

    # function.delay() is used to trigger function as celery task
    detect.delay(manifest_url, model)
    return f"Run detect task triggered with Celery! Check terminal to see the logs..."


@app.route('/detect_all', methods=['POST'])
@key_required
def detect_all():
    """
    To download images from a list of IIIF manifest and launch detection
    """
    # Get manifest URL file from the request
    url_file = request.files['url_file']
    model = request.form.get('model')
    callback_url = request.form.get('callback')

    if not exists(MANIFESTS_PATH):
        os.mkdir(MANIFESTS_PATH)

    # Save a copy of the file on the GPU
    url_file.save(f'{MANIFESTS_PATH}/{url_file.filename}')

    # Read file to list manifest URLs to be processed
    with open(f'{MANIFESTS_PATH}/{url_file.filename}', mode='r') as f:
        manifest_urls = f.read().splitlines()
    manifest_urls = list(filter(None, manifest_urls))

    for manifest_url in manifest_urls:
        detect.delay(manifest_url, model, callback_url)

    return 'Success'


@app.route('/delete_detect', methods=['POST'])
@key_required
def delete_detect():
    """
    To delete images for a witness and relaunch detection
    """
    # Get manifest URL from the request form
    manifest_url = request.form['manifest_url']
    model = request.form.get('model')
    callback_url = request.form.get('callback')

    manifest = IIIFDownloader(manifest_url)
    dir_path = os.path.join(IMG_PATH, IIIFDownloader.get_dir_name(manifest))

    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
        # function.delay() is used to trigger function as celery task
        detect.delay(manifest_url, model, callback_url)
        return f"Image deletion and detection task triggered with Celery! Check terminal to see the logs..."

    else:
        return f"No images to delete."


# TODO: create endpoint to get all available models
@app.route('/models', methods=['GET'])
def get_models():
    models_info = {}

    for filename in os.listdir(MODEL_PATH):
        if filename.endswith(".pt"):
            full_path = os.path.join(MODEL_PATH, filename)
            modification_date = os.path.getmtime(full_path)
            models_info[filename] = modification_date

    return jsonify(models_info)


import zipfile


@app.route('/send_dataset', methods=['POST'])
def send_dataset():
    img_zip = request.files['img_zip']
    anno_zip = request.files['anno_zip']
    yaml_file = request.files['yaml_file']
    dataset_name = request.form.get('dataset_name')
    action = request.form.get('action')

    imgs_dir = DATASETS_PATH / dataset_name / "images" / action
    anno_dir = DATASETS_PATH / dataset_name / "labels" / action
    yaml_file_path = DATA_PATH / f"{dataset_name}.yaml"

    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)

    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    with zipfile.ZipFile(img_zip, 'r') as imgs:
        imgs.extractall(imgs_dir)

    with zipfile.ZipFile(anno_zip, 'r') as annos:
        annos.extractall(anno_dir)

    with open(yaml_file_path, 'w') as yaml:
        yaml.write(yaml_file.read().decode('utf-8'))

    # YAML file
    return f"{dataset_name} sent for {action}"
