import os
import zipfile
from flask import request, jsonify
# from os.path import exists

from app.app import app
from app.utils.security import key_required
from app.utils.paths import DATA_PATH, DATASETS_PATH


@app.route('/send-dataset', methods=['POST'])
# @key_required
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

    return f"{dataset_name} sent for {action}"
