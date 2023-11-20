import os
import time
import zipfile
from flask import request, jsonify
from os.path import exists

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

    if not exists(imgs_dir):
        os.makedirs(imgs_dir)

    if not exists(anno_dir):
        os.makedirs(anno_dir)

    with zipfile.ZipFile(img_zip, 'r') as imgs:
        imgs.extractall(imgs_dir)

    with zipfile.ZipFile(anno_zip, 'r') as annos:
        annos.extractall(anno_dir)

    with open(yaml_file_path, 'w') as yaml:
        yaml.write(yaml_file.read().decode('utf-8'))

    return f"{dataset_name} sent for {action}"


# @app.route('/test-model', methods=['POST'])
# # @key_required
# def test_model():
#     model = request.form.get('model')
#     dataset = request.form.get('dataset')
    # Lancer script test en asynchrone
    # Retourner résultats ?


# @app.route('/train-model', methods=['POST'])
# @key_required
# def train_model():
    # Lancer script entrainement
    # Param pour décider du modèle + du dataset


@app.route('/datasets', methods=['GET'])
def get_datasets():
    datasets_info = {}

    for dirname in os.listdir(DATASETS_PATH):
        full_path = os.path.join(DATASETS_PATH, dirname)
        modification_date = os.path.getmtime(full_path)
        datasets_info[dirname] = time.ctime(modification_date)

    return jsonify(datasets_info)
