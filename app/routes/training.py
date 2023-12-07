import os
import time
import zipfile
from flask import request, jsonify
from os.path import exists

from app.app import app
from app.utils.security import key_required
from app.utils.tasks import validate, training, save_dataset
from app.utils.paths import DATA_PATH, DATASETS_PATH


@app.route('/send-data', methods=['POST'])
@key_required
def send_data():
    img_zip = request.files['img_zip']
    anno_zip = request.files['anno_zip']
    yaml_file = request.files['yaml_file']
    action = request.form.get('action')

    dataset_name = yaml_file.filename.replace('.yaml', '')
    imgs_dir = DATASETS_PATH / dataset_name / "images" / action
    anno_dir = DATASETS_PATH / dataset_name / "labels" / action
    yaml_file_path = DATA_PATH / f"{dataset_name}.yaml"

    try:
        if not exists(imgs_dir):
            os.makedirs(imgs_dir)

        if not exists(anno_dir):
            os.makedirs(anno_dir)

        if action == 'test':
            # Create val directories for tests to avoid an error
            val_imgs_dir = DATASETS_PATH / dataset_name / "images" / "val"
            val_anno_dir = DATASETS_PATH / dataset_name / "labels" / "val"
            if not exists(val_imgs_dir):
                os.makedirs(val_imgs_dir)

            if not exists(val_anno_dir):
                os.makedirs(val_anno_dir)

        try:
            with zipfile.ZipFile(img_zip, 'r') as imgs:
                imgs.extractall(imgs_dir)
        except Exception as e:
            return f'An error occurred while extracting images: {e}'

        try:
            with zipfile.ZipFile(anno_zip, 'r') as annos:
                annos.extractall(anno_dir)
        except Exception as e:
            return f'An error occurred while extracting annotations: {e}'

        try:
            with open(yaml_file_path, 'w') as yaml:
                yaml.write(yaml_file.read().decode('utf-8'))
        except Exception as e:
            return f'An error occurred while writing YAML file: {e}'

        return f"{dataset_name} sent for {action}"

    except Exception as e:
        return f'An error occurred: {e}'


@app.route('/send-dataset', methods=['POST'])
@key_required
def send_dataset():
    dataset_zip = request.files['data_zip']
    yaml_file = request.files['yaml_file']

    save_dataset.delay(dataset_zip, yaml_file)
    return f'Downloading dataset with Celery!'


@app.route('/test-model', methods=['POST'])
@key_required
def test_model():
    model = request.form.get('model')
    data = request.form.get('data')
    name = request.form.get('name')

    validate.delay(model, data, name)
    return f"Validation task triggered with Celery!"


@app.route('/train-model', methods=['POST'])
@key_required
def train_model():
    model = request.form.get('model')
    data = request.form.get('data')

    training.delay(model, data)
    return f"Training task triggered with Celery!"


@app.route('/datasets', methods=['GET'])
def get_datasets():
    datasets_info = {}

    try:
        for dirname in os.listdir(DATASETS_PATH):
            full_path = os.path.join(DATASETS_PATH, dirname)
            modification_date = os.path.getmtime(full_path)
            datasets_info[dirname] = time.ctime(modification_date)

        return jsonify(datasets_info)

    except Exception:
        return jsonify("No dataset.")
