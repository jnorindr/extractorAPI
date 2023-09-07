from flask import request
import os

from os.path import exists

from app import app, detect
from utils.paths import MANIFESTS_PATH


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
