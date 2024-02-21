import os
from flask import request

from app.app import app
from app.similarity.const import FEAT_NET
from app.utils.paths import ENV
from app.utils.tasks import similarity
from app.utils.security import key_required

from app.utils.logger import console


@app.route("/run_similarity", methods=['POST'])
@key_required
def run_similarity():
    """
    documents = {
        "wit3_man186_anno181": "https://eida.obspm.fr/eida/wit3_man186_anno181/list/",
        "wit87_img87_anno87": "https://eida.obspm.fr/eida/wit87_img87_anno87/list/",
        "wit2_img2_anno2": "https://eida.obspm.fr/eida/wit2_img2_anno2/list/"
    }
    Compute similarity for images from a list of URLs.
    Each URL corresponds to a document and contains a list of images to download:
    {
        "img_name": "https://domain-name.com/image_name.jpg",
        "img_name": "https://other-domain.com/image_name.jpg",
        "img_name": "https://iiif-server.com/.../coordinates/size/rotation/default.jpg",
        "img_name": "..."
    }
    Each document is compared to itself and other documents resulting in a list a comparison pairs
    """

    if not request.is_json:
        console("[run_similarity] Request does contain correct payload")
        return "Similarity task aborted!"

    # dict of document ids with a URL containing a list of images
    documents = request.get_json().get('documents', {})
    # which feature extraction backbone to use
    model = request.get_json().get('model', FEAT_NET)
    # which url to send back the similarity data
    callback = request.get_json().get('callback', f"{ENV.str('CLIENT_APP_URL')}/similarity")

    similarity.delay(documents, model, callback)
    return f"Similarity task triggered for {list(documents.keys())} with {model}!"


@app.route('/delete_similarity', methods=['POST'])
@key_required
def delete_similarity():
    """
    Delete features and pairs of scores related to a given document
    """
    # TODO
    return False
