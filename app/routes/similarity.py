import os
from flask import request

from app.app import app
from app.similarity.const import FEAT_NET
from app.utils.tasks import similarity
from app.utils.security import key_required


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
        "images": [
            "https://domain-name.com/image_name.jpg",
            "https://other-domain.com/image_name.jpg",
            "https://iiif-server.com/.../coordinates/size/rotation/default.jpg",
            "..."
        ]
    }
    Each document is compared to itself and other documents resulting in a list a comparison pairs
    """
    documents = request.get_json().get('documents', {})  # dict of document ids with a URL containing a list of images
    model = request.get_json().get('model', FEAT_NET) # which feature extraction backbone to use
    callback = request.form.get('callback', None)  # which url to send back the similarity data

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