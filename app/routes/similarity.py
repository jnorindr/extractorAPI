import os
from flask import request

from app.app import app
from app.utils.tasks import similarity
from app.utils.security import key_required


@app.route("/run_similarity", methods=['POST'])
@key_required
def run_similarity():
    """
    Compute similarity for images from a list of URLs.
    Each URL corresponds to a document and contains a list of images to download:
    {
        images: [
            "https://domain-name.com/image_name.jpg",
            "https://other-domain.com/image_name.jpg",
            "https://iiif-server.com/.../coordinates/size/rotation/default.jpg",
            "..."
        ]
    }
    Each document is compared to itself and other documents resulting in a list a comparison pairs
    """
    documents = request.form['documents']  # list of url
    model = request.form.get('model')  # which feature extraction backbone to use
    callback = request.form.get('callback')  # which url to send back the similarity data

    similarity.delay(documents, model, callback)
    return f"Detection task triggered with Celery!"


@app.route('/delete_similarity', methods=['POST'])
@key_required
def delete_similarity():
    """
    Delete features and pairs of scores related to a given document
    """
    return False