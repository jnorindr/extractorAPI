from app.db_models.apps import AppsModel
import functools
from hmac import compare_digest
from flask import request


def is_valid(api_key):
    client_app = AppsModel.find_key(api_key)
    if client_app and compare_digest(client_app.app_key, api_key):
        return True


def key_required(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if "X-API-Key" in request.headers:
            api_key = request.headers.get("X-API-Key")
        else:
            return {"message": "Please provide an API key"}, 400

        if request.method == "POST" and is_valid(api_key):
            return func(*args, **kwargs)
        else:
            return {"message": "The provided API key is not valid"}, 403

    return decorator
