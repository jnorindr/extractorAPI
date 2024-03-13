from app.db_models.apps import AppsModel
import functools
from hmac import compare_digest
from flask import request


def get_client_app(api_key):
    client_app = AppsModel.find_key(api_key)
    if client_app and compare_digest(client_app.app_key, api_key):
        return client_app
    return None


def key_required(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if "X-API-Key" in request.headers:
            api_key = request.headers.get("X-API-Key")
        else:
            return {"message": "Please provide an API key"}, 400

        client_app = get_client_app(api_key)
        if request.method == "POST" and client_app:
            return func(client_app.app_name, *args, **kwargs)
        else:
            return {"message": "The provided API key is not valid"}, 403

    return decorator
