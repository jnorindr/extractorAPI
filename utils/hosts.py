from flask import request
from utils.paths import ENV


ALLOWED_HOSTS = ENV.list("ALLOWED_HOSTS")
# TODO: add host restriction


def allow_hosts(func):
    def wrapper(*args, **kwargs):
        if request.remote_addr not in ALLOWED_HOSTS:
            return "Access denied", 403
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper
