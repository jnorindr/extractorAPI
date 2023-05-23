import environ
from flask import request

from utils.paths import API_ROOT

ENV = environ.Env()
environ.Env.read_env(env_file=f"{API_ROOT}/.env")
ALLOWED_HOSTS = ENV.list("ALLOWED_HOSTS")


def allow_hosts(func):
    def wrapper(*args, **kwargs):
        if request.remote_addr not in ALLOWED_HOSTS:
            return "Access denied", 403
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper
