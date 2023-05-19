import os
import environ
from flask import request

allowed_hosts = ['obspm.fr']
env = environ.Env()
environ.env.readenv


def allow_hosts(func):
    def wrapper(*args, **kwargs):
        if request.remote_addr not in allowed_hosts:
            return "Access denied", 403
        return func(*args, **kwargs)
    return wrapper
