import os
from flask import request
from dotenv import load_dotenv

load_dotenv()
allowed_hosts = os.getenv("ALLOWED_HOSTS")


def allow_hosts(func):
    def wrapper(*args, **kwargs):
        if request.remote_addr not in allowed_hosts:
            return "Access denied", 403
        return func(*args, **kwargs)
    return wrapper
