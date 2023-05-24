import json
from functools import wraps
from pathlib import Path
from utils.logger import log

import requests


def create_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_if_dir_exists(path):
    path = Path(path)
    if not path.exists():
        create_dir(path)
        return False
    return True


def get_json(url):
    try:
        response = requests.get(url)
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        log(e)
        return None


def pprint(o):
    if type(o) == str:
        try:
            return json.dumps(json.loads(o), indent=4, sort_keys=True)
        except ValueError:
            return o
    elif type(o) == dict or type(o) == list:
        return json.dumps(o, indent=4, sort_keys=True)
    else:
        return str(o)
