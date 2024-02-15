import json
import os
from pathlib import Path


def create_dir(path:Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_dir(path):
    path = Path(path)
    if not path.exists():
        create_dir(path)
        return False
    return True


def create_dirs_if_not(paths):
    for path in paths:
        check_dir(path)
    return paths


def create_file_if_not(path):
    path = Path(path)
    if not path.exists():
        path.touch(mode=0o666)
    return path


def create_files_if_not(paths):
    for path in paths:
        create_file_if_not(path)
    return paths


def sanitize_url(string):
    return string.replace(" ", "+").replace(" ", "+")


def sanitize_str(string):
    return (string.replace("/", "").replace(".", "").replace("https", "").replace("http", "")
            .replace("www", "").replace(" ", "_").replace(":", ""))


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
