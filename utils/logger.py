import json
import logging
import os
import time
import environ

from utils.paths import API_ROOT, LOG_PATH

ENV = environ.Env()
environ.Env.read_env(env_file=f"{API_ROOT}/.env")
DEBUG = ENV.bool("DEBUG")


class TerminalColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_color(msg_type=None):
    if msg_type == "error":
        return TerminalColors.FAIL
    if msg_type == "warning":
        return TerminalColors.WARNING
    return TerminalColors.OKBLUE


def log(msg):
    """
    Record an error message in the system log
    """
    import logging
    import traceback

    trace = traceback.format_exc()
    if trace == "NoneType: None\n":
        # trace = traceback.extract_stack(limit=10)
        trace = ""

    if not os.path.isfile(LOG_PATH):
        f = open(LOG_PATH, "x")
        f.close()

    # Create a logger instance
    logger = logging.getLogger("django")
    # get_time() is already printed by the logger object
    logger.error(f"\n{pprint(msg)}{trace}\n")


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


def console(msg="🚨🚨🚨", msg_type=None):
    msg = f"\n\n\n{get_time()}\n{get_color(msg_type)}{TerminalColors.BOLD}{pprint(msg)}{TerminalColors.ENDC}\n\n\n"

    if not DEBUG:
        log(msg)
        return

    logger = logging.getLogger("django")
    if msg_type == "error":
        logger.error(msg)
    elif msg_type == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)
