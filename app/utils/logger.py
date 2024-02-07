import logging
import time

from app.utils.paths import ENV, LOG_PATH
from app.utils import pprint

DEBUG = ENV.list("DEBUG")


class TerminalColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_color(msg_type=None):
    if msg_type == "error":
        return TerminalColors.FAIL
    if msg_type == "warning":
        return TerminalColors.WARNING
    return TerminalColors.OKBLUE


def console(msg="🚨🚨🚨", msg_type=None):
    msg = f"\n\n\n{get_time()}\n{get_color(msg_type)}{TerminalColors.BOLD}{pprint(msg)}{TerminalColors.ENDC}\n\n\n"

    logger = logging.getLogger("exapi")
    file_handler = logging.FileHandler(LOG_PATH)
    logger.addHandler(file_handler)

    if msg_type == "error":
        logger.error(msg)
    elif msg_type == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)
