import logging
import os
import time

from utils.paths import ENV, LOG_PATH
from utils import pprint

DEBUG = ENV.list("DEBUG")


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


def log(msg, color=TerminalColors.OKBLUE):
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
    logger.error(f"\n{pprint(msg)}{trace}\n")
