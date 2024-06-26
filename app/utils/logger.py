import logging
import os
import time
import traceback

from app.utils.paths import ENV, IMG_LOG
from app.utils import pprint

DEBUG = ENV.list("DEBUG")

class TerminalColors:
    """
    Last digit
    0	black
    1	red
    2	green
    3	yellow
    4	blue
    5	magenta
    6	cyan
    7	white
    """
    black = "\033[90m"
    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    cyan = "\033[96m"
    white = "\033[97m"
    bold = "\033[1m"
    underline = "\033[4m"
    end = "\033[0m"


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_color(color=None):
    return getattr(TerminalColors, color, "\033[94m")


def console(msg="🚨🚨🚨", color="blue", error: Exception = None):
    # logger = logging.getLogger("exapi")

    if error:
        color = "red"
    stack_trace = f"\nStack Trace:\n{get_color(color)}{traceback.format_exc()}{TerminalColors.end}\n" if error else ""
    msg = f"\n\n[{get_time()}]\n{get_color(color)}{TerminalColors.bold}{pprint(msg)}{TerminalColors.end}\n{stack_trace}"

    # if error:
    #     logger.error(msg, exc_info=error)
    # elif color == "yellow":
    #     logger.warning(msg)
    # else:
    #     logger.info(msg)
    print(msg)


def log(msg="🚨🚨🚨", color="blue", error: Exception = None):
    """
    Record an error message in the system log
    """
    trace = traceback.format_exc()
    if trace == "NoneType: None\n":
        # trace = traceback.extract_stack(limit=10)
        trace = ""

    # # Create a logger instance
    # logger = logging.getLogger("exapi")
    #
    # if error:
    #     logger.error(f"\n\n[{get_time()}]\n{get_color(color)}{TerminalColors.bold}{pprint(msg)}{TerminalColors.end}\n{trace}\n", exc_info=error)
    # elif color == "yellow":
    #     logger.warning(f"\n\n[{get_time()}]\n{get_color(color)}{TerminalColors.bold}{pprint(msg)}{TerminalColors.end}\n{trace}\n")
    # else:
    #     logger.info(f"\n\n[{get_time()}]\n{get_color(color)}{TerminalColors.bold}{pprint(msg)}{TerminalColors.end}\n{trace}\n")

    print(f"\n\n[{get_time()}]\n{get_color(color)}{TerminalColors.bold}{pprint(msg)}{TerminalColors.end}\n{trace}\n")


def log_failed_img(img_name, img_url):
    if not os.path.isfile(IMG_LOG):
        f = open(IMG_LOG, "x")
        f.close()

    with open(IMG_LOG, "a") as f:
        f.write(f"{img_name} {img_url}\n")
