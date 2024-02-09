import logging
import os
import time
import traceback

from app.app import app_logger
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


def console(msg="🚨🚨🚨", color="blue", error:Exception=None):
    if error:
        color = "red"
        stack_trace = traceback.format_exc()
        msg += f"\n\nStack Trace:\n{stack_trace}"

    msg = f"\n\n{get_time()}\n{get_color(color)}{TerminalColors.bold}{pprint(msg)}{TerminalColors.end}\n\n"

    if error:
        app_logger.logger.error(msg, exc_info=error)
    elif color == "yellow":
        app_logger.logger.warning(msg)
    else:
        app_logger.logger.info(msg)


def log_failed_img(img_name, img_url):
    if not os.path.isfile(IMG_LOG):
        f = open(IMG_LOG, "x")
        f.close()

    with open(IMG_LOG, "a") as f:
        f.write(f"{img_name} {img_url}\n")
