import logging
import os
from pathlib import Path

from flask import Flask
from celery import Celery
from flask_sqlalchemy import SQLAlchemy

from .config import Config
from app.utils.paths import ENV, APP_LOG


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

class AppLogger:
    def __init__(self, log_file_path):
        self.logger = logging.getLogger("exapi")
        # self.logger.setLevel(logging.DEBUG)

        # file_handler = logging.FileHandler(log_file_path)
        # file_handler.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # file_handler.setFormatter(formatter)

        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.DEBUG)
        # console_handler.setFormatter(formatter)

        self.logger.addHandler(logging.FileHandler(log_file_path))
        self.logger.addHandler(logging.StreamHandler())


app_logger = AppLogger(APP_LOG)

celery = Celery(
    app.import_name,
    backend=ENV.str("CELERY_BROKER_URL"),
    broker=ENV.str("CELERY_BROKER_URL")
)
celery.conf.update(app.config)


from app.routes import detection, training, similarity
