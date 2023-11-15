from flask import Flask
from celery import Celery
from flask_sqlalchemy import SQLAlchemy

from .config import Config
from app.utils.paths import ENV


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

celery = Celery(
    app.import_name,
    backend=ENV.str("CELERY_BROKER_URL"),
    broker=ENV.str("CELERY_BROKER_URL")
)
celery.conf.update(app.config)


from app.routes import detection, training
