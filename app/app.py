from flask import Flask
from celery import Celery
from flask_sqlalchemy import SQLAlchemy

from .config import Config
from app.utils.paths import ENV


app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

REDIS_BROKER = f"{ENV.str('CELERY_BROKER_URL')}/{1 if ENV.bool('DEBUG') else 0}"

celery = Celery(
    app.import_name,
    backend=REDIS_BROKER,
    broker=REDIS_BROKER
)
celery.conf.update(app.config)


from app.routes import detection, training, similarity
