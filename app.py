from flask import Flask
from celery import Celery
from flask_sqlalchemy import SQLAlchemy

from utils.paths import ENV


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = ENV.str('SQLALCHEMY_DATABASE_URI')
db = SQLAlchemy(app)

celery = Celery(
    app.import_name,
    backend=ENV.str("CELERY_BROKER_URL"),
    broker=ENV.str("CELERY_BROKER_URL")
)
celery.conf.update(app.config)


from routes import detection

if __name__ == '__main__':
    app.run()
