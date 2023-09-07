from celery import Celery
from utils.paths import ENV

CELERY_BROKER_URL = ENV.str("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = ENV.str("CELERY_BROKER_URL")


def get_celery_app_instance(app):
    celery = Celery(
        app.import_name,
        backend=CELERY_BROKER_URL,
        broker=CELERY_BROKER_URL
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
