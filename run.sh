#! /bin/bash
(trap 'kill 0' SIGINT;
    (venv/bin/celery -A app.app.celery worker -B -c 1 --loglevel=info -P threads) &
    (venv/bin/python run.py);
);
