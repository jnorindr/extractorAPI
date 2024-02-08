#! /bin/bash
(trap 'kill 0' SIGINT; 
    (celery -A app.app.celery worker -B -c 1 --loglevel=info -P threads) &
    (python run.py);
);
