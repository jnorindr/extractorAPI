# Extractor API

> API that does blablabli

## Requirements

> Pre-requirements: sudo privileges, git, Python 3.10

System dependencies
```shell
sudo apt-get install redis-server python3-venv python3-dev
```

Python dependencies
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Setting the environment variables
```bash
cp .env{.template,}
```

## Run the application
```shell
sudo systemctl start redis && celery -A app.celery worker -c 1 --loglevel=info && FLASK_ENV=development FLASK_APP=app.py flask run
```

## Launch annotation
```shell
curl -X POST -F manifest_url='<url-manifest>' http://127.0.0.1:5000/run_detect
```