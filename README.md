# Element extraction on a GPU

extractoAPI is a tool to run on a GPU an algorithm using a model for element extraction in images.

extractorAPI retrieves images from a IIIF manifest or a list of manifests and uses a vision model to extract objects from images and return annotations in text files. 

## Requirements :hammer_and_wrench:

> - **Sudo** privileges
> - **Git**
> - **Python:** 3.10

#### System dependencies
```shell
sudo apt-get install redis-server python3-venv python3-dev
```
#### Repository
```shell
git clone https://github.com/jnorindr/extractorAPI
cd extractorAPI
```
#### Python dependencies
```shell
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
#### Setting the environment variables
Copy the content of the template file
```bash
cp .env{.template,}
```
Change the content according to your host restrictions, Celery backend and the client app
```
ALLOWED_HOSTS="<host1>,<host2>"
CELERY_BROKER_URL="redis://<localhost-port>"
DEBUG=True
CLIENT_APP_URL="<url>"
```
## Run the application 
Start Redis and Celery
```shell
sudo systemctl start redis && celery -A app.celery worker -B -c 1 --loglevel=info
```
```shell
FLASK_ENV=development FLASK_APP=app.py flask run
```
## Launch annotation :rocket:
One manifest
```shell
curl -X POST -F manifest_url='<url-manifest>' http://127.0.0.1:5000/run_detect
```
Manifest list in a text file
```shell
curl -X POST -F url_file=@iiif/test-manifests.txt http://127.0.0.1:5000/detect_all
```
To use a different model from the default model
```shell
curl -X POST -F manifest_url='<url-manifest>' model='<filename>' http://127.0.0.1:5000/run_detect
```
```shell
curl -X POST -F url_file=@iiif/test-manifests.txt model='<filename>' http://127.0.0.1:5000/detect_all
```
