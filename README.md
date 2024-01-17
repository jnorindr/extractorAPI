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
#### API Keys Database
Create a SQLite database at the root of the API repository to store API keys
```
sqlite3 DatabaseName.db
```
```
CREATE TABLE table_name(
   id INT PRIMARY KEY NOT NULL,
   app_name CHAR(50) NOT NULL,
   app_key CHAR(80) NOT NULL
);
```
Add an app and its key to the database
```
INSERT INTO table_name VALUES ('value', 'value2', 'value3');
```
Exit the SQLite shell
```
.exit
```
#### Setting the environment variables
Copy the content of the template file
```bash
cp .env{.template,}
```
Change the content according to your Celery backend, client app and API keys database
```
CELERY_BROKER_URL="redis://:<redis-password>@<localhost-port>/0"
DEBUG=True
CLIENT_APP_URL="<url>"
SQLALCHEMY_DATABASE_URI=sqlite:////<database-path>
```
If you want to use the API for training and use Comet as a tracker, add to your `.env`:
```
COMET_API_KEY=<comet-API-key>
COMET_PROJECT_NAME=<project-name>
```
#### Enabling authentication for Redis instance
Open the Redis configuration file
```
vim /etc/redis/redis.conf
```
Uncomment and set a password
```
requirepass <your_password>
```
Restart Redis
```
sudo systemctl restart redis-server
```
Test the password
```
redis-cli -a <your_password>
```
## Run the application 
Start Redis and Celery
```shell
sudo systemctl start redis && celery -A app.app.celery worker -B -c 1 --loglevel=info -P threads
```
Run the app
```shell
python run.py
```
## Launch annotation :rocket:
One manifest
```shell
curl -X POST -H "X-API-Key: <api-key>" -F manifest_url='<url-manifest>' http://127.0.0.1:5000/run_detect
```
Manifest list in a text file
```shell
curl -X POST -H "X-API-Key: <api-key>" -F url_file=@iiif/test-manifests.txt http://127.0.0.1:5000/detect_all
```
To use a different model from the default model
```shell
curl -X POST -H "X-API-Key: <api-key>" -F manifest_url='<url-manifest>' model='<filename>' http://127.0.0.1:5000/run_detect
```
```shell
curl -X POST -H "X-API-Key: <api-key>" -F url_file=@iiif/test-manifests.txt model='<filename>' http://127.0.0.1:5000/detect_all
```