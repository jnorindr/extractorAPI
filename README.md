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
Set your app and database variables:
```bash
DB_NAME="<db-name>"
APP_NAME="<front-app-name-that-will-access-api>"
APP_KEY="$(openssl rand -base64 32 | tr -d '/\n')"
```
Create a SQLite database at the root of the API repository to store API keys
```shell
sqlite3 $DB_NAME.db <<EOF
CREATE TABLE apps (
   id INTEGER PRIMARY KEY AUTOINCREMENT,
   app_name CHAR(50) NOT NULL,
   app_key CHAR(80) NOT NULL
);
EOF
```
Add app name and key to the database
```bash
sqlite3 $DB_NAME.db <<EOF
INSERT INTO apps (app_name, app_key) VALUES ('$APP_NAME', '$APP_KEY');
EOF
```
Show content of the `apps` table:
```bash
sqlite3 -header -column $DB_NAME.db <<EOF
SELECT * FROM apps;
EOF
```

#### Setting the environment variables
Copy the content of the template file
```bash
cp .env{.template,}
```
Change the content according to your Celery backend, client app and API keys database
```bash
CELERY_BROKER_URL="redis://localhost:<redis-port>" # default port: 6379
API_PORT=<api-port> # default port: 5000
DEBUG=True # False for production
CLIENT_APP_URL="<url-of-front-app-connected-to-API>"
DB_NAME="<db-name-without-extension>"
```
[//]: # (If you use another port than `6379` for Redis &#40;e.g. multiple celery instances on the same server&#41;, update the `redis.conf`:)
[//]: # (```bash)
[//]: # (REDIS_PORT=<redis-port>)
[//]: # (REDIS_CONF=$&#40;redis-cli INFO | grep config_file | awk -F: '{print $2}' | tr -d '[:space:]'&#41;)
[//]: # (sudo sed -i -e "/^port 6379/a\port $REDIS_PORT" "$REDIS_CONF" # append new port to listen to)
[//]: # (sudo systemctl restart redis)
[//]: # (```)
If you want to use the API for training and use [Comet](https://www.comet.com/) as a tracker, add to your `.env`:
```bash
COMET_API_KEY=<comet-API-key>
COMET_PROJECT_NAME=<project-name>
```

#### Enabling authentication for Redis instance
> :warning: Be sure to not override a previously defined redis password

Get the path of Redis config file
```bash
REDIS_CONF=$(redis-cli INFO | grep config_file | awk -F: '{print $2}' | tr -d '[:space:]')
```
Generate a password
```bash
REDIS_PSW="$(openssl rand -base64 32 | tr -d '/\n')"
```
Update the redis configuration
```bash
sudo sed -i -e "s/^requirepass [^ ]*/requirepass $REDIS_PSW/" "$REDIS_CONF"
sudo sed -i -e "s/# requirepass [^ ]*/requirepass $REDIS_PSW/" "$REDIS_CONF"
```
Update the `CELERY_BROKER_URL` inside the `.env` file:
```bash
sed -i '' -e "s~^CELERY_BROKER_URL=.*~CELERY_BROKER_URL=\"redis://:$REDIS_PSW@localhost:6379\"~" .env
```
Restart Redis
```bash
sudo systemctl restart redis-server
```
Test the password
```bash
redis-cli -a $REDIS_PSW
```

## Run the application
If not already, start redis:
```bash
sudo systemctl start redis
```
Launch Celery
```bash
celery -A app.app.celery worker -B -c 1 --loglevel=info -P threads
```
Run the app
```bash
python run.py
```

Or run everything at once:
```bash
bash run.sh
```

## Use API :rocket:

### Load variables
```bash
# Choose app to use for request
APP_NAME="<your_app_name>"
# Load environment variables ($DB_NAME and $API_PORT)
source .env
# Get API_KEY
API_KEY=$(sqlite3 $DB_NAME.db <<EOF
SELECT app_key FROM apps WHERE app_name = '$APP_NAME';
EOF
)
```

### Extract annotations
One manifest
```bash
curl -X POST -H "X-API-Key: $API_KEY" -F manifest_url='<url-manifest>' http://127.0.0.1:$API_PORT/run_detect
```
Manifest list in a text file
```bash
curl -X POST -H "X-API-Key: $API_KEY" -F url_file=@iiif/test-manifests.txt http://127.0.0.1:$API_PORT/detect_all
```
To use a different model from the default model
```bash
curl -X POST -H "X-API-Key: $API_KEY" -F model='<model-filename>' manifest_url='<url-manifest>'  http://127.0.0.1:$API_PORT/run_detect
```
Get the list of available extraction models filenames
```bash
curl http://127.0.0.1:$API_PORT/models
```

### Compute similarity
Compute similarity scores for pairs of documents (here: `(doc1,doc1)`, `(doc1,doc2)`, `(doc1,doc3)`, `(doc2,doc2)`, `(doc2,doc3)`, `(doc3,doc3)`)
```bash
curl -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{
    "documents": {
        "doc1_id": "doc1_url",
        "doc2_id": "doc2_url",
        "doc3_id": "doc3_url"
    }
}' http://127.0.0.1:$API_PORT/run_similarity
```

Choose the backbone model for feature extraction (between: `resnet34`, `moco_v2_800ep_pretrain`, `dino_deitsmall16_pretrain`, `dino_vitbase8_pretrain`)
```bash
curl -X POST -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{
    "documents": {"doc1_id": "doc1_url"},
    "model": "dino_vitbase8_pretrain"
}' http://127.0.0.1:$API_PORT/run_similarity
```
