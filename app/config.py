import os
from app.utils.paths import ENV, API_ROOT


class Config():
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{API_ROOT}/{ENV('DB_NAME')}.db"
    API_PORT = ENV('API_PORT', default=5000)
