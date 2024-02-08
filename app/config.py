import os
from app.utils.paths import ENV


class Config():
    SQLALCHEMY_DATABASE_URI = ENV.str('SQLALCHEMY_DATABASE_URI')
    APP_PORT = ENV('APP_PORT', default=5000)
