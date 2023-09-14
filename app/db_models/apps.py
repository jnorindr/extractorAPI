from app.app import db


class AppsModel(db.Model):
    __tablename__ = 'apps'

    id = db.Column(db.Integer, primary_key=True)
    app_name = db.Column(db.String(50))
    app_key = db.Column(db.String(80))

    def __init__(self, app_name, app_key=None):
        self.app_name = app_name
        self.app_key = app_key

    def json(self):
        return {
            'app_name': self.app_name,
            'app_key': self.app_key
        }

    @classmethod
    def find_name(cls, app_name):
        return cls.query.filter_by(app_name=app_name).first()

    @classmethod
    def find_key(cls, app_key):
        return cls.query.filter_by(app_key=app_key).first()
