from flask import Flask
from flask_cors import CORS
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from config import Config
from auth_routes import register_auth_routes
from models import db, bcrypt
from logging_config import setup_logging
import os

app = Flask(__name__)
app.config.from_object(Config)
app.logger.info(f"JWT Secret Key Loaded: {'YES' if app.config.get('JWT_SECRET_KEY') else 'NO --- PROBLEM!'}")

CORS(app, resources={r"/api/*": {"origins": Config.CORS_ORIGINS}}, supports_credentials=True)
db.init_app(app)
bcrypt.init_app(app)
jwt = JWTManager(app)
migrate = Migrate(app, db)

setup_logging()

register_auth_routes(app)

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)