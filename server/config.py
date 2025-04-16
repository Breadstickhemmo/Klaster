import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    CORS_ORIGINS = ["http://localhost:3000"]
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY')