import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///symptom_checker.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')
    MEDICAL_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    @classmethod
    def init_app(cls, app):
        # Ensure data directories exist
        data_dirs = [
            cls.MEDICAL_DATA_PATH,
            os.path.join(cls.MEDICAL_DATA_PATH, 'medical_data'),
            os.path.join(cls.MEDICAL_DATA_PATH, 'bpm_sessions'),
            os.path.join(cls.MEDICAL_DATA_PATH, 'reports'),
            os.path.join(cls.MEDICAL_DATA_PATH, 'analysis')
        ]
        for dir_path in data_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path) 