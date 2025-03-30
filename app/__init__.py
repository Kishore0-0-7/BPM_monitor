from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os
from config import Config

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'main.index'  # Redirect to home page if not logged in
    
    @login_manager.user_loader
    def load_user(user_id):
        from app.models.user import User
        return User.query.get(int(user_id))
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Create database tables
    with app.app_context():
        db.create_all()
        
        # Ensure data directory exists
        data_dir = app.config.get('MEDICAL_DATA_PATH')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    # Initialize BPM monitor
    try:
        # Import here to avoid circular imports
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app_bpm import create_bpm_monitor
        app = create_bpm_monitor(app)
        print("BPM monitor integration successful")
    except Exception as e:
        print(f"Error initializing BPM monitor: {e}")
        print("The application will start without BPM monitoring capabilities")
        
    return app