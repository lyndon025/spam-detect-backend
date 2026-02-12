from flask import Flask
from flask_cors import CORS
from src.config import Config

def create_app():
    app = Flask(__name__)
    # Explicitly allow the frontend origin
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    from src.routes import main_bp
    app.register_blueprint(main_bp)

    return app
