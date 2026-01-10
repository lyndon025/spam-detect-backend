from flask import Flask
from flask_cors import CORS
from src.config import Config

def create_app():
    app = Flask(__name__)
    # Explicitly allow the frontend origin
    CORS(app, resources={r"/*": {"origins": [
        "https://spam-detectph.vercel.app",
        "http://127.0.0.1:5500", 
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000"
    ]}})
    
    from src.routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
