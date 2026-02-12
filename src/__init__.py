from flask import Flask
from flask_cors import CORS
from src.config import Config

def create_app():
    app = Flask(__name__)
    # Explicitly allow the frontend origin
    CORS(app, resources={r"/*": {
        "origins": [
            "https://spam-detectph.vercel.app",
            "http://localhost:5173",  # For local testing if needed
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }})
    
    from src.routes import main_bp
    app.register_blueprint(main_bp)

    return app
