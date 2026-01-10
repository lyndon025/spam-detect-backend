import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    PORT = int(os.environ.get("PORT", 5000))
    
    if not OPENROUTER_API_KEY:
        print("⚠️ WARNING: OPENROUTER_API_KEY not set. AI features will fail.")
