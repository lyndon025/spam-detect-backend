# 🛡️ Spam Detect PH - Backend API

This is the Python/Flask API for the **Spam Detect PH** capstone project. It serves the AI model predictions and connects to Google Gemini for advanced analysis.

## 🚀 Tech Stack
- **Framework:** Flask (Python)
- **ML Model:** Scikit-Learn (MLPClassifier + TF-IDF)
- **Explainability:** LIME (Local Interpretable Model-agnostic Explanations)
- **AI Integration:** Google Gemini 2.5 Flash via OpenRouter
- **Deployment:** Fly.io (Primary), Render (Alternative)

## 📂 Folder Structure
- `app.py`: Main server entry point.
- `src/`: Application source code (routes, services, utils).
- `models/`: Trained ML models (`.pkl`).
- `fly.toml`: Fly.io configuration (set for 24/7 uptime).

## 🌍 Deployment (Fly.io)
The backend is configured to run on **Fly.io** with a 24/7 uptime configuration (~$3/mo).

1. **Install Fly CLI:** [https://fly.io/docs/hands-on/install-flyctl/](https://fly.io/docs/hands-on/install-flyctl/)
2. **Login:** `fly auth login`
3. **Deploy:** `fly deploy`

*Note: The `fly.toml` is configured with `min_machines_running = 1` and `auto_stop_machines = false` to prevent cold starts.*

## 🔧 Local Development
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Env Vars:** Create `.env` with `OPENROUTER_API_KEY=your_key_here`
3. **Run:** `python app.py`
   - Runs on `http://localhost:5000`
   - CORS is configured to allow `localhost:3000`, `localhost:5500`, and the live Vercel app.
