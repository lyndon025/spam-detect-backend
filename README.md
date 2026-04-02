# 🛡️ Spam Detect PH - Backend API

This is the Python/Flask API for the **Spam Detect PH** capstone project. It serves the AI model predictions and connects to Google Gemini for advanced analysis.

## 🚀 Tech Stack
- **Framework:** Flask (Python)
- **ML Model:** Scikit-Learn (MLPClassifier + TF-IDF)
- **Explainability:** LIME (Local Interpretable Model-agnostic Explanations)
- **AI Integration:** Google Gemini 2.5 Flash via OpenRouter
- **Deployment:** Render (Primary), Fly.io (Legacy)

## 📂 Folder Structure
- `app.py`: Main server entry point.
- `src/`: Application source code (routes, services, utils).
- `models/`: Trained ML models (`.pkl`).
- `render.yaml`: Render blueprint for automated deployment.

## 🌍 Deployment (Render)
The backend is configured to run on **Render** (using `render.yaml`).

1. **Connect to GitHub:** Link your repository to Render.
2. **Auto-Deploy:** Render will automatically build via Docker using the provided `Dockerfile`.

*Note: Free instances on Render may spin down after inactivity. The frontend includes a "waking up" message to handle this.*

## 🏗️ Legacy Deployment (Fly.io)
Previously hosted on Fly.io. To restart:
1. `fly deploy`

## 🔧 Local Development
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Env Vars:** Create `.env` with `OPENROUTER_API_KEY=your_key_here`
3. **Run:** `python app.py`
   - Runs on `http://localhost:5000`
   - CORS is configured to allow `localhost:3000`, `localhost:5500`, and the live Vercel app.
