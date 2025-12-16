# 🛡️ Spam Detect PH - Backend API

This is the Python/Flask API for the **Spam Detect PH** capstone project. It serves the AI model predictions and connects to Google Gemini for advanced analysis.

## 🚀 Tech Stack
- **Framework:** Flask (Python)
- **ML Model:** Scikit-Learn (MLPClassifier + TF-IDF)
- **Explainability:** LIME (Local Interpretable Model-agnostic Explanations)
- **AI Integration:** Google Gemini 2.5 Flash via OpenRouter
- **Deployment:** Render

## 📂 Folder Structure
- `app.py`: Main server logic.
- `models/`: Contains the trained `.pkl` model files.
- `requirements.txt`: Python dependencies.

## 🔧 Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Create `.env` file with `OPENROUTER_API_KEY`.
3. Run server: `python app.py`
