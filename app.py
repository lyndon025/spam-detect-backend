import os
import re
import string
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from lime.lime_text import LimeTextExplainer

# NEW: Import OpenAI for OpenRouter
from openai import OpenAI

# 1. LOAD SECRETS
load_dotenv()  # Reads .env file

# NEW: OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY not set. AI features will fail.")

# Initialize Client (OpenRouter uses standard OpenAI interface)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# 3. SETUP FLASK
app = Flask(__name__)
CORS(app)

# 4. LOAD MODEL & VECTORIZER
try:
    model = joblib.load("src/models/spam_mlp_model.pkl")
    vectorizer = joblib.load("src/models/vectorizer.pkl")

    # Initialize LIME Explainer once to save time
    explainer = LimeTextExplainer(class_names=model.classes_)

    print("✅ Model, Vectorizer, and LIME loaded.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None


# 5. CLEANING FUNCTION (Must match training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


# 6. PIPELINE HELPER FOR LIME
def predict_proba_pipeline(texts):
    """
    LIME needs a function that takes raw text list -> returns probabilities
    """
    cleaned_texts = [clean_text(t) for t in texts]
    vec_texts = vectorizer.transform(cleaned_texts)
    return model.predict_proba(vec_texts)


# --- ROUTES ---


@app.route("/", methods=["GET"])
def home():
    return "✅ Spam Detector Backend is Running"


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    raw_text = data.get("text", "")

    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # A. PREDICTION
        cleaned = clean_text(raw_text)
        vec_text = vectorizer.transform([cleaned])

        # Get probabilities to find the winning class index
        probs = model.predict_proba(vec_text)[0]
        prediction_idx = np.argmax(probs)  # Index of the highest probability
        prediction = model.classes_[prediction_idx]  # Name of the class
        confidence = probs[prediction_idx] * 100

        # B. TRAFFIC LIGHT LOGIC
        danger_labels = ["spam", "scam", "smishing", "finance_scam"]
        caution_labels = ["ads", "promo"]

        category = "safe"
        if prediction in danger_labels:
            category = "danger"
        elif prediction in caution_labels:
            category = "caution"

        if confidence < 50.0:
            category = "safe"

        # C. LINK DETECTION (Regex)
        link_pattern = r"(http|https|www|\.com|\.ph|\.net|\.org|\.gov|\.ly)"
        has_link = bool(re.search(link_pattern, raw_text, re.IGNORECASE))

        # D. LIME EXPLAINABILITY (The "Why")
        exp = explainer.explain_instance(
            raw_text,
            predict_proba_pipeline,
            num_features=6,
            num_samples=1000,
            labels=(prediction_idx,),
        )

        # Get features for the predicted class
        lime_features = exp.as_list(label=prediction_idx)

        return jsonify(
            {
                "status": "success",
                "prediction": prediction.upper(),
                "confidence": f"{confidence:.2f}",
                "category": category,
                "has_link": has_link,
                "lime_data": lime_features,  # Send weighted words to frontend
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask-gemini", methods=["POST"])
def ask_gemini():
    """
    Replaced Google Gemini SDK with OpenRouter (via OpenAI SDK).
    Kept route name '/ask-gemini' so frontend doesn't break.
    """
    if not OPENROUTER_API_KEY:
        return jsonify({"analysis": "Server missing OpenRouter API Key."}), 500

    data = request.get_json()
    text = data.get("text", "")

    try:
        prompt = (
            f"Analyze this SMS message: '{text}'. No need to repeat it. "
            "1. Is it a Scam, Spam, or Safe? "
            "2. Explain why in 1 short sentence. "
            "3. If it's a scam, what tactic could it be (e.g., Urgency, Phishing)?"
        )

        # Using OpenRouter
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.4,
            extra_headers={
                "HTTP-Referer": "https://spam-detect-ph.vercel.app",
                "X-Title": "Spam Detect PH",
            },
        )

        analysis_text = response.choices[0].message.content
        return jsonify({"analysis": analysis_text})

    except Exception as e:
        print(f"OpenRouter Error: {e}")
        return jsonify({"analysis": f"AI Error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
