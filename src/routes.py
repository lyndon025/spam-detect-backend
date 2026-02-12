from flask import Blueprint, request, jsonify
from src.services import model_service, ai_service
from src.utils import clean_text
import re
import numpy as np

main_bp = Blueprint('main', __name__)

@main_bp.route("/", methods=["GET"])
def home():
    return "✅ Spam Detector Backend is Running"

@main_bp.route("/test-cors", methods=["GET"])
def test_cors():
    return jsonify({"message": "CORS is working!", "status": "success"})

@main_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        raw_text = data.get("text", "")

        if not raw_text:
            return jsonify({"error": "No text provided"}), 400

        prediction, confidence, prediction_idx = model_service.predict(raw_text)
        category = model_service.get_category(prediction, confidence)

        # Link Detection
        from src.utils import detect_link
        has_link = detect_link(raw_text)

        # LIME Explainability
        lime_features = model_service.explain(raw_text, prediction_idx)

        return jsonify(
            {
                "status": "success",
                "prediction": prediction.upper(),
                "confidence": f"{confidence:.2f}",
                "category": category,
                "has_link": has_link,
                "lime_data": lime_features,
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@main_bp.route("/ask-gemini", methods=["POST"])
def ask_gemini():
    try:
        data = request.get_json()
        text = data.get("text", "")
        
        analysis_text = ai_service.analyze_sms(text)
        return jsonify({"analysis": analysis_text})

    except Exception as e:
        print(f"OpenRouter Error: {e}")
        return jsonify({"analysis": f"AI Error: {str(e)}"}), 500
