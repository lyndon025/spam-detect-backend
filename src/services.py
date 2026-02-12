import joblib
import os
import numpy as np
from lime.lime_text import LimeTextExplainer
from openai import OpenAI
from src.config import Config
from src.utils import clean_text

class ModelService:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.explainer = None
        self._load_models()

    def _load_models(self):
        try:
            # Paths relative to the root where the app is run
            self.model = joblib.load("src/models/spam_mlp_model.pkl")
            self.vectorizer = joblib.load("src/models/vectorizer.pkl")
            self.explainer = LimeTextExplainer(class_names=self.model.classes_)
            print("✅ Model, Vectorizer, and LIME loaded.")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model = None

    def predict_proba_pipeline(self, texts):
        """
        LIME needs a function that takes raw text list -> returns probabilities
        """
        cleaned_texts = [clean_text(t) for t in texts]
        vec_texts = self.vectorizer.transform(cleaned_texts)
        return self.model.predict_proba(vec_texts)

    def predict(self, raw_text):
        if not self.model:
            raise Exception("Model not loaded")

        cleaned = clean_text(raw_text)
        vec_text = self.vectorizer.transform([cleaned])
        
        probs = self.model.predict_proba(vec_text)[0]
        prediction_idx = np.argmax(probs)
        prediction = self.model.classes_[prediction_idx]
        confidence = probs[prediction_idx] * 100
        
        return prediction, confidence, prediction_idx

    def get_category(self, prediction, confidence):
        danger_labels = ["spam", "scam", "smishing", "finance_scam"]
        caution_labels = ["ads", "promo"]

        if confidence < 50.0:
            return "safe"
            
        if prediction in danger_labels:
            return "danger"
        elif prediction in caution_labels:
            return "caution"
        
        return "safe"

    def explain(self, raw_text, prediction_idx):
        exp = self.explainer.explain_instance(
            raw_text,
            self.predict_proba_pipeline,
            num_features=6,
            num_samples=1000,
            labels=(prediction_idx,),
        )
        return exp.as_list(label=prediction_idx)

class AIService:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=Config.OPENROUTER_API_KEY,
        )

    def analyze_sms(self, text):
        if not Config.OPENROUTER_API_KEY:
            raise Exception("Server missing OpenRouter API Key.")

        prompt = (
            f"Analyze this SMS message: '{text}'. No need to repeat it. "
            "1. Is it a Scam, Spam, or Safe? "
            "2. Explain why in 1 short sentence. "
            "3. If it's a scam, what tactic could it be (e.g., Urgency, Phishing)?"
        )

        try:
            print(f"DEBUG: Calling OpenRouter for model: google/gemini-2.5-flash-lite")
            response = self.client.chat.completions.create(
                model="google/gemini-2.5-flash-lite",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.4,
                extra_headers={
                    "HTTP-Referer": "https://spam-detectph.vercel.app",
                    "X-Title": "Spam Detect PH",
                },
            )
            print("DEBUG: OpenRouter response received.")
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenRouter API Error: {e}")
            return "AI Analysis is currently unavailable. Please try again later."

# Singleton instances
model_service = ModelService()
ai_service = AIService()
