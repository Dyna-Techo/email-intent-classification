"""
Mail Logic - Prediction Engine

Standalone script for email intent classification.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime


# CONFIGURATION
MODEL_PATH = "model/email_intent_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
RESULTS_DIR = "results"
CONFIDENCE_THRESHOLD = 0.45
FALLBACK_INTENT = "General"


# LOAD MODEL & VECTORIZER
print("🔄 Loading model...")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✓ Model loaded successfully\n")
except FileNotFoundError as e:
    print(f"✗ Error: Could not load model files.")
    print(f"  Please run: python src/train_model.py")
    exit(1)


# PREDICTION FUNCTION
def predict_intent(email_text: str):
    """Predict the intent category of an email."""
    X_vec = vectorizer.transform([email_text])
    probabilities = model.predict_proba(X_vec)[0]
    classes = model.classes_

    sorted_idx = np.argsort(probabilities)[::-1]

    top_2 = [
        (classes[i], round(float(probabilities[i]), 3))
        for i in sorted_idx[:2]
    ]

    best_intent, best_confidence = top_2[0]

    if best_confidence < CONFIDENCE_THRESHOLD:
        final_intent = FALLBACK_INTENT
    else:
        final_intent = best_intent

    return final_intent, round(best_confidence, 3), top_2


# LOGGING FUNCTION
def log_prediction(email_text, intent, confidence):
    """Save prediction to results directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Main text log
    with open(os.path.join(RESULTS_DIR, "result.txt"), "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] | {intent} | {confidence} | {email_text}\n")

    # Combined log
    combined_dir = os.path.join(RESULTS_DIR, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    with open(os.path.join(combined_dir, "combined.txt"), "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] | {intent} | {confidence} | {email_text}\n")

    # Intent-specific Excel
    intent_dir = os.path.join(RESULTS_DIR, intent)
    os.makedirs(intent_dir, exist_ok=True)

    excel_path = os.path.join(intent_dir, f"{intent}.xlsx")

    row = pd.DataFrame([{
        "timestamp": timestamp,
        "intent": intent,
        "confidence": confidence,
        "email_text": email_text
    }])

    if os.path.exists(excel_path):
        existing = pd.read_excel(excel_path)
        updated = pd.concat([existing, row], ignore_index=True)
    else:
        updated = row

    updated.to_excel(excel_path, index=False)


# INTERACTIVE PREDICTION
def interactive_predict():
    """Run interactive prediction loop."""
    
    print("\n" + "=" * 60)
    print("EMAIL INTENT CLASSIFICATION - PREDICTION ENGINE")
    print("=" * 60)
    print("\n📝 Instructions:")
    print("  - Paste or type an email body")
    print("  - The system will classify it into an intent category")
    print("  - Type 'quit' to exit\n")

    while True:
        try:
            print("-" * 60)
            print("\n✉️  Paste email text below (type 'quit' to exit):\n")
            
            lines = []
            while True:
                line = input()
                if line.lower() == 'quit':
                    print("\n👋 Goodbye!")
                    return
                if line.lower() == 'done':
                    break
                lines.append(line)
            
            email_text = "\n".join(lines).strip()
            
            if not email_text:
                print("⚠️  Empty input. Please try again.\n")
                continue

            print("\n🔄 Classifying email...")
            intent, confidence, top_2 = predict_intent(email_text)

            log_prediction(email_text, intent, confidence)

            print("\n" + "=" * 60)
            print("PREDICTION RESULT")
            print("=" * 60)
            print(f"✓ Final Intent      : {intent}")
            print(f"  Confidence        : {confidence * 100:.2f}%")
            
            print(f"\n🔀 Top 2 Predictions:")
            for i, (label, score) in enumerate(top_2, 1):
                print(f"   {i}. {label:15} : {score * 100:6.2f}%")
            
            print(f"\n💾 Result saved to: {RESULTS_DIR}/")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return
        except Exception as e:
            print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    interactive_predict()
