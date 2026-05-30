"""
Mail Logic - Model Training Script

Trains a Logistic Regression classifier using TF-IDF vectorization
to classify emails into different intent categories.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# CONFIGURATION
DATA_PATH = "data/combined_dataset.csv"
MODEL_DIR = "model"
CONFIDENCE_THRESHOLD = 0.45

os.makedirs(MODEL_DIR, exist_ok=True)


# STEP 1: LOAD DATASET
print("=" * 50)
print("EMAIL INTENT CLASSIFICATION - MODEL TRAINING")
print("=" * 50)
print("\n📂 Loading dataset...")

try:
    data = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded successfully")
except FileNotFoundError:
    print(f"✗ Error: Dataset not found at {DATA_PATH}")
    exit(1)

print(f"\nDataset Shape: {data.shape}")
print(f"\n📊 Intent Distribution:")
print(data["intent"].value_counts())


# STEP 2: TRAIN-TEST SPLIT
print("\n" + "=" * 50)
print("PREPARING DATA")
print("=" * 50)

X = data["text"]
y = data["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train/Test Split (80/20)")
print(f"  Training samples  : {len(X_train)}")
print(f"  Testing samples   : {len(X_test)}")


# STEP 3: TF-IDF VECTORIZATION
print("\n" + "=" * 50)
print("FEATURE EXTRACTION")
print("=" * 50)

print("\n📝 TF-IDF Vectorization...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\n✓ Vectorization complete")
print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())} features")


# STEP 4: TRAIN LOGISTIC REGRESSION MODEL
print("\n" + "=" * 50)
print("MODEL TRAINING")
print("=" * 50)

print("\n🤖 Training Logistic Regression...")

model = LogisticRegression(
    max_iter=4000,
    C=3.0,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_vec, y_train)
print("\n✓ Model training complete")


# STEP 5: EVALUATION
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📈 Accuracy: {accuracy:.4f}")
print("\nDetailed Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred))


# STEP 6: SAVE MODEL
print("\n" + "=" * 50)
print("SAVING MODEL")
print("=" * 50)

model_path = os.path.join(MODEL_DIR, "email_intent_model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"\n✓ Model saved to: {model_path}")
print(f"✓ Vectorizer saved to: {vectorizer_path}")

print("\n" + "=" * 50)
print("✓ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)
print("\nYou can now use gui_app.py for predictions.")      
