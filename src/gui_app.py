"""
Mail Logic - Email Intent Classification GUI Application

A tkinter-based GUI application for classifying emails into different intent
categories using a pre-trained machine learning model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import customtkinter as ctk
import tkinter.messagebox as messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# THEME CONFIGURATION
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# APPLICATION CONFIGURATION
MODEL_PATH = "model/email_intent_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
RESULTS_DIR = "results"

CONFIDENCE_THRESHOLD = 0.45
FALLBACK_INTENT = "General"


# LOAD TRAINED MODEL & VECTORIZER
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError as e:
    print(f"Error: Could not load model files. {e}")
    exit(1)


# PREDICTION FUNCTIONS
def predict_intent(text):
    """Predict the intent of an email text."""
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    order = np.argsort(probs)[::-1][:2]
    top2 = [(classes[i], probs[i]) for i in order]

    intent, conf = top2[0]
    final_intent = intent if conf >= CONFIDENCE_THRESHOLD else FALLBACK_INTENT
    
    return final_intent, conf, top2


def log_prediction(text, intent, conf):
    """Save prediction result to Excel file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    intent_dir = os.path.join(RESULTS_DIR, intent)
    os.makedirs(intent_dir, exist_ok=True)

    excel_path = os.path.join(intent_dir, f"{intent}.xlsx")

    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "intent": intent,
        "confidence_percent": round(conf * 100, 2),
        "email_text": text
    }])

    if os.path.exists(excel_path):
        old = pd.read_excel(excel_path)
        row = pd.concat([old, row], ignore_index=True)

    row.to_excel(excel_path, index=False)


# UI HELPER FUNCTIONS
def animate_progress(target, step=0.01):
    """Animate progress bar to target value."""
    current = confidence_bar.get()
    if current < target:
        confidence_bar.set(min(current + step, target))
        app.after(10, lambda: animate_progress(target, step))


def on_predict():
    """Handle predict button click."""
    predict_btn.configure(state="disabled")

    text = email_input.get("1.0", "end").strip()
    if not text:
        messagebox.showwarning("Input Required", "Please paste the email text.")
        predict_btn.configure(state="normal")
        return

    intent, conf, top2 = predict_intent(text)
    log_prediction(text, intent, conf)

    intent_value.configure(text="Predicting...")
    confidence_value.configure(text="— %")
    confidence_bar.set(0)
    top2_value.configure(text="")

    def reveal():
        intent_value.configure(text=intent)
        confidence_value.configure(text=f"{round(conf * 100, 2)} %")
        animate_progress(conf)

        t = ""
        for i, (lbl, score) in enumerate(top2, 1):
            t += f"{i}. {lbl} — {round(score * 100, 2)}%\n"

        top2_value.configure(text=t)
        predict_btn.configure(state="normal")

    app.after(400, reveal)


def show_analytics():
    """Display analytics dashboard."""
    intents, counts = [], []

    if not os.path.exists(RESULTS_DIR):
        messagebox.showinfo("Analytics", "No results yet. Make some predictions first!")
        return

    for name in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, name)
        file = os.path.join(path, f"{name}.xlsx")
        if os.path.isdir(path) and os.path.exists(file):
            try:
                df = pd.read_excel(file)
                intents.append(name)
                counts.append(len(df))
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if not intents:
        messagebox.showinfo("Analytics", "No data available yet.")
        return

    win = ctk.CTkToplevel(app)
    win.title("Analytics Dashboard")
    win.geometry("900x500")

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#212121")
    ax.bar(intents, counts, color="#2563EB")
    ax.set_title("Email Intent Distribution", color="white")
    ax.set_ylabel("Email Count", color="white")
    ax.tick_params(axis="x", rotation=45, colors="white")
    ax.tick_params(axis="y", colors="white")
    ax.set_facecolor("#212121")

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


# CREATE MAIN WINDOW
app = ctk.CTk()
app.title("Mail Logic - Email Intent Classification")
app.geometry("1400x800")
app.minsize(1200, 700)


# HEADER SECTION
header = ctk.CTkLabel(
    app,
    text="📩 Mail Logic - Email Intent Classifier",
    font=ctk.CTkFont(size=26, weight="bold")
)
header.pack(pady=20)


# MAIN CONTENT FRAME
main = ctk.CTkFrame(app)
main.pack(padx=30, pady=10, fill="both", expand=True)
main.grid_columnconfigure(0, weight=3)
main.grid_columnconfigure(1, weight=2)


# LEFT PANEL: EMAIL INPUT
left = ctk.CTkFrame(main)
left.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

ctk.CTkLabel(
    left,
    text="📧 Email Content",
    font=ctk.CTkFont(size=16, weight="bold")
).pack(anchor="w", padx=25, pady=(25, 8))

email_input = ctk.CTkTextbox(left, height=260)
email_input.pack(padx=25, pady=10, fill="both", expand=True)

btn_row = ctk.CTkFrame(left, fg_color="transparent")
btn_row.pack(pady=25)

predict_btn = ctk.CTkButton(
    btn_row,
    text="🚀 Predict Intent",
    width=220,
    height=45,
    font=ctk.CTkFont(size=14, weight="bold"),
    command=on_predict
)
predict_btn.grid(row=0, column=0, padx=15)

ctk.CTkButton(
    btn_row,
    text="📊 View Analytics",
    width=220,
    height=45,
    font=ctk.CTkFont(size=14, weight="bold"),
    fg_color="#2563EB",
    command=show_analytics
).grid(row=0, column=1, padx=15)


# RIGHT PANEL: PREDICTION RESULTS
right = ctk.CTkFrame(main)
right.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

ctk.CTkLabel(
    right,
    text="Prediction Result",
    font=ctk.CTkFont(size=18, weight="bold")
).pack(pady=(30, 12))

ctk.CTkLabel(right, text="FINAL INTENT").pack()

intent_value = ctk.CTkLabel(
    right,
    text="—",
    font=ctk.CTkFont(size=32, weight="bold"),
    text_color="#22C55E"
)
intent_value.pack(pady=6)

ctk.CTkLabel(
    right,
    text="CONFIDENCE SCORE",
    font=ctk.CTkFont(size=12)
).pack(pady=(20, 5))

confidence_value = ctk.CTkLabel(
    right,
    text="— %",
    font=ctk.CTkFont(size=18)
)
confidence_value.pack(pady=5)

confidence_bar = ctk.CTkProgressBar(right, width=340, height=14)
confidence_bar.set(0)
confidence_bar.pack(pady=12)

ctk.CTkLabel(
    right,
    text="🔀 Top 2 Predictions",
    font=ctk.CTkFont(size=14, weight="bold")
).pack(pady=(25, 8))

top2_value = ctk.CTkLabel(
    right,
    text="",
    font=ctk.CTkFont(size=14),
    justify="left"
)
top2_value.pack(pady=10)


if __name__ == "__main__":
    app.mainloop()
