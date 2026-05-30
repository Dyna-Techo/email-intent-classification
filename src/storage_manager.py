"""
Mail Logic - Storage Manager

Handles file I/O operations for storing and managing classification results.
"""

import os
import pandas as pd
from datetime import datetime


BASE_DIR = "results"


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def log_combined_txt(email_link, subject, intent):
    """Log prediction to combined text file."""
    path = os.path.join(BASE_DIR, "combined")
    ensure_dir(path)

    file = os.path.join(path, "all_predictions.txt")

    with open(file, "a", encoding="utf-8") as f:
        f.write(f"Email Link: {email_link}\n")
        f.write(f"Subject: {subject}\n")
        f.write(f"Intent: {intent}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")


def log_intent_excel(intent, email_link, subject, confidence=None):
    """Log prediction to Excel file organized by intent category."""
    intent_lower = intent.lower()
    folder = os.path.join(BASE_DIR, intent_lower)
    ensure_dir(folder)

    file = os.path.join(folder, f"{intent_lower}.xlsx")

    row_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Email Link": email_link,
        "Subject": subject,
        "Intent": intent.capitalize()
    }
    
    if confidence is not None:
        row_data["Confidence"] = round(confidence * 100, 2)

    row = pd.DataFrame([row_data])

    if os.path.exists(file):
        df = pd.read_excel(file)
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row

    df.to_excel(file, index=False)


def get_all_results(intent=None):
    """Retrieve all classification results."""
    results = {}
    
    if not os.path.exists(BASE_DIR):
        return results
    
    if intent:
        intent_dir = os.path.join(BASE_DIR, intent.lower())
        intent_file = os.path.join(intent_dir, f"{intent.lower()}.xlsx")
        
        if os.path.exists(intent_file):
            results[intent] = pd.read_excel(intent_file)
    else:
        for folder in os.listdir(BASE_DIR):
            folder_path = os.path.join(BASE_DIR, folder)
            if os.path.isdir(folder_path):
                file = os.path.join(folder_path, f"{folder}.xlsx")
                if os.path.exists(file):
                    results[folder] = pd.read_excel(file)
    
    return results


def get_statistics():
    """Calculate statistics from all results."""
    results = get_all_results()
    
    stats = {
        "total_predictions": 0,
        "categories": {}
    }
    
    for intent, df in results.items():
        count = len(df)
        stats["total_predictions"] += count
        stats["categories"][intent] = {
            "count": count,
            "percentage": 0
        }
    
    total = stats["total_predictions"]
    if total > 0:
        for intent in stats["categories"]:
            percentage = (stats["categories"][intent]["count"] / total) * 100
            stats["categories"][intent]["percentage"] = round(percentage, 2)
    
    return stats
