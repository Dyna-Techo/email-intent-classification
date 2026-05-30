"""
Mail Logic - Email Intent Classification System

A machine learning application that automatically classifies emails into 
different intent categories using scikit-learn and provides a GUI interface.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Email Intent Classification System"

from .storage_manager import (
    log_intent_excel,
    log_combined_txt,
    get_all_results,
    get_statistics
)

__all__ = [
    "log_intent_excel",
    "log_combined_txt",
    "get_all_results",
    "get_statistics"
]
