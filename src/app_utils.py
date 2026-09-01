"""
Streamlit App Data Loading and Caching Utilities.
"""

import os

def get_data_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

def get_model_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
