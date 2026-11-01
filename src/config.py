"""
Centralized Configuration and Settings Module.
Loads environment variables and sets standard directory paths and constants.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core Directories
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", PROJECT_ROOT / "static"))

# API Settings
JOLPICA_BASE_URL = os.getenv("JOLPICA_API_BASE_URL", "http://api.jolpi.ca/ergast/f1")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "10"))

# Server Settings
PORT = int(os.getenv("PORT", "8501"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Required Model Files
MODEL_FILES = {
    "podium": MODEL_DIR / "podium_model.pkl",
    "winner": MODEL_DIR / "winner_model.pkl",
    "position": MODEL_DIR / "position_model.pkl",
    "feature_cols": MODEL_DIR / "feature_cols.pkl",
}

# Required Data Files
DATA_FILES = {
    "driver_stats": DATA_DIR / "latest_driver_stats.csv",
    "team_stats": DATA_DIR / "latest_team_stats.csv",
    "circuit_stats": DATA_DIR / "latest_circuit_stats.csv",
    "races": DATA_DIR / "races.csv",
    "master": DATA_DIR / "master_dataset.csv",
}
