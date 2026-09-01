"""
Integration Tests for Trained ML Models and Pipeline Inference.
"""

import os

def test_model_files_exist():
    model_dir = "models"
    for m in ["podium_model.pkl", "winner_model.pkl", "position_model.pkl", "feature_cols.pkl"]:
        path = os.path.join(model_dir, m)
        assert os.path.exists(path), f"{path} does not exist"
