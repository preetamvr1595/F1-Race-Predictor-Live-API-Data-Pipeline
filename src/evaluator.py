"""
Model Performance Evaluation Metrics Utility.
"""

from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from typing import Dict

def evaluate_classifier(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }

def evaluate_regressor(y_true, y_pred) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred))
    }
