"""
Unit Tests for Feature Engineering Modules.
"""

import pandas as pd
from src.features.driver_features import compute_driver_recent_form
from src.features.reliability_features import compute_dnf_rate

def test_driver_recent_form():
    data = pd.DataFrame({
        "driverId": ["hamilton", "hamilton"],
        "season": [2024, 2024],
        "round": [1, 2],
        "position_numeric": [1, 3],
        "points": [25.0, 15.0]
    })
    res = compute_driver_recent_form(data)
    assert "recent_form_avg_position" in res.columns
    assert res["recent_form_avg_position"].iloc[-1] == 2.0

def test_dnf_rate():
    data = pd.DataFrame({
        "driverId": ["verstappen", "verstappen"],
        "status": ["Finished", "Engine"]
    })
    res = compute_dnf_rate(data)
    assert res["driver_dnf_rate"].iloc[0] == 0.5
