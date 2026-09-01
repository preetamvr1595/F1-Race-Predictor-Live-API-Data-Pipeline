"""
Driver Form and Performance Feature Extractors.
"""

import pandas as pd

def compute_driver_recent_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["driverId", "season", "round"]).copy()
    df["recent_form_avg_position"] = df.groupby("driverId")["position_numeric"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df["recent_form_avg_points"] = df.groupby("driverId")["points"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return df
