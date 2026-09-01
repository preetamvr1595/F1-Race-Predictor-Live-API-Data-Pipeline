"""
Constructor Team Performance and Pace Metrics.
"""

import pandas as pd

def compute_team_pace(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["constructorId", "season", "round"]).copy()
    df["team_recent_avg_points"] = df.groupby("constructorId")["points"].transform(
        lambda x: x.rolling(window * 2, min_periods=1).mean()
    )
    return df
