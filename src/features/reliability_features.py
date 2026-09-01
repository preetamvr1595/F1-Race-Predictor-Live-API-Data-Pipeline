"""
Driver DNF (Did Not Finish) and Grid Penalty Feature Extractors.
"""

import pandas as pd

def compute_dnf_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_dnf"] = (df["status"] != "Finished").astype(int)
    df["driver_dnf_rate"] = df.groupby("driverId")["is_dnf"].transform("mean")
    return df
