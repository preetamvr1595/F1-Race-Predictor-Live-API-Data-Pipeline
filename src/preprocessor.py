"""
Data Preprocessing and Imputation Functions.
"""

import pandas as pd

def preprocess_race_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "position_numeric" not in df.columns:
        df["position_numeric"] = pd.to_numeric(df["position"], errors="coerce").fillna(20).astype(int)
    if "grid" in df.columns:
        df["grid"] = pd.to_numeric(df["grid"], errors="coerce").fillna(20).astype(int)
    return df
