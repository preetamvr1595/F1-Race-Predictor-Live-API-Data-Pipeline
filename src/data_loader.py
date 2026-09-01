"""
Data Loading and Merging Utilities for F1 Historical Datasets.
"""

import pandas as pd
import os

def load_master_dataset(data_path: str = "data/master_dataset.csv") -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    return pd.read_csv(data_path)

def filter_seasons(df: pd.DataFrame, start_season: int = 2010, end_season: int = 2024) -> pd.DataFrame:
    return df[(df["season"] >= start_season) & (df["season"] <= end_season)].copy()
