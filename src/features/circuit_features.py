"""
Circuit Specific Driver History Feature Extractor.
"""

import pandas as pd

def compute_circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["driver_circuit_avg_position"] = df.groupby(["driverId", "circuitId"])["position_numeric"].transform("mean")
    return df
