import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame, date_col: str = "date", target_col: str = "sales") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(date_col)
    # time features
    df["month"] = df[date_col].dt.month
    df["dayofweek"] = df[date_col].dt.dayofweek

    # Feature 1: rolling mean of past 7 days
    df["lag_1"] = df[target_col].shift(1)
    df["rolling_mean_7"] = df[target_col].shift(1).rolling(window=7, min_periods=1).mean()

    # Feature 2: rolling std of past 14 days
    df["rolling_std_14"] = df[target_col].shift(1).rolling(window=14, min_periods=1).std().fillna(0)

    # Classification target: holiday vs non-holiday (assume 'is_holiday' exists or derive)
    if "is_holiday" not in df.columns:
        # crude heuristic: weekends as holidays
        df["is_holiday"] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)

    # Drop rows with NA target
    df = df.dropna(subset=[target_col])
    return df
