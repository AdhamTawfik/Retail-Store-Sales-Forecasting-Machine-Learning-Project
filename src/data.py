import os
from typing import Tuple
import pandas as pd
from .utils import set_seed


def load_data(path: str) -> pd.DataFrame:
    # Load CSV from data/raw directory. Expects a train.csv with date and sales columns.
    # Try sample first for faster processing, fall back to full data
    sample_path = path.replace('train.csv', 'train_sample.csv')
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path, parse_dates=["date"])
        print(f"Using sampled data: {len(df):,} rows (for faster processing)")
    elif os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        print(f"Using full data: {len(df):,} rows")
    else:
        df = pd.DataFrame()
    return df


def time_split(df: pd.DataFrame, date_col: str = "date", test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test
