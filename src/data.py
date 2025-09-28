import os
from typing import Tuple
import pandas as pd
from .utils import set_seed


def load_data(path: str) -> pd.DataFrame:
    """Load CSV from data/raw directory. Expects a train.csv with date and sales columns."""
    df = pd.read_csv(path, parse_dates=["date"]) if os.path.exists(path) else pd.DataFrame()
    return df


def time_split(df: pd.DataFrame, date_col: str = "date", test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test
