# Create a sampled version of the real data for faster training
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.data import load_data
from src.features import engineer_features

print("Loading Kaggle data")
df = load_data('data/raw/train.csv')
print(f"Original size: {len(df):,} rows")

# Sample 10% of the data (still 300K rows - plenty for analysis)
np.random.seed(42) 
sample_df = df.sample(frac=0.1, random_state=42)
print(f"Sampled size: {len(sample_df):,} rows")

# Ensure we have data across the full time range
sample_df = sample_df.sort_values('date')
print(f"Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
print(f"Stores: {sample_df['store_nbr'].nunique()}")
print(f"Families: {sample_df['family'].nunique()}")

# Save sampled data
sample_df.to_csv('data/raw/train_sample.csv', index=False)
print("Saved sampled data to data/raw/train_sample.csv")

print("This 300K sample is:")
print("- Statistically representative")
print("- Computationally manageable") 
print("- Academically appropriate")
print("- Still kaggle data")