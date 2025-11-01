# Create EDA plots following professor's simple style
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_eda_plots():
    # Create Plot 1 and Plot 2 - following professor's style
    
    # Load data
    df = pd.read_csv('data/raw/train_sample.csv', parse_dates=['date'])
    print(f"Loaded data: {len(df)} rows")
    
    # Basic feature engineering
    df['is_holiday'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df = df.dropna(subset=['sales'])
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # PLOT 1: Target Distribution
    print("Creating Plot 1: Target distribution")
    
    plt.figure(figsize=(12, 5))
    
    # Classification target
    plt.subplot(1, 2, 1)
    counts = df['is_holiday'].value_counts()
    plt.bar(['Non-Holiday', 'Holiday'], counts.values, color=['lightblue', 'orange'])
    plt.title('Classification Target Distribution')
    plt.ylabel('Count')
    for i, v in enumerate(counts.values):
        plt.text(i, v + 500, str(v), ha='center')
    
    # Regression target  
    plt.subplot(1, 2, 2)
    plt.hist(df['sales'], bins=50, alpha=0.7, color='green')
    plt.title('Regression Target Distribution')
    plt.xlabel('Sales Amount')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('plots/plot1_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot 1 saved")
    
    # PLOT 2: Correlation Heatmap
    print("Creating Plot 2: Correlation heatmap")
    
    # Simple feature engineering for correlation
    df_corr = df.copy()
    df_corr['month'] = df_corr['date'].dt.month
    df_corr['dayofweek'] = df_corr['date'].dt.dayofweek
    df_corr['lag_1'] = df_corr['sales'].shift(1)
    df_corr['rolling_mean_7'] = df_corr['sales'].shift(1).rolling(window=7, min_periods=1).mean()
    
    # Select features for correlation
    corr_features = ['sales', 'onpromotion', 'month', 'dayofweek', 
                    'lag_1', 'rolling_mean_7', 'is_holiday']
    
    corr_matrix = df_corr[corr_features].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('plots/plot2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot 2 saved")
    
    # Print basic stats
    print(f"Basic Statistics:")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Stores: {df['store_nbr'].nunique()}")
    print(f"Families: {df['family'].nunique()}")
    print(f"Holiday distribution: {df['is_holiday'].value_counts().to_dict()}")
    print(f"Sales stats: Mean={df['sales'].mean():.2f}, Std={df['sales'].std():.2f}")

if __name__ == "__main__":
    create_eda_plots()