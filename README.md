# Retail Store Sales Forecasting - CS-4120 Midpoint Project

Student: Adham Tawfik

# Project Overview
This project implements machine learning models for retail sales forecasting using the Kaggle Store Sales dataset. We solve two tasks:
1. Classification: Predict holiday vs non-holiday sales periods
2. Regression: Predict sales amounts

# Dataset
- Source: Store Sales - Time Series Forecasting (Kaggle)
- Original Size: 3,000,888 records (2013-2017)
- Working Sample: 150,044 records (5% stratified sample)
- Features: date, store_nbr, family, sales, onpromotion

# Setup Instructions

## Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Data Setup
Place the Kaggle dataset in `data/raw/train.csv`
- Download from: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
- Extract train.csv to `data/raw/` folder

# Reproduction Steps

## Generate Sample Data (if needed)
```bash
python create_sample.py
```

## Run Complete Analysis
```bash
# 1. Generate EDA plots
python create_eda_plots.py

# 2. Train baseline models
python src/train_baselines.py
```

# Model Results

# Regression Models (MAE/RMSE)
- Linear Regression: 3.71 / 12.52
- Random Forest: 2.93 / 8.41

# Classification Models (Accuracy/F1/AUC)
- Logistic Regression: 0.864 / 0.731 / 0.889
- Random Forest: 0.878 / 0.765 / 0.920

# Files Generated
- `plots/plot1_target_distribution.png` - Target analysis
- `plots/plot2_correlation_heatmap.png` - Feature correlations
- `plots/plot3_confusion_matrix.png` - Classification results
- `plots/plot4_residuals.png` - Regression diagnostics

# MLflow Tracking
Experiments are tracked in `mlruns/` directory with:
- Model parameters and hyperparameters
- Performance metrics
- Model artifacts

# Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- mlflow

# Random Seed
All experiments use fixed random seed (42) for reproducibility.

# Notes
- Data preprocessing and feature engineering
- Baseline model implementation
- Results analysis and reporting
- MLflow experiment tracking setup
