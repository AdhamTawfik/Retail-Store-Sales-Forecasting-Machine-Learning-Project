Download instructions

This project uses the "Store Sales - Time Series Forecasting" dataset from Kaggle that's been approved and provided in the project description PDF.

1. Go to https://www.kaggle.com/competitions/store-sales-time-series-forecasting and download the dataset.
2. Extract the CSV files and place them in `data/raw/`.

Expected files (example):
- `data/raw/train.csv`
- `data/raw/test.csv`

Retail Sales Forecasting - CS-4120

 Setup
1. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)
2. Install dependencies:
   pip install -r requirements.txt

Dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset

Place it in `data/raw/` folder.

Run
 Train classical ML models:
  python src/train_baselines.py
 Train neural networks:
  python src/train_nn.py
 Evaluate models:
  python src/evaluate.py

AI Disclosure
ChatGPT was used for ideas on the structure of files and code.
