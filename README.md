Retail Store Sales Forecasting

This repository contains code for holiday vs non-holiday sales classification and sales amount regression using the "Store Sales - Time Series Forecasting" dataset (Kaggle).

Setup

- Create a Python virtual environment (recommended) and install requirements:

  pip install -r requirements.txt

- Download the dataset as described in `data/README.md` and place the extracted CSV(s) in the `data/` folder.

Reproduce training

- Train baselines (classical ML):

  python src\train_baselines.py

- Train neural nets:

  python src\train_nn.py

Both scripts log parameters, metrics, and artifacts to MLflow (local `mlruns/` by default).

- Random seeds are set in code for reproducibility. See `src/utils.py`.
- Large raw data is not committed. See `data/README.md` for download instructions.
- AI assistance: small snippets and guidance were used from public resources and private AI tools; all code is written/assembled here in the project style.
# Retail-Store-Sales-Forecasting-Machine-Learning-Project
A ML project to predict retail sales from a dataset of a store's sales. Dataset taken from kaggle. Project done for the UPEI Machine Learning, Data Mining (CS-4120-01) course.
