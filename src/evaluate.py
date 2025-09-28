import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             mean_absolute_error, mean_squared_error,
                             confusion_matrix)


def classification_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    return {"accuracy": acc, "f1": f1, "roc_auc": roc}


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"mae": mae, "rmse": rmse}


def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()


def plot_residuals(y_true, y_pred, out_path):
    res = y_true - y_pred
    plt.figure()
    sns.histplot(res, kde=True)
    plt.title("Residuals")
    plt.savefig(out_path)
    plt.close()
