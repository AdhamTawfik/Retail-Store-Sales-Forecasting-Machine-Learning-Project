"""Train baseline classical models for classification and regression and log to MLflow."""
import os
import mlflow
from mlflow import sklearn as mlflow_sklearn
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data import load_data, time_split
from src.features import engineer_features
from src.evaluate import classification_metrics, regression_metrics, plot_confusion, plot_residuals
from src.utils import set_seed, makedirs


def run():
    set_seed()
    makedirs("models")
    # Expecting data/raw/train.csv
    df = load_data("data/raw/train.csv")
    if df.empty:
        print("No data found at data/raw/train.csv - please download as instructed in data/README.md")
        return

    df = engineer_features(df)
    train, test = time_split(df, test_frac=0.2)

    features = ["month", "dayofweek", "lag_1", "rolling_mean_7", "rolling_std_14"]

    X_train = train[features]
    X_test = test[features]

    # Regression target
    y_train_reg = train["sales"]
    y_test_reg = test["sales"]

    # Classification target
    y_train_clf = train["is_holiday"]
    y_test_clf = test["is_holiday"]

    mlflow.set_experiment("baselines")
    with mlflow.start_run(run_name="baselines_run"):
        # Regression
        rfr = RandomForestRegressor(n_estimators=50, random_state=42)
        rfr.fit(X_train, y_train_reg)
        y_pred_reg = rfr.predict(X_test)
        reg_metrics = regression_metrics(y_test_reg, y_pred_reg)
        mlflow.log_params({"rfr_n_estimators": 50})
        mlflow.log_metrics(reg_metrics)
        joblib.dump(rfr, "models/rfr_baseline.joblib")
        mlflow_sklearn.log_model(rfr, "rfr_baseline")
        plot_residuals(y_test_reg, y_pred_reg, "models/rfr_residuals.png")
        mlflow.log_artifact("models/rfr_residuals.png")

        # Classification
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train_clf)
        y_pred_clf = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
        clf_metrics = classification_metrics(y_test_clf, y_pred_clf, y_proba)
        mlflow.log_params({"clf_n_estimators": 50})
        mlflow.log_metrics({k: v for k, v in clf_metrics.items() if v is not None})
        joblib.dump(clf, "models/clf_baseline.joblib")
        mlflow_sklearn.log_model(clf, "clf_baseline")
        plot_confusion(y_test_clf, y_pred_clf, "models/clf_confusion.png")
        mlflow.log_artifact("models/clf_confusion.png")


if __name__ == "__main__":
    run()
