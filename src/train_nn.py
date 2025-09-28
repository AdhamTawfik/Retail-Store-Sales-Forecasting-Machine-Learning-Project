"""Train simple neural network models (MLP) for regression and classification and log to MLflow."""
import mlflow
from mlflow import sklearn as mlflow_sklearn
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib

from src.data import load_data, time_split
from src.features import engineer_features
from src.evaluate import classification_metrics, regression_metrics, plot_confusion, plot_residuals
from src.utils import set_seed, makedirs


def run():
    set_seed()
    makedirs("models")
    df = load_data("data/raw/train.csv")
    if df.empty:
        print("No data found at data/raw/train.csv - please download as instructed in data/README.md")
        return

    df = engineer_features(df)
    train, test = time_split(df, test_frac=0.2)
    features = ["month", "dayofweek", "lag_1", "rolling_mean_7", "rolling_std_14"]

    X_train = train[features]
    X_test = test[features]

    y_train_reg = train["sales"]
    y_test_reg = test["sales"]
    y_train_clf = train["is_holiday"]
    y_test_clf = test["is_holiday"]

    mlflow.set_experiment("nn_models")
    with mlflow.start_run(run_name="mlp_run"):
        mlp_r = MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
        mlp_r.fit(X_train, y_train_reg)
        y_pred_reg = mlp_r.predict(X_test)
        reg_metrics = regression_metrics(y_test_reg, y_pred_reg)
        mlflow.log_params({"mlp_reg_layers": (50,)})
        mlflow.log_metrics(reg_metrics)
        joblib.dump(mlp_r, "models/mlp_reg.joblib")
        mlflow_sklearn.log_model(mlp_r, "mlp_reg")
        plot_residuals(y_test_reg, y_pred_reg, "models/mlp_reg_residuals.png")
        mlflow.log_artifact("models/mlp_reg_residuals.png")

        mlp_c = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
        mlp_c.fit(X_train, y_train_clf)
        y_pred_clf = mlp_c.predict(X_test)
        y_proba = mlp_c.predict_proba(X_test)[:, 1] if hasattr(mlp_c, "predict_proba") else None
        clf_metrics = classification_metrics(y_test_clf, y_pred_clf, y_proba)
        mlflow.log_params({"mlp_clf_layers": (50,)})
        mlflow.log_metrics({k: v for k, v in clf_metrics.items() if v is not None})
        joblib.dump(mlp_c, "models/mlp_clf.joblib")
        mlflow_sklearn.log_model(mlp_c, "mlp_clf")
        plot_confusion(y_test_clf, y_pred_clf, "models/mlp_clf_confusion.png")
        mlflow.log_artifact("models/mlp_clf_confusion.png")


if __name__ == "__main__":
    run()
