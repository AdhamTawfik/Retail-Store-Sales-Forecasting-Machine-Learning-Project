# Train baseline models for classification and regression tasks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import os

def load_and_prepare_data():
    # Load data and prepare features
    # Load data
    df = pd.read_csv('data/raw/train_sample.csv', parse_dates=['date'])
    print(f"Loaded data: {len(df)} rows")
    
    # Basic feature engineering
    df = df.sort_values('date')
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    
    # Add lag features
    df['lag_1'] = df['sales'].shift(1)
    df['rolling_mean_7'] = df['sales'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_std_14'] = df['sales'].shift(1).rolling(window=14, min_periods=1).std().fillna(0)
    
    # Classification target: use weekend as holiday 
    df['is_holiday'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Remove missing values
    df = df.dropna(subset=['sales'])
    print(f"After preprocessing: {len(df)} rows")
    
    return df

def run_baselines():
    # Train baseline models
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Features for modeling
    features = ['month', 'dayofweek', 'lag_1', 'rolling_mean_7', 'rolling_std_14']
    X = df[features].fillna(0)  # fill missing values with 0
    
    # Targets
    y_reg = df['sales']  # regression target
    y_clf = df['is_holiday']  # classification target
    
    print(f"Features: {features}")
    print(f"Data shape: {X.shape}")
    print(f"Class balance: {y_clf.value_counts().to_dict()}")
    
    # Train/Test split
    # Split chronologically (80/20)
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_reg_train = y_reg.iloc[:split_idx]
    y_reg_test = y_reg.iloc[split_idx:]
    y_clf_train = y_clf.iloc[:split_idx]
    y_clf_test = y_clf.iloc[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Start MLflow experiment
    mlflow.set_experiment("retail_baselines")
    
    with mlflow.start_run():
        # Regression models
        print("Training Regression Models")
        
        # Linear Regression Pipeline
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        lr_pipeline.fit(X_train, y_reg_train)
        y_reg_pred_lr = lr_pipeline.predict(X_test)
        
        mae_lr = mean_absolute_error(y_reg_test, y_reg_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_lr))
        print(f"Linear Regression - MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")
        
        # Random Forest Regressor Pipeline
        rfr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
        ])
        rfr_pipeline.fit(X_train, y_reg_train)
        y_reg_pred_rfr = rfr_pipeline.predict(X_test)
        
        mae_rfr = mean_absolute_error(y_reg_test, y_reg_pred_rfr)
        rmse_rfr = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_rfr))
        print(f"Random Forest Reg - MAE: {mae_rfr:.2f}, RMSE: {rmse_rfr:.2f}")
        
        # Classification models
        print("Training Classification Models")
        
        # Logistic Regression Pipeline
        log_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        log_pipeline.fit(X_train, y_clf_train)
        y_clf_pred_log = log_pipeline.predict(X_test)
        y_clf_proba_log = log_pipeline.predict_proba(X_test)[:, 1]
        
        acc_log = metrics.accuracy_score(y_clf_test, y_clf_pred_log)
        f1_log = metrics.f1_score(y_clf_test, y_clf_pred_log)
        auc_log = metrics.roc_auc_score(y_clf_test, y_clf_proba_log)
        print(f"Logistic Regression - Accuracy: {acc_log:.3f}, F1: {f1_log:.3f}, AUC: {auc_log:.3f}")
        
        # Random Forest Classifier Pipeline
        rfc_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
        ])
        rfc_pipeline.fit(X_train, y_clf_train)
        y_clf_pred_rfc = rfc_pipeline.predict(X_test)
        y_clf_proba_rfc = rfc_pipeline.predict_proba(X_test)[:, 1]
        
        acc_rfc = metrics.accuracy_score(y_clf_test, y_clf_pred_rfc)
        f1_rfc = metrics.f1_score(y_clf_test, y_clf_pred_rfc)
        auc_rfc = metrics.roc_auc_score(y_clf_test, y_clf_proba_rfc)
        print(f"Random Forest Clf - Accuracy: {acc_rfc:.3f}, F1: {f1_rfc:.3f}, AUC: {auc_rfc:.3f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'lr_mae': mae_lr, 'lr_rmse': rmse_lr,
            'rfr_mae': mae_rfr, 'rfr_rmse': rmse_rfr,
            'log_accuracy': acc_log, 'log_f1': f1_log, 'log_auc': auc_log,
            'rfc_accuracy': acc_rfc, 'rfc_f1': f1_rfc, 'rfc_auc': auc_rfc
        })
        
        # Create plots
        create_plots(y_reg_test, y_reg_pred_rfr, y_clf_test, y_clf_pred_rfc)
        
        print("Results Summary")
        print(f"Best Regression Model: {'Random Forest' if rmse_rfr < rmse_lr else 'Linear Regression'}")
        print(f"Best Classification Model: {'Random Forest' if f1_rfc > f1_log else 'Logistic Regression'}")

def create_plots(y_reg_test, y_reg_pred, y_clf_test, y_clf_pred):
    # Create required plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_clf_test, y_clf_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Best Classification Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('plots/plot3_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Residuals
    residuals = y_reg_test - y_reg_pred
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_reg_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    
    plt.tight_layout()
    plt.savefig('plots/plot4_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plots saved to plots/ directory")

if __name__ == "__main__":
    run_baselines()
