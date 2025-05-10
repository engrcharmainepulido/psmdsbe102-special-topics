
import optuna
import mlflow  # ✅ import mlflow before calling its functions
mlflow.set_experiment("HeartDisease_RF")

import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
from config import OUTPUT_DIR
from data import load_data
from models import get_model

def objective(trial):
    df = load_data()
    X = df.drop(["Unnamed: 0", "target"], axis=1, errors='ignore')
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train = X_train.astype('float64')
    X_val = X_val.astype('float64')

    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 300, 500]),
        "max_depth": trial.suggest_categorical("max_depth", [4, 8, 16, 24, 32, None]),
        "min_samples_split": trial.suggest_float("min_samples_split", 0.01, 0.3),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 6]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False])
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        mlflow.log_metric("val_accuracy", acc)

        signature = infer_signature(X_train, preds)
        mlflow.sklearn.log_model(model, "model", input_example=X_train.iloc[:1], signature=signature)

    return acc

def run_hyperparameter_tuning(output_dir):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_trial.params
    best_config_df = pd.DataFrame([best_params])
    best_config_df.to_csv(os.path.join(output_dir, "best_config.csv"), index=False)
    print("Best validation accuracy:", study.best_value)
    print("Best params saved to best_config.csv")
