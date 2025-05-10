
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import OUTPUT_DIR, DATA_PATH

def evaluate_best_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(["Unnamed: 0", "target"], axis=1, errors='ignore')
    y = df["target"]
    _, X_val, _, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Set MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("HeartDisease_RF")
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("HeartDisease_RF")
    if not experiment:
        print("No experiment found.")
        return

    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        print("No runs found.")
        return

    # Get best model
    best_run = max(runs, key=lambda r: r.data.metrics.get("val_accuracy", 0))
    model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
    preds = model.predict(X_val)

    # === Save confusion matrix ===
    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # ✅ Replace this line:
    # plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # ✅ With this debug version:
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")

    # === Save model ===
    joblib_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    joblib.dump(model, joblib_path)
    print(f"Best model saved to: {joblib_path}")