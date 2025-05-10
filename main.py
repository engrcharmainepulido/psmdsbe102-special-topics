from tune import run_hyperparameter_tuning
from evaluate import evaluate_best_model
import os

if __name__ == "__main__":
    print(">>> Starting script...")
    output_dir = "heart_disease_results"

    os.makedirs(output_dir, exist_ok=True)

    print(">>> Running hyperparameter tuning with Optuna...")
    run_hyperparameter_tuning(output_dir)

    print(">>> Evaluating best model...")
    evaluate_best_model()

    print(">>> Done. You can now run `python serve.py` to deploy the model.")
