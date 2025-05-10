# Heart Disease Prediction Model Pipeline

A complete machine learning pipeline for heart disease prediction, featuring hyperparameter tuning, model evaluation, and deployment as a REST API.

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting heart disease using a Random Forest classifier. The pipeline includes:

- Data loading and preprocessing
- Hyperparameter tuning with Optuna
- Model evaluation and metrics visualization
- Model serving via FastAPI
- MLflow experiment tracking
- Command-line interface for easy operation

## Repository Structure

```
model-pipeline/
├── heart_disease_results/       # Output directory for model artifacts
│   ├── best_config.csv          # Best hyperparameters
│   ├── best_model.pkl           # Serialized model
│   └── confusion_matrix.png     # Evaluation visualization
├── mlruns/                      # MLflow tracking data
├── cli.py                       # Command-line interface
├── config.py                    # Configuration settings
├── data.py                      # Data loading utilities
├── evaluate.py                  # Model evaluation
├── heart_disease.csv            # Dataset
├── main.py                      # Main execution script
├── models.py                    # Model definitions
├── predict.py                   # Prediction utilities
├── requirements.txt             # Dependencies
├── serve.py                     # FastAPI server
├── tune.py                      # Hyperparameter tuning
└── utils.py                     # Utility functions
```

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd model-pipeline
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Using the Command Line Interface

The project provides a convenient CLI for all operations:

```bash
# Run hyperparameter tuning and train the model
python cli.py train

# Evaluate the best model and generate metrics
python cli.py evaluate

# Start the prediction API server
python cli.py serve
```

### Running the Complete Pipeline

To run the entire pipeline (tuning and evaluation):

```bash
python main.py
```

### Making Predictions

Once the model is trained and the server is running, you can make predictions:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "age": 63.0,
  "sex": 1,
  "cp": 3,
  "trestbps": 145.0,
  "chol": 233.0,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150.0,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}'
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: Tuned using Optuna with 20 trials
- **Metrics**: Accuracy, Confusion Matrix
- **Tracking**: MLflow experiment tracking

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Prediction endpoint that accepts patient data and returns heart disease prediction

### Input Format

```json
{
  "age": 63.0,
  "sex": 1,
  "cp": 3,
  "trestbps": 145.0,
  "chol": 233.0,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150.0,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

### Output Format

```json
{
  "prediction": 1,
  "confidence": 0.9245
}
```

## Dependencies

- scikit-learn
- pandas
- matplotlib
- seaborn
- joblib
- mlflow
- fastapi
- uvicorn
- pyngrok
- optuna
- typer

## Future Improvements

- Add more model options
- Create a web interface for predictions
- Containerize the application with Docker
