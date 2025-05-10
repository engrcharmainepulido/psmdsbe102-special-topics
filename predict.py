
import joblib
import numpy as np

def predict(model_path, input_features):
    model = joblib.load(model_path)
    prediction = model.predict([input_features])[0]
    confidence = model.predict_proba([input_features]).max()
    return prediction, confidence
