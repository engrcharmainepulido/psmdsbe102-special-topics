
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("heart_disease_results/best_model.pkl")

class HeartDiseaseInput(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Heart Disease Predictor is running!"}

@app.post("/predict")
async def predict(input_data: HeartDiseaseInput):
    input_array = np.array([[input_data.age, input_data.sex, input_data.cp,
                             input_data.trestbps, input_data.chol, input_data.fbs,
                             input_data.restecg, input_data.thalach, input_data.exang,
                             input_data.oldpeak, input_data.slope, input_data.ca,
                             input_data.thal]])
    prediction = model.predict(input_array)[0]
    confidence = model.predict_proba(input_array).max()
    return {"prediction": int(prediction), "confidence": round(float(confidence), 4)}

# ✅ Add this to run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)