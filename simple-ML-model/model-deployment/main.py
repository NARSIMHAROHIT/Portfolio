from typing import List
import joblib
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Model API")

# Load trained model
model = joblib.load("model.pkl")

class PredictRequest(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return {"message": "Welcome to the ML model deployment API"}

@app.post("/predict")
def predict(request: PredictRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
