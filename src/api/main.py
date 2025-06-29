from fastapi import FastAPI, HTTPException
from pydantic_models import CustomerData, PredictionResponse
import numpy as np
import mlflow.sklearn

app = FastAPI()

def load_model(model_name="credit_risk_model", stage="Production"):
    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # Convert incoming data to numpy array matching model input
        features = np.array([[data.feature_1, data.feature_2, data.feature_3]])
        prob = model.predict_proba(features)[0][1]  # Probability of class 1
        return PredictionResponse(risk_probability=prob)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
