from pydantic import BaseModel

class CustomerData(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    # Add all your model input features here

class PredictionResponse(BaseModel):
    risk_probability: float
