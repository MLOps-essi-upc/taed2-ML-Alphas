from fastapi import FastAPI, Query
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

# Create a Pydantic class to represent the payload for /models/{type}
class PredictPayload(BaseModel):
    feature1: float
    feature2: float
    # Add more features as needed

# Create a dictionary to store model information
models_info = {
    "model1": {"type": "model1", "metric": 0.95},
    "model2": {"type": "model2", "metric": 0.92},
    # Add more models as needed
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the API!"}

@app.get("/models")
def get_models(type: Optional[str] = None):
    if type:
        # Filter models by type if type query parameter is provided
        filtered_models = {key: value for key, value in models_info.items() if value["type"] == type}
        return filtered_models
    return models_info

@app.get("/models/{type}")
def predict(type: str, payload: PredictPayload):
    # Use type to look up the model and perform prediction
    if type not in models_info:
        return {"error": "Model not found"}
    
    # Simulate model prediction (replace with your actual model code)
    prediction_result = f"Prediction for {type}: {payload.feature1}, {payload.feature2}"
    
    return {"prediction": prediction_result}
