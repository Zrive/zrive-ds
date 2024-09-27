import os
import joblib
import pandas as pd
import boto3
import time
import psutil
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict
from module_6.basket_model.exceptions import UserNotFoundException
from module_6.basket_model.feature_store import FeatureStore
from module_6.basket_model.basket_model import BasketModel
from datetime import datetime

app = FastAPI()

feature_store = FeatureStore()
basket_model = BasketModel()

LOG_FILE = "/home/ebacigalupe/zrive-ds/zrive-ds/src/module_6/service_metrics.txt"
error_count = 0
request_count = 0

def log_metrics(message: str) -> None:
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{message}\n")

@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    global request_count, error_count

    start_time = time.time()
    request_count += 1
    
    try:
        response = await call_next(request)
        latency = time.time() - start_time
        log_metrics(f"Timestamp: {datetime.now()}, Latency: {latency:.4f}s, Traffic: {request_count}, Errors: {error_count}, Saturation: {psutil.cpu_percent()}%")
        return response
    except Exception as e:
        error_count += 1
        log_metrics(f"Timestamp: {datetime.now()}, Error: {str(e)}")
        raise e

class PredictionRequest(BaseModel):
    user_id: str

@app.get("/status")
def status() -> Dict[str, Any]:
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest) -> Dict[str, Any]:
    user_id = request.user_id
    try:
        features = feature_store.get_features(user_id)
        prediction = basket_model.predict(features.values.reshape(1, -1))
        log_metrics(f"Prediction for user {user_id}: {prediction[0]}")
        return {"user_id": user_id, "predicted_price": prediction[0]}
    except UserNotFoundException as e:
        log_metrics(f"Error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_metrics(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
