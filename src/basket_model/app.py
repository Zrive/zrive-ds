import uvicorn
from fastapi import FastAPI
from src.basket_model.feature_store import FeatureStore
from src.basket_model.basket_model import BasketModel
import logging
from datetime import datetime
import json
from fastapi.responses import JSONResponse
from fastapi import Request
from src.exceptions import UserNotFoundException, PredictionException

app = FastAPI()

logging.basicConfig(
    filename="service_logs.txt",
    level=logging.INFO,
    format="%(message)s"
)

def log_prediction(user_id, features, prediction):
    entry = {
        "type": "prediction",
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": str(user_id),
        "features": features.tolist() if hasattr(features, "tolist") else features,
        "prediction": float(prediction)
    }
    logging.info(json.dumps(entry))

def log_error(error_type, message, user_id=None):
    entry = {
        "type": "error",
        "timestamp": datetime.utcnow().isoformat(),
        "error_type": error_type,
        "user_id": str(user_id) if user_id is not None else None,
        "message": message
    }
    logging.error(json.dumps(entry))

fs = FeatureStore()
model = BasketModel()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/status")
def read_root():
    return {"message": "Code 200: Service is up and running."}

@app.post("/predict")
def predict(payload: dict):
    try:
        user_id = payload["user_id"]
        features = fs.get_features(user_id)
        features_array = features.values.reshape(1, -1)
        pred = model.predict(features_array)
        log_prediction(user_id, features_array, pred[0])
        return {"prediction": float(pred[0])}
    except UserNotFoundException as exc:
        log_error("UserNotFound", str(exc), payload.get("user_id"))
        return JSONResponse(status_code=404, content={"error": "User not found"})
    except PredictionException as exc:
        log_error("PredictionError", str(exc), payload.get("user_id"))
        return JSONResponse(status_code=500, content={"error": "Prediction failed"})
    except Exception as exc:
        log_error("GeneralError", str(exc), payload.get("user_id"))
        return JSONResponse(status_code=500, content={"error": "Internal error"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
