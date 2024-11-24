import os
import joblib
import numpy as np

from src.exceptions import PredictionException


MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "bin/model.joblib")
)


class BasketModel:
    def __init__(self):
        self.model = joblib.load(MODEL)

    def predict(self, features: np.ndarray) -> np.ndarray:
        try:
            pred = self.model.predict(features)
        except Exception as exception:
            raise PredictionException("Error during model inference") from exception
        return pred
