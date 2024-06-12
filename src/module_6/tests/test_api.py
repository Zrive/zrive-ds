from fastapi.testclient import TestClient
from module_6.app import app

client = TestClient(app)

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_user_not_found():
    response = client.post("/predict", json={"user_id": "unknown_user"})
    assert response.status_code == 404
    assert response.json() == {"detail": "User not found in feature store"}

def test_predict_success():
    response = client.post("/predict", json={"user_id": "known_user"})
    assert response.status_code == 200
    assert "predicted_price" in response.json()