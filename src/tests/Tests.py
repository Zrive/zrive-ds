import requests
url = "http://localhost:8000/predict"
payload = {"user_id": "75550bb2c8f44796dc817aadeb8d3ddcd223d568ccddfd4df9e5f315173ba62a298484421d366d557b1cf6b8945e7c3e41f33cb71d0b51ea3cc633a63a4bd855"}
response = requests.post(url, json=payload)
print("Status code:", response.status_code)
print("Response:", response.json())
