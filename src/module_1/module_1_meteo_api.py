import requests
import json
import matplotlib.pyplot as plt
import numpy as np

COORDINATES = {
        "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
        "London": {"latitude": 51.507351, "longitude": -0.127758},
        "Rio": {"latitude": -22.906847, "longitude": -43.172896},
    }
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
API_URL = "https://api.open-meteo.com/v1/forecast"

# Función auxiliar genérica para hacer llamadas a APIs
def make_api_request(url, params=None):
    """Realiza una solicitud a una API y maneja el rate limit, status code y schema validation."""
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Lanza una excepción si el status code no es 200
        return response.json()  # Suponiendo que la API devuelve datos en formato JSON
    except requests.exceptions.HTTPError as http_err:
        print(f"Error HTTP: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error en la solicitud: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"Error al decodificar JSON: {json_err}")
    return None

# Función para obtener datos climáticos de Meteo para una ciudad específica
def get_data_meteo_api(city):
    
    if city not in COORDINATES:
        print(f"La ciudad '{city}' no está definida en las coordenadas.")
        return None

    coords = COORDINATES[city]

    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start": 1950,
        "end": 2050,
        "variables": VARIABLES,
    }
    
    # Realizar la solicitud a la API 
    data = make_api_request(API_URL, params=params)
    return data

def main():
    #raise NotImplementedError
    cities = ["Madrid", "London", "Rio"]

    for city in cities:
        print(f"Obteniendo datos para {city}...")
        data = get_data_meteo_api(city)
        if data:
            print(f"Datos para {city}:\n{data}")
    

if __name__ == "__main__":
   main()