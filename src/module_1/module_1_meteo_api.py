import requests
import time
from typing import Dict, Union, Optional, Any

# import pandas as pd

API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

MODELS = [
    "CMCC_CM2_VHR4",
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]


def get_api_data(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Optional[Dict[str, Any]]:
    for _ in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud a la API: {e}")
            time.sleep(retry_delay)
    return None


def get_data_meteo_api(city, start_date, end_date, models=MODELS):
    params = {
        **COORDINATES[city],
        "start_date": start_date,
        "end_date": end_date,
        "models": models,
        "daily": VARIABLES.split(","),
    }
    data = get_api_data(API_URL, params=params)
    """
    cosas que me importan de la respuesta:
    - daily_units (dict) -> Es un diccionario con las unidades de las variables
    - daily (dict) -> Es un diccionario con las variables

    precauciones con:
    - Hay que tener en cuenta que las variables pueden no estar disponibles
        para todos los modelos
    """
    print(data)


def main():
    response = get_data_meteo_api("Madrid", "2020-01-01", "2020-01-31", MODELS[:2])
    # print(response)


if __name__ == "__main__":
    main()
