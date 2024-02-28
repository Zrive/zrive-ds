import pandas as pd
import requests
import json

API_URL = "https://api.open-meteo.com/v1/forecast"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"


# función genérica para llamar a la API
def request_API(url: str, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Lanza una excepción si el status code no es 200
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"Error HTTP: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error en la solicitud: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"Error al decodificar JSON: {json_err}")
    return None


# función get_data_meteo_api para sacar los datos dada la ciudad


def get_data_meteo_api(
    longitude: float, latitude: float, start_date: str, end_date: str
):
    headers = {}  # empty because we don't need any headers for this API
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "models": [
            "CMCC_CM2_VHR4",
            "FGOALS_f3_H",
            "HiRAM_SIT_HR",
            "MRI_AGCM3_2_S",
            "EC_Earth3P_HR",
            "MPI_ESM1_2_XR",
            "NICAM16_8S",
        ],
        "daily": VARIABLES,
    }

    data = request_API(API_URL, params=params)
    return data


def main():
    print(f"Obteniendo datos para 'Madrid'...")
    data = get_data_meteo_api(
        COORDINATES["Madrid"]["latitude"],
        COORDINATES["Madrid"]["longitude"],
        start_date=1950,
        end_date=2025,
    )
    print(f"Datos para Madrid:\n{data}")


# estoy obteniendo un error 400
# faltan procesar los datos + plotting
# y los test unitarios

if __name__ == "__main__":
    main()
