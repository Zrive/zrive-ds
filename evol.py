import requests
import numpy as np
import matplotlib.pyplot as plt


API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
}
VARIABLES = "temperature_2m_mean"


#primera funcion, recogida de datos
#conceptos clave: definicion de la funcion con argumentos de entrada, diccionario (par clave-valor)

def get_data_meteo_api(city, start_year, end_year):
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start": f"{start_year}-01-01",
        "end": f"{end_year}-12-31",
        "models": ["MRI_AGCM3_2_S"],
        "daily": VARIABLES
    }
    response = requests.get(API_URL, params=params)
    data = response.json()
    return data



def main():
    cities = ["Madrid"]
    start_year = 1950
    end_year = 2050

    #bucle sobre las ciudades
    for city in cities:
        data = get_data_meteo_api(city, start_year, end_year)
        print(data)


if __name__ == "__main__":
    main()



