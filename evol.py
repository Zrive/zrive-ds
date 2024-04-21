import requests
import numpy as np
import matplotlib.pyplot as plt




API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"


#primera funcion, recogida de datos
#conceptos clave: definicion de la funcion con argumentos de entrada, diccionario (par clave-valor)

def get_data_meteo_api(city, start_year, end_year):
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start": f"{start_year}-01-01",
        "end": f"{end_year}-12-31",
        "models": ["MRI_AGCM3_2_S", "EC_Earth3P_HR"],
        "daily": VARIABLES
    }
    response = requests.get(API_URL, params=params)
    data = response.json()
    return data



def main():
    cities = ["Madrid", "London", "Rio"]
    start_year = 1950
    end_year = 2050

    #bucle sobre las ciudades
    for city in cities:
        data = get_data_meteo_api(city, start_year, end_year)
        print(data)

    #hola
    print("hola")
    


if __name__ == "__main__":
    main()



