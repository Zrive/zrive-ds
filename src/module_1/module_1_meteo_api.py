################################################
### Semana 1: Recoger datos API Meteorologia ###
################################################
import requests
import pandas as pd
import matplotlib as plt
#import backoff


#@backoff.on_exception(backoff.expo, request.error.RateLimitError)
def call_api(url, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        return response()
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud a la API: {e}")
        return None

def get_data_meteo_api(city, API_URL):
    params = {"city": city}
    data = call_api(API_URL, params)
    if data:
        return data
    else:
        return None



def main():
    COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
    }
    VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
    raise NotImplementedError

if __name__ == "__main__":
    main()
