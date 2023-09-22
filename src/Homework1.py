import requests
import numpy as np


API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

def get_data_meteo_api(city, start_year, end_year):
    data = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start": start_year,
        "end": end_year,
        "variables": VARIABLES,
    }
    
    response = requests.get(API_URL, params = data)

    if response.status.code == 200:
        return response.json()
    else:
        print("Failed to obtain the data for {city}: {response_status_code}")
        
def calculo_estadistico(data):
    data_array = np.array(data)
    mean = np.mean(data_array, axis=0)
    std_dev = np.std(data_array, axis=0)
    return mean, std_dev



def main():
    raise NotImplementedError

if __name__ == "__main__":
    start_year = 1950
    end_year = 2050
