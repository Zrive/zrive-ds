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

def plot_climate_data(city_data, city_name):
    years = [int(date[:4]) for date in city_data['dates']]
    temperature_data = np.array(city_data['temperature_2m_mean']['data'])
    precipitation_data = np.array(city_data['precipitation_sum']['data'])
    soil_moisture_data = np.array(city_data['soil_moisture_0_to_10cm_mean']['data'])

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(years, temperature_data.mean(axis=1), label='Temperature (Mean)')
    plt.fill_between(years, temperature_data.min(axis=1), temperature_data.max(axis=1), alpha=0.2)
    plt.title(f'Climate Data for {city_name}')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(years, precipitation_data.mean(axis=1), label='Precipitation (Mean)')
    plt.fill_between(years, precipitation_data.min(axis=1), precipitation_data.max(axis=1), alpha=0.2)
    plt.ylabel('Precipitation (mm)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(years, soil_moisture_data.mean(axis=1), label='Soil Moisture (Mean)')
    plt.fill_between(years, soil_moisture_data.min(axis=1), soil_moisture_data.max(axis=1), alpha=0.2)
    plt.xlabel('Year')
    plt.ylabel('Soil Moisture (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    raise NotImplementedError

if __name__ == "__main__":
    start_year = 1950
    end_year = 2050

    for city_name in COORDINATES.keys():
        city_data = get_data_meteo_api(city_name, start_year, end_year)
        if city_data:
            plot_climate_data(city_data, city_name)