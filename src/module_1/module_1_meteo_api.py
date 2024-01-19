import requests
import pandas as pd
from datetime import datetime, timedelta

API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"]

def get_data_meteo_api(city, start_date, end_date):
    if city not in COORDINATES:
        return f"City '{city}' not found."

    coords = COORDINATES[city]
    params = {
        'latitude': coords['latitude'],
        'longitude': coords['longitude'],
        'start_date': start_date,
        'end_date': end_date,
        'daily': VARIABLES
    }
    response = requests.get(API_URL, params=params)
    if response.status_code != 200:
        return f"Error: Received status code {response.status_code}"

    data = response.json()
    # Assuming the API returns data in a structure that includes the variables under a 'data' key
    daily_data = {
        'date': pd.date_range(start=start_date, end=end_date, freq='D').tolist(),
        VARIABLES[0]: data.get('data', {}).get(VARIABLES[0], []),
        VARIABLES[1]: data.get('data', {}).get(VARIABLES[1], []),
        VARIABLES[2]: data.get('data', {}).get(VARIABLES[2], []),
    }
    
    daily_dataframe = pd.DataFrame(data=daily_data)
    return daily_dataframe

# Example usage:
# Set the date range for the last month up to January 10th
end_date = "2024-01-10"
start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

all_city_data = {}
for city in COORDINATES.keys():
    all_city_data[city] = get_data_meteo_api(city, start_date, end_date)

# Print the data for each city
for city, data in all_city_data.items():
    print(f"Data for {city}:")
    print(data)
    print("\n")

