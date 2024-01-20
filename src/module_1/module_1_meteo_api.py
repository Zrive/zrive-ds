import requests
import time
from datetime import datetime, timedelta

API_URL = "https://climate-api.open-meteo.com/v1/climate?"
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

def get_data_meteo_api():
    results = {}
    end_date = "2024-01-10"
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    
    for city, coords in COORDINATES.items():
        params = {
            'latitude': coords['latitude'],
            'longitude': coords['longitude'],
            'start_date': start_date,
            'end_date': end_date,
            'variables': VARIABLES
        }
        url = API_URL + '&'.join(f'{key}={value}' for key, value in params.items())

        try:
            response = requests.get(url)
            if response.status_code == 429:
                time.sleep(10)  # Wait longer if rate limit is reached
                response = requests.get(url)  # Retry the request
            if response.status_code != 200:
                results[city] = f"Error: Received status code {response.status_code}"
            else:
                data = response.json()
                print(f"Raw JSON Data for {city}: {data}")  # Print the raw JSON data
                # Attempt to extract the desired data
                desired_data = {
                    'temperature_2m_mean': data.get('temperature_2m_mean', 'Not available'),
                    'precipitation_sum': data.get('precipitation_sum', 'Not available'),
                    'soil_moisture_0_to_10cm_mean': data.get('soil_moisture_0_to_10cm_mean', 'Not available')
                }
                results[city] = desired_data
        except requests.exceptions.RequestException as e:
            results[city] = f"API request failed: {e}"
        finally:
            time.sleep(1)  # Regular cool off period

    return results

# Ejemplo de uso
all_city_data = get_data_meteo_api()
print(all_city_data)