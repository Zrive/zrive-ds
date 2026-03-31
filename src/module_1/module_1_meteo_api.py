"""
Module 1 - Meteo API
"""
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
from typing import Dict, Any

# Global variables
API_URL = "https://archive-api.open-meteo.com/v1/archive?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

def fetch_api_data(
    url: str, params: Dict[str, Any], max_retries: int = 3, backoff_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Fetch data from an API endpoint with retries and backoff.
    
    Params:
        - url: str, the API endpoint URL.
        - params: Dict[str, Any], the query parameters to send.
        - max_retries: int, the maximum number of retries before giving up.
        - backoff_factor: float, the factor to increase the sleep time between retries.
    Returns:
        - Dict[str, Any], the JSON response from the API.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                print(f"Rate limited. Retrying after {retry_after} seconds...")
                time.sleep(retry_after * backoff_factor)
                retries += 1
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            retries += 1
            if retries < max_retries:
                sleep_time = backoff_factor * (2 ** retries)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise Exception(f"Max retries ({max_retries}) exceeded.")
    raise Exception(f"API request failed after {max_retries} retries.")

def validate_response_schema(response: Dict[str, Any]):
    """
    Validates the schema of the API response.
    
    Params:
        - response: Dict[str, Any], the JSON response from the API.
    Raises:
        - ValueError: if the response schema is invalid.
    """
    required_keys = [
        'daily', 'daily_units', 'latitude', 'longitude', 'generationtime_ms'
    ]
    for key in required_keys:
        if key not in response:
            raise ValueError(f"Invalid API response: missing key {key}")
        
    daily = response['daily']
    required_vars = VARIABLES + ['time']
    for var in required_vars:
        if var not in daily:
            raise ValueError(
                f"Invalid API response: missing variable {var} in daily data"
            )

def get_data_meteo_api(city: str) -> Dict[str, Any]:
    """
    Getter function to fetch weather data from the Open-Meteo API.
    
    Params:
        - city: str, the name of the city to fetch data for.
    Returns:
        - Dict[str, Any], the JSON response from the API.
    """
    coords = COORDINATES.get(city)
    if not coords:
        raise ValueError(f"City {city} not found in COORDINATES")
    
    start_date = "2010-01-01"
    end_date = "2020-12-31"
    
    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
    }
    
    response = fetch_api_data(url=API_URL,params=params)
    validate_response_schema(response)
    return response

def process_daily_data(response: Dict[str, Any], city: str) -> pd.DataFrame:
    """
    Processes the daily weather data from the API response.
    
    Params:
        - response: Dict[str, Any], the JSON response from the API.
        - city: str, the name of the city.
    Returns:
        - pd.DataFrame, the processed weather data.
    """
    daily_data = response['daily']
    df = pd.DataFrame({
        'time': daily_data['time'],
        'temperature': daily_data['temperature_2m_mean'],
        'precipitation': daily_data['precipitation_sum'],
        'wind_speed': daily_data['wind_speed_10m_max']
    })
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Resample to monthly data
    monthly_df = pd.DataFrame()
    monthly_df['temperature'] = df['temperature'].resample('ME').mean()
    monthly_df['precipitation'] = df['precipitation'].resample('ME').sum()
    monthly_df['wind_speed'] = df['wind_speed'].resample('ME').max()
    monthly_df['city'] = city
    return monthly_df

def plot_combined_data(combined_df: pd.DataFrame):
    """
    Plots the combined weather data for multiple cities.
    
    Params:
        - combined_df: pd.DataFrame, the combined weather data.
    """
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    variables = ['temperature', 'precipitation', 'wind_speed']
    titles = [
        'Monthly Mean Temperature (°C)',
        'Monthly Total Precipitation (mm)',
        'Monthly Maximum Wind Speed (m/s)'
    ]
    
    for i, (var, title) in enumerate(zip(variables, titles)):
        ax = axes[i]
        for city in combined_df['city'].unique():
            city_data = combined_df[combined_df['city'] == city]
            ax.plot(city_data.index, city_data[var], label=city)
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('weather_evolution.png')
    print("Plot saved to 'weather_evolution.png'")
    plt.close()

def main():
    cities = COORDINATES.keys()
    all_data = []
    
    for city in cities:
        print(f"Fetching data for {city}...")
        try:
            response = get_data_meteo_api(city)
            processed_df = process_daily_data(response, city)
            all_data.append(processed_df)
        except Exception as e:
            print(f"Error processing {city}: {e}")
            continue
    combined_df = pd.concat(all_data)
    plot_combined_data(combined_df)

if __name__ == "__main__":
    main()