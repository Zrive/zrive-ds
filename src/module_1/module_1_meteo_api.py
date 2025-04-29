# src/module_1/module_1_meteo_api.py
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from typing import Any


COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
    }

API_URL = "https://archive-api.open-meteo.com/v1/archive?"


def use_api(url: str, params: dict):
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    try:
        responses = openmeteo.weather_api(url, params=params)
        return responses[0]
    except Exception as e:
        print(f"API error: {e}")
        raise

def get_data_meteo_api(city: str):
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": "2010-01-01",
        "end_date": "2019-12-31",
        "daily": ["precipitation_sum", "temperature_2m_mean", "wind_speed_10m_max"]
    }
    response = use_api(API_URL, params)
    daily = response.Daily()
    return daily

def organize_data(daily: Any) -> pd.DataFrame:
    ''' get the data from the meteo API for every day and transform it for mothly with mean
    '''
    daily_precipitation_sum = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
    }
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_dataframe = pd.DataFrame(data=daily_data)
    daily_dataframe.set_index("date", inplace=True)
    monthly_data = daily_dataframe.resample("ME").agg({
        "precipitation_sum": "sum",
        "temperature_2m_mean": "mean",
        "wind_speed_10m_max": "mean"
    })
    monthly_data.reset_index(inplace=True)
    return monthly_data

def create_chart(all_data: list[pd.DataFrame], cities: list[str]):
    plt.figure(figsize=(12, 8))
    
    # Temperature
    plt.subplot(3, 1, 1)
    
    for data, city in zip(all_data, cities):
        plt.plot(data["date"], data["temperature_2m_mean"], label=city)
        
    plt.title("Monthly Mean Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  
    
    # Precipitation
    plt.subplot(3, 1, 2)
    
    for data, city in zip(all_data, cities):
        plt.plot(data["date"], data["precipitation_sum"], label=city)
    
    plt.title("Monthly Precipitation Sum")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (mm)")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")) 
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  
    
    # Wind Speed
    plt.subplot(3, 1, 3)
    
    for data, city in zip(all_data, cities):
        plt.plot(data["date"], data["wind_speed_10m_max"], label=city)
    
    plt.title("Monthly Max Wind Speed")
    plt.xlabel("Date")
    plt.ylabel("Wind Speed (km/h)")
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m")) 
    plt.gca().xaxis.set_major_locator(mdates.YearLocator()) 
    plt.tight_layout()
    plt.savefig("weather_comparison.png")
    plt.show()

def main():
    cities = ["Madrid", "London", "Rio"]
    all_data = []
    for city in cities:
        daily = get_data_meteo_api(city)
        monthly_data = organize_data(daily)
        all_data.append(monthly_data)
    create_chart(all_data, cities)

if __name__ == "__main__":
    main()