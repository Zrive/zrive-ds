""" This is a dummy example """

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import matplotlib

# Api call data funtion

def get_data_meteo_api(city: str, model: str):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = "https://climate-api.open-meteo.com/v1/climate?"  
    params = {
    "latitude": COORDINATES[city_to_call]["latitude"],
    "longitude": COORDINATES[city_to_call]["longitude"],
    "start_date": "1950-01-01",
    "end_date": "2050-12-31",
    "models": [model_to_use],
    "daily": ["temperature_2m_mean","precipitation_sum","soil_moisture_0_to_10cm_mean"]}

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Data
    daily = response.Daily()
    temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    precipitation_sum = daily.Variables(1).ValuesAsNumpy()
    soil_moisture_0_to_10cm_mean = daily.Variables(2).ValuesAsNumpy()

    # Parse date
    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left")}


    # Data into dataset columns
    daily_data["City"] = city
    daily_data["Model"] = model
    daily_data["temperature_2m_mean"] = temperature_2m_mean
    daily_data["precipitation_sum"] = precipitation_sum
    daily_data["soil_moisture_0_to_10cm_mean"] = soil_moisture_0_to_10cm_mean

    # Create dataframe
    daily_dataframe = pd.DataFrame(data = daily_data)
    return daily_dataframe


COORDINATES = {
"Madrid": {"latitude": 40.416775, "longitude": -3.703790},
"London": {"latitude": 51.507351, "longitude": -0.127758},
"Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
Moldels = ["MRI_AGCM3_2_S", "EC_Earth3P_HR"]

city_to_call = "Rio"
model_to_use = "MRI_AGCM3_2_S"

daily_dataframe = get_data_meteo_api(city_to_call, model_to_use)

# Data overview function

def data_overview(dataset):
    print(dataset.describe())

dataset_to_analyze = daily_dataframe
model_to_use = "temperature_2m_mean"

data_overview(dataset_to_analyze)

# Plot functions

def plot_function(data):
    daily_dataframe.plot.line(x = "date", y = data)

def general_plot():
    daily_dataframe.plot.line(x = "date", subplots = True)

data_to_plot = "precipitation_sum"

plot_function(data_to_plot)
general_plot()

def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
