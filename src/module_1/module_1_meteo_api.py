import openmeteo_requests
import requests_cache
from retry_requests import retry
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse

import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters needed to call the API
API_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
START_DATE = "2010-01-01"
END_DATE = "2020-12-31"


def get_data_meteo_api(city: str) -> WeatherApiResponse:
    "Main function to call the meteo API and get the data for the specified city."
    coordinates = _get_coordinates_of_city(city)
    city_data = _call_api(coordinates, START_DATE, END_DATE, VARIABLES)
    if city_data:
        logger.info(f"Successfully retrieved data for {city}.")
        return city_data
    else:
        logger.error(f"API response for {city} is invalid.")
        raise ValueError(f"API response for {city} is invalid.")


def _get_coordinates_of_city(city: str) -> dict[float, float]:
    "Helper function to get the coordinates of the city from the COORDINATES"
    "dictionary."
    if city in COORDINATES:
        return COORDINATES[city]
    else:
        raise ValueError(f"City '{city}' not found in COORDINATES dictionary.")


def _call_api(
    coordinates: dict[float, float],
    start_date: str,
    end_date: str,
    variables: list[str],
) -> WeatherApiResponse:
    "Helper function to call the API and get the data for the coordinates provided."
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(variables),
    }

    try:
        responses = openmeteo.weather_api(API_URL, params)
        return responses[0]
    except Exception as e:
        raise Exception(f"API call failed for coordinates {coordinates}: {e}.")


def process_meteo_data(data: WeatherApiResponse) -> pd.DataFrame:
    "Function to process the raw data from the API and convert it into"
    "a pandas DataFrame."
    daily = data.Daily()

    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()

    daily_data = {
        "time": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }

    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

    daily_dataframe = pd.DataFrame(data=daily_data)
    return daily_dataframe


def visualize_meteo_data(
    df: pd.DataFrame,
    city: str,
    column: str,
    label: str,
    variable: str,
    ylabel: str,
    color: str,
) -> None:
    "Function to visualize the evolution of meteo data for a city."
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df[column], label=label, color=color)
    plt.title(f"{variable} data evolution for {city}, {START_DATE} to {END_DATE}")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    cities = ["Madrid", "London", "Rio"]
    for city in cities:
        city_meteo_data = get_data_meteo_api(city)
        df_meteo_data_per_city = process_meteo_data(city_meteo_data)

        visualize_meteo_data(
            df_meteo_data_per_city,
            city,
            "temperature_2m_mean",
            "Temperature (°C)",
            "Temperature",
            "°C",
            "orange",
        )
        visualize_meteo_data(
            df_meteo_data_per_city,
            city,
            "precipitation_sum",
            "Precipitation (mm)",
            "Precipitation",
            "mm",
            "blue",
        )
        visualize_meteo_data(
            df_meteo_data_per_city,
            city,
            "wind_speed_10m_max",
            "Wind Speed (km/h)",
            "Wind Speed",
            "km/h",
            "gray",
        )
