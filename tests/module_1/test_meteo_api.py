import pytest
import pandas as pd

from src.module_1.module_1_meteo_api import (
    _get_coordinates_of_city,
    _call_api,
    get_data_meteo_api,
    process_meteo_data,
    visualize_meteo_data,
)
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse


def test_get_coordinates_of_city():
    city = "Madrid"
    expected_coordinates = {"latitude": 40.416775, "longitude": -3.703790}
    assert (
        _get_coordinates_of_city(city) == expected_coordinates
    ), f"Expected {expected_coordinates} but got {_get_coordinates_of_city(city)}"


def test_get_coordinates_of_city_invalid_city():
    city = "San Francisco"
    expected_error_message = f"City '{city}' not found in COORDINATES dictionary."
    with pytest.raises(ValueError) as exception_info:
        _get_coordinates_of_city(city)
    assert (
        str(exception_info.value) == expected_error_message
    ), f"Expected error: '{expected_error_message}', got '{str(exception_info.value)}'"


def test_call_api():
    coordinates = {"latitude": 40.416775, "longitude": -3.703790}
    start_date = "2010-01-01"
    end_date = "2020-12-31"
    variables = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
    response = _call_api(coordinates, start_date, end_date, variables)
    assert isinstance(
        response, WeatherApiResponse
    ), f"Expected response to be a WeatherApiResponse but got {type(response)}"


def test_call_api_invalid_coordinates():
    coordinates = {"latitude": 100.0, "longitude": 200.0}
    start_date = "2010-01-01"
    end_date = "2020-12-31"
    variables = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
    with pytest.raises(Exception) as exception_info:
        _call_api(coordinates, start_date, end_date, variables)
    assert "API call failed for coordinates" in str(exception_info.value), (
        "Expected error message to contain 'API call failed for coordinates' "
        f"but got '{str(exception_info.value)}'"
    )


def test_get_data_meteo_api():
    city = "Madrid"
    data = get_data_meteo_api(city)
    assert isinstance(
        data, WeatherApiResponse
    ), f"Expected data to be a WeatherApiResponse but got {type(data)}"


def test_get_data_meteo_api_invalid_city():
    city = "San Francisco"
    expected_error_message = f"City '{city}' not found in COORDINATES dictionary."
    with pytest.raises(ValueError) as exception_info:
        get_data_meteo_api(city)
    assert (
        str(exception_info.value) == expected_error_message
    ), f"Expected error: '{expected_error_message}', got '{str(exception_info.value)}'"


def test_process_meteo_data():
    data = get_data_meteo_api("Madrid")
    df = process_meteo_data(data)
    assert isinstance(
        df, pd.DataFrame
    ), f"Expected output to be a DataFrame but got {type(df)}"
    assert not df.empty, "Expected DataFrame is empty."


def test_process_meteo_data_invalid_data():
    data = {
        "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "temperature_2m_mean": [20.0, 21.0, 22.0],
    }
    with pytest.raises(AttributeError):
        process_meteo_data(data)


def test_visualize_meteo_data():
    data = pd.DataFrame(
        {
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "temperature_2m_mean": [20.0, 21.0, 22.0],
            "precipitation_sum": [0.0, 1.0, 2.0],
            "wind_speed_10m_max": [5.0, 6.0, 7.0],
        }
    )
    try:
        visualize_meteo_data(
            df=data,
            city="Madrid",
            column="temperature_2m_mean",
            label="Temperature (°C)",
            variable="Temperature",
            ylabel="°C",
            color="blue",
        )
    except Exception as e:
        pytest.fail(f"visualize_meteo_data raised an exception: {e}")
