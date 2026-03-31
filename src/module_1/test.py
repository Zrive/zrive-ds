import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.module_1.module_1_meteo_api import (fetch_api_data,
                                             validate_response_schema,
                                             get_data_meteo_api,
                                             process_daily_data)


@patch('requests.get')
def test_fetch_api_data_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "latitude": 40.416775,
        "longitude": -3.703790,
        "generationtime_ms": 1.0,
        "daily": {
            "time": ["2010-01-01"],
            "temperature_2m_mean": [10.0],
            "precipitation_sum": [5.0],
            "wind_speed_10m_max": [3.0]
        },
        "daily_units": {}
    }
    mock_get.return_value = mock_response
    result = fetch_api_data("http://test.com", {})
    assert result == mock_response.json()

def test_validate_response_schema_valid():
    valid_response = {
        "latitude": 40.416775,
        "longitude": -3.703790,
        "generationtime_ms": 1.0,
        "daily": {
            "time": [],
            "temperature_2m_mean": [],
            "precipitation_sum": [],
            "wind_speed_10m_max": []
        },
        "daily_units": {}
    }
    validate_response_schema(valid_response)

def test_validate_response_schema_invalid():
    invalid_response = {
        "latitude": 40.416775,
        "generationtime_ms": 1.0,
        "daily": {}
    }
    with pytest.raises(ValueError):
        validate_response_schema(invalid_response)

@patch('src.module_1.module_1_meteo_api.fetch_api_data')
def test_get_data_meteo_api_success(mock_fetch):
    mock_fetch.return_value = {
        "latitude": 40.416775,
        "longitude": -3.703790,
        "generationtime_ms": 1.0,
        "daily": {
            "time": ["2010-01-01"],
            "temperature_2m_mean": [10.0],
            "precipitation_sum": [5.0],
            "wind_speed_10m_max": [3.0]
        },
        "daily_units": {}
    }
    result = get_data_meteo_api("Madrid")
    assert "daily" in result

def test_process_daily_data():
    sample_response = {
        "daily": {
            "time": ["2010-01-01", "2010-01-02"],
            "temperature_2m_mean": [10.0, 12.0],
            "precipitation_sum": [5.0, 0.0],
            "wind_speed_10m_max": [3.0, 4.0]
        }
    }
    df = process_daily_data(sample_response, "Madrid")
    assert isinstance(df, pd.DataFrame)