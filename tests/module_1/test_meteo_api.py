""" This is a dummy example to show how to import code from src/ for testing"""

from src.module_1.module_1_meteo_api import (
    get_data_meteo_api,
    process_data,
    mean_calc,
    sum_calc,
    max_calc,
    plotting_line,
    main,
)
import pandas as pd
from unittest.mock import patch

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
API_URL = "https://archive-api.open-meteo.com/v1/archive"


def test_mean_calc() -> None:
    df = pd.DataFrame(
        {
            "time": pd.date_range(start="2020-01-01", periods=5, freq="D"),
            "values": [2.0, 2.5, 3.0, 4.0, 5.0],
        }
    ).set_index("time")
    mean_df = mean_calc(df, "2D")
    # Assert the length of the result
    assert len(mean_df) == 3
    expected_values = [2.25, 3.5, 5.0]  # Means of (2.0, 2.5), (3.0, 4.0), (5.0)
    assert list(mean_df["values"]) == expected_values


def test_sum_calc() -> None:
    # Sample data
    df = pd.DataFrame(
        {
            "time": pd.date_range(start="2020-01-01", periods=5, freq="D"),
            "values": [2.0, 2.5, 3.0, 4.0, 5.0],
        }
    ).set_index("time")
    # Call sum_calc function with a 2-day period
    sum_df = sum_calc(df, "2D")
    # Assert the length of the result
    assert len(sum_df) == 3  # Two 2-day periods and one 1-day period
    # Assert that the sum values are calculated correctly
    expected_values = [4.5, 7.0, 5.0]  # Sums of (2.0 + 2.5), (3.0 + 4.0), and (5.0)
    assert list(sum_df["values"]) == expected_values


def test_max_calc() -> None:
    # Sample data
    df = pd.DataFrame(
        {
            "time": pd.date_range(start="2020-01-01", periods=5, freq="D"),
            "values": [2.0, 2.5, 3.0, 4.0, 5.0],
        }
    ).set_index("time")

    # Call max_calc function with a 2-day period
    max_df = max_calc(df, "2D")
    # Assert the length of the result
    assert len(max_df) == 3  # Two 2-day periods and one 1-day period
    # Assert that the max values are calculated correctly
    expected_values = [2.5, 4.0, 5.0]  # Max of (2.0, 2.5), (3.0, 4.0), and (5.0)
    assert list(max_df["values"]) == expected_values


def test_plotting_line() -> None:
    # Create a simple DataFrame for plotting
    df = pd.DataFrame(
        {
            "time": pd.date_range(start="2020-01-01", periods=2, freq="D"),
            "temperature_2m": [2.0, 2.5],
        }
    ).set_index("time")
    # Call the plotting function
    fig = plotting_line(df, "temperature_2m", "Madrid")
    # Assert that the figure contains the correct data
    assert fig.data[0].x.tolist() == df.index.tolist()
    assert fig.data[0].y.tolist() == [2.0, 2.5]


def test_process_data() -> None:
    # Mock input data similar to the API response structure
    data = {
        "hourly": {
            "time": ["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-01T02:00"],
            "temperature_2m": [5.0, 6.0, 7.0],
        }
    }
    magnitude = "temperature_2m"
    df = process_data(data, magnitude)
    # Assert the DataFrame has the correct shape and data
    assert isinstance(df, pd.DataFrame)

    assert (
        list(df[magnitude]) == data["hourly"][magnitude]
    )  # Ensure temperature data matches
    # Check if the DataFrame index is a DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)

    # Check if the DataFrame columns include the magnitude (temperature_2m)
    assert magnitude in df.columns


def test_get_data_meteo_api() -> None:
    # Mock data to simulate the expected response
    mock_data = {
        "hourly": {
            "time": ["2020-01-01T00:00", "2020-01-02T00:00"],
            "temperature_2m": [2.0, 2.5],
        }
    }

    # Patch to replace the requests.get with a mock object (mock_get)
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = mock_data

        data = get_data_meteo_api("Madrid", "temperature_2m")

        # Checks if the data returned by get_data_meteo_api is equal to the mock_data
        assert data == mock_data
        # Checks if the API is called with the expected arguments
        mock_get.assert_called_once_with(
            API_URL,
            params={
                "latitude": COORDINATES["Madrid"]["latitude"],
                "longitude": COORDINATES["Madrid"]["longitude"],
                "start_date": "2010-01-01",
                "end_date": "2019-12-31",
                "hourly": "temperature_2m",
            },
        )


def test_main() -> None:
    main()
