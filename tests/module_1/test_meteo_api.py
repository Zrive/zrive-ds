import src.module_1.module_1_meteo_api as meteo
import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock


class TestMeteoAPI:
    """Test suite for the Meteo API script."""

    @patch("requests.get")
    def test_get_data_meteo_api_success(self, mock_get):
        """Test successful data retrieval from the API."""
        # Create mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200

        # Sample API response data structure
        sample_data = {
            "latitude": 40.416775,
            "longitude": -3.70379,
            "generationtime_ms": 2.7571,
            "utc_offset_seconds": 0,
            "timezone": "GMT",
            "timezone_abbreviation": "GMT",
            "elevation": 667.0,
            "daily_units": {
                "time": "iso8601",
                "temperature_2m_mean": "°C",
                "precipitation_sum": "mm",
                "wind_speed_10m_max": "km/h",
            },
            "daily": {
                "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "temperature_2m_mean": [10.5, 11.2, 9.8],
                "precipitation_sum": [0.0, 5.2, 2.1],
                "wind_speed_10m_max": [15.3, 20.1, 18.5],
            },
        }

        mock_response.json.return_value = sample_data
        mock_get.return_value = mock_response

        # Call the function to test
        result = meteo.get_data_meteo_api("Madrid", "2020-01-01", "2020-01-03")

        # Assert the function was called with correct parameters
        mock_get.assert_called_once()

        # Verify the result is a DataFrame with expected structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 3 days of data
        assert "temperature_2m_mean" in result.columns
        assert "precipitation_sum" in result.columns
        assert "wind_speed_10m_max" in result.columns
        assert "city" in result.columns
        assert result["city"].iloc[0] == "Madrid"

    @patch("requests.get")
    def test_get_data_meteo_api_city_validation(self, mock_get):
        """Test city validation in the get_data_meteo_api function."""
        # Test with an invalid city
        with pytest.raises(KeyError):
            meteo.get_data_meteo_api("InvalidCity")

        # Ensure the API call was not made
        mock_get.assert_not_called()

    @patch("requests.get")
    def test_get_data_meteo_api_api_error(self, mock_get):
        """Test handling of API errors in get_data_meteo_api."""
        # Configure mock to raise an HTTP error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response

        # Test that the function properly raises the exception
        with pytest.raises(Exception):
            meteo.get_data_meteo_api("Madrid")

    def test_process_weather_data(self):
        """Test the process_weather_data function."""
        # Create test data
        dates = pd.date_range(start="2020-01-01", periods=60)
        test_data = pd.DataFrame(
            {
                "time": dates,
                "city": ["Madrid"] * 30 + ["London"] * 30,
                "temperature_2m_mean": np.random.uniform(5, 20, 60),
                "precipitation_sum": np.random.uniform(0, 10, 60),
                "wind_speed_10m_max": np.random.uniform(5, 30, 60),
            }
        )

        # Process data monthly
        result_monthly = meteo.process_weather_data(test_data, "ME")

        # Verify results
        assert isinstance(result_monthly, pd.DataFrame)
        assert "time" in result_monthly.columns
        assert "city" in result_monthly.columns

        # Verify cities are preserved
        cities = result_monthly["city"].unique()
        assert len(cities) == 2
        assert "Madrid" in cities
        assert "London" in cities

        # Verify the resampling reduced the number of rows
        # 60 daily records should become approximately 2 monthly records per city
        # (2 months × 2 cities = around 4 records total)
        assert len(result_monthly) < len(test_data)

        # Process data yearly
        result_yearly = meteo.process_weather_data(test_data, "YE")

        # Verify yearly aggregation is even more compact
        assert len(result_yearly) < len(result_monthly)

    def test_plot_weather_data(self, tmp_path):
        """Test the plot_weather_data function."""
        # Create test directory within the pytest temporary directory
        test_output_dir = tmp_path / "test_plots"

        # Create test data
        dates = pd.date_range(start="2020-01-01", periods=12, freq="ME")
        test_data = pd.DataFrame(
            {
                "time": dates.tolist() * 3,
                "city": ["Madrid"] * 12 + ["London"] * 12 + ["Rio"] * 12,
                "temperature_2m_mean": np.random.uniform(5, 25, 36),
                "precipitation_sum": np.random.uniform(0, 100, 36),
                "wind_speed_10m_max": np.random.uniform(5, 30, 36),
            }
        )

        # Generate plots
        plot_files = meteo.plot_weather_data(test_data, str(test_output_dir))

        # Verify plot files were created
        for file_path in plot_files.values():
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) > 0  # File should not be empty


if __name__ == "__main__":
    pytest.main(["-v"])
