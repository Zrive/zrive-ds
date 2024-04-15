import unittest
from unittest.mock import patch
import pandas as pd
from io import StringIO
import os
from module_1_meteo_api import main

"""
Debería estar en test pero me da error:
    from src.module_1.module_1_meteo_api import main
    ModuleNotFoundError: No module named 'src'
"""

# Example data that might come from the API
api_response_data = {
    "data": {
        "time": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "temperature_2m_mean": [5.0, 6.0, 4.0],
        "precipitation_sum": [0.0, 0.2, 0.0],
        "soil_moisture_0_to_10cm_mean": [0.1, 0.1, 0.1]
    }
}

def mock_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self):
            self.status_code = 200
            self.reason = 'OK'

        def json(self):
            return api_response_data

    return MockResponse()

class TestWeatherDataProcessing(unittest.TestCase):
    @patch('requests.get', side_effect=mock_requests_get)
    def test_complete_workflow(self, mock_get):
        # Assuming 'main' function is available and properly set up to use mocked `requests.get`
        # You might need to adjust the `main` function to be more testable or to better fit this setup.

        # Run the main function to process everything
        main()

        # Check if files were created (assuming output directory and file names are correct)
        expected_files = ['outputs/Madrid_daily_data.csv', 'outputs/Madrid_statistics.csv', 'outputs/Madrid_climate_data.png']
        for file_name in expected_files:
            with self.subTest(file=file_name):
                self.assertTrue(os.path.exists(file_name), f"{file_name} does not exist")

        # Load one of the output files to verify contents (for example, statistics)
        with open('outputs/Madrid_statistics.csv', 'r') as file:
            content = pd.read_csv(file)
            self.assertTrue(not content.empty, "Statistics file is empty")
            print(content.head())

# Running the tests
if __name__ == '__main__':
    unittest.main()
