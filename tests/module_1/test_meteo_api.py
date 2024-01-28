import unittest
from unittest.mock import patch, Mock
import pandas as pd 
from src.module_1.module_1_meteo_api import main, validate_response, get_data_meteo_api, calculate_mean_std, call_api

class TestValidateResponse(unittest.TestCase):

    def test_validate_response_valid(self):
        response = {
            "latitude": 40.416775,
            "longitude": -3.703790,
            "generationtime_ms": 2.2119,
            "timezone": "Europe/Madrid",
            "timezone_abbreviation": "CEST",
            "daily": {},
            "daily_units": {}
        }
        self.assertTrue(validate_response(response))

    def test_validate_response_invalid(self):
        response = {"invalid_key": "invalid_value"}
        self.assertFalse(validate_response(response))


class TestCalculateMeanStd(unittest.TestCase):

    def test_calculate_mean_std(self):
        data = {
            'time': ['2022-01-01', '2022-01-02'],
            'temperature_2m_AAAAAAAA': [10, 20],
            'temperature_2m_BBBBBBBB': [15, 25],
            'city': ['City1', 'City1']
        }
        df = pd.DataFrame(data)
        result = calculate_mean_std(df)
        self.assertIn('temperature_mean', result.columns)
        self.assertIn('temperature_std', result.columns)
        self.assertEqual(result['temperature_mean'][0], 12.5)
        self.assertEqual(result['temperature_mean'][1], 22.5)

class TestCallAPI(unittest.TestCase):

    @patch('src.module_1.module_1_meteo_api.requests.get')
    def test_call_api_successful(self, mock_get):
        # Mock a successful API response
        mock_response = Mock()
        mock_get.return_value = mock_response
        mock_response.status_code = 200
        mock_response.json.return_value = {
            # Mocked JSON response
        }
        params = {
            # Mocked parameters
        }
        result = call_api(params)
        self.assertEqual(result.status_code, 200)

    @patch('src.module_1.module_1_meteo_api.requests.get')
    def test_call_api_failure(self, mock_get):
        # Mock a failed API response
        mock_response = Mock()
        mock_get.return_value = mock_response
        mock_response.status_code = 404

        result = call_api({})
        self.assertEqual(result.status_code, 404)





def test_main():
    raise NotImplementedError

if __name__ == "__main__":
    unittest.main()