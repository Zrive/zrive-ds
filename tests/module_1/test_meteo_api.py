""" This is a dummy example to show how to import code from src/ for testing"""
import unittest

from src.module_1.module_1_meteo_api import main
from src.module_1.module_1_meteo_api import get_statistics
from src.module_1.module_1_meteo_api import reduce_temporal_resolution

import pandas as pd

class TestMeteoAPI(unittest.TestCase):

    def test_reduce_temporal_resolution(self):
        # Create a sample daily data DataFrame
        self.dates = pd.date_range(start="2000-01-01", end="2000-08-31", freq='D')
        data = {
            'temperature_2m_mean': [10 + i for i in range(len(self.dates))],
            'precipitation_sum': [20 + i  for i in range(len(self.dates))],    
            'soil_moisture_0_to_10cm_mean': [5 + i for i in range(len(self.dates))] 
        }
        self.daily_df = pd.DataFrame(data, index=self.dates)


        monthly_data = reduce_temporal_resolution(self.daily_df, "CityX")

        # Check if the resulting DataFrame is monthly
        self.assertTrue(monthly_data.index.freqstr == 'ME', "Index should be set to monthly frequency.")

        # Check the shape is 8 months
        self.assertEqual(len(monthly_data), 8, "There should be 8 months of data.")

        # Check that the resampling is calculating means correctly
        expected_temperature_mean = self.daily_df['temperature_2m_mean'].resample('ME').mean()
        pd.testing.assert_series_equal(monthly_data['temperature_2m_mean'], expected_temperature_mean, "Monthly temperature means are incorrect.")


    def test_get_statistics(self):
        # Prepare a sample DataFrame
        data = {
            'temperature_2m_mean': [10, 10, 10, 10, 10],
            'precipitation_sum': [30, 30, 30, 30, 30],
            'soil_moisture_0_to_10cm_mean': [20, 20, 20, 20, 20]
        }
        df = pd.DataFrame(data)

        expected_result = {
            'Statistic': ['Mean', 'Deviation'],
            'Temperature': [10, 0],
            'Precipitation': [30, 0],
            'Soil Moisture': [20, 0]
        }
        result = get_statistics(df, 'Madrid')
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()