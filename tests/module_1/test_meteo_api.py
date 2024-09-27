""" This is a dummy example to show how to import code from src/ for testing"""
import unittest
from unittest.mock import patch
import pandas as pd
from src.module_1.module_1_meteo_api import get_data_meteo_api, process_data

#def test_main():
 #   raise NotImplementedError

class TestWeatherAPI(unittest.TestCase):

    @patch('src.module_1.module_1_meteo_api.requests.get')
    def test_get_data_meteo_api_success(self, mock_get):
        """Prueba que la función obtiene datos correctamente."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "daily": [
                {"time": "2010-01-01", "temperature_2m_mean": 5.0, "precipitation_sum": 0.0, "wind_speed_10m_max": 3.0},
                {"time": "2010-01-02", "temperature_2m_mean": 6.0, "precipitation_sum": 1.0, "wind_speed_10m_max": 2.5},
            ]
        }
        
        # Llamar a la función
        data = get_data_meteo_api("Madrid", 40.416775, -3.703790)
        
        # Verificar que se obtienen los datos correctamente
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 2)  # Verificamos que hay dos días de datos
        self.assertEqual(data[0]['temperature_2m_mean'], 5.0)  # Comprobamos la temperatura del primer día

    @patch('src.module_1.module_1_meteo_api.requests.get')
    def test_get_data_meteo_api_failure(self, mock_get):
        """Prueba que la función maneja errores correctamente."""
        mock_get.return_value.status_code = 404  # Simular un error 404
        
        # Llamar a la función
        data = get_data_meteo_api("Madrid", 40.416775, -3.703790)
        
        # Verificar que no se obtienen datos
        self.assertIsNone(data)

    def test_process_data(self):
        """Prueba que la función de procesamiento de datos funciona correctamente."""
        sample_data = [
            {"time": "2010-01-01", "temperature_2m_mean": 5.0, "precipitation_sum": 0.0, "wind_speed_10m_max": 3.0},
            {"time": "2010-01-02", "temperature_2m_mean": 6.0, "precipitation_sum": 1.0, "wind_speed_10m_max": 2.5},
        ]
        
        # Procesar los datos
        df_monthly = process_data(sample_data)
        
        # Verificar que se genera el DataFrame
        self.assertIsInstance(df_monthly, pd.DataFrame)
        self.assertEqual(len(df_monthly), 1)  # Debería haber solo un mes en los datos de muestra
        self.assertAlmostEqual(df_monthly['temperature_2m_mean'].mean(), 5.5)  # Verificar promedio de temperatura

if __name__ == "__main__":
    unittest.main()

    