""" This is a dummy example to show how to import code from src/ for testing"""

from src.module_1.module_1_meteo_api import main
import unittest
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
def media_anual(df):
    """
    Pasa de datos diarios/semanales/mensuales a datos anuales
    
    Parámetros:
        - df: DataFrame con la primera columna time.
        
    Devuelve:
        - DataFrame, el DataFrame con datos anualizados de su media
    """
    # Convertir la columna 'time' a tipo datetime
    df['time'] = pd.to_datetime(df['time'])

    # Extraer el año de la columna 'time'
    df['year'] = df['time'].dt.year

    # Calcular la media anual de cada variable climatológica
    df_anual = df.groupby('year').mean()
    

    
    return df_anual

class TestMediaAnual(unittest.TestCase):
    def test_media_anual(self):
        # Crear un DataFrame de ejemplo
        data = {
            'time': pd.date_range(start='2022-01-01', end='2022-12-31'),
            'variable1': [1] * 365,  # Ejemplo de datos para la variable 1
            'variable2': [2] * 365,  # Ejemplo de datos para la variable 2
            
        }
        df = pd.DataFrame(data)
        
        # Llamar a la función media_anual
        df_resultante = media_anual(df)
        
        # Verificar que el DataFrame resultante tenga el número correcto de columnas
        self.assertEqual(len(df_resultante.columns), 3)  # 'year', 'variable1', 'variable2'
        
        # Verificar que todas las columnas tengan valores
        self.assertFalse(df_resultante.isnull().values.any())
        
        # Verificar que las columnas 'year' y 'variable1' existan
        self.assertIn('year', df_resultante.columns)
        self.assertIn('variable1', df_resultante.columns)
        
        # Verificar que las columnas 'year' y 'variable2' no existan
        self.assertNotIn('variable2', df_resultante.columns)

if __name__ == '__main__':
    unittest.main()

def test_main():
    raise NotImplementedError