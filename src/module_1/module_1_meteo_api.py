import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


API_URL = "https://climate-api.open-meteo.com/v1/climate?"
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
# Funcion para descargar los datos de la API
def get_data_meteo_api():
    results = {}
    end_date = "2050-01-01"
    start_date = "1950-01-01"
    for city, coords in COORDINATES.items():
        params = {
            'latitude': coords['latitude'],
            'longitude': coords['longitude'],
            'start_date': start_date,
            'end_date': end_date,
            "models": ["MRI-AGCM3-2-S"], # Mejorar incluyendo el resto de modelos
            'daily': VARIABLES
        }
        try:
            response = requests.get(API_URL, params=params)
            if response.status_code == 429:
                time.sleep(10) 
                response = requests.get(API_URL, params=params)
            if response.status_code != 200:
                results[city] = f"Error: Received status code {response.status_code}"
            else:
                data = response.json()
                data = data['daily']
                results[city] = pd.DataFrame(data)
        except requests.exceptions.RequestException as e:
            results[city] = f"API request failed: {e}"
        finally:
            time.sleep(1)

    return results


# Dataset creado:
all_city_data = get_data_meteo_api()
for city, df in all_city_data.items():
  print(city)
  print(df)

# Función para representar el gráfico de dispersión - No funciona
 def plot_city_data(city_name, df):
    df_city = df[df['city_name'] == city_name]
    df_city['time'] = pd.to_datetime(df_city['time'])
    variables = ['temperature_2m_mean', 'precipitation_sum', 'soil_moisture_0_to_10cm_mean']
    
    for variable in variables:
        df_var = df_city[['time', variable]].dropna()
        
        # Gráfico de dispersión
        plt.figure(figsize=(10, 6))
        plt.scatter(df_var['time'], df_var[variable], label=f'{variable} en {city_name}')
        plt.title(f'Evolución de {variable} en {city_name}')
        plt.xlabel('Tiempo')
        plt.ylabel(variable)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

plot_city_data('Madrid', df)

#Funcion principal por definir
def main():

if __name__ == "__main__":
main()

# Primera versión del código a mejorar: Incompleto.