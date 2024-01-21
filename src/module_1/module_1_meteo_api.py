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
    end_date = "2024-01-20"
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d") # Vamos a trabajar con los datos de los ultimos 30 dias
    for city, coords in COORDINATES.items():
        params = {
            'latitude': coords['latitude'],
            'longitude': coords['longitude'],
            'start_date': start_date,
            'end_date': end_date,
            "models": ["FGOALS_f3_H"], # Mejorar incluyendo el resto de modelos
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
  #  print(city)
  #  print(df)
    
# Función para representar el gráfico de dispersión
 def plot_city_data(city, df):
    df_city = df[df['city'] == city]
    df_city['time'] = pd.to_datetime(df_city['time'])
    variables = ['temperature_2m_mean', 'precipitation_sum', 'soil_moisture_0_to_10cm_mean']
    
    for variable in variables:
        df_var = df_city[['time', variable]].dropna()
        
        # Gráfico de dispersión
        plt.figure(figsize=(10, 6))
        plt.scatter(df_var['time'], df_var[variable], label=f'{variable} en {city}')
        plt.title(f'Evolución de {variable} en {city}')
        plt.xlabel('Tiempo')
        plt.ylabel(variable)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

plot_city_data('Madrid', df)

# Primera versión del código a mejorar: Incompleto.