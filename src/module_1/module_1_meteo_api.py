import requests  # Esta librería nos permite hacer solicitudes a la API.

# Definimos las coordenadas de las ciudades:
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

# Definimos las variables que queremos (temperatura, precipitación, viento):
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

# Definimos la función para obtener los datos de la API:
def get_data_meteo_api(city_name, latitude, longitude):
    """Obtiene los datos meteorológicos de la API para una ciudad."""
    url = "https://archive-api.open-meteo.com/v1/archive"  # URL de la API
    params = {
        "latitude": latitude,  # Latitud de la ciudad
        "longitude": longitude,  # Longitud de la ciudad
        "start_date": "2010-01-01",  # Fecha de inicio
        "end_date": "2020-12-31",  # Fecha de fin
        "daily": VARIABLES,  # Variables que queremos
        "timezone": "Europe/Madrid"  # Zona horaria (para tener horas consistentes)
    }
    
    # Hacemos la solicitud a la API:
    response = requests.get(url, params=params)
    
    # Si la respuesta es correcta (status code 200):
    if response.status_code == 200:
        data = response.json()  # Convertimos la respuesta a formato JSON
        return data["daily"]  # Devolvemos solo la parte "daily" que contiene los datos
    else:
        print(f"Error al obtener los datos para {city_name}")
        return None

import pandas as pd  # pandas es una librería que nos ayuda a trabajar con tablas de datos.

def process_data(data):
    """Procesa los datos para convertirlos en promedios mensuales."""
    # Convertimos los datos en un DataFrame (una tabla de datos):
    df = pd.DataFrame(data)
    
    # Convertimos la columna "time" a formato de fecha para que pandas lo entienda:
    df['time'] = pd.to_datetime(df['time'])
    
    # Hacemos que "time" sea la columna de índices (las fechas serán las filas):
    df.set_index('time', inplace=True)
    
    # Hacemos un promedio de los datos por mes:
    df_monthly = df.resample('M').mean()  # 'M' es el código para "mes"
    
    return df_monthly

import matplotlib.pyplot as plt  # matplotlib nos permite dibujar gráficos.

def plot_data(df, city_name):
    """Genera gráficos de la temperatura, precipitación y viento de una ciudad."""
    # Creamos una ventana para los gráficos:
    plt.figure(figsize=(10, 5))
    
    # Dibujamos la temperatura media en rojo:
    plt.plot(df.index, df['temperature_2m_mean'], label='Temperatura Media (°C)', color='red')
    
    # Dibujamos la precipitación en azul:
    plt.plot(df.index, df['precipitation_sum'], label='Precipitación (mm)', color='blue')
    
    # Dibujamos la velocidad del viento en verde:
    plt.plot(df.index, df['wind_speed_10m_max'], label='Viento Máximo (m/s)', color='green')
    
    # Añadimos un título y etiquetas:
    plt.title(f"Evolución meteorológica en {city_name} (2010-2020)")
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    
    # Mostramos una leyenda para identificar cada línea:
    plt.legend()
    
    # Mostramos el gráfico:
    plt.show()

def main():
    # Iteramos sobre cada ciudad y obtenemos los datos:
    for city, coords in COORDINATES.items():
        print(f"Obteniendo datos para {city}...")
        
        # Llamamos a la función para obtener los datos de la API:
        data = get_data_meteo_api(city, coords["latitude"], coords["longitude"])
        
        # Si los datos son correctos, los procesamos:
        if data:
            df = process_data(data)
            
            # Dibujamos los gráficos:
            plot_data(df, city)
        else:
            print(f"No se pudieron obtener datos para {city}")

# Si ejecutamos este archivo, se llamará a la función main():
if __name__ == "__main__":
    main()