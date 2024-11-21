
# Este script resuelve la tarea del módulo 1 en el que el objetivo es crear un script que se 
# conecte a la API Meteo, sacar los datos de variables meteorológicas para distintas ciudades
#  entre 2010 y 2020. Una vez hecho eso se adaptan esas fechas para su representación.

# Librerías utilizadas

import pandas as pd
import matplotlib.pyplot as plt

# Definición de variables globales

API_URL = "<https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

RESOLUTION ='daily'

start_date = '2010-01-01'
end_date   = '2020-12-31'

# Función que crea la URL y se conecta a la API

# Por cada ciudad se hace una llamada (para recolectar datos de varias ciudades se deben hacer varias llamadas)
# Como la entrada es el nombre de la ciudad, puedo recorrer en un bucle el diccionario de coordenadas y obtener 
# los valores de ciudades para todas las variables

def get_data_meteo_api(nombre_ciudad, start_date, end_date, lista_variables, resolution):
    '''
    resolution: posibles valores 'daily' o 'hourly'.
    '''
    import requests
    import json

    nombre_ciudad = nombre_ciudad.capitalize()
    
    # Tomar variables y pasarlas a "formato url"
    latitude_url   = 'latitude=' + str(COORDINATES[nombre_ciudad]["latitude"])
    longitude_url  = 'longitude='+ str(COORDINATES[nombre_ciudad]["longitude"])
    start_date_url = 'start_date=' + start_date
    end_date_url   = 'end_date=' + end_date
    resolution_url = resolution
    
    # variables_url
    variables_url = ''
    for variable in lista_variables:
        variables_url+=variable
        variables_url+=','
    variables_url = variables_url[:-1] #Elimino la última coma residual

    # Construir URL
    API_url = 'https://archive-api.open-meteo.com/v1/archive?'+latitude_url+'&'+longitude_url+'&'+start_date_url+'&'+end_date_url+'&'+resolution+'='+variables_url

    # Llamada a la API (construir función)
    res = requests.get(API_url)

    return res.json()


# Bulce para hacer las llamadas a la API

data_list = []

for ciudad, coordenadas in COORDINATES.items():

    API_data = get_data_meteo_api(ciudad, start_date, end_date, VARIABLES, RESOLUTION)['daily']
    pd_data = pd.DataFrame(API_data, index=API_data['time'])
    pd_data.index = pd.to_datetime(pd_data.index)
    pd_data.drop('time', axis=1, inplace=True)
    pd_data['city'] = ciudad

    data_list.append(pd_data)



# Bucle para transformar y representar los datos

# Quiero sacar un único gráfico por ciudad y variable con datos máximos y mínimos
# Los gráficos se van a guardar en la carpeta "gráficas"

!mkdir -p module_1_plots

for data_ciudad in data_list:
    for variable in VARIABLES:
        nombre_ciudad = data_ciudad['city'].iloc[0]
        data_ciudad_variable =data_ciudad[variable].resample('MS').agg(['max', 'mean', 'min'])

        plt.figure()
        plt.plot(data_ciudad_variable.index, data_ciudad_variable['mean'])
        plt.fill_between(data_ciudad_variable.index, data_ciudad_variable['min'], data_ciudad_variable['max'], alpha=0.2 )
        plt.grid()
        plt.xlabel('Fecha (mes)')
        plt.ylabel(variable)
        plt.title(nombre_ciudad)
        
        plt.savefig(f"module_1_plots/{nombre_ciudad}" + "_" + str(variable))



# def main():
#     raise NotImplementedError

# if __name__ == "__main__":
#     main()

