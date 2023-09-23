################################################
### Semana 1: Recoger datos API Meteorologia ###
################################################
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
import backoff
import time
import json
import scipy.stats as stats

from jsonschema import validate, SchemaError


# Carga el archivo JSON en la variable schema_validation
with open("/home/ramon/Zrive/zrive-ds/src/module_1/schema.json", 'r') as file:
    schema_to_validate = json.load(file)

@backoff.on_exception(backoff.expo, requests.exceptions.Timeout)
def call_api(url, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Error al hacer la solicitud a la API: {e}')
        return None

def get_data_meteo_api(city, coordinates, start_date, end_date, vars):
    URL = f"https://climate-api.open-meteo.com/v1/climate?latitude={coordinates['latitude']}&longitude={coordinates['longitude']}&start_date={start_date}&end_date={end_date}&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily={vars}"
    data = call_api(URL)
    try:
        validate(data, schema_to_validate)
    except SchemaError as e:
        print(f"Error en el API_JSON schema{e}")
    finally:
        df=pd.DataFrame(data=data['daily'])
        df['city'] = city
        time.sleep(3)
    return df

def data_parse(df, variables):
    vars = [x.strip() for x in variables.split(',')]
    resumen_df = df[['time']].copy()
    for variable in vars:
        modelos = [col for col in df.columns if variable in col]
        promedio_por_dia = df[modelos].mean(axis=1)
        resumen_df[variable] = promedio_por_dia
    return resumen_df


def graficar(df, nombre_archivo, city):
    # Filtrar el DataFrame para incluir solo datos del primer día de cada mes
    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby(df['time'].dt.year).mean()
    promedios = df.mean()
    desviaciones = df.std()
    
    plt.figure(figsize=(10, 6))  # Tamaño del gráfico
    
    for variable in ['temperature_2m_mean', 'precipitation_sum', 'soil_moisture_0_to_10cm_mean']:
        plt.plot(promedios.index, promedios[variable], label=variable, linestyle='-', marker='o')
        ci = 1.96 * (desviaciones[variable] / (len(df) ** 0.5))
        plt.fill_between(promedios.index, promedios[variable] - ci, promedios[variable] + ci, alpha=0.3)

    # Configurar el gráfico
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(f'Meteo: {city}')
    plt.legend()

    # Guardar el gráfico como un archivo JPG
    plt.savefig(f'{nombre_archivo}.jpg', format='jpg')

    # Mostrar el gráfico en pantalla (opcional)
    #plt.show()

def main():
        # Variables iniciales
    COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790}
    }
    start_date='1950-01-01'
    end_date='2023-08-30'
    VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
        #####
    for i in COORDINATES.keys():
        data_cities = get_data_meteo_api(city = i, coordinates = COORDINATES[i], start_date = start_date,
                                              end_date = end_date, vars = VARIABLES)
        meteo_data_final = data_parse(data_cities, variables = VARIABLES)
        graficar(df =meteo_data_final, nombre_archivo = f'{i}_meteo', city = i)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print (f'Timepo de ejecutción: {end_time-start_time}s')