import requests
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
from pandas import Timestamp

import numpy as np

import matplotlib.pyplot as plt

''' 
NOTAS DE MEJORA: 
1. Usar la librería requests, más general que la que viene en la web de la API. Su uso está más generalizado y se recomienda a largo plazo.
Usando la librería requests, podemos crear una función general para llamar API's que se pueda reutilizar en un futuro. 
2. Falta hacer los tests unitarios (aprender a hacerlos). 
3. Estudiar otro tipo de visualizaciones para que sean más interpretables los gráficos. 
 '''

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

API_URL = "https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
 "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
 "London": {"latitude": 51.507351, "longitude": -0.127758},
 "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
FECHA_INICIO = "2010-01-01"
FECHA_FINAL = "2020-12-31"

def get_meteo_api(ciudad: str, fecha_inicio: str, fecha_final: str, variables: list[str]):
    params = {
        "start_date": fecha_inicio,
        "end_date":   fecha_final,
        "latitude":   COORDINATES[ciudad]["latitude"],
        "longitude":  COORDINATES[ciudad]["longitude"],
        "daily":   VARIABLES,
    }

    response = openmeteo.weather_api(API_URL, params=params)[0]
    daily = response.Daily()
    daily_data = {"date": pd.date_range(
    start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
    end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
    freq = pd.Timedelta(seconds = daily.Interval()),
    inclusive = "left")}

    for i in range(len(VARIABLES)):
        daily_data[VARIABLES[i]] = daily.Variables(i).ValuesAsNumpy()

    daily_data["ciudad"] = ciudad

    daily_dataframe = pd.DataFrame(data = daily_data)
    return daily_dataframe

def df_resolucion_mensual(df: pd.DataFrame, ciudad: str, variable: list[str]):
    df_grouped = df[df["ciudad"] == ciudad].copy()
    df_grouped["date"] = df_grouped["date"].dt.tz_localize(None)
    df_grouped = (df_grouped.groupby(df_grouped["date"].dt.to_period("M"))[variable].agg(["mean", "max", "min", "median", "std"]).reset_index())
    df_grouped['date'] = df_grouped['date'].dt.to_timestamp()
    df_grouped.columns = ['date'] + [f"{stat}_{var}" for var in variable for stat in ['mean', 'max', 'min', 'median', 'std']]
    return df_grouped

def plot_auxiliar(df: pd.DataFrame, ciudad: str, variable: str, f_inicio: str, f_final: str):
    interval = df['date'].diff().median()
    width = interval / np.timedelta64(1, 'D')*0.8
    plt.figure(figsize=(10, 6))
    plt.bar(df['date'], df[variable],width=width)
    plt.title(f'{ciudad} - {variable} entre {f_inicio} y {f_final}')
    plt.xlabel('Tiempo')
    plt.ylabel(f'{variable}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def grafica_entre_fechas(df: pd.DataFrame, ciudades: list[str], variables: list[str], f_inicio: str, f_final: str, dif_max_dias: int, function: str):
    f_inicio = pd.to_datetime(f_inicio)
    f_final = pd.to_datetime(f_final)
    fi_date = f_inicio.date()
    ff_date = f_final.date()
    if f_inicio<pd.to_datetime(FECHA_INICIO) or f_final>pd.to_datetime(FECHA_FINAL):
        raise ValueError("No se seleccionan las fechas correctas.")
    elif (f_final-f_inicio).days<dif_max_dias:
        for ciudad in ciudades:
            for variable in variables:
                ciudad_df = df[df["ciudad"] == ciudad]
                ciudad_df = ciudad_df[(ciudad_df["date"].dt.date >= fi_date) & (ciudad_df["date"].dt.date <= ff_date)]
                plot_auxiliar(df=ciudad_df, ciudad=ciudad, variable=variable, f_inicio=f_inicio, f_final=f_final)
    else:
        for ciudad in ciudades:
            for variable in variables:
                fi_date=Timestamp(f_inicio).to_period('M').to_timestamp()
                ff_date=Timestamp(f_final).to_period('M').to_timestamp()
                df_grouped = df_resolucion_mensual(df, ciudad, [variable])
                df_grouped = df_grouped[(df_grouped["date"] >= fi_date) & (df_grouped["date"] <= ff_date)]
                plot_auxiliar(df=df_grouped, ciudad=ciudad, variable=f"{function}_{variable}", f_inicio=f_inicio, f_final=f_final)

def comparativa_entre_meses(df: pd.DataFrame, ciudades: list[str], variables: list[str], mes: int):
  df_mes_unido = []
  for ciudad in ciudades:
    dfs = df_resolucion_mensual(df, ciudad, variables)
    dfs["ciudad"] = ciudad
    df_mes_unido.append(dfs)
  df_mes = pd.concat(df_mes_unido, ignore_index=True)
  df_mes = df_mes[df_mes["date"].dt.month == mes]
  for variable in variables:
    n = len(ciudades)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, ciudad in zip(axes, ciudades):
        df_ciudad = df_mes[df_mes["ciudad"] == ciudad]

        ax.plot(df_ciudad["date"], df_ciudad[f"mean_{variable}"], label=f"Media {ciudad}", color='b')

        ax.fill_between(df_ciudad["date"], df_ciudad[f"min_{variable}"], df_ciudad[f"max_{variable}"],
                        color='lightblue', alpha=0.3, label=f"Rango {ciudad}")

        ax.set_title(f"{ciudad} - {variable}")
        ax.set_xlabel("Año")
        ax.set_ylabel(f"{variable.capitalize()}")

    plt.tight_layout()
    plt.show()

def main():
  dfs = []
  for ciudad in COORDINATES:
    df = get_meteo_api(ciudad, fecha_inicio=FECHA_INICIO, fecha_final=FECHA_FINAL, variables=VARIABLES)
    dfs.append(df)
  df_todo = pd.concat(dfs, ignore_index=True)
  grafica_entre_fechas(df_todo, ["Madrid", "Rio", "London"], VARIABLES, "2010-01-01", "2020-03-01", 100,  "mean" )
  comparativa_entre_meses(df_todo, ["Madrid", "Rio", "London"], VARIABLES, 7)

main()