################################################
### Semana 1: Recoger datos API Meteorologia ###
################################################
import requests
import pandas as pd
import matplotlib.pyplot as plt
import backoff
import time
import json
import scipy.stats as stats
from jsonschema import validate, SchemaError


# Carga el archivo JSON en la variable schema_validation
with open("/home/ramon/Zrive/zrive-ds/src/module_1/schema.json", "r") as file:
    schema_to_validate = json.load(file)


@backoff.on_exception(backoff.expo, requests.exceptions.Timeout)
def call_api(url, params=None):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud a la API: {e}")
        return None


def get_data_meteo_api(coordinates, start_date, end_date, vars):
    URL = f"https://climate-api.open-meteo.com/v1/climate?latitude={coordinates['latitude']}&longitude={coordinates['longitude']}&start_date={start_date}&end_date={end_date}&models=CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S&daily={vars}"
    data = call_api(URL)
    try:
        validate(data, schema_to_validate)
    except SchemaError as e:
        print(f"Error en el API_JSON schema{e}")
    finally:
        df = pd.DataFrame(data=data["daily"])
    return df


def data_parse(df, variables):
    vars = [x.strip() for x in variables.split(",")]
    resumen_df = df[["time"]].copy()
    for variable in vars:
        modelos = [col for col in df.columns if variable in col]
        promedio_por_dia = df[modelos].mean(axis=1)
        resumen_df[variable] = promedio_por_dia
    return resumen_df


def conf_interval(df):
    df["time"] = pd.to_datetime(df["time"])
    df = df.melt(id_vars=["time"], var_name="Indicadores", value_name="Value")
    df["Year"] = df["time"].dt.year
    result = (
        df.groupby(["Year", "Indicadores"])["Value"].agg(["mean", "std"]).reset_index()
    )
    result["Lower_CI"] = result.apply(
        lambda row: row["mean"]
        - stats.t.ppf(1 - 0.05 / 2, df=len(df) - 1) * row["std"] / (len(df) ** 0.5),
        axis=1,
    )
    result["Upper_CI"] = result.apply(
        lambda row: row["mean"]
        + stats.t.ppf(1 - 0.05 / 2, df=len(df) - 1) * row["std"] / (len(df) ** 0.5),
        axis=1,
    )
    return result


def graficar(df, variables):
    vars = [x.strip() for x in variables.split(",")]
    num_vars = len(vars)
    # Configurar el grid de subplots
    num_cols = 1  # Una columna por gráfico
    num_rows = num_vars  # Una fila por variable
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 6 * num_vars), sharex=True)
    for i, variable in enumerate(vars):
        df_filtered = df[df["Indicadores"] == variable]
        cities = df_filtered["city"].unique()
        # Seleccionar el eje adecuado en el grid
        if num_vars > 1:
            ax = axs[i]
        else:
            ax = axs
        # Configurar el gráfico
        ax.set_title(variable)  # Establecer el título del gráfico como la variable
        for city in cities:
            city_data = df_filtered[df_filtered["city"] == city]
            # Dibujar el gráfico de líneas
            ax.plot(city_data["Year"], city_data["mean"], label=city)
            # Colorear el intervalo de confianza
            ax.fill_between(
                city_data["Year"],
                city_data["Lower_CI"],
                city_data["Upper_CI"],
                alpha=0.1,
                facecolor="k",
            )
        # Agregar leyenda
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend(loc="lower center")
        ax.set_title(f"{variable} with Confidence Intervals by City")
    plt.tight_layout()
    fig.savefig(f"Meteo.jpg", format="jpg")


def main():
    # Variables iniciales
    COORDINATES = {
        "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
        "London": {"latitude": 51.507351, "longitude": -0.127758},
        "Rio": {"latitude": -22.906847, "longitude": -43.172896},
    }
    start_date = "1950-01-01"
    end_date = "2050-01-01"
    VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
    data_cities = []
    for i in COORDINATES.keys():
        city_data = get_data_meteo_api(
            coordinates=COORDINATES[i],
            start_date=start_date,
            end_date=end_date,
            vars=VARIABLES,
        )
        viariable_data = data_parse(df=city_data, variables=VARIABLES)
        variable_intervals = conf_interval(df=viariable_data)
        variable_intervals["city"] = i
        data_cities.append(variable_intervals)
        time.sleep(3)
    meteo_data = pd.concat(data_cities, axis=0)
    graficar(df=meteo_data, variables=VARIABLES)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Timepo de ejecutción: {end_time-start_time}s")
