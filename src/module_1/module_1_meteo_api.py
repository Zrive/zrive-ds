import requests
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd

# Constantes
API_URL = "https://climate-api.open-meteo.com/v1/climate?"
LATITUD = "latitude"
LONGITUD = "longitude"
START_DATE = "start_date"
END_DATE = "end_date"
DAILY = "daily"
COORDINATES = {
    "Madrid": {LATITUD: 40.416775, LONGITUD: -3.703790},
    "London": {LATITUD: 51.507351, LONGITUD: -0.127758},
    "Rio": {LATITUD: -22.906847, LONGITUD: -43.172896},
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
MODELS = [
    "CMCC_CM2_VHR4",
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]


def f_construye_params(
    ciudades: dict[str, dict[str, float]],
    start_date: str,
    end_date: str,
    data: str,
    models: list[str],
) -> list[dict[str, float | str]]:
    l_params = list()
    for ciudad in ciudades.keys():
        l_params.append(
            {
                LATITUD: ciudades[ciudad][LATITUD],
                LONGITUD: ciudades[ciudad][LONGITUD],
                START_DATE: start_date,
                END_DATE: end_date,
                "models": models,
                DAILY: data,
            }
        )
    return l_params


def f_verifica_esquema_basic(paquete: dict) -> bool:
    L_BASICOS = [
        LATITUD,
        LONGITUD,
        "utc_offset_seconds",
        "timezone",
        "timezone_abbreviation",
        "elevation",
        "daily_units",
    ]
    paquete_correcto = False
    required_keys_set = set(L_BASICOS)
    data_keys_set = set(paquete.keys())
    common_keys = required_keys_set & data_keys_set
    if len(common_keys) != len(required_keys_set):
        paquete_correcto = False
    else:
        paquete_correcto = True
    return paquete_correcto


def f_check_codes(paquete: requests.models.Response) -> bool:
    paquete_correcto = False
    if paquete.status_code == 400:
        paquete_correcto = False
    elif paquete.status_code == 500:
        print("CCCC")
        time.sleep(5)
        paquete_correcto = False
    elif paquete.status_code == 200:
        paquete_correcto = True
    else:
        paquete_correcto = False
    return paquete_correcto


def get_data_meteo_api(
    url: str,
    cities: dict[str, int],
    start_date: str,
    end_date: str,
    data: str,
    models: list[str],
) -> list[dict]:
    parametros_get = f_construye_params(
        ciudades=cities,
        start_date=start_date,
        end_date=end_date,
        data=data,
        models=models,
    )
    l_datos_ciudad = list()
    N_INTENTOS = 4
    for llamada in parametros_get:
        done = False
        intentos = 0
        while not done and intentos < N_INTENTOS:
            raw_data = requests.get(url=url, params=llamada)
            print(f_check_codes(raw_data), f_verifica_esquema_basic(raw_data.json()))
            if f_check_codes(raw_data) and f_verifica_esquema_basic(raw_data.json()):
                l_datos_ciudad.append(raw_data.json())
                done = True
            else:
                intentos += 1
            # print("wrong esquema", raw_data.json())
            time.sleep(5)

        if intentos == N_INTENTOS:
            raise RuntimeError("Problemas con la api")
    return l_datos_ciudad


def main():
    return 0


if __name__ == "__main__":
    paquetes = get_data_meteo_api(
        url=API_URL,
        cities=COORDINATES,
        start_date="2023-01-01",
        end_date="2023-02-01",
        models=MODELS,
        data=VARIABLES,
    )
    agg_dic = dict()
    order_ciudades = ("madrid", "london", "rio")
    for index, paquete in enumerate(paquetes):
        mod_medidas = 1
        lista_agg = list()
        basic_df = dict()
        lista_std = list()
        lista_avg = list()
        modelos_por_medida = list(paquetes[index]["daily"].keys())[1:]
        for tipo_medida in modelos_por_medida:
            if mod_medidas % 4 == 0:
                df = pd.DataFrame(basic_df)
                lista_agg.append(df)
                lista_avg.append(df.mean())
                lista_std.append(df.std())
                basic_df = dict()
            else:
                basic_df[tipo_medida] = paquetes[index]["daily"][tipo_medida]
            mod_medidas += 1

            agg_dic[f"{order_ciudades[index]}_avg"] = lista_avg
            agg_dic[f"{order_ciudades[index]}_std"] = lista_std

    ### PLOTTING

    madrid_avg = pd.DataFrame(agg_dic["madrid_avg"])
    madrid_std = pd.DataFrame(agg_dic["madrid_std"])

    # Extract variable names for plotting. Assuming all models report the same variables.
    variables = madrid_avg.columns

    # Create plots for average values
    plt.figure(figsize=(14, 7))
    for var in variables:
        plt.plot(madrid_avg.index, madrid_avg[var], label=var + " Avg")
    plt.title("Madrid Climate Variables Average")
    plt.xlabel("Time")
    plt.ylabel("Average Value")
    plt.legend()
    plt.show()

    # Create plots for standard deviation values
    plt.figure(figsize=(14, 7))
    for var in variables:
        plt.plot(madrid_std.index, madrid_std[var], label=var + " Std")
    plt.title("Madrid Climate Variables Standard Deviation")
    plt.xlabel("Time")
    plt.ylabel("Std Value")
    plt.legend()
    plt.show()
