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


def tratamiento_pd_series_a_df(serie: pd.core.series.Series, key_val: str, date_month: str) -> pd.core.frame.DataFrame:
    ciudad = key_val.split('_')[0]
    pdf = pd.DataFrame(serie, columns=['valores'])
    pdf = pdf.reset_index() 
    pdf.columns = ['modelo', 'valores']
    pdf['ciudad'] = ciudad
    pdf['date_month'] = date_month
    print(date_month)
    return pdf

def concat_datos_a_tablon(diccionario_df_agg: dict, target_month: str) -> pd.core.frame.DataFrame:
    dic_tab = dict()
    for key in diccionario_df_agg.keys():
        for item in diccionario_df_agg[key]:
            dic_tab[key] = tratamiento_pd_series_a_df(serie=item
                                                            , key_val=key
                                                            , date_month=target_month)
    return pd.concat([dic_tab[key] for key in dic_tab.keys()])

def tratar_fechas(date_str: str) -> str:
    from datetime import datetime, timedelta
    date_obj = datetime.strptime(date_str, "%d-%m-%Y")
    print(date_obj)
    new_date_obj = date_obj - timedelta(days=1)
    new_date_str = new_date_obj.strftime("%Y-%m-%d")
    return new_date_str

def main():
    return 0


if __name__ == "__main__":
    
    START_DATE = "1950-01-01"
    END_DATE = "2050-01-01"
    
    lista_meses = pd.date_range(START_DATE, END_DATE, freq='MS').strftime("%m-%Y").tolist()
    
    lista_starts = ['01-'+ mes for mes in lista_meses]
    
    lista_tup_comienzo_fin = [(lista_starts[index-1], tratar_fechas(lista_starts[index])) for index in range(1, len(lista_starts))]
    
    l_paquetes = list()
    
    for fecha in lista_tup_comienzo_fin:
        paquete_raw = get_data_meteo_api(
            url=API_URL,
            cities=COORDINATES,
            start_date=fecha[0],
            end_date=fecha[1],
            models=MODELS,
            data=VARIABLES,
        )
        l_paquetes.append(paquete_raw)
    
    
    general_avg = list()
    general_std = list()
    order_ciudades = ("madrid", "london", "rio")
    for index_date, paquetes in enumerate(l_paquetes):
        agg_avg_dic = dict()
        agg_std_dic = dict()
        for index, paquete in enumerate(paquetes):
            mod_medidas = 1
            lista_agg = list()
            basic_df = dict()
            lista_std = list()
            lista_avg = list()
            modelos_por_medida = list(paquetes[index]["daily"].keys())[1:]
            for tipo_medida in modelos_por_medida:
                basic_df[tipo_medida] = paquetes[index]["daily"][tipo_medida]

                if mod_medidas % 3 == 0:
                    df = pd.DataFrame(basic_df)
                    lista_agg.append(df)
                    lista_avg.append(df.mean())
                    lista_std.append(df.std())
                    basic_df = dict()
                mod_medidas += 1

            agg_avg_dic[f"{order_ciudades[index]}_avg"] = lista_avg
            agg_std_dic[f"{order_ciudades[index]}_std"] = lista_std
            
        pdf_avg_agg_mes = concat_datos_a_tablon(diccionario_df_agg=agg_avg_dic
                                                , target_month=lista_meses[index_date])
        pdf_std_agg_mes = concat_datos_a_tablon(diccionario_df_agg=agg_std_dic
                                                , target_month=lista_meses[index_date])

        general_avg.append(pdf_avg_agg_mes)
        general_std.append(pdf_std_agg_mes)
        
    df_avg = pd.concat(general_avg)
    df_std = pd.concat(general_std)
    
    df_avg.reset_index(drop=True, inplace=True)
    df_std.reset_index(drop=True, inplace=True)

    cities = df_avg['ciudad'].unique()    
    for df in [df_avg, df_std]:
        
        for medida in VARIABLES.split(','):
            df_temperature = df[df['modelo'].str.startswith(f"{medida}_")]
    
            for city in cities:
                city_data = df_temperature[df_temperature['ciudad'] == city]
                plt.figure(figsize=(10, 6))
                plt.plot(city_data['date_month'], city_data['valores'], marker='o', linestyle='-')
                plt.title(f'Temperature Time Series for {city}')
                plt.xlabel(medida)
                plt.ylabel('valor')
                plt.xticks(rotation=45)
                plt.grid(True)
                plt.tight_layout()
                plt.show()


# TODO Pendiente revision nulos

# TODO pendiente revision constantes

# TODO crear tests

# TODO meter en main las cosas y modularizar un poco el codigo