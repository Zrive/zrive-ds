import requests
import time
from typing import Dict, List, Optional, Any
import pandas as pd
import matplotlib.pyplot as plt


API_URL = "https://climate-api.open-meteo.com/v1/climate?"

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
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


def get_api_data(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> Optional[Dict[str, Any]]:
    for _ in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error en la solicitud a la API: {e}")
            time.sleep(retry_delay)
    return None


def get_data_meteo_api(
    city: str, start_date: str, end_date: str, models: List[str] = MODELS
) -> pd.DataFrame:
    params = {
        **COORDINATES[city],
        "start_date": start_date,
        "end_date": end_date,
        "models": models,
        "daily": VARIABLES.split(","),
    }

    api_data = get_api_data(API_URL, params=params)
    if api_data is None:
        return
    df = pd.DataFrame(api_data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df_yearly = df.groupby("year").mean().reset_index()
    print(df_yearly)
    return df_yearly


def plot_data():
    pass


def main():
    START_DATE = "1950-01-01"
    END_DATE = "2050-12-31"
    madrid_df = get_data_meteo_api("Madrid", START_DATE, END_DATE, MODELS)
    print(madrid_df)
    london_df = get_data_meteo_api("London", START_DATE, END_DATE, MODELS)
    print(london_df)
    rio_df = get_data_meteo_api("Rio", START_DATE, END_DATE, MODELS)
    print(rio_df)

    # print(response)


if __name__ == "__main__":
    main()
