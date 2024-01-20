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

UNITS = {
    "temperature_2m_mean": "ºC",
    "precipitation_sum": "mm",
    "soil_moisture_0_to_10cm_mean": r"$m^3/m^3$",
}


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
            print(f"API request error: {e}. Reason: {response.reason}")
            time.sleep(retry_delay)
    return None


def get_data_meteo_api(
    city: str, start_date: str, end_date: str, models: List[str] = MODELS
) -> Optional[Dict[str, Any]]:
    params = {
        **COORDINATES[city],
        "start_date": start_date,
        "end_date": end_date,
        "models": models,
        "daily": VARIABLES.split(","),
    }

    api_data = get_api_data(API_URL, params=params)
    if api_data is None:
        return None
    return api_data


def join_models_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    df_returned = df[["city", "year"]].copy()

    for var in VARIABLES.split(","):
        variable_names = [c for c in df.columns if c.startswith(var)]
        df_returned[f"{var}_mean"] = df[variable_names].mean(axis=1)
        df_returned[f"{var}_std"] = df[variable_names].std(axis=1)

    df_returned = df_returned.fillna(0)

    return df_returned


def plot_variable_by_model_city(
    df: pd.DataFrame, variable: str, stat: str, param_dict: Dict[str, Any] = {}
):  # works well for temperature and precipitation due to models restrictions
    rows = 4
    cols = 2
    fig, ax = plt.subplots(rows, cols, figsize=(15, 15))
    ax = ax.flatten()
    for idx, model in enumerate(MODELS):  # a plot for each model
        col_to_plot = f"{variable}_{model}"

        # a color for each city
        for city in df["city"].unique():
            # select the data for the city and get the statistic
            df_city = df[df["city"] == city][["year", col_to_plot]].groupby("year")

            if stat == "mean":
                df_city = df_city.mean()
            elif stat == "std":
                df_city = df_city.std()
            df_city = df_city.reset_index()

            ax[idx].plot(df_city["year"], df_city[col_to_plot], **param_dict)

        ax[idx].set_xlabel("Year")
        ax[idx].set_ylabel(f"{variable}")
        ax[idx].set_title(f"{variable} {stat} ({UNITS[variable]})\n({model} model)")
        ax[idx].grid(True)
        ax[idx].legend(df["city"].unique())

    fig.savefig(f"./src/module_1/plot_images/{stat}_{variable}_yearly.png")


def plot_variable_time_series(
    df: pd.DataFrame, variable: str, param_dict: Dict[str, Any] = {}
):
    # plot the mean and mean +- std for each city
    cities = df["city"].unique()
    rows = 1
    cols = len(cities)
    fig, ax = plt.subplots(rows, cols, figsize=(15, 5))
    ax = ax.flatten()
    for idx, city in enumerate(cities):
        df_city = df[df["city"] == city][
            ["year", f"{variable}_mean", f"{variable}_std"]
        ]

        df_yearly = df_city.groupby("year").mean().reset_index()

        df_yearly["mean"] = df_yearly[f"{variable}_mean"]
        df_yearly["lower"] = (
            df_yearly[f"{variable}_mean"] - df_yearly[f"{variable}_std"]
        )
        df_yearly["upper"] = (
            df_yearly[f"{variable}_mean"] + df_yearly[f"{variable}_std"]
        )

        ax[idx].plot(df_yearly["year"], df_yearly["mean"], **param_dict)
        ax[idx].fill_between(
            df_yearly["year"], df_yearly["lower"], df_yearly["upper"], alpha=0.3
        )
        ax[idx].set_xlabel("Year")
        ax[idx].set_ylabel(f"{variable}")
        ax[idx].set_title(f"{variable} ({UNITS[variable]}) in {city}")
        ax[idx].grid(True)

    fig.tight_layout()
    fig.savefig(f"./src/module_1/plot_images/{variable}_time_series.png")


def main():
    START_DATE = "1950-01-01"
    END_DATE = "2050-12-31"
    city_df_list = []
    for city in COORDINATES:
        city_data = get_data_meteo_api(city, START_DATE, END_DATE, MODELS)
        if city_data is not None:
            city_df = pd.DataFrame(city_data["daily"])
            city_df["city"] = city
            city_df["time"] = pd.to_datetime(city_df["time"])
            city_df["year"] = city_df["time"].dt.year
            city_df_list.append(city_df)

    if len(city_df_list) == 0:
        print("No data retrieved from the API")
        return
    df = pd.concat(city_df_list)

    # Plot the data for each model
    # temperature
    plot_variable_by_model_city(df, VARIABLES.split(",")[0], "mean")
    plot_variable_by_model_city(df, VARIABLES.split(",")[0], "std")

    # precipitation
    plot_variable_by_model_city(df, VARIABLES.split(",")[1], "mean")
    plot_variable_by_model_city(df, VARIABLES.split(",")[1], "std")

    # transform the dataset to merge the models data into one variable
    df_unified = join_models_mean_std(df)

    # Plot the data for each city
    # temperature
    plot_variable_time_series(df_unified, VARIABLES.split(",")[0])
    # precipitation
    plot_variable_time_series(df_unified, VARIABLES.split(",")[1])
    # soil moisture
    plot_variable_time_series(df_unified, VARIABLES.split(",")[2])


if __name__ == "__main__":
    main()
