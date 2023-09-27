import pandas as pd
import requests
import matplotlib.pyplot as plt
import time
import itertools
from typing import List

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

MODELS = "CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR"

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean" 


class APIConnector:
    base = ""

    # Silly constructor
    def __init__(self):
        self.base = "https://climate-api.open-meteo.com/v1/climate?"

    # Call api method, we'll see how to manage status_code
    def call_api(self, city: str):
        # URL inputs
        lat = float(COORDINATES[city]["latitude"])
        long = float(COORDINATES[city]["longitude"])
        coords = f"latitude={lat}&longitude={long}"
        date_span = "&start_date=1950-01-01&end_date=2050-12-31"
        models = f"&models={MODELS}"
        mode = f"&daily={VARIABLES}"

        # Final URL
        url = self.base + coords + date_span + models + mode

        # Response
        response = requests.get(url)
        if response.status_code == 409:
            for i in range(10):
                sleepy_cooloff = 1.5 * i
                time.sleep(sleepy_cooloff)
                response = requests.get(url)
                if response.status_code == 409:
                    continue
                else:
                    break
        return response.json()

    # Schema validation
    def validate_schema(json: dict) -> bool:
        # Base schema
        schema = {
            "latitude": float,
            "longitude": float,
            "generationtime_ms": float,
            "utc_offset_seconds": int,
            "timezone": object,
            "timezone_abbreviation": object,
            "elevation": float,
            "daily_units": object,
            "daily": object,
        }
        # Checks on keys and dtypes
        for key, d_type in schema.items():
            if key not in json:
                return False
            if not isinstance(json[key], d_type):
                return False
        return True

    # get_data_meteo_api, should return treated response
    def get_data_meteo_api(self, city: str) -> pd.DataFrame:
        # Response JSON
        response = self.call_api(city)
        # if not APIConnector.validate_schema(response):
        # raise Exception(f"Invalid schema for response {response}")
        # Treat response
        df_response = pd.DataFrame(response)
        print(df_response.head())
        return df_response


def get_df_index() -> List:
    models = MODELS.split(',')
    variables = VARIABLES.split(',')
    unjoined_index = list(itertools.product(variables, models))
    joined_index = []
    for item in unjoined_index:
        joined_index.append('_'.join(item))
    return joined_index


# Calculate mean and dispersion (std & variance)
    # Calculate mean and dispersion
def calc_stats(df: pd.DataFrame) -> pd.DataFrame:
    index = get_df_index()

    models = MODELS.split(',')
    # Get data from df
    time = df["daily"]["time"]
    df_models = pd.DataFrame() 
    df_models["Date"] = time
    df_models["Year"] = pd.to_datetime(df_models["Date"]).dt.year
    df_models.drop(columns="Date", inplace=True)
    
    # Get data from index
    for item in index:
        if models[0] in item:
            col_name = item.replace(models[0], 'model1')
            df_models[col_name] = df["daily"][item]
        elif models[1] in item:
            col_name = item.replace(models[1], 'model2')
            df_models[col_name] = df["daily"][item]
        elif models[2] in item:
            col_name = item.replace(models[2], 'model3')
            df_models[col_name] = df["daily"][item]
    
    # Group by year and get stats
    grouped_df = df_models.groupby("Year").agg(['mean', 'std']).reset_index()

    return grouped_df


def paint_plots(dfs: list, plot: str, freq="monthly"):

    # Since the grouped df is multi indexed and we want to compare
    # every metric on their own (mean against mean, etc):
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    fig3, ax3 = plt.subplots(figsize=(9, 6))

    ax1.set_title(f"Temperature: {plot}")
    ax2.set_title(f"Rain: {plot}")
    ax3.set_title(f"Moisture: {plot}")

    if plot == "mean":
        for df in dfs:
            y1 = df.loc[:, df.columns[1]]
            y2 = df.loc[:, df.columns[4]]
            y3 = df.loc[:, df.columns[7]]

            ax1.plot(x, y1)
            ax2.plot(x, y2)
            ax3.plot(x, y3)

    elif plot == "var":
        for df in dfs:
            y1 = df.loc[:, df.columns[2]]
            y2 = df.loc[:, df.columns[5]]
            y3 = df.loc[:, df.columns[8]]

            ax1.plot(x, y1)
            ax2.plot(x, y2)
            ax3.plot(x, y3)
    else:
        for df in dfs:
            y1 = df.loc[:, df.columns[3]]
            y2 = df.loc[:, df.columns[6]]
            y3 = df.loc[:, df.columns[9]]

            ax1.plot(x, y1)
            ax2.plot(x, y2)
            ax3.plot(x, y3)

    ax1.legend(["Madrid", "London", "Rio"])
    ax2.legend(["Madrid", "London", "Rio"])
    ax3.legend(["Madrid", "London", "Rio"])


def main():
    connector = APIConnector()

    df_madrid = connector.get_data_meteo_api("Madrid")
    #df_londres = connector.get_data_meteo_api("London")
    #df_rio = connector.get_data_meteo_api("Rio")

    calc_stats(df_madrid)
    # londres = calc_stats(df_londres)
    #rio = calc_stats(df_rio)


    #paint_plots([madrid, londres, rio], "mean")
    #paint_plots([madrid, londres, rio], "var")
    #paint_plots([madrid, londres, rio], "std")


if __name__ == "__main__":
    main()
