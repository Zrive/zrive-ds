""" This is a dummy example """
import pandas as pd
import requests
import matplotlib.pyplot as plt
import time
import numpy as np

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

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
        mode = f"&daily={VARIABLES}"

        # Final URL
        url = self.base + coords + date_span + mode
        
        # Response
        response = requests.get(url)
        if response.status_code == 409:
            sleepy_cooloff = 0
            time.sleep(sleepy_cooloff)
            response = requests.get(url)
        return response.json()
    
    # Schema validation
    def validate_schema(json: dict) -> bool:
        # Base schema
        schema = {
            'latitude':float,
            'longitude':float,
            'generationtime_ms':float,
            'utc_offset_seconds':int,
            'timezone': object,
            'timezone_abbreviation': object,
            'elevation':float,
            'daily_units':str,
            'daily': list
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
        if not APIConnector.validate_schema(response):
            raise Exception(f"Invalid schema for response {response}")
        # Treat response
        df_response = pd.DataFrame(response)
        # print(df_response.head())

        return df_response
    
    # Calculate mean and dispersion (std & variance)
    def calc_stats(df: pd.DataFrame, freq:str ="monthly") -> pd.DataFrame:
        # Get data from df
        time = df['daily'][0]
        temp = df['daily'][1]
        prec = df['daily'][2]
        moist = df['daily'][3]
        
        # New df with data in cols
        working_df = pd.DataFrame([time,temp,prec,moist],
                                columns=["date","temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"])
        
        # Check frequency of data
        if freq == "monthly":
            # Maybe parse the string using .split('-') could be another option, not sure what's best
            working_df['date'] = pd.to_datetime(working_df['date'])
            working_df['month'] = working_df['date'].dt.month
            working_df = working_df.drop(columns='date')

            # Group by month and get stats
            grouped_df = working_df.groupby('month').agg([np.mean, np.var, np.std]).reset_index()

        # elif freq == "yearly":
        # In case there's a chance to introduce more cases (semesters, trimesters, etc)
        else:
            working_df['date'] = pd.to_datetime(working_df['date'])
            working_df['year'] = working_df['date'].dt.year
            working_df = working_df.drop(columns='date')

            # Group by year and get stats
            grouped_df = working_df.groupby('year').agg([np.mean, np.var, np.std]).reset_index()

        return grouped_df


    def paint_plots(df: pd.DataFrame, freq="monthly"):
        if freq == "monthly":
            x = df['month']
        else:
            x = df['year']
        
        y1 = df[df.columns[0]] 
        y2 = df[df.columns[1]]
        y3 = df[df.columns[2]]
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(x, y1)
        ax.plot(x, y2)
        ax.plot(x, y3)

        return fig

def main():
    connector = APIConnector()

    df_madrid = connector.get_data_meteo_api("Madrid")
    df_londres = connector.get_data_meteo_api("London")
    df_rio = connector.get_data_meteo_api("Rio")

    madrid = APIConnector.calc_stats(df_madrid)
    londres = APIConnector.calc_stats(df_londres)
    rio = APIConnector.calc_stats(df_rio)

    madrid_fig = APIConnector.paint_plots(madrid)
    londres_fig = APIConnector.paint_plots(londres)
    rio_fig = APIConnector.paint_plots(rio)

    


if __name__ == "__main__":
    main()
