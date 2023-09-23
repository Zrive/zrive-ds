""" This is a dummy example """
import pandas as pd
import requests
import matplotlib.pyplot as plt
import time

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


def main():
    connector = APIConnector()
    df_madrid = connector.get_data_meteo_api("Madrid")
    df_londres = connector.get_data_meteo_api("London")
    df_rio = connector.get_data_meteo_api("Rio")
    # raise NotImplementedError

if __name__ == "__main__":
    main()
