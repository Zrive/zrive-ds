""" This is a dummy example """
import pandas as pd
import requests
import matplotlib.pyplot as plt

COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"

class APIConnector:

    lat, long = 0, 0

    def __init__(self, city: str):
        self.lat = COORDINATES[city]["latitude"]
        self.long = COORDINATES[city]["longitude"]
        self.base = "https://climate-api.open-meteo.com/v1/climate?"
        coords = f"latitude{self.lat}&longitude{self.long}"
        date_span = "&start_date=1950-01-01&end_date=2050-12-31"
        # models = ¿? Dudoso
        mode = f"&daily={VARIABLES}"
        self.url = self.base + coords + date_span + mode
       


def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
