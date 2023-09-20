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

    API_URL = "https://climate-api.open-meteo.com/v1/climate?"

    def __init__(self, url: str):
        self.API_URL = url 
    
    

        


def main():
    raise NotImplementedError

if __name__ == "__main__":
    main()
