

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry



API_URL = "https://archive-api.open-meteo.com/v1/archive"

COORDINATES = {
 "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
 "London": {"latitude": 51.507351, "longitude": -0.127758},
 "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


#def get_data_meteo_api():
    #return
    



#def main():
 
 #raise NotImplementedError

 
#if __name__ == "__main__":
# main()