import requests

API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"]
START_DATE = "1950-01-01"
END_DATE = "2050-12-31"


def main():
    print(get_data_meteo_api("Madrid"))


def get_data_meteo_api(city):
    param_city = COORDINATES[city]
    param_city["start_date"] = START_DATE
    param_city["end_date"] = END_DATE
    param_city["daily"] = VARIABLES[0]
    data = api_connection(param_city)
    return data


def api_connection(query_params):
    try:
        response = requests.get(API_URL, params=query_params)
        response.raise_for_status()
        # Additional code will only run if the request is successful
        return response.text
    except requests.exceptions.HTTPError as error:
        print(error)
        # This code will run if there is a 404 error.


if __name__ == "__main__":
    main()
