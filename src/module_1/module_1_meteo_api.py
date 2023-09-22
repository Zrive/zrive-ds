import requests
import matplotlib.pyplot as plt
import numpy as np

# import time

API_URL = "https://climate-api.open-meteo.com/v1/climate?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = "temperature_2m_mean,precipitation_sum,soil_moisture_0_to_10cm_mean"
MODELS = (
    "CMCC_CM2_VHR4,FGOALS_f3_H,HiRAM_SIT_HR,"
    "MRI_AGCM3_2_S,EC_Earth3P_HR,MPI_ESM1_2_XR,NICAM16_8S"
)


def get_data_meteo_api(city, start_date, end_date):
    city_dict = COORDINATES[city]
    latitude = city_dict["latitude"]
    longitude = city_dict["longitude"]

    api_url = (
        f"https://climate-api.open-meteo.com/v1/climate?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&models={MODELS}"
        f"&daily={VARIABLES}"
    )

    return api_url


def api_request(api_url):
    response = requests.get(api_url)

    if response.status_code == 200:
        return response


def plot_mean_and_std(data, time, data_key):
    mean_value = np.mean(data)
    # dispersion = abs(data - mean_value)  

    plt.figure(figsize=(10, 6))
    plt.scatter(time, data, label="Dispersion", color="b")
    plt.axhline(y=mean_value, color="r", linestyle="--", label="Mean value")
    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.title(data_key)
    plt.legend()
    plt.show()


def main():
    city = "Madrid"
    start_date = "2020-01-01"
    end_date = "2021-01-01"

    list_variables = VARIABLES.split(",")
    list_models = MODELS.split(",")

    variable = list_variables[1]
    model = list_models[1]
    data_key = f"{variable}_{model}"

    api_url = get_data_meteo_api(city, start_date, end_date)

    response = api_request(api_url)

    if response.status_code == 200:
        data_dict = response.json()
        data = data_dict["daily"][data_key]
        time = data_dict["daily"]["time"]

        plot_mean_and_std(data, time, data_key)


if __name__ == "__main__":
    main()
