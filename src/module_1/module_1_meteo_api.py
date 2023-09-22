import requests
import matplotlib.pyplot as plt
import numpy as np
import os

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


DATA_FOLDER_NAME = "Plots"

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


def plot_mean_and_std(mean_data, anual_mean_values, time, city, variable, model, plot_folder): 

    y_values = anual_mean_values.values()
    x_values = anual_mean_values.keys()


    init_year = int(list(x_values)[0].split("-")[0])
    end_year = int(list(x_values)[-1].split("-")[0])

    custom_x_labels = [str(year) for year in range(init_year, end_year, 2)]

    plt.figure(figsize=(12, 8))
    plt.scatter(x_values, y_values, label="Dispersion", color="r")
    plt.axhline(y=mean_data, color="b", linestyle="--", label="Mean value")
    plt.xlabel("Time")
    plt.ylabel("Data")
    plt.xticks(fontsize=10)
    plt.xticks(rotation=45)
    plt.xticks(custom_x_labels, custom_x_labels)
    plt.title(f"Yearly mean deviation. City: {city}. Variable measured: {variable}. Model: {model}")
    pic_filename = f"{plot_folder}/{city}_{variable}_{model}.png"
    plt.legend()
    plt.savefig(pic_filename)
    plt.close()


def main():

    start_date = "1950-01-01"
    end_date = "2045-12-31"


    cities = COORDINATES.keys()

    for city in cities:

        plot_folder = f"{DATA_FOLDER_NAME}/{city}"
        if os.path.exists(plot_folder) is False:
            os.makedirs(plot_folder)


        list_variables = VARIABLES.split(",")
        list_models = MODELS.split(",")

        for variable in list_variables:
            for model in list_models:

                #model = "CMCC_CM2_VHR4"
                #variable = "soil_moisture_0_to_10cm_mean"

                print(model, variable)

                data_key = f"{variable}_{model}"

                api_url = get_data_meteo_api(city, start_date, end_date)

                response = api_request(api_url)

                if response.status_code == 200:

                    data_dict = response.json()
                    data = data_dict["daily"][data_key]
                    time = data_dict["daily"]["time"]
                    try:
                        mean_data = np.mean(data)
                    except TypeError:
                        print(f"{variable} for {model} does not exist. Skip")
                        break


                    sum_year_values = {}
                    count_year_values = {}

                    for value,date in zip(data, time):
                        year = date.split("-")[0] # Output: [2020, 01, 01]

                        if year in sum_year_values.keys():
                            sum_year_values[year] += value
                            count_year_values[year] +=1

                        else:
                            sum_year_values[year] = value
                            count_year_values[year] = 1

                    anual_mean_values = {year: (sum/count) for year, sum, count in zip(sum_year_values.keys(), sum_year_values.values(), count_year_values.values())} # yearly dispersion from mean

                    plot_mean_and_std(mean_data, anual_mean_values, time, city, variable, model, plot_folder)

                else:
                    print(response.status_code)


if __name__ == "__main__":
    main()
