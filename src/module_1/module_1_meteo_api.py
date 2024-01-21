import requests
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


# def main():
#     raise NotImplementedError

# if __name__ == "__main__":
#     main()


def connect_to_api(url, method="GET", headers=None, params=None, data=None):
    """
    Connect to an API using the requests library.

    :param url: URL of the API endpoint.
    :param method: HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE', etc.). Default is 'GET'.
    :param headers: Dictionary of HTTP headers to send with the request.
    :param params: Dictionary of URL parameters to append to the URL.
    :param data: Dictionary of data to send in the body of the request (for POST, PUT, DELETE).
    :return: Response object from requests.
    """
    try:
        response = requests.request(
            method, url, headers=headers, params=params, data=data
        )
        if response.status_code == 429:
            # Extract the Retry-After header and sleep for that duration
            retry_after = int(
                response.headers.get("Retry-After", 10)
            )  # Default to 60 seconds if header is missing
            print(f"Rate limit hit. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
        else:
            return response
    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")


def get_data_meteo_api(city, VARIABLES):
    """
    Connect to an API using the requests library.

    :param city: URL of the API endpoint.
    """

    # Define Climate API URL
    API_URL = "https://climate-api.open-meteo.com/v1/climate?"

    # Define coordinate of available cities
    COORDINATES = {
        "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
        "London": {"latitude": 51.507351, "longitude": -0.127758},
        "Rio": {"latitude": -22.906847, "longitude": -43.172896},
    }

    # Define variables to retrieve from API
    # VARIABLES = ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"]

    # Define API parameters
    params = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": "1950-01-01",
        "end_date": "2050-12-31",
        "models": [
            "CMCC_CM2_VHR4",
            "FGOALS_f3_H",
            "HiRAM_SIT_HR",
            "MRI_AGCM3_2_S",
            "EC_Earth3P_HR",
            "MPI_ESM1_2_XR",
            "NICAM16_8S",
        ],
        "daily": VARIABLES,
    }

    # Call API
    response = connect_to_api(API_URL, params=params)
    data = response.json()

    # Create a DataFrame with the retrieved data
    columns_df = list(data["daily_units"].keys())

    dicts = {}
    for column in columns_df:
        dicts[column] = data["daily"][column]

    df_city_climate = pd.DataFrame(dicts)
    df_city_climate["city"] = city

    # Save dataframe as csv file
    data_path = os.getcwd() + "/data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    save_path = f"{data_path}/{str.lower(city)}_climate.csv"
    df_city_climate.to_csv(save_path, index=False)

    return df_city_climate


def calculate_meteo_mean_max_min_values(df_meteo_0, variables):
    df_meteo = df_meteo_0.set_index("time")

    for var in variables:
        columns_var = [col for col in df_meteo.columns if var in col]

        # Get mean, max and mean values of all models available per variable.
        # Join them together in a single dataframe

        df_join = pd.concat(
            [
                pd.DataFrame(
                    df_meteo[columns_var].mean(axis=1), columns=[var + "_avg"]
                ),
                pd.DataFrame(df_meteo[columns_var].max(axis=1), columns=[var + "_max"]),
                pd.DataFrame(df_meteo[columns_var].min(axis=1), columns=[var + "_min"]),
            ],
            axis=1,
        )

        df_meteo = pd.concat([df_meteo, df_join], axis=1)

    # Select only the columns of interest
    columns_final = ["time", "city"] + [
        col
        for col in df_meteo.columns
        if col.endswith("_avg") or col.endswith("_max") or col.endswith("_min")
    ]
    df_meteo = df_meteo.reset_index()[columns_final]

    return df_meteo


def process_data_meteo(cities, variables):
    # Connect to meteo api or load csv file with data for each city. Concat all info in a single dataframe
    list_all_cities = []

    for city in cities:
        data_path = f"{os.getcwd()}/data/{str.lower(city)}_climate.csv"

        # Check if data is available or connect to API to download it
        if os.path.isfile(data_path):
            df_city_0 = pd.read_csv(data_path, index_col=False)
        else:
            df_city_0 = get_data_meteo_api(city, variables)

        # Call function to calculate mean, max and min values per meteo variable
        df_city = calculate_meteo_mean_max_min_values(
            df_meteo_0=df_city_0, variables=variables
        )
        columns = df_city.columns

        # Aggregate monthly data
        df_city["time"] = pd.to_datetime(df_city["time"], format="%d/%m/%Y")
        df_city["time"] = df_city["time"].dt.to_period("Y")
        df_city = df_city.groupby(["time", "city"]).agg(["mean"]).reset_index()

        # Reset column names to avoid 'mean' header
        df_city.columns = columns

        # Append resulting dataframe in list
        list_all_cities.append(df_city)

    df_all_cities = pd.concat(list_all_cities)

    return df_all_cities


def plot_meteo_variables(cities, variables):
    df_data = process_data_meteo(cities, variables)

    for city in cities:
        for var in variables:
            df = df_data[df_data["city"] == city]

            x = df["time"].astype(str).values
            y_mean = df[var + "_avg"].values
            y_max = df[var + "_max"].values
            y_min = df[var + "_min"].values

            # Plot
            plt.figure(figsize=(16, 10), dpi=80)
            plt.ylabel(var, fontsize=16)
            plt.plot(x, y_mean, color="white", lw=2)
            plt.fill_between(x, y_min, y_max, color="#3F5D7D")

            # Decorations
            # Lighten borders
            plt.gca().spines["top"].set_alpha(0)
            plt.gca().spines["bottom"].set_alpha(1)
            plt.gca().spines["right"].set_alpha(0)
            plt.gca().spines["left"].set_alpha(1)
            plt.xticks(x[::5], [str(d) for d in x[::5]], fontsize=12)
            # plt.title("Daily Order Quantity of Brazilian Retail with Error Bands (95% confidence)", fontsize=20)

            # Axis limits
            plt.xlim(x.min(), x.max())

            plt.grid(axis="y", color="black", linestyle="--", linewidth=0.5)

            plt.show()


cities = ["Madrid", "London", "Rio"]
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"]


plot_meteo_variables(cities, VARIABLES)

### Aplicar Black y el otro

### Añadir tipos a las funciones

### Gestionar el connect fail de la api

### Gestionar si metes una ciudad del que no tienes las coordenadas
