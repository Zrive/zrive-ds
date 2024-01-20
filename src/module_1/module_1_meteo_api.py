import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors


VARIABLES = ["temperature_2m_mean", "precipitation_sum", "soil_moisture_0_to_10cm_mean"]
MODELS = [
    "CMCC_CM2_VHR4",
    "FGOALS_f3_H",
    "HiRAM_SIT_HR",
    "MRI_AGCM3_2_S",
    "EC_Earth3P_HR",
    "MPI_ESM1_2_XR",
    "NICAM16_8S",
]

years = pd.date_range(start="1950-01-01", end="2050-12-31")

API_URL = "https://climate-api.open-meteo.com/v1/climate"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}


def get_data_meteo_api(city: str) -> None:
    df = model_data(city)
    df["date"] = pd.to_datetime(df["date"])
    df = df.resample("M", on="date").mean()
    plot_climate_data(df, city)
    plt.show()


def call_api(city: str, model: str) -> dict:
    parameters = {
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "start_date": "1950-01-01",
        "end_date": "2050-12-31",
        "model": model,
        "timezone": "Europe/Berlin",
        "daily": VARIABLES,
    }

    response = requests.get(url=API_URL, params=parameters)

    # Check if the status code is 200
    if response.status_code != 200:
        if response.status_code == 429:
            reset_time = int(response.headers.get("x-ratelimit-reset", 0))
            print(
                f"API rate limit exceeded. "
                f"Requests will be available after {reset_time} seconds."
            )
            raise Exception("API rate limit exceeded")
        else:
            print(f"Unexpected status code: {response.status_code}")
            raise Exception("Unexpected status code")

    # Parse the JSON response
    try:
        data = response.json()["daily"]
    except (ValueError, KeyError) as e:
        print(f"Error parsing API response: {e}")
        raise
    return data


def model_data(city: str) -> pd.DataFrame:
    data = {"date": years}

    for model in MODELS:
        results = call_api(city, model)
        data[f"temperature_2m_mean_{model}"] = np.asarray(
            results["temperature_2m_mean"]
        )
        data[f"precipitation_sum_{model}"] = np.asarray(results["precipitation_sum"])
        data[f"soil_moisture_0_to_10cm_mean_{model}"] = np.asarray(
            results["soil_moisture_0_to_10cm_mean"]
        )

    df = pd.DataFrame(data)

    # Calculate mean
    temp_columns = [col for col in df.columns if "temperature_2m_mean" in col]
    precip_columns = [col for col in df.columns if "precipitation_sum" in col]
    soil_moisture_columns = [
        col for col in df.columns if "soil_moisture_0_to_10cm_mean" in col
    ]

    df["temperature_2m_mean"] = df[temp_columns].mean(axis=1)
    df["precipitation_sum"] = df[precip_columns].mean(axis=1)
    df["soil_moisture_0_to_10cm_mean"] = df[soil_moisture_columns].mean(axis=1)

    # Select necessary columns
    df = df[
        [
            "date",
            "temperature_2m_mean",
            "precipitation_sum",
            "soil_moisture_0_to_10cm_mean",
        ]
    ]

    return df


def plot_climate_data(df: pd.DataFrame, city: str) -> plt:
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.rolling(window=12).mean()

    # Create a new figure with two subplots, sharing the x axis
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Create a color palette
    tableau_palette = list(mcolors.TABLEAU_COLORS.values())

    # First subplot
    ax2 = ax1.twinx()
    ax1.plot(
        numeric_df.index,
        numeric_df["temperature_2m_mean"],
        color=tableau_palette[0],
        label="Mean Temperature (ºC)",
    )
    ax2.plot(
        numeric_df.index,
        numeric_df["soil_moisture_0_to_10cm_mean"],
        color=tableau_palette[1],
        label="Mean Soil Moisture 0-10cm (m^3/m^3)",
    )

    # Set the y-axis labels
    ax1.set_ylabel("Mean Temperature (ºC)", color=tableau_palette[0])
    ax2.set_ylabel("Mean Soil Moisture 0-10cm (m^3/m^3)", color=tableau_palette[1])

    # Set the x-axis major ticks to the decades.
    years = mdates.YearLocator(10)
    ax1.xaxis.set_major_locator(years)

    # Add a legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    # Second subplot
    ax3.bar(
        numeric_df.index,
        numeric_df["precipitation_sum"],
        color=tableau_palette[2],
        label="Precipitation Sum (mm)",
    )
    ax3.set_ylabel("Precipitation Sum (mm)", color=tableau_palette[2])

    # Add a legend
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines3, labels3, loc=0)

    # Set the titles
    fig.suptitle(f"Avg. Climate Change Data in {city} (1950-2050)")
    ax1.set_title("Temperature and Soil Moisture")
    ax3.set_title("Precipitation")

    # Rotate the x-axis labels for readability
    plt.xticks(rotation=45)

    return plt


def main():
    for city in COORDINATES.keys():
        get_data_meteo_api(city)


if __name__ == "__main__":
    main()
