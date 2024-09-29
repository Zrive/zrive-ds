import requests
import pandas as pd
from pandas import DataFrame
import plotly.express as px  # type: ignore
import streamlit as st
import time
from typing import Any, Dict

API_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m"]


PLOT_INFO = {
    "temperature_2m": {"color": "red", "y_label": "Mean Temperature (ºC)"},
    "precipitation": {"color": "blue", "y_label": "Accumulated precipitation (mm)"},
    "wind_speed_10m": {"color": "green", "y_label": "Max. Wind speed 10 meters (km/h)"},
}


def get_data_meteo_api(city: str, magnitude: str) -> Dict[str, Any]:
    params = {
        "latitude": str(COORDINATES[city]["latitude"]),
        "longitude": str(COORDINATES[city]["longitude"]),
        "start_date": "2010-01-01",
        "end_date": "2019-12-31",
        "hourly": magnitude,
    }

    r = requests.get(API_URL, params=params)
    print(r.status_code)
    data = r.json()
    return data


# --------------------------------------------
# Generic function for APIs conection
# Not used in the main()
# -------------------------------------------
def make_api_request(url, params=None, headers=None, retries=3, cooldown=10):
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, params=params, headers=headers)
            status_code = response.status_code

            if status_code == 200:
                return response.json()

            elif status_code == 429:  # Rate limit alcanzado
                print(f"Rate limit alcanzado. Esperando {cooldown} segundos...")
                time.sleep(cooldown)

            elif 500 <= status_code < 600:  # Error en el servidor
                print(f"Error en el servidor ({status_code}). Reintentando...")
                attempt += 1
                time.sleep(cooldown)

            else:
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            print(f"Error al realizar la solicitud: {e}")
            attempt += 1
            time.sleep(cooldown)

    raise Exception(f"La solicitud a la API falló después de {retries} intentos.")


# -----------------------
# Params: data = API schema
# Return: data_proc = DataFrame[time,temperature_2m]
# ----------------------
def process_data(data: Dict[str, Any], magnitude: str) -> DataFrame:
    time = data["hourly"]["time"]
    temperature = data["hourly"][magnitude]

    # DataFrame transformation
    df = pd.DataFrame({"time": pd.to_datetime(time), magnitude: temperature})
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    return df


def mean_calc(df: DataFrame, period: str) -> DataFrame:
    mean_df = df.resample(period).mean()
    return mean_df


def sum_calc(df: DataFrame, period: str) -> DataFrame:
    sum_df = df.resample(period).sum()
    return sum_df


def max_calc(df: DataFrame, period: str) -> DataFrame:
    max_df = df.resample(period).max()
    return max_df


def plotting_line(df, magnitude, city):
    color = PLOT_INFO[magnitude]["color"]
    # Create a line plot using plotly express
    fig = px.line(
        df,
        x=df.index,
        y=magnitude,
        markers=True,  # Adds markers to the line plot
        title=f"{city}",
        labels={
            "x": "Date (Quarter)",
            magnitude: f'{PLOT_INFO[magnitude]["y_label"]} ',
        },
    )

    # Customize the layout for the plot
    fig.update_traces(line_color=color)
    fig.update_layout(
        xaxis_title="Date (Quarter)",
        yaxis_title=f'{PLOT_INFO[magnitude]["y_label"]}',
        title_font_size=18,
        xaxis_tickangle=-45,
        autosize=True,
        width=800,
        height=500,
    )

    # fig.show()
    return fig


def main() -> None:
    st.set_page_config(
        layout="wide",
    )

    st.title("Meteorological Data Visualization")
    cols = st.columns(3)
    index = 0
    for city in COORDINATES.keys():
        print(f"Processing data for {city}")
        # Get API data [time, temperature_2m]
        data = get_data_meteo_api(city, VARIABLES[0])
        # Process data
        data_process = process_data(data, VARIABLES[0])
        data_rep = mean_calc(data_process, "3ME")
        # Plotting
        with cols[index % 3]:  # Distribute plots in columns
            fig = plotting_line(data_rep, VARIABLES[0], city)
            st.plotly_chart(fig)

        # Precipitation
        data = get_data_meteo_api(city, VARIABLES[1])
        data_process = process_data(data, VARIABLES[1])
        data_rep = sum_calc(data_process, "3ME")
        with cols[index % 3]:  # Distribute plots in columns
            fig = plotting_line(data_rep, VARIABLES[1], city)
            st.plotly_chart(fig)

        # Wind
        data = get_data_meteo_api(city, VARIABLES[2])
        data_process = process_data(data, VARIABLES[2])
        data_rep = max_calc(data_process, "3ME")
        with cols[index % 3]:  # Distribute plots in columns
            fig = plotting_line(data_rep, VARIABLES[2], city)
            st.plotly_chart(fig)

        index += 1  # Increment index for next column


if __name__ == "__main__":
    main()
